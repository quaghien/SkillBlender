# SPDX-License-Identifier: BSD-3-Clause
# HRL Actor-Critic V2 - Unified Version
# 
# Combines:
# - Pretrained skills loading from actor_critic_hrl.py
# - Hold skill logic (PPO-correct) from actor_critic_hrl_simple.py
# - Hard option (1 skill per step, no blend)
# - Proper log_prob calculation with is_held masking

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from copy import deepcopy
import os
from rsl_rl.utils import class_to_dict


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    else:
        return nn.ELU()


class HoldTimeController:
    """
    Hold skill for K steps to reduce jitter.
    PPO-correct: khi held, log_prob_gating = 0 (không phạt decision cũ)
    """
    
    def __init__(self, hold_steps=5, num_envs=4096, device='cuda'):
        self.hold_steps = hold_steps
        self.num_envs = num_envs
        self.device = device
        self.timer = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.skill_latched = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.is_held = torch.zeros(num_envs, dtype=torch.bool, device=device)
    
    def should_sample(self):
        """Returns mask where timer == 0 (should sample new skill)"""
        return self.timer == 0
    
    def update(self, skill_new, sample_mask):
        """
        Update skill based on sampling mask.
        Returns: (skill_executed, is_held_mask)
        """
        # Decrement timer where active
        active = self.timer > 0
        self.timer[active] -= 1
        
        # Where we sampled new: update skill and reset timer
        self.skill_latched[sample_mask] = skill_new[sample_mask]
        self.timer[sample_mask] = self.hold_steps
        
        # Track which envs are "held" (didn't sample this step)
        self.is_held = ~sample_mask
        
        return self.skill_latched.clone(), self.is_held.clone()
    
    def set_hold_steps(self, K):
        """Update hold duration (from curriculum)"""
        self.hold_steps = K
    
    def reset(self, env_ids):
        """Reset on episode done"""
        self.timer[env_ids] = 0
        self.skill_latched[env_ids] = 0
        self.is_held[env_ids] = False
    
    def reset_all(self, num_envs):
        """Reset all envs (e.g. when batch size changes)"""
        self.num_envs = num_envs
        self.timer = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.skill_latched = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.is_held = torch.zeros(num_envs, dtype=torch.bool, device=self.device)


class ActorCriticHRL(nn.Module):
    """
    HRL Actor-Critic V2 - Unified Version
    
    Key Features:
    1. HARD OPTION: Chọn 1 skill, giữ K steps (default K=5)
    2. PRETRAINED SKILLS: Load 4 frozen low-level policies
    3. PPO-CORRECT: log_prob = 0 khi is_held (không phạt decision cũ)
    4. NO BLEND: Mỗi step chỉ chạy 1 skill, không weighted sum
    
    Architecture:
        High-Level:  obs → [skill_logits(4), commands(sum of cmd_dims)]
        Low-Level:   4 pretrained frozen policies
        Output:      action = skill_i(obs, command_i)
    
    Skills (4 pretrained):
        0: walking   - vx, vy, omega (3D)
        1: reaching  - wrist targets (6D)
        2: squatting - root height (2D)
        3: stepping  - foot targets (3D)
    """
    
    is_recurrent = False
    
    def __init__(self,
                 num_actor_obs,          # Observation dim (105)
                 num_critic_obs,         # Privileged observation dim (303)
                 num_actions,            # 19 (H1 DOFs)
                 num_skills=4,           # 4 pretrained skills
                 obs_context_len=1,
                 actor_hidden_dims=[256, 256, 256],
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 device='cpu',
                 args=None,
                 frame_stack=1,
                 command_dim=3,          # Not used, computed from skills
                 num_dofs=19,
                 skill_dict=None,
                 hold_steps=5,           # Hold skill for K steps
                 **kwargs):
        
        super(ActorCriticHRL, self).__init__()
        
        self.device = device
        self.num_actions = num_actions
        self.obs_context_len = obs_context_len
        self.args = args
        self.frame_stack = frame_stack
        self.num_dofs = num_dofs
        self.hold_steps = hold_steps
        
        # Activation
        activation_fn = get_activation(activation)
        
        # ====================================================================
        # LOAD PRETRAINED LOW-LEVEL SKILLS (FROZEN)
        # ====================================================================
        if skill_dict is None:
            # Default skill_dict with low_high ranges matching original training
            # low_high: clamp range for high-level output before passing to skill
            skill_dict = {
                'h1_walking': {
                    'experiment_name': 'h1_walking',
                    'load_run': '0000_best',
                    'checkpoint': -1,
                    'low_high': (-1, 1)  # Clamp to [-1,1] then scale by [2,2,1]
                },
                'h1_reaching': {
                    'experiment_name': 'h1_reaching',
                    'load_run': '0000_best',
                    'checkpoint': -1,
                    'low_high': (-1, 1)  # wrist diff range
                },
                'h1_squatting': {
                    'experiment_name': 'h1_squatting',
                    'load_run': '0000_best',
                    'checkpoint': -1,
                    'low_high': (-1, 1)  # height diff range
                },
                'h1_stepping': {
                    'experiment_name': 'h1_stepping',
                    'load_run': '0000_best',
                    'checkpoint': -1,
                    'low_high': (-1, 1)  # feet diff range
                },
            }
        
        self._get_low_level_policies(args, device, skill_dict)
        
        # Compute total command dim from loaded skills
        self.total_command_dim = sum([self.env_cfg_list[i].env.command_dim for i in range(self.num_skills)])
        
        # ====================================================================
        # HIGH-LEVEL ACTOR: Skill-Aware Architecture (Option A)
        # ====================================================================
        # Stage 1: obs → hidden → skill_logits (select which skill)
        # Stage 2: hidden + skill_embed → commands (conditioned on skill)
        
        # Shared encoder: obs → hidden
        self.hidden_dim = actor_hidden_dims[-1]  # Last hidden dim
        encoder_layers = []
        mlp_input_dim_a = num_actor_obs
        encoder_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        encoder_layers.append(activation_fn)
        for l in range(len(actor_hidden_dims) - 1):
            encoder_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
            encoder_layers.append(activation_fn)
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Skill selection head: hidden → skill_logits
        self.skill_head = nn.Linear(self.hidden_dim, self.num_skills)
        
        # Skill embedding: skill_id → embed_dim
        self.skill_embed_dim = 16  # Compact embedding
        self.skill_embedding = nn.Embedding(self.num_skills, self.skill_embed_dim)
        
        # Command head: hidden + skill_embed → commands (CONDITIONED on selected skill)
        self.command_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.skill_embed_dim, 128),
            activation_fn,
            nn.Linear(128, self.total_command_dim)
        )
        
        # Legacy: keep self.actor for compatibility (not used in forward)
        self.actor = None
        
        # ====================================================================
        # CRITIC: privileged_obs → value
        # ====================================================================
        critic_layers = []
        mlp_input_dim_c = num_critic_obs
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation_fn)
        self.critic = nn.Sequential(*critic_layers)
        
        # ====================================================================
        # COMMAND STD (learnable, separate per skill)
        # ====================================================================
        self.command_log_std = nn.Parameter(torch.zeros(self.total_command_dim))
        
        # ====================================================================
        # HOLD TIME CONTROLLER
        # ====================================================================
        self.hold_controller = None  # Initialized in init_option_state
        
        # ====================================================================
        # STATE TRACKING
        # ====================================================================
        self.current_skill = None       # [num_envs] current executed skill
        self.step_in_option = None      # Legacy compatibility
        
        # Curriculum params
        self.K = hold_steps
        self.epsilon = 0.18
        self.tau = 2.0
        
        # Distribution caching for log_prob
        self.last_gating_dist = None
        self.last_command_dist = None
        self.last_info = None
        
        # Compatibility with old code
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False
        
        print(f"\n{'='*70}")
        print(f"[ActorCriticHRL V2] Initialized with SKILL-AWARE Command Head")
        print(f"{'='*70}")
        print(f"Skills loaded: {self.skill_names}")
        print(f"  - Command dims per skill: {[self.env_cfg_list[i].env.command_dim for i in range(self.num_skills)]}")
        print(f"  - Total command dim: {self.total_command_dim}")
        print(f"Architecture:")
        print(f"  - Encoder: obs({num_actor_obs}) → hidden({self.hidden_dim})")
        print(f"  - Skill head: hidden → skill_logits({self.num_skills})")
        print(f"  - Skill embed: skill_id → embed({self.skill_embed_dim})")
        print(f"  - Command head: hidden+embed → commands({self.total_command_dim})")
        print(f"Hold steps (K): {hold_steps}")
        print(f"{'='*70}\n")
    
    def _get_one_policy(self, args, device, task, experiment_name, load_run, checkpoint):
        """Load a single pretrained low-level policy"""
        from legged_gym.utils import task_registry
        from legged_gym.utils.helpers import get_load_path
        from legged_gym import LEGGED_GYM_ROOT_DIR
        from rsl_rl.modules import ActorCritic
        
        # Get skill arguments
        skill_args = deepcopy(args) if args is not None else type('Args', (), {
            'task': task,
            'experiment_name': experiment_name,
            'load_run': load_run,
            'checkpoint': checkpoint
        })()
        
        skill_args.task = task
        skill_args.experiment_name = experiment_name
        skill_args.load_run = load_run
        skill_args.checkpoint = checkpoint
        
        # Get config
        skill_env_cfg, skill_train_cfg = task_registry.get_cfgs(
            name=skill_args.task, 
            load_run=skill_args.load_run, 
            experiment_name=skill_args.experiment_name
        )
        
        # Create policy
        skill_policy = ActorCritic(
            skill_env_cfg.env.num_observations,
            skill_env_cfg.env.num_privileged_obs,
            skill_env_cfg.env.num_actions,
            obs_context_len=1,
            **class_to_dict(skill_train_cfg)["policy"]
        ).to(device)
        
        # Load weights
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', skill_train_cfg.runner.experiment_name)
        skill_resume_path = get_load_path(log_root, load_run=skill_args.load_run, checkpoint=skill_args.checkpoint)
        print(f"Loading {skill_args.task} policy from: {skill_resume_path}")
        
        try:
            loaded_dict = torch.load(skill_resume_path, map_location=device)
        except:
            loaded_dict = torch.load(skill_resume_path, map_location="cuda:0")
        
        skill_policy.load_state_dict(loaded_dict['model_state_dict'])
        skill_train_cfg.runner.resume_path = skill_resume_path
        
        # Freeze and extract actor only
        skill_policy.freeze()
        skill_policy = skill_policy.actor
        
        return skill_policy, skill_env_cfg, skill_train_cfg
    
    def _get_low_level_policies(self, args, device, skill_dict):
        """Load all pretrained low-level policies"""
        self.skill_names = list(skill_dict.keys())
        self.policy_list = []
        self.env_cfg_list = []
        self.train_cfg_list = []
        self.low_high_list = []
        
        for key, value in skill_dict.items():
            policy, env_cfg, train_cfg = self._get_one_policy(
                args, device, key, 
                value['experiment_name'], 
                value['load_run'], 
                value['checkpoint']
            )
            self.policy_list.append(policy)
            self.env_cfg_list.append(env_cfg)
            self.train_cfg_list.append(train_cfg)
            self.low_high_list.append(value.get('low_high', None))
        
        self.num_skills = len(self.policy_list)
    
    def _construct_skill_obs(self, hrl_obs, command, skill_idx):
        """Construct observation for a specific skill from HRL observation.
        
        This follows the ORIGINAL ActorCriticHierarchical approach:
        - High-level outputs raw command (clamped to low_high)
        - Command is placed directly in skill observation (no error calculation)
        - Skills receive command in their expected format
        
        HRL obs (105D): 
            state(69) = base_lin_vel*2.0(3) + base_ang_vel*0.25(3) + gravity(3) + 
                        commands(3) + dof_pos*1.0(19) + dof_vel*0.05(19) + actions(19)
            + goal(14) + mask(14) + task_id(8)
        
        Skill obs format: command + dof_pos + dof_vel + actions + base_ang_vel + base_euler
        
        SCALING for skills (matching original training):
            - Walking (skill 0): command = [vel_x, vel_y, yaw] * [2.0, 2.0, 1.0]
            - Reaching (skill 1): command = raw 14D (wrist targets, clamped to [-1,1])
            - Squatting (skill 2): command = raw 1D (height target, clamped to [-1,1])
            - Stepping (skill 3): command = raw 4D (feet targets, clamped to [-1,1])
        """
        B = hrl_obs.shape[0]
        
        # Extract from HRL obs
        base_lin_vel_scaled = hrl_obs[:, 0:3]     # scaled by 2.0
        base_ang_vel_scaled = hrl_obs[:, 3:6]     # scaled by 0.25
        gravity = hrl_obs[:, 6:9]                  # projected gravity (proxy for euler)
        dof_pos = hrl_obs[:, 12:31]                # 19, scaled by 1.0
        dof_vel = hrl_obs[:, 31:50]                # 19, scaled by 0.05
        actions = hrl_obs[:, 50:69]                # 19
        
        # Rescale ang_vel: HRL has *0.25, skill expects *1.0
        # (Skills use: base_ang_vel * obs_scales.ang_vel where ang_vel=1.0)
        base_ang_vel = base_ang_vel_scaled * 4.0
        
        # For base_euler: skills use base_euler_xyz * obs_scales.quat (=1.0)
        # We use projected_gravity as proxy since HRL env doesn't have real euler
        # This is an approximation: gravity ≈ [-roll, -pitch, -1] for small angles
        # TODO: If needed, h1_hrl.py should compute and include real base_euler_xyz
        base_euler = gravity
        
        # ====================================================================
        # SKILL-SPECIFIC COMMAND PROCESSING (matching original training)
        # ====================================================================
        
        # Clamp command if low_high specified
        low_high = self.low_high_list[skill_idx]
        if low_high is not None:
            low, high = low_high
            if isinstance(low, (list, tuple)):
                low = torch.tensor(low, device=command.device, dtype=command.dtype)
            if isinstance(high, (list, tuple)):
                high = torch.tensor(high, device=command.device, dtype=command.dtype)
            command = torch.clamp(command, low, high)
        
        # Apply skill-specific scaling
        if skill_idx == 0:  # Walking
            # Walking was trained with: commands[:,:3] * commands_scale
            # where commands_scale = [obs_scales.lin_vel, obs_scales.lin_vel, obs_scales.ang_vel]
            #                       = [2.0, 2.0, 1.0]
            # HRL actor outputs in tanh range [-1, 1], so we scale to match training:
            # After clamp to (-2, 2), we apply the same scale as original
            walking_scale = torch.tensor([2.0, 2.0, 1.0], device=command.device, dtype=command.dtype)
            command = command * walking_scale
        
        # For reaching/squatting/stepping: command is raw (error signal interpretation)
        # The original ActorCriticHierarchical also passes raw command without computing error
        # Skills were trained with error = current - target, but here we pass raw command
        # The high-level learns to output values that work as "pseudo-error" signals
        
        # Construct skill-specific observation
        skill_obs = torch.cat([
            command,        # cmd_dim (varies per skill)
            dof_pos,        # 19
            dof_vel,        # 19
            actions,        # 19
            base_ang_vel,   # 3
            base_euler,     # 3
        ], dim=-1)
        
        return skill_obs
    
    def init_option_state(self, num_envs):
        """Initialize hold controller and state tracking"""
        self.hold_controller = HoldTimeController(
            hold_steps=self.K,
            num_envs=num_envs,
            device=self.device
        )
        self.current_skill = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.step_in_option = torch.zeros(num_envs, dtype=torch.long, device=self.device)
    
    def update_curriculum_params(self, K, epsilon, tau):
        """Update curriculum parameters"""
        self.K = K
        self.epsilon = epsilon
        self.tau = tau
        
        # Update hold controller
        if self.hold_controller is not None:
            self.hold_controller.set_hold_steps(K)
    
    def _actor_forward(self, observations, is_training=True):
        """
        High-level policy forward pass with SKILL-AWARE Command Head.
        
        Architecture:
        1. Encoder: obs → hidden
        2. Skill head: hidden → skill_logits → sample skill
        3. Command head: hidden + skill_embed → commands (CONDITIONED on skill)
        4. Execute ONLY the selected skill
        
        Args:
            observations: [B, obs_dim]
            is_training: If True, use hold logic; else sample freely
        
        Returns:
            dict with:
                - actions_mean: [B, num_actions]
                - skill_exec: [B] executed skill ID
                - is_held: [B] whether skill was held
                - command_sampled: [B, total_cmd_dim] sampled commands
                - gating_probs: [B, num_skills] gating probabilities
        """
        B = observations.shape[0]
        
        # Initialize hold controller if needed
        if self.hold_controller is None or self.hold_controller.num_envs != B:
            self.init_option_state(B)
        
        # ====================================================================
        # Stage 1: Encode observation and select skill
        # ====================================================================
        hidden = self.encoder(observations)  # [B, hidden_dim]
        gating_logits = self.skill_head(hidden)  # [B, num_skills]
        
        # Apply temperature to gating
        gating_logits_scaled = gating_logits / self.tau
        gating_probs = F.softmax(gating_logits_scaled, dim=-1)
        gating_dist = Categorical(probs=gating_probs)
        self.last_gating_dist = gating_dist
        
        # Skill selection with hold logic (BEFORE command generation)
        if is_training and self.hold_controller is not None:
            sample_mask = self.hold_controller.should_sample()
            
            # Epsilon-greedy exploration
            if self.epsilon > 0:
                explore_mask = torch.rand(B, device=self.device) < self.epsilon
                sample_mask = sample_mask | explore_mask
            
            skill_new = gating_dist.sample()
            skill_exec, is_held = self.hold_controller.update(skill_new, sample_mask)
        else:
            # Inference: just sample
            skill_exec = gating_dist.sample()
            is_held = torch.zeros(B, dtype=torch.bool, device=self.device)
        
        self.current_skill = skill_exec
        
        # ====================================================================
        # Stage 2: Generate commands CONDITIONED on selected skill
        # ====================================================================
        skill_embed = self.skill_embedding(skill_exec)  # [B, embed_dim]
        command_input = torch.cat([hidden, skill_embed], dim=-1)  # [B, hidden_dim + embed_dim]
        command_means = self.command_head(command_input)  # [B, total_cmd_dim]
        
        # Sample command
        command_std = torch.exp(self.command_log_std)
        command_dist = Normal(command_means, command_std)
        command_sampled = command_dist.sample()
        self.last_command_dist = command_dist
        
        # Execute SINGLE skill (HARD OPTION - no blend)
        actions_mean = torch.zeros(B, self.num_actions, device=self.device)
        
        for skill_id in range(self.num_skills):
            mask = (skill_exec == skill_id)
            if not mask.any():
                continue
            
            # Get command for this skill
            prev_cmd_dim = sum([self.env_cfg_list[j].env.command_dim for j in range(skill_id)])
            curr_cmd_dim = self.env_cfg_list[skill_id].env.command_dim
            command_i = command_sampled[mask, prev_cmd_dim:prev_cmd_dim + curr_cmd_dim]
            
            # Construct observation for skill
            obs_for_skill = self._construct_skill_obs(observations[mask], command_i, skill_id)
            
            # Forward through pretrained policy (frozen)
            with torch.no_grad():
                action_i = self.policy_list[skill_id](obs_for_skill)
            
            actions_mean[mask] = action_i
        
        # Compute skill entropy for diversity bonus
        skill_entropy = gating_dist.entropy()  # [B] - higher = more diverse
        
        # Store info for log_prob
        self.last_info = {
            'skill_exec': skill_exec,
            'is_held': is_held,
            'command_sampled': command_sampled,
            'gating_probs': gating_probs,
            'skill_entropy': skill_entropy,  # Add entropy
        }
        
        return {
            'actions_mean': actions_mean,
            'skill_exec': skill_exec,
            'is_held': is_held,
            'command_sampled': command_sampled,
            'gating_probs': gating_probs,
            'skill_entropy': skill_entropy,  # Return entropy
        }
    
    def act(self, observations, **kwargs):
        """Sample actions from policy"""
        if self.obs_context_len != 1:
            observations = observations[..., -1, :]
        
        result = self._actor_forward(observations, is_training=True)
        actions_mean = result['actions_mean']
        
        # Add action noise
        self.distribution = Normal(actions_mean, self.std)
        actions = self.distribution.sample()
        
        return actions
    
    def get_actions_log_prob(self, actions):
        """
        PPO-CORRECT log_prob:
        - gating: log_prob = 0 if is_held (don't penalize old decision)
        - command: only log_prob of used slice
        - action: standard log_prob (for low-level noise)
        """
        if self.last_info is None:
            raise RuntimeError("get_actions_log_prob called before act()")
        
        skill_exec = self.last_info['skill_exec']
        is_held = self.last_info['is_held']
        command_sampled = self.last_info['command_sampled']
        
        B = skill_exec.shape[0]
        
        # 1. Gating log_prob (0 if held)
        log_prob_gating = self.last_gating_dist.log_prob(skill_exec)
        log_prob_gating = log_prob_gating * (~is_held).float()  # Zero out held
        
        # 2. Command log_prob (only used slice per skill)
        log_prob_cmd = torch.zeros(B, device=self.device)
        for skill_id in range(self.num_skills):
            mask = (skill_exec == skill_id)
            if not mask.any():
                continue
            
            prev_cmd_dim = sum([self.env_cfg_list[j].env.command_dim for j in range(skill_id)])
            curr_cmd_dim = self.env_cfg_list[skill_id].env.command_dim
            
            cmd_used = command_sampled[mask, prev_cmd_dim:prev_cmd_dim + curr_cmd_dim]
            mean_slice = self.last_command_dist.mean[mask, prev_cmd_dim:prev_cmd_dim + curr_cmd_dim]
            std_slice = self.last_command_dist.stddev[mask, prev_cmd_dim:prev_cmd_dim + curr_cmd_dim]
            
            log_prob_slice = Normal(mean_slice, std_slice).log_prob(cmd_used).sum(dim=-1)
            log_prob_cmd[mask] = log_prob_slice
        
        # 3. Action log_prob (standard for low-level noise)
        log_prob_action = self.distribution.log_prob(actions).sum(dim=-1)
        
        # Total log_prob
        total_log_prob = log_prob_gating + log_prob_cmd + log_prob_action
        return torch.clamp(total_log_prob, -100, 100)
    
    def act_inference(self, observations):
        """Deterministic action for inference"""
        if self.obs_context_len != 1:
            observations = observations[..., -1, :]
        
        result = self._actor_forward(observations, is_training=False)
        return result['actions_mean']
    
    def evaluate(self, critic_obs, **kwargs):
        """Evaluate value function"""
        return self.critic(critic_obs)
    
    def reset(self, dones=None):
        """Reset on episode done"""
        if dones is not None and dones.any() and self.hold_controller is not None:
            env_ids = dones.nonzero(as_tuple=False).flatten()
            self.hold_controller.reset(env_ids)
    
    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        if self.distribution is not None:
            return self.distribution.mean
        return torch.zeros(1, device=self.device)
    
    @property
    def action_std(self):
        return self.std
    
    @property
    def entropy(self):
        """Total entropy: gating + command + action"""
        if self.last_gating_dist is None:
            return torch.zeros(1, device=self.device)
        
        # Gating entropy
        ent_gating = self.last_gating_dist.entropy()
        
        # Command entropy (full, not sliced for simplicity)
        ent_command = self.last_command_dist.entropy().sum(dim=-1)
        
        # Action entropy
        ent_action = self.distribution.entropy().sum(dim=-1)
        
        return ent_gating + ent_command + ent_action
    
    def get_skill_switch_rate(self):
        """Calculate skill switch rate (1 - hold_rate)"""
        if self.hold_controller is None:
            return 0.0
        
        # Switch rate = fraction of envs that sampled new skill this step
        # is_held = True means held (didn't switch), so switch = ~is_held
        if self.last_info is not None and 'is_held' in self.last_info:
            is_held = self.last_info['is_held']
            switch_rate = (~is_held).float().mean().item()
            return switch_rate
        return 1.0 / self.K  # Default to theoretical rate
