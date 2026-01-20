# SPDX-License-Identifier: BSD-3-Clause
# HRL Hierarchical Actor-Critic Network
# Using PRETRAINED low-level skills: walking, reaching, squatting, stepping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from copy import deepcopy
import os
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils.helpers import class_to_dict


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


class ActorCriticHRL(nn.Module):
    """
    Hierarchical RL Actor-Critic for 8-task meta-learning.
    Uses PRETRAINED low-level skill policies (walking, reaching, squatting, stepping).
    
    Architecture:
        1. High-Level Policy: obs → [commands_per_skill, blend_weights]
        2. LOW-LEVEL SKILLS: 4 pretrained frozen policies
        3. Blend: action = sum(weight_i * skill_i(command_i))
        4. Critic: privileged_obs → value
    
    Skills (4 pretrained):
        0: walking   - Locomotion skill
        1: reaching  - Arm reaching skill  
        2: squatting - Lower body movement
        3: stepping  - Foot placement skill
    """
    
    is_recurrent = False
    
    def __init__(self,
                 num_actor_obs,          # Observation dim
                 num_critic_obs,         # Privileged observation dim
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
                 command_dim=3,
                 num_dofs=19,
                 skill_dict=None,
                 **kwargs):
        
        super(ActorCriticHRL, self).__init__()
        
        self.device = device
        self.num_actions = num_actions
        self.obs_context_len = obs_context_len
        self.args = args
        self.frame_stack = frame_stack
        self.command_dim = command_dim  # Original command dim from env (e.g., 3)
        self.num_dofs = num_dofs
        
        # Default skill_dict if not provided
        if skill_dict is None:
            skill_dict = {
                'h1_walking': {
                    'experiment_name': 'h1_walking',
                    'load_run': '0000_best',
                    'checkpoint': -1,
                    'low_high': None
                },
                'h1_reaching': {
                    'experiment_name': 'h1_reaching',
                    'load_run': '0000_best',
                    'checkpoint': -1,
                    'low_high': None
                },
                'h1_squatting': {
                    'experiment_name': 'h1_squatting',
                    'load_run': '0000_best',
                    'checkpoint': -1,
                    'low_high': None
                },
                'h1_stepping': {
                    'experiment_name': 'h1_stepping',
                    'load_run': '0000_best',
                    'checkpoint': -1,
                    'low_high': None
                },
            }
        
        # Activation
        activation_fn = get_activation(activation)
        
        # ====================================================================
        # LOAD PRETRAINED LOW-LEVEL SKILLS
        # ====================================================================
        self._get_low_level_policies(args, device, skill_dict)
        
        # Calculate output dimension for high-level policy
        # Output = command for each skill + blend weights for each skill
        num_output = self._get_num_output(frame_stack)
        
        # ====================================================================
        # HIGH-LEVEL POLICY (Actor)
        # obs → [commands_per_skill, blend_weights_per_skill]
        # ====================================================================
        actor_layers = []
        mlp_input_dim_a = num_actor_obs
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation_fn)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_output))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation_fn)
        self.actor = nn.Sequential(*actor_layers)
        
        # ====================================================================
        # CRITIC: Value Network
        # privileged_obs → value
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
        
        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        
        # Disable args validation for speedup
        Normal.set_default_validate_args = False
        
        # HRL state tracking
        self.step_in_option = None
        self.current_skill = None  # Dominant skill for logging
        self.K = 10  # Option duration (will be updated)
        self.epsilon = 0.18
        self.tau = 2.0
        
        # Cache for entropy
        self._cached_entropy = None
        self.entropy_skill = None
        self.entropy_command = None
        self.entropy_action = None
        
        print(f"\n{'='*70}")
        print(f"[ActorCriticHRL] Initialized with PRETRAINED Low-Level Skills")
        print(f"{'='*70}")
        print(f"Skills loaded: {self.skill_names}")
        print(f"High-level output dim: {num_output}")
        print(f"  - Commands per skill: {[self.env_cfg_list[i].env.command_dim for i in range(self.num_skills)]}")
        print(f"  - Blend weights: {self.num_skills} x {self.num_dofs} = {self.num_skills * self.num_dofs}")
        print(f"Actor Network: {self.actor}")
        print(f"Critic Network: {self.critic}")
        print(f"{'='*70}\n")
    
    def _get_one_policy(self, args, device, task, experiment_name, load_run, checkpoint):
        """Load a single pretrained low-level policy"""
        from legged_gym.utils import task_registry
        from rsl_rl.modules import ActorCritic
        from legged_gym.utils.helpers import get_load_path
        
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
    
    def _get_num_output(self, frame_stack=1):
        """Calculate high-level policy output dimension"""
        # Output = sum of command dims for each skill + blend weights (num_skills * num_dofs)
        num_output = 0
        for i in range(self.num_skills):
            num_output += self.env_cfg_list[i].env.command_dim
        num_output += self.num_skills * self.num_dofs  # Blend weights
        return num_output * frame_stack
    
    def _construct_skill_obs(self, hrl_obs, command, skill_idx):
        """Construct observation for a specific skill from HRL observation.
        
        HRL obs (105D): 
            state(69) = base_lin_vel*2.0(3) + base_ang_vel*0.25(3) + gravity(3) + 
                        commands(3) + dof_pos*1.0(19) + dof_vel*0.05(19) + actions(19)
            + goal(14) + mask(14) + task_id(8)
        
        Skill obs format: command + dof_pos + dof_vel + actions + base_ang_vel + base_euler
        Scales: dof_pos*1.0, dof_vel*0.05, ang_vel*1.0, euler*1.0
        
        Args:
            hrl_obs: [B, 105] - HRL observation
            command: [B, cmd_dim] - Generated command for this skill
            skill_idx: Index of the skill
        
        Returns:
            skill_obs: [B, skill_obs_dim] - Properly formatted observation for the skill
        """
        B = hrl_obs.shape[0]
        
        # Extract components from HRL obs (state portion is first 69 dims)
        # HRL state: base_lin_vel*2(3) + base_ang_vel*0.25(3) + gravity(3) + commands(3) + dof_pos(19) + dof_vel(19) + actions(19)
        base_lin_vel_scaled = hrl_obs[:, 0:3]     # 3, scaled by 2.0
        base_ang_vel_scaled = hrl_obs[:, 3:6]     # 3, scaled by 0.25
        gravity = hrl_obs[:, 6:9]                  # 3 (projected gravity)
        # hrl_commands = hrl_obs[:, 9:12]          # 3 (skip, use generated command)
        dof_pos = hrl_obs[:, 12:31]                # 19, already scaled by 1.0 ✓
        dof_vel = hrl_obs[:, 31:50]                # 19, already scaled by 0.05 ✓
        actions = hrl_obs[:, 50:69]                # 19
        
        # Rescale ang_vel: HRL has *0.25, skill expects *1.0
        # So: skill_ang_vel = hrl_ang_vel * (1.0 / 0.25) = hrl_ang_vel * 4.0
        base_ang_vel = base_ang_vel_scaled * 4.0  # [B, 3]
        
        # Use projected gravity as euler proxy (both represent orientation)
        # Note: This is an approximation. projected_gravity = [gx, gy, gz] in body frame
        # base_euler = [roll, pitch, yaw]. They encode similar info but not identical.
        # For pretrained policies, this might cause some degradation but should work.
        base_euler = gravity  # [B, 3]
        
        # Clamp command if low_high specified
        low_high = self.low_high_list[skill_idx]
        if low_high is not None:
            low, high = low_high
            if isinstance(low, (list, tuple)) and not isinstance(low, torch.Tensor):
                low = torch.tensor(low, device=command.device, dtype=command.dtype)
            if isinstance(high, (list, tuple)) and not isinstance(high, torch.Tensor):
                high = torch.tensor(high, device=command.device, dtype=command.dtype)
            command = torch.clamp(command, low, high)
        
        # Construct skill-specific observation
        # All skills use format: command + dof_pos + dof_vel + actions + base_ang_vel + base_euler
        skill_obs = torch.cat([
            command,        # cmd_dim (varies per skill)
            dof_pos,        # 19
            dof_vel,        # 19
            actions,        # 19
            base_ang_vel,   # 3
            base_euler,     # 3
        ], dim=-1)  # Total: cmd_dim + 63
        
        return skill_obs
    
    def _actor(self, observations):
        """
        High-level policy forward pass.
        
        Returns:
            actions_mean: Blended action from all skills
            masks: Blend weights [B, num_skills, num_dofs]
        """
        raw_mean = self.actor(observations)  # [B, num_output]
        
        # Split into commands and weights
        # raw_mean = [cmd_skill_0, cmd_skill_1, ..., weights_skill_0, weights_skill_1, ...]
        total_command_dim = sum([self.env_cfg_list[i].env.command_dim for i in range(self.num_skills)])
        
        input_to_low_level_policies = raw_mean[:, :total_command_dim]
        mask_to_low_level_policies = raw_mean[:, total_command_dim:]
        
        # Parse weights into per-skill masks
        masks = []
        for i in range(self.num_skills):
            mask = mask_to_low_level_policies[:, i*self.num_dofs:(i+1)*self.num_dofs]
            masks.append(mask)
        masks = torch.stack(masks, dim=1)  # [B, num_skills, num_dofs]
        masks = torch.softmax(masks, dim=1)  # Softmax across skills
        
        # Run each low-level skill and blend
        means = []
        for i in range(self.num_skills):
            # Get command for this skill
            prev_command_dim_sum = sum([self.env_cfg_list[j].env.command_dim for j in range(i)])
            curr_command_dim = self.env_cfg_list[i].env.command_dim
            command_i = input_to_low_level_policies[:, prev_command_dim_sum:prev_command_dim_sum+curr_command_dim]
            
            # Construct proper observation for this skill
            obs_for_skill_i = self._construct_skill_obs(observations, command_i, i)
            
            # Forward through pretrained policy (frozen)
            with torch.no_grad():
                action_i = self.policy_list[i](obs_for_skill_i)
            
            # Weight by blend mask
            weighted_action = action_i * masks[:, i]
            means.append(weighted_action)
        
        # Blend all skills
        actions_mean = sum(means)
        
        # Track dominant skill for logging
        if self.current_skill is None or self.current_skill.shape[0] != masks.shape[0]:
            self.current_skill = torch.zeros(masks.shape[0], dtype=torch.long, device=self.device)
        
        # Dominant skill = max average weight
        avg_weights = masks.mean(dim=-1)  # [B, num_skills]
        self.current_skill = avg_weights.argmax(dim=-1)  # [B]
        
        return {
            "actions_mean": actions_mean,
            "masks": masks
        }
    
    def init_option_state(self, num_envs):
        """Initialize option state"""
        self.step_in_option = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.current_skill = torch.zeros(num_envs, dtype=torch.long, device=self.device)
    
    def update_curriculum_params(self, K, epsilon, tau):
        """Update curriculum parameters (for compatibility)"""
        self.K = K
        self.epsilon = epsilon
        self.tau = tau
    
    def reset(self, dones=None):
        """Reset (for compatibility)"""
        pass
    
    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, observations):
        """Update action distribution from observations"""
        mean = self._actor(observations)['actions_mean']
        self.distribution = Normal(mean, mean*0. + self.std)
    
    def act(self, observations, **kwargs):
        """Sample actions from policy"""
        if self.obs_context_len != 1:
            observations = observations[..., -1, :]
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        """Get log probability of actions"""
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        """Deterministic action for inference"""
        if self.obs_context_len != 1:
            observations = observations[..., -1, :]
        result = self._actor(observations)
        actions_mean = result['actions_mean']
        # Store blend weights for visualization
        self.blend_weights = result['masks']  # [B, num_skills, num_dofs]
        return actions_mean
    
    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """Compute value from critic"""
        value = self.critic(critic_observations)
        return value
    
    # ========================================================================
    # HRL-specific methods for logging compatibility
    # ========================================================================
    
    def get_skill_histogram(self):
        """Get skill distribution for logging"""
        if self.current_skill is None:
            return None
        counts = torch.bincount(self.current_skill, minlength=self.num_skills).float()
        return counts / counts.sum()
    
    def get_skill_switch_rate(self):
        """Get skill switch rate (approximate from dominant skill changes)"""
        # For blending approach, this is less meaningful
        # Return 1/K as expected rate
        return 1.0 / self.K if self.K > 0 else 0.0
    
    def freeze(self):
        """Freeze all parameters"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze high-level policy parameters (keep low-level frozen)"""
        # Only unfreeze actor and critic
        for param in self.actor.parameters():
            param.requires_grad = True
        for param in self.critic.parameters():
            param.requires_grad = True
        self.std.requires_grad = True
