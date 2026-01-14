# Hierarchical RL Actor-Critic - SIMPLIFIED PPO-CORRECT VERSION
# Key principles:
# 1. Hard Option: chọn 1 skill (discrete), không blend
# 2. Command slicing: chỉ sample/logprob dims tương ứng skill
# 3. No EMA in training: tránh làm lệch action distribution
# 4. PPO log_prob = log_prob của ĐÚNG những gì thực thi

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import os


class HoldTimeController:
    """
    Hold skill for N steps to reduce jitter.
    PPO-correct: khi held, log_prob_gating = 0 (không phạt decision cũ)
    """
    
    def __init__(self, hold_steps=3, num_envs=4096, device='cuda'):
        self.hold_steps = hold_steps
        self.num_envs = num_envs
        self.device = device
        self.timer = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.skill_latched = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.is_held = torch.ones(num_envs, dtype=torch.bool, device=device)  # Track if currently held
    
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
    
    def reset(self, env_ids):
        self.timer[env_ids] = 0
        self.skill_latched[env_ids] = 0
        self.is_held[env_ids] = False


class GatingHead(nn.Module):
    """Skill selector: 4 skills"""
    
    def __init__(self, feature_dim=128, num_skills=4):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_skills)
    
    def forward(self, features):
        logits = self.fc(features)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


class CommandHead(nn.Module):
    """Command generator - outputs mean/std for ALL dims, but only relevant slice is used"""
    
    # Clamp log_std to prevent extreme values
    LOG_STD_MIN = -5.0  # std >= 0.0067
    LOG_STD_MAX = 2.0   # std <= 7.39
    
    def __init__(self, feature_dim=128, command_dim=14):
        super().__init__()
        self.fc_mean = nn.Linear(feature_dim, command_dim)
        self.fc_log_std = nn.Linear(feature_dim, command_dim)
    
    def forward(self, features):
        mean = self.fc_mean(features)
        log_std = self.fc_log_std(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std


class ResidualHead(nn.Module):
    """Motor correction - init near-zero"""
    
    # Clamp log_std for stable training
    LOG_STD_MIN = -5.0  # std >= 0.0067
    LOG_STD_MAX = 0.0   # std <= 1.0 (residual should be small)
    
    def __init__(self, feature_dim=128, action_dim=19):
        super().__init__()
        self.fc_mean = nn.Linear(feature_dim, action_dim)
        self.fc_log_std = nn.Linear(feature_dim, action_dim)
        
        # Init near zero
        nn.init.uniform_(self.fc_mean.weight, -1e-4, 1e-4)
        nn.init.zeros_(self.fc_mean.bias)
        nn.init.uniform_(self.fc_log_std.weight, -1e-4, 1e-4)
        nn.init.constant_(self.fc_log_std.bias, -2.0)  # std ~ 0.135
    
    def forward(self, features):
        mean = self.fc_mean(features)
        log_std = self.fc_log_std(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std


class ActorCriticHRLSimple(nn.Module):
    """
    SIMPLIFIED HRL Policy - PPO-correct version
    
    Key changes from original:
    1. Hard option: chỉ chạy 1 skill (không blend)
    2. Command slicing: chỉ sample & logprob dims tương ứng skill
    3. No EMA in training: action_exec = clamp(a_low + residual)
    4. PPO log_prob tính trên (gating, cmd_slice, residual)
    5. COMMAND RANGES: clamp output to valid ranges per skill
    """
    
    # Command slices per skill (total 14D)
    # NOTE: These correspond to the actual training ranges from author's configs!
    SKILL_CMD_SLICES = {
        0: slice(0, 3),    # Walk: vx, vy, omega
        1: slice(3, 9),    # Reach: l_wrist_xyz(3) + r_wrist_xyz(3)
        2: slice(9, 11),   # Squat: root_height(1) + placeholder(1)
        3: slice(11, 14),  # Step: l_foot_xy(2) + r_foot_xy_flag(1)
    }
    SKILL_CMD_DIMS = {0: 3, 1: 6, 2: 2, 3: 3}
    
    # === COMMAND RANGES PER SKILL (from author's training configs) ===
    # These are the VALID ranges the low-level skills were trained on!
    # Going outside these ranges will cause the robot to fall.
    COMMAND_RANGES = {
        # Walking: [vx_min, vx_max, vy_min, vy_max, omega_min, omega_max]
        # NOTE: vx trained up to 2.0 but only stable within [-1, 1]
        0: {
            'min': torch.tensor([-1.0, -1.0, -1.0]),
            'max': torch.tensor([1.0, 1.0, 1.0]),
        },
        # Reaching: [l_x, l_y, l_z, r_x, r_y, r_z] - RELATIVE to default wrist pos
        1: {
            'min': torch.tensor([-0.10, -0.10, -0.25, -0.10, -0.25, -0.25]),
            'max': torch.tensor([0.25, 0.25, 0.25, 0.25, 0.10, 0.25]),
        },
        # Squatting: [root_height, placeholder] - root_height in meters
        2: {
            'min': torch.tensor([0.2, 0.0]),
            'max': torch.tensor([1.1, 1.0]),
        },
        # Stepping: [l_foot_x, l_foot_y, r_foot_x] - RELATIVE feet movement
        # feet_max_radius = 0.25
        3: {
            'min': torch.tensor([-0.25, -0.25, -0.25]),
            'max': torch.tensor([0.25, 0.25, 0.25]),
        },
    }
    
    def __init__(
        self,
        num_obs=105,
        num_critic_obs=303,
        num_actions=19,
        num_skills=4,
        command_dim=14,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        num_envs=4096,
        device='cuda',
        **kwargs
    ):
        super().__init__()
        
        self.num_obs = num_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.num_skills = num_skills
        self.command_dim = command_dim
        self.num_envs = num_envs
        self.device = device
        
        self.is_recurrent = False
        
        # === INITIALIZE COMMAND RANGES ON DEVICE ===
        # Pre-move tensors to device to avoid repeated transfers
        self._cmd_ranges_device = {}
        for skill_id, ranges in self.COMMAND_RANGES.items():
            self._cmd_ranges_device[skill_id] = {
                'min': ranges['min'].to(device),
                'max': ranges['max'].to(device),
            }
        
        # === ACTOR BACKBONE ===
        actor_layers = []
        prev_dim = num_obs
        for dim in actor_hidden_dims:
            actor_layers.extend([nn.Linear(prev_dim, dim), nn.ELU()])
            prev_dim = dim
        self.actor_backbone = nn.Sequential(*actor_layers)
        self.feature_dim = actor_hidden_dims[-1]
        
        # === OUTPUT HEADS ===
        self.gating_head = GatingHead(self.feature_dim, num_skills)
        self.command_head = CommandHead(self.feature_dim, command_dim)
        self.residual_head = ResidualHead(self.feature_dim, num_actions)
        
        # === CRITIC ===
        critic_layers = []
        prev_dim = num_critic_obs
        for dim in critic_hidden_dims:
            critic_layers.extend([nn.Linear(prev_dim, dim), nn.ELU()])
            prev_dim = dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # === CONTROL ===
        self.hold_time = HoldTimeController(hold_steps=3, num_envs=num_envs, device=device)
        
        # EMA for evaluation only (disabled in training)
        self.ema_alpha = 0.6
        self.action_prev = torch.zeros(num_envs, num_actions, device=device)
        self.training_mode = True  # Toggle for EMA
        
        # === STATE FOR LOG PROB ===
        self.last_gating_dist = None
        self.last_command_dist = None  
        self.last_residual_dist = None
        self.last_info = None
        
        # Residual curriculum
        self.current_step = 0
        self.residual_clip = 0.0
        
        # Compatibility
        self.std = nn.Parameter(0.5 * torch.ones(num_actions))
    
    def set_training_step(self, step):
        """Residual curriculum: frozen for first 50k iterations"""
        self.current_step = step
        PHASE2_THRESHOLD = 4_915_200_000  # 50k iterations
        if step < PHASE2_THRESHOLD:
            self.residual_clip = 0.0
        else:
            self.residual_clip = 0.05
    
    def forward(self, obs):
        """
        SIMPLIFIED forward pass - PPO correct:
        1. Gating: sample 1 skill (with hold)
        2. Command: only slice for that skill
        3. Action: only run 1 skill (no blend)
        4. No EMA in training
        """
        batch_size = obs.shape[0]
        is_training = (batch_size == self.num_envs)
        
        # 1. Features
        features = self.actor_backbone(obs)
        
        # 2. Gating
        gating_logits, gating_probs = self.gating_head(features)
        gating_dist = Categorical(probs=gating_probs)
        self.last_gating_dist = gating_dist
        
        # 3. Hold time logic
        if is_training:
            sample_mask = self.hold_time.should_sample()
            skill_new = gating_dist.sample()
            skill_exec, is_held = self.hold_time.update(skill_new, sample_mask)
        else:
            skill_exec = gating_dist.sample()
            is_held = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # 4. Command - sample ALL dims but only USE relevant slice
        cmd_mean, cmd_std = self.command_head(features)
        cmd_dist = Normal(cmd_mean, cmd_std)
        self.last_command_dist = cmd_dist
        command_full_raw = cmd_dist.sample()  # [batch, 14]
        
        # 4b. CLAMP command to valid ranges per skill
        # This is CRITICAL - low-level skills were trained on specific ranges!
        command_full = self._clamp_command_to_valid_ranges(command_full_raw, skill_exec)
        
        # 5. Execute SINGLE skill (HARD OPTION - no blend)
        base_action = self._execute_single_skill(obs, command_full, skill_exec)
        
        # 6. Residual
        res_mean, res_std = self.residual_head(features)
        res_dist = Normal(res_mean, res_std)
        self.last_residual_dist = res_dist
        residual_raw = res_dist.sample()
        
        if self.residual_clip > 0:
            residual = torch.clamp(residual_raw, -self.residual_clip, self.residual_clip)
        else:
            residual = torch.zeros_like(residual_raw)
        
        # 7. Final action (NO EMA in training)
        action_raw = base_action + residual
        action_exec = torch.clamp(action_raw, -1.0, 1.0)
        
        # Apply EMA only in evaluation
        if not self.training_mode and not is_training:
            action_exec = self.ema_alpha * self.action_prev + (1 - self.ema_alpha) * action_exec
            self.action_prev = action_exec.clone()
        
        # 8. Info for PPO
        # NOTE: Store command_full (clamped) for execution, but command_full_raw for log_prob
        # PPO needs log_prob of what was SAMPLED, but we execute the CLAMPED version
        info = {
            'skill_exec': skill_exec,
            'is_held': is_held,
            'command_full': command_full,         # Clamped - used for execution
            'command_full_raw': command_full_raw, # Raw sampled - used for log_prob
            'residual_raw': residual_raw,
            'gating_probs': gating_probs,
        }
        self.last_info = info
        
        return action_exec, None, info
    
    def _execute_single_skill(self, obs, command_full, skill_exec):
        """
        Execute ONLY the selected skill (HARD OPTION).
        No blending - each env runs exactly 1 skill.
        """
        batch_size = obs.shape[0]
        base_action = torch.zeros(batch_size, self.num_actions, device=self.device)
        state = obs[:, :69]  # Extract state from obs
        
        for skill_id in range(self.num_skills):
            mask = (skill_exec == skill_id)
            if not mask.any():
                continue
            
            # Get command slice for this skill
            cmd_slice = self.SKILL_CMD_SLICES[skill_id]
            cmd = command_full[mask, cmd_slice]  # Only relevant dims
            
            # Simple action mapping (replace with loaded policies later)
            action = torch.zeros(mask.sum(), self.num_actions, device=self.device)
            
            if skill_id == 0:  # Walk
                action[:, :10] = cmd[:, :1].expand(-1, 10) * 0.1
            elif skill_id == 1:  # Reach
                action[:, 11:17] = cmd  # Arm joints
            elif skill_id == 2:  # Squat
                action[:, 2:5] = cmd[:, :1].expand(-1, 3) * 0.5
            elif skill_id == 3:  # Step
                action[:, :3] = cmd * 0.2
            
            base_action[mask] = action
        
        return base_action
    
    def _clamp_command_to_valid_ranges(self, command_full, skill_exec):
        """
        Clamp command values to the valid ranges that low-level skills were trained on.
        
        This is CRITICAL because:
        - Walking trained on vx∈[-1,2], vy∈[-1,1], omega∈[-1,1]
        - Reaching trained on wrist deltas within ±0.25m  
        - Squatting trained on root_height∈[0.2, 1.1]m
        - Stepping trained on feet deltas within ±0.25m
        
        If commands go outside these ranges, the robot will fall!
        """
        command_clamped = command_full.clone()
        
        for skill_id in range(self.num_skills):
            mask = (skill_exec == skill_id)
            if not mask.any():
                continue
            
            cmd_slice = self.SKILL_CMD_SLICES[skill_id]
            ranges = self._cmd_ranges_device[skill_id]
            
            # Use pre-initialized device tensors
            min_vals = ranges['min']
            max_vals = ranges['max']
            
            # Clamp the relevant slice
            cmd_vals = command_clamped[mask, cmd_slice]
            cmd_clamped = torch.clamp(cmd_vals, min_vals, max_vals)
            command_clamped[mask, cmd_slice] = cmd_clamped
        
        return command_clamped
    
    def get_log_prob(self, skill_exec, command_full, residual_raw, is_held):
        """
        PPO-CORRECT log_prob:
        - gating: log_prob = 0 if held (don't penalize old decision)
        - command: only log_prob of USED slice
        - residual: full 19D
        """
        batch_size = skill_exec.shape[0]
        
        # 1. Gating log_prob (0 if held)
        log_prob_gating = self.last_gating_dist.log_prob(skill_exec)
        log_prob_gating = log_prob_gating * (~is_held).float()  # Zero out held
        
        # 2. Command log_prob (only used slice)
        log_prob_cmd = torch.zeros(batch_size, device=self.device)
        for skill_id in range(self.num_skills):
            mask = (skill_exec == skill_id)
            if not mask.any():
                continue
            
            cmd_slice = self.SKILL_CMD_SLICES[skill_id]
            cmd_used = command_full[mask, cmd_slice]
            mean_slice = self.last_command_dist.mean[mask, cmd_slice]
            std_slice = self.last_command_dist.stddev[mask, cmd_slice]
            
            log_prob_slice = Normal(mean_slice, std_slice).log_prob(cmd_used).sum(dim=-1)
            log_prob_cmd[mask] = log_prob_slice
        
        # 3. Residual log_prob (full)
        log_prob_res = self.last_residual_dist.log_prob(residual_raw).sum(dim=-1)
        
        total = log_prob_gating + log_prob_cmd + log_prob_res
        return torch.clamp(total, -100, 100)
    
    def get_actions_log_prob(self, actions):
        """Required by PPO - compute log_prob from last forward"""
        if self.last_info is None:
            raise RuntimeError("get_actions_log_prob called before forward()")
        
        # Use command_full_raw (what was sampled) for log_prob calculation
        # NOT command_full (which is clamped)
        command_for_logprob = self.last_info.get('command_full_raw', self.last_info['command_full'])
        
        return self.get_log_prob(
            self.last_info['skill_exec'],
            command_for_logprob,
            self.last_info['residual_raw'],
            self.last_info['is_held']
        )
    
    def evaluate_actions(self, obs, skill_exec, command_full, residual_raw, is_held):
        """
        Re-evaluate for PPO update phase.
        Must recompute distributions from obs.
        """
        # 1. Backbone
        features = self.actor_backbone(obs)
        
        # 2. Gating
        _, gating_probs = self.gating_head(features)
        gating_dist = Categorical(probs=gating_probs)
        
        # 3. Command
        cmd_mean, cmd_std = self.command_head(features)
        
        # 4. Residual
        res_mean, res_std = self.residual_head(features)
        res_dist = Normal(res_mean, res_std)
        
        batch_size = obs.shape[0]
        
        # 5. Log probs
        log_prob_gating = gating_dist.log_prob(skill_exec)
        log_prob_gating = log_prob_gating * (~is_held).float()
        
        log_prob_cmd = torch.zeros(batch_size, device=self.device)
        for skill_id in range(self.num_skills):
            mask = (skill_exec == skill_id)
            if not mask.any():
                continue
            cmd_slice = self.SKILL_CMD_SLICES[skill_id]
            cmd_used = command_full[mask, cmd_slice]
            lp = Normal(cmd_mean[mask, cmd_slice], cmd_std[mask, cmd_slice]).log_prob(cmd_used).sum(-1)
            log_prob_cmd[mask] = lp
        
        log_prob_res = res_dist.log_prob(residual_raw).sum(dim=-1)
        
        log_prob_total = log_prob_gating + log_prob_cmd + log_prob_res
        
        # 6. Entropy
        entropy_gating = gating_dist.entropy()
        
        # Command entropy: only for used dims
        entropy_cmd = torch.zeros(batch_size, device=self.device)
        for skill_id in range(self.num_skills):
            mask = (skill_exec == skill_id)
            if not mask.any():
                continue
            cmd_slice = self.SKILL_CMD_SLICES[skill_id]
            ent = Normal(cmd_mean[mask, cmd_slice], cmd_std[mask, cmd_slice]).entropy().sum(-1)
            entropy_cmd[mask] = ent
        
        entropy_res = res_dist.entropy().sum(dim=-1)
        entropy_total = entropy_gating + entropy_cmd + entropy_res
        
        # 7. Value
        value = self.critic(obs).squeeze(-1)
        
        return log_prob_total, entropy_total, value
    
    def act(self, obs, masks=None, hidden_states=None):
        """Inference only"""
        with torch.no_grad():
            action, _, info = self.forward(obs)
            self.last_info = info
        return action
    
    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)
    
    def evaluate(self, obs, masks=None, hidden_states=None):
        """Return [batch, 1] for PPO"""
        return self.critic(obs)
    
    def reset(self, dones):
        if dones.any():
            env_ids = dones.nonzero(as_tuple=False).flatten()
            self.hold_time.reset(env_ids)
            self.action_prev[env_ids] = 0
    
    @property
    def action_mean(self):
        if self.last_residual_dist is not None:
            return self.last_residual_dist.mean
        return torch.zeros(self.num_envs, self.num_actions, device=self.device)
    
    @property
    def action_std(self):
        if self.last_residual_dist is not None:
            return self.last_residual_dist.stddev
        return torch.ones(self.num_envs, self.num_actions, device=self.device)
    
    @property
    def entropy(self):
        if self.last_gating_dist is None:
            return torch.zeros(1, device=self.device)
        gating_ent = self.last_gating_dist.entropy()
        cmd_ent = self.last_command_dist.entropy().sum(dim=-1)
        res_ent = self.last_residual_dist.entropy().sum(dim=-1)
        return gating_ent + cmd_ent + res_ent


def create_hrl_policy_simple(train_cfg, num_envs, device):
    """Factory function for simplified HRL policy"""
    
    policy = ActorCriticHRLSimple(
        num_obs=105,
        num_critic_obs=303,
        num_actions=19,
        num_skills=4,
        command_dim=14,
        actor_hidden_dims=train_cfg.policy.actor_hidden_dims,
        critic_hidden_dims=train_cfg.policy.critic_hidden_dims,
        num_envs=num_envs,
        device=device,
    )
    
    return policy.to(device)
