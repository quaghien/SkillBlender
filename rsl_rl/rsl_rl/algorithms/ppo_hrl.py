# SPDX-License-Identifier: BSD-3-Clause
# HRL PPO with Curriculum Learning
# 2-Stage Curriculum: Stage 1 (explore skills) → Stage 2 (refine commands)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from rsl_rl.storage import RolloutStorage
# Note: ActorCriticHRL is passed in, no need to import here (avoids circular import)


class CurriculumController:
    """
    Manages 2-stage curriculum learning for HRL training.
    
    Stage 1 (0 - 20k iterations):
        - K = 10 (long option duration)
        - ε = 0.18 → 0 (linear decay)
        - τ = 2.0 → 1.0 (linear decay)
        - c_ent_skill = 0.05 (constant, for skill diversity)
        - lr = fixed
        
    Incremental Task Training:
        - Each task gets 1000 iterations
        - Focus task = 70%, old tasks share 30% equally
    """
    
    def __init__(self,
                 total_iterations=8000,
                 num_tasks=8,
                 iters_per_task=1000,
                 focus_ratio=0.7,
                 K_start=5,
                 K_end=5,
                 epsilon_start=0.18,
                 epsilon_end=0.0,
                 tau_start=2.0,
                 tau_end=1.0,
                 c_ent_skill=0.05,
                 **kwargs):  # Accept extra args for compatibility
        
        self.total_iterations = total_iterations
        self.num_tasks = num_tasks
        self.iters_per_task = iters_per_task
        self.focus_ratio = focus_ratio  # 0.7 = 70% for focus task
        
        # Parameters (linear decay over entire training)
        self.K_start = K_start
        self.K_end = K_end
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.c_ent_skill = c_ent_skill  # Constant for skill diversity
        
        # Current state
        self.current_iteration = 0
        
        print(f"\n{'='*70}")
        print(f"[CurriculumController] Incremental Task Training")
        print(f"{'='*70}")
        print(f"Total iterations: {total_iterations}")
        print(f"Tasks: {num_tasks}, Iters/task: {iters_per_task}")
        print(f"Focus ratio: {focus_ratio} (old tasks share {1-focus_ratio})")
        print(f"  K: {K_start} → {K_end} (linear)")
        print(f"  ε: {epsilon_start} → {epsilon_end} (linear)")
        print(f"  τ: {tau_start} → {tau_end} (linear)")
        print(f"  c_ent_skill: {c_ent_skill} (constant)")
        print(f"{'='*70}\n")
    
    def update(self, iteration):
        """Update current iteration"""
        self.current_iteration = iteration
    
    def _get_alpha(self):
        """Get interpolation factor [0, 1]"""
        if self.total_iterations <= 0:
            return 1.0
        return min(self.current_iteration / self.total_iterations, 1.0)
    
    def get_current_phase(self):
        """Get current phase (0-indexed) = which task is focus"""
        return min(self.current_iteration // self.iters_per_task, self.num_tasks - 1)
    
    def get_active_tasks(self):
        """Get list of active task IDs"""
        phase = self.get_current_phase()
        return list(range(phase + 1))
    
    def get_task_weights(self):
        """
        Get sampling weights for each task.
        Focus task = focus_ratio (70%), old tasks share (1 - focus_ratio) equally.
        
        Returns: list of 8 weights (sum = 1.0)
        """
        phase = self.get_current_phase()
        weights = [0.0] * self.num_tasks
        
        if phase == 0:
            # Only task 0
            weights[0] = 1.0
        else:
            # Old tasks share (1 - focus_ratio) equally
            old_task_weight = (1 - self.focus_ratio) / phase
            for i in range(phase):
                weights[i] = old_task_weight
            # Focus task gets focus_ratio
            weights[phase] = self.focus_ratio
        
        return weights
    
    def get_K(self):
        """Get current option duration K (linear decay)"""
        alpha = self._get_alpha()
        K_float = self.K_start + alpha * (self.K_end - self.K_start)
        return int(np.round(K_float))
    
    def get_epsilon(self):
        """Get current exploration epsilon (linear decay)"""
        alpha = self._get_alpha()
        return self.epsilon_start + alpha * (self.epsilon_end - self.epsilon_start)
    
    def get_tau(self):
        """Get current temperature tau (linear decay)"""
        alpha = self._get_alpha()
        return self.tau_start + alpha * (self.tau_end - self.tau_start)
    
    def get_c_ent_skill(self):
        """Get skill entropy coefficient (constant)"""
        return self.c_ent_skill
    
    def get_all_params(self):
        """Get all curriculum parameters as dict"""
        phase = self.get_current_phase()
        active_tasks = self.get_active_tasks()
        weights = self.get_task_weights()
        return {
            'K': self.get_K(),
            'epsilon': self.get_epsilon(),
            'tau': self.get_tau(),
            'c_ent_skill': self.get_c_ent_skill(),
            'phase': phase,
            'active_tasks': active_tasks,
            'focus_task': phase,
            'task_weights': weights,
        }


class PPO_HRL:
    """
    PPO with HRL-specific modifications:
    - Skill-aware architecture
    - Skill entropy bonus (for diversity)
    - Option duration handling
    - Incremental task training curriculum
    """
    
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.001,  # Base entropy (for action/command)
                 c_ent_skill=0.05,    # Skill entropy (constant, for diversity)
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 # HRL curriculum
                 total_iterations=8000,
                 **curriculum_kwargs):
        
        self.device = device
        
        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.transition = RolloutStorage.Transition()
        
        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.c_ent_skill = c_ent_skill  # Constant skill entropy
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.learning_rate = learning_rate
        
        # Curriculum controller (simple linear decay)
        self.curriculum = CurriculumController(
            total_iterations=total_iterations,
            c_ent_skill=c_ent_skill,
            **curriculum_kwargs
        )
        
        # Create optimizer (simple, single learning rate)
        self.optimizer = optim.AdamW(
            self.actor_critic.parameters(),
            lr=learning_rate
        )
        
        print(f"\n[PPO_HRL] Initialized with pretrained low-level skills")
        print(f"  Base entropy: {entropy_coef}")
        print(f"  Skill entropy: {c_ent_skill}")
        print(f"  Learning rate: {learning_rate}\n")
    
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, actor_obs_vision_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env,
            actor_obs_shape, actor_obs_vision_shape,
            critic_obs_shape, action_shape,
            self.device
        )
    
    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()
    
    def update_curriculum(self, iteration):
        """Update curriculum parameters and actor_critic"""
        self.curriculum.update(iteration)
        params = self.curriculum.get_all_params()
        
        # Update actor_critic
        self.actor_critic.update_curriculum_params(
            K=params['K'],
            epsilon=params['epsilon'],
            tau=params['tau']
        )
        
        return params
    
    def act(self, obs, critic_obs):
        """Forward pass through HRL network"""
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        
        # Compute actions using HRL policy
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.std.detach()
        
        # Store obs
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1
            )
        
        # Record transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        # Clone to avoid inference tensor issue
        if isinstance(last_critic_obs, tuple):
            last_critic_obs = tuple(x.clone() for x in last_critic_obs)
        else:
            last_critic_obs = last_critic_obs.clone()
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
    
    def update(self):
        """
        PPO update with HRL-specific loss:
        
        Loss = loss_skill + loss_command + loss_low + loss_value - c_ent*entropy
        
        Where:
        - loss_skill: PPO loss for skill selection (only when sampled)
        - loss_command: PPO loss for command generation (only when sampled)
        - loss_low: PPO loss for low-level actions (always)
        - loss_value: Value function MSE
        - entropy: Weighted sum (skill + command + action)
        """
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_skill_loss = 0
        mean_command_loss = 0
        mean_low_loss = 0
        mean_entropy = 0
        mean_entropy_skill = 0
        mean_entropy_action = 0
        
        # Get current curriculum params
        c_ent_skill = self.curriculum.get_c_ent_skill()
        
        # Debug metrics
        self.debug_ratio_mean = 0
        self.debug_ratio_max = 0
        self.debug_kl_approx = 0
        self.debug_clipfrac = 0
        
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, \
            old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, hrl_info_batch in generator:
            
            # Standard PPO (with pretrained skills, no special HRL log_prob needed)
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0] if hid_states_batch[0] is not None else None)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1] if hid_states_batch[0] is not None else None)
            
            # Entropy components
            entropy_action = self.actor_critic.entropy.mean()  # Action entropy
            
            # Skill entropy from gating distribution (for diversity)
            if hasattr(self.actor_critic, 'last_gating_dist') and self.actor_critic.last_gating_dist is not None:
                entropy_skill = self.actor_critic.last_gating_dist.entropy().mean()
            else:
                entropy_skill = torch.tensor(0.0, device=self.device)
            
            # PPO surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            
            # Debug metrics
            with torch.no_grad():
                self.debug_ratio_mean = ratio.mean().item()
                self.debug_ratio_max = ratio.max().item()
                log_ratio = actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
                self.debug_kl_approx = ((torch.exp(log_ratio) - 1) - log_ratio).mean().item()
                self.debug_clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
            
            # Clipped surrogate
            adv = torch.squeeze(advantages_batch)
            surrogate = ratio * adv
            surrogate_clipped = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
            surrogate_loss = -torch.min(surrogate, surrogate_clipped).mean()
            
            # Value loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            
            # Total loss = surrogate + value - entropy bonuses
            # Skill entropy bonus: encourage diverse skill selection
            # Action entropy bonus: standard exploration
            loss = (
                surrogate_loss +
                self.value_loss_coef * value_loss -
                c_ent_skill * entropy_skill -     # Skill diversity
                self.entropy_coef * entropy_action  # Action exploration
            )
            
            # Gradient update
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Accumulate stats
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_action.item() if isinstance(entropy_action, torch.Tensor) else entropy_action
            mean_entropy_skill += entropy_skill.item()
            mean_entropy_action += entropy_action.item()
        
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_entropy_skill /= num_updates
        mean_entropy_action /= num_updates
        
        # Store for logging
        self.mean_value_loss = mean_value_loss
        self.mean_surrogate_loss = mean_surrogate_loss
        self.mean_entropy = mean_entropy
        self.mean_entropy_skill = mean_entropy_skill
        self.mean_entropy_action = mean_entropy_action
        
        return mean_value_loss, mean_surrogate_loss
