# SPDX-License-Identifier: BSD-3-Clause
# HRL PPO with Curriculum Learning
# 2-Stage Curriculum: Stage 1 (explore skills) → Stage 2 (refine commands)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from rsl_rl.modules import ActorCriticHRL
from rsl_rl.storage import RolloutStorage


class CurriculumController:
    """
    Manages 2-stage curriculum learning for HRL training.
    
    Stage 1 (0 - 20k iterations):
        - K = 10 (long option duration)
        - ε = 0.18 (constant exploration)
        - τ = 2.0 (high temperature)
        - c_ent_skill = 0.02 (high skill entropy bonus)
        - lr_cmd = 0.2 × lr_base (slow command learning)
    
    Stage 2 (20k - 100k iterations):
        - K = 10 → 5 (linear anneal over 2k iterations)
        - ε = 0.18 → 0.0 (linear anneal)
        - τ = 2.0 → 1.0 (linear anneal)
        - c_ent_skill = 0.02 → 0.005 (exp decay, half-life 10k)
        - lr_cmd = lr_base (normal learning)
    """
    
    def __init__(self,
                 stage1_end=20000,
                 total_iterations=100000,
                 K_start=10,
                 K_end=5,
                 epsilon_start=0.18,
                 epsilon_end=0.0,
                 tau_start=2.0,
                 tau_end=1.0,
                 c_ent_skill_start=0.02,
                 c_ent_skill_end=0.005,
                 lr_cmd_ratio_stage1=0.2,
                 lr_cmd_ratio_stage2=1.0):
        
        self.stage1_end = stage1_end
        self.total_iterations = total_iterations
        self.stage2_start = stage1_end
        self.stage2_end = total_iterations
        
        # Parameters
        self.K_start = K_start
        self.K_end = K_end
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.c_ent_skill_start = c_ent_skill_start
        self.c_ent_skill_end = c_ent_skill_end
        self.lr_cmd_ratio_stage1 = lr_cmd_ratio_stage1
        self.lr_cmd_ratio_stage2 = lr_cmd_ratio_stage2
        
        # Current stage
        self.current_stage = 1
        self.current_iteration = 0
        
        # K annealing (over 2k iterations in stage 2)
        self.K_anneal_start = self.stage2_start
        self.K_anneal_end = self.stage2_start + 2000
        
        print(f"\n{'='*70}")
        print(f"[CurriculumController] Initialized 2-Stage Curriculum")
        print(f"{'='*70}")
        print(f"Stage 1: 0 → {stage1_end}")
        print(f"  K={K_start}, ε={epsilon_start}, τ={tau_start}")
        print(f"  c_ent_skill={c_ent_skill_start}, lr_cmd_ratio={lr_cmd_ratio_stage1}")
        print(f"")
        print(f"Stage 2: {stage1_end} → {total_iterations}")
        print(f"  K={K_start}→{K_end} (anneal over {self.K_anneal_end - self.K_anneal_start} iters)")
        print(f"  ε={epsilon_start}→{epsilon_end} (linear)")
        print(f"  τ={tau_start}→{tau_end} (linear)")
        print(f"  c_ent_skill={c_ent_skill_start}→{c_ent_skill_end} (exp decay)")
        print(f"  lr_cmd_ratio={lr_cmd_ratio_stage2}")
        print(f"{'='*70}\n")
    
    def update(self, iteration):
        """Update curriculum parameters based on current iteration"""
        self.current_iteration = iteration
        
        # Determine stage
        if iteration < self.stage1_end:
            self.current_stage = 1
        else:
            self.current_stage = 2
    
    def get_K(self):
        """Get current option duration K"""
        if self.current_stage == 1:
            return self.K_start
        else:
            # Linear anneal from K_start to K_end over 2k iterations
            if self.current_iteration < self.K_anneal_end:
                alpha = (self.current_iteration - self.K_anneal_start) / (self.K_anneal_end - self.K_anneal_start)
                K_float = self.K_start + alpha * (self.K_end - self.K_start)
                return int(np.round(K_float))
            else:
                return self.K_end
    
    def get_epsilon(self):
        """Get current exploration epsilon"""
        if self.current_stage == 1:
            return self.epsilon_start
        else:
            # Linear anneal
            alpha = (self.current_iteration - self.stage2_start) / (self.stage2_end - self.stage2_start)
            alpha = min(alpha, 1.0)
            return self.epsilon_start + alpha * (self.epsilon_end - self.epsilon_start)
    
    def get_tau(self):
        """Get current temperature tau"""
        if self.current_stage == 1:
            return self.tau_start
        else:
            # Linear anneal
            alpha = (self.current_iteration - self.stage2_start) / (self.stage2_end - self.stage2_start)
            alpha = min(alpha, 1.0)
            return self.tau_start + alpha * (self.tau_end - self.tau_start)
    
    def get_c_ent_skill(self):
        """Get current skill entropy coefficient"""
        if self.current_stage == 1:
            return self.c_ent_skill_start
        else:
            # Exponential decay with half-life 10k iterations
            iterations_in_stage2 = self.current_iteration - self.stage2_start
            half_life = 10000
            decay_factor = 0.5 ** (iterations_in_stage2 / half_life)
            c_ent = self.c_ent_skill_start * decay_factor
            return max(c_ent, self.c_ent_skill_end)
    
    def get_lr_cmd_ratio(self):
        """Get command learning rate ratio"""
        if self.current_stage == 1:
            return self.lr_cmd_ratio_stage1
        else:
            return self.lr_cmd_ratio_stage2
    
    def get_all_params(self):
        """Get all curriculum parameters as dict"""
        return {
            'stage': self.current_stage,
            'K': self.get_K(),
            'epsilon': self.get_epsilon(),
            'tau': self.get_tau(),
            'c_ent_skill': self.get_c_ent_skill(),
            'lr_cmd_ratio': self.get_lr_cmd_ratio(),
        }


class PPO_HRL:
    """
    PPO with HRL-specific modifications:
    - 3-part loss: skill + command + low policy
    - Separated entropy bonuses
    - Option duration handling
    - Curriculum learning
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
                 c_ent_skill=0.02,    # Skill entropy (higher for exploration)
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 # HRL curriculum
                 stage1_end=20000,
                 total_iterations=100000,
                 **curriculum_kwargs):
        
        self.device = device
        
        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.actor_critic.parameters()),
            lr=learning_rate
        )
        self.transition = RolloutStorage.Transition()
        
        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.c_ent_skill_base = c_ent_skill
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.learning_rate = learning_rate
        
        # Curriculum controller
        # Remove c_ent_skill_start from kwargs if exists (we use c_ent_skill param)
        curriculum_kwargs.pop('c_ent_skill_start', None)
        curriculum_kwargs.pop('c_ent_skill', None)  # Also remove if passed
        self.curriculum = CurriculumController(
            stage1_end=stage1_end,
            total_iterations=total_iterations,
            c_ent_skill_start=c_ent_skill,
            **curriculum_kwargs
        )
        
        # Standard optimizer (no separate param groups needed with pretrained skills)
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.actor_critic.parameters()),
            lr=learning_rate
        )
        
        print(f"\n[PPO_HRL] Initialized with pretrained low-level skills")
        print(f"  Base entropy: {entropy_coef}")
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
        
        # Update command LR
        lr_cmd = self.learning_rate * params['lr_cmd_ratio']
        for param_group in self.optimizer.param_groups:
            if param_group.get('name') == 'command':
                param_group['lr'] = lr_cmd
        
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
        mean_entropy_command = 0
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
            entropy_batch = self.actor_critic.entropy.mean()  # Mean over batch
            entropy_skill = torch.tensor(0.0, device=self.device)
            entropy_command = torch.tensor(0.0, device=self.device)
            entropy_action = entropy_batch  # Already scalar
            weight_sample = 1.0
            
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
            
            # Total loss with HRL entropy
            # Use HIGHER entropy bonus for skills (encourage exploration)
            # Use LOWER entropy bonus for commands/actions (allow convergence)
            loss = (
                surrogate_loss +
                self.value_loss_coef * value_loss -
                c_ent_skill * entropy_skill * weight_sample -  # Higher weight for skill
                self.entropy_coef * entropy_command * weight_sample -  # Lower for command
                self.entropy_coef * entropy_action  # Lower for action
            )
            
            # Gradient update
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Accumulate stats
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.item() if isinstance(entropy_batch, torch.Tensor) else entropy_batch
            mean_entropy_skill += entropy_skill.item()
            mean_entropy_command += entropy_command.item()
            mean_entropy_action += entropy_action.item()
        
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_entropy_skill /= num_updates
        mean_entropy_command /= num_updates
        mean_entropy_action /= num_updates
        
        # Store for logging
        self.mean_value_loss = mean_value_loss
        self.mean_surrogate_loss = mean_surrogate_loss
        self.mean_entropy = mean_entropy
        self.mean_entropy_skill = mean_entropy_skill
        self.mean_entropy_command = mean_entropy_command
        self.mean_entropy_action = mean_entropy_action
        
        return mean_value_loss, mean_surrogate_loss
