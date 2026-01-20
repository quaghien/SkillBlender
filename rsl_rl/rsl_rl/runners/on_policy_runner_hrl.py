# SPDX-License-Identifier: BSD-3-Clause
# HRL On-Policy Runner with Curriculum Learning and Enhanced Logging

import time
import os
from collections import deque
import statistics

import torch
import numpy as np
import wandb

from rsl_rl.algorithms import PPO_HRL
from rsl_rl.modules import ActorCriticHRL


class OnPolicyRunnerHRL:
    """
    Training orchestrator for HRL with curriculum learning and comprehensive logging.
    
    Differences from standard OnPolicyRunner:
    - Uses ActorCriticHRL network
    - Uses PPO_HRL algorithm with curriculum
    - Logs HRL-specific metrics:
      - Skill histogram
      - Skill switch rate
      - Separated entropy (skill, command, action)
      - Per-task rewards and metrics
      - Curriculum parameters (K, ε, τ)
    """
    
    def __init__(self,
                 env,
                 train_cfg,
                 log_dir=None,
                 device='cpu',
                 **kwargs):
        
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        
        self.use_vision = self.env.cfg.sensor.enable_sensor if hasattr(self.env.cfg, 'sensor') else False
        
        # Actor-Critic initialization
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCriticHRL
        if hasattr(self.env, 'obs_context_len'):
            obs_context_len = self.env.obs_context_len
        else:
            obs_context_len = 1
        
        args = kwargs.get('args', None)
        actor_critic = actor_critic_class(
            self.env.num_obs,
            num_critic_obs,
            self.env.num_actions,
            obs_context_len=obs_context_len,
            **self.policy_cfg,
            device=self.device,
            args=args,
        ).to(self.device)
        
        # PPO_HRL initialization
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO_HRL
        self.alg = alg_class(
            actor_critic,
            device=self.device,
            **self.alg_cfg
        )
        
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        
        # Init storage
        if self.use_vision:
            obs_vision_shape = [obs_context_len, 3, self.env.cfg.sensor.camera.height, self.env.cfg.sensor.camera.width] if obs_context_len != 1 else [3, self.env.cfg.sensor.camera.height, self.env.cfg.sensor.camera.width]
        else:
            obs_vision_shape = None
        obs_shape = [obs_context_len, self.env.num_obs] if obs_context_len != 1 else [self.env.num_obs]
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            obs_shape,
            obs_vision_shape,
            [self.env.num_privileged_obs],
            [self.env.num_actions]
        )
        
        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        
        # HRL-specific logging buffers
        self.skill_histogram_buffer = deque(maxlen=50)
        self.skill_switch_rate_buffer = deque(maxlen=50)
        
        # Reward per skill tracking
        self.num_skills = self.alg.actor_critic.num_skills
        self.skill_names = [name.replace('h1_', '') for name in self.alg.actor_critic.skill_names]
        self.skill_reward_sum = torch.zeros(self.num_skills, device=self.device)
        self.skill_step_count = torch.zeros(self.num_skills, device=self.device)
        
        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """
        Main training loop with curriculum learning.
        
        Flow per iteration:
        1. Update curriculum parameters
        2. Collect rollout (60 steps × 4096 envs)
        3. Compute returns (GAE)
        4. PPO update (2 epochs × 4 mini-batches)
        5. Log metrics
        6. Save checkpoint
        """
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length)
            )
        
        obs = self.env.get_observations()
        if self.use_vision:
            obs_vision = self.env.get_visual_observations().to(self.device)
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        
        self.alg.actor_critic.train()  # Training mode
        
        ep_infos = []
        ep_metrics = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        donebuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        self.tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_iter = self.current_learning_iteration
        
        # Track stage transitions
        last_stage = 1
        
        for it in range(self.start_iter, self.tot_iter):
            start = time.time()
            
            # === UPDATE CURRICULUM ===
            curriculum_params = self.alg.update_curriculum(it)
            
            # Log stage transition
            if curriculum_params['stage'] != last_stage:
                print(f"\n{'='*70}")
                print(f"🎯 STAGE TRANSITION: Stage {last_stage} → Stage {curriculum_params['stage']}")
                print(f"{'='*70}")
                print(f"  K: {curriculum_params['K']}")
                print(f"  ε: {curriculum_params['epsilon']:.4f}")
                print(f"  τ: {curriculum_params['tau']:.2f}")
                print(f"  c_ent_skill: {curriculum_params['c_ent_skill']:.4f}")
                print(f"  lr_cmd_ratio: {curriculum_params['lr_cmd_ratio']:.2f}")
                print(f"{'='*70}\n")
                last_stage = curriculum_params['stage']
            
            # === ROLLOUT ===
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    if self.use_vision:
                        actions = self.alg.act((obs, obs_vision), (critic_obs, obs_vision))
                    else:
                        actions = self.alg.act(obs, critic_obs)
                    
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    if self.use_vision:
                        obs_vision = self.env.get_visual_observations().to(self.device)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    
                    # Track reward per skill
                    current_skills = self.alg.actor_critic.current_skill  # [num_envs]
                    for skill_id in range(self.num_skills):
                        skill_mask = (current_skills == skill_id)
                        if skill_mask.any():
                            self.skill_reward_sum[skill_id] += rewards[skill_mask].sum()
                            self.skill_step_count[skill_id] += skill_mask.sum()
                    
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        if 'episode_metrics' in infos:
                            ep_metrics.append(infos['episode_metrics'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        donebuffer.append(len(new_ids) / self.env.num_envs)
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
            
            stop = time.time()
            collection_time = stop - start
            
            # === LEARNING STEP ===
            start = stop
            if self.use_vision:
                self.alg.compute_returns((critic_obs, obs_vision))
            else:
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            self.alg.storage.clear()  # Reset storage for next rollout
            stop = time.time()
            learn_time = stop - start
            
            # === LOGGING ===
            if self.log_dir is not None:
                self.log_hrl(locals(), curriculum_params)
            
            # === SAVE CHECKPOINT ===
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)), curriculum_params)
            
            ep_infos.clear()
            ep_metrics.clear()
            
            self.current_learning_iteration += 1
        
        # Final save
        self.save(
            os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)),
            curriculum_params
        )
    
    def log_hrl(self, locs, curriculum_params, width=80, pad=35):
        """
        Comprehensive HRL logging with:
        - Standard PPO metrics
        - HRL-specific metrics (skill histogram, switch rate, separated entropy)
        - Per-task rewards and metrics
        - Curriculum parameters
        """
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']
        
        ep_string = f''
        wandb_dict = {}
        
        # Episode infos
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_dict['Episode/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        
        if locs['ep_metrics']:
            for key in locs['ep_metrics'][0]:
                info = []
                for ep_metric in locs['ep_metrics']:
                    info.append(ep_metric[key])
                value = np.mean(info)
                wandb_dict['Metric/' + key] = value
                ep_string += f"""{f'Mean episode metric {key}:':>{pad}} {value:.4f}\n"""
        
        # Standard metrics
        std = self.alg.actor_critic.std.cpu().detach().numpy()
        mean_std = std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        
        wandb_dict['Loss/value_function'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        
        # HRL-specific entropy logging
        wandb_dict['Entropy/total'] = self.alg.mean_entropy
        wandb_dict['Entropy/skill'] = self.alg.mean_entropy_skill
        wandb_dict['Entropy/command'] = self.alg.mean_entropy_command
        wandb_dict['Entropy/action'] = self.alg.mean_entropy_action
        
        # Curriculum parameters
        wandb_dict['Curriculum/stage'] = curriculum_params['stage']
        wandb_dict['Curriculum/K'] = curriculum_params['K']
        wandb_dict['Curriculum/epsilon'] = curriculum_params['epsilon']
        wandb_dict['Curriculum/tau'] = curriculum_params['tau']
        wandb_dict['Curriculum/c_ent_skill'] = curriculum_params['c_ent_skill']
        wandb_dict['Curriculum/lr_cmd_ratio'] = curriculum_params['lr_cmd_ratio']
        
        # HRL skill metrics
        skill_hist = self.alg.actor_critic.get_skill_histogram()
        if skill_hist is not None:
            for i, prob in enumerate(skill_hist):
                wandb_dict[f'Skill/histogram_{self.skill_names[i]}'] = prob.item()
            self.skill_histogram_buffer.append(skill_hist.cpu().numpy())
        
        switch_rate = self.alg.actor_critic.get_skill_switch_rate()
        wandb_dict['Skill/switch_rate'] = switch_rate
        wandb_dict['Skill/expected_switch_rate'] = 1.0 / curriculum_params['K']
        self.skill_switch_rate_buffer.append(switch_rate)
        
        # Reward per skill
        for skill_id in range(self.num_skills):
            if self.skill_step_count[skill_id] > 0:
                avg_reward = (self.skill_reward_sum[skill_id] / self.skill_step_count[skill_id]).item()
                wandb_dict[f'SkillReward/{self.skill_names[skill_id]}'] = avg_reward
        # Reset skill reward tracking for next iteration
        self.skill_reward_sum.zero_()
        self.skill_step_count.zero_()
        
        # Per-task rewards (if env has get_task_stats)
        if hasattr(self.env, 'get_task_stats'):
            task_stats = self.env.get_task_stats()
            for key, value in task_stats.items():
                if key not in ['rewards', 'counts']:  # Skip raw dicts
                    wandb_dict[key] = value
        
        # Performance
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection_time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        wandb_dict['Std/mean_std'] = mean_std
        
        # Per-dim std
        for i, std_val in enumerate(self.alg.actor_critic.std):
            wandb_dict[f'Std/std_dim_{i}'] = std_val
        
        # Training metrics
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            wandb_dict['Train/dones'] = statistics.mean(locs['donebuffer'])
        
        # Log to wandb
        wandb.log(wandb_dict, step=locs['it'])
        
        # === CONSOLE OUTPUT ===
        str_title = f" \033[1m Learning iteration {locs['it']}/{self.tot_iter} \033[0m "
        
        if len(locs['rewbuffer']) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_title.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                f"""\n"""
                f"""{'--- HRL Metrics ---':>{pad}}\n"""
                f"""{'Stage:':>{pad}} {curriculum_params['stage']}\n"""
                f"""{'K (option duration):':>{pad}} {curriculum_params['K']}\n"""
                f"""{'Epsilon (exploration):':>{pad}} {curriculum_params['epsilon']:.4f}\n"""
                f"""{'Tau (temperature):':>{pad}} {curriculum_params['tau']:.2f}\n"""
                f"""{'Skill entropy:':>{pad}} {self.alg.mean_entropy_skill:.4f}\n"""
                f"""{'Command entropy:':>{pad}} {self.alg.mean_entropy_command:.4f}\n"""
                f"""{'Action entropy:':>{pad}} {self.alg.mean_entropy_action:.4f}\n"""
                f"""{'Switch rate:':>{pad}} {switch_rate:.4f} (target: {1.0/curriculum_params['K']:.4f})\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_title.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"""
            )
        
        log_string += ep_string
        
        # ETA
        eta = self.tot_time / (locs['it'] + 1 - self.start_iter) * (locs['num_learning_iterations'] - (locs['it'] - self.start_iter))
        eta_hrs, eta_mins, eta_secs = eta // 3600, (eta % 3600) // 60, eta % 60
        tot_hrs, tot_mins, tot_secs = self.tot_time // 3600, (self.tot_time % 3600) // 60, self.tot_time % 60
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Experiment name:':>{pad}} {self.cfg['experiment_name']}\n"""
            f"""{'Run name:':>{pad}} {self.cfg['run_name']}\n"""
            f"""{'Progress:':>{pad}} {self.start_iter}+{locs['it']-self.start_iter}/{self.tot_iter-self.start_iter}+{self.start_iter}\n"""
            f"""{'Device:':>{pad}} {self.device}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {tot_hrs:.0f} hrs {tot_mins:.0f} mins {tot_secs:.1f} s\n"""
            f"""{'ETA:':>{pad}} {eta_hrs:.0f} hrs {eta_mins:.0f} mins {eta_secs:.1f} s\n"""
        )
        print(log_string)
    
    def save(self, path, infos=None):
        """Save checkpoint with curriculum state"""
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'curriculum_state': {
                'iteration': self.alg.curriculum.current_iteration,
                'stage': self.alg.curriculum.current_stage,
            },
            'infos': infos,
        }, path)
    
    def load(self, path, load_optimizer=True):
        """Load checkpoint with curriculum state"""
        try:
            loaded_dict = torch.load(path)
        except:
            loaded_dict = torch.load(path, map_location="cuda:0")
        
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'], strict=False)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        
        # Restore curriculum state
        if 'curriculum_state' in loaded_dict:
            curr_state = loaded_dict['curriculum_state']
            self.alg.curriculum.current_iteration = curr_state['iteration']
            self.alg.curriculum.current_stage = curr_state['stage']
            print(f"[OnPolicyRunnerHRL] Restored curriculum: iteration={curr_state['iteration']}, stage={curr_state['stage']}")
        
        return loaded_dict['infos']
    
    def get_inference_policy(self, device=None):
        """Get inference policy for evaluation"""
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act
