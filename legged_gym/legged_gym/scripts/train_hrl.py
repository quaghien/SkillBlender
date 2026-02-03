#!/usr/bin/env python3
"""
Training script for Hierarchical RL (4 Tasks Joint Training)

Usage:
    python train_hrl.py --task h1_hrl --run_name hrl_v1 --num_envs 4096
"""

import os
import sys

# Extract custom args before importing to avoid gymutil error
num_envs_override = -1
max_iterations_override = -1

if "--num_envs" in sys.argv:
    try:
        idx = sys.argv.index("--num_envs")
        num_envs_override = int(sys.argv[idx + 1])
        del sys.argv[idx:idx+2]
    except (IndexError, ValueError):
        pass

if "--max_iterations" in sys.argv:
    try:
        idx = sys.argv.index("--max_iterations")
        max_iterations_override = int(sys.argv[idx + 1])
        del sys.argv[idx:idx+2]
    except (IndexError, ValueError):
        pass

# IMPORTANT: Import isaacgym BEFORE torch
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import torch

# HRL modules are imported via registry (ActorCriticHRL from rsl_rl.modules)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def train_hrl(args):
    """Main training function for HRL"""
    
    # Set num_envs override
    if num_envs_override > 0:
        args.num_envs = num_envs_override
    
    # Initialize wandb
    if HAS_WANDB and args.wandb:
        wandb.init(
            project=args.wandb,
            name=args.run_name,
            entity=getattr(args, 'entity', None),
            config=vars(args),
        )
    
    # Create environment
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    # Create PPO runner (this also creates HRL policy via registry)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
    )
    
    # Get policy from runner (already created by registry with ActorCriticHRL)
    policy = ppo_runner.alg.actor_critic
    device = args.rl_device
    
    # DEBUG: Print checkpoint load status
    print(f"\n>>> CHECKPOINT STATUS <<<")
    print(f"Current learning iteration: {ppo_runner.current_learning_iteration}")
    print(f"Log dir: {ppo_runner.log_dir}")
    print(f"This is a FRESH training session (iter=0)" if ppo_runner.current_learning_iteration == 0 
          else f"⚠️ RESUMED from iteration {ppo_runner.current_learning_iteration}")
    print(f">>> END CHECKPOINT STATUS <<<\n")
    
    # Training loop with simple linear decay curriculum
    max_iterations = train_cfg.runner.max_iterations
    if max_iterations_override > 0:
        max_iterations = max_iterations_override
    
    steps_per_iter = env.num_envs * train_cfg.runner.num_steps_per_env
    
    print(f"\n{'='*60}")
    print("HIERARCHICAL RL TRAINING - Simple Curriculum")
    print(f"{'='*60}")
    print(f"Envs: {env.num_envs}")
    print(f"Steps/iter: {steps_per_iter}")
    print(f"Tasks: reach, button, cabinet, ball, box, lift, transfer, carry (Easy→Hard)")
    print(f"Skills: walk, reach, squat, step")
    print(f"Max iterations: {max_iterations} ({max_iterations // 8} per task if incremental)")
    print(f"Curriculum: K={train_cfg.algorithm.K_start}→{train_cfg.algorithm.K_end}, "
          f"ε={train_cfg.algorithm.epsilon_start}→{train_cfg.algorithm.epsilon_end}, "
          f"τ={train_cfg.algorithm.tau_start}→{train_cfg.algorithm.tau_end}")
    print(f"Skill entropy coef: {train_cfg.algorithm.c_ent_skill}")
    print(f"{'='*60}\n")
    
    for iteration in range(max_iterations):
        # Update curriculum based on ITERATION (linear decay)
        total_steps = iteration * steps_per_iter
        ppo_runner.alg.update_curriculum(iteration)
        
        # Get current curriculum params
        curriculum_params = ppo_runner.alg.curriculum.get_all_params()
        
        # Training step
        ppo_runner.learn(num_learning_iterations=1, init_at_random_ep_len=(iteration == 0))
        
        # Logging every 10 iters (more frequent for debugging)
        if iteration % 10 == 0:
            # Get per-task stats
            task_stats = env.get_task_stats()
            
            # Task distribution
            task_counts = {name: (env.task_ids == i).sum().item() 
                          for i, name in enumerate(env.task_names)}
            
            # Get learning metrics from PPO
            current_lr = ppo_runner.alg.optimizer.param_groups[0]['lr']
            
            # Collect gradient norm
            total_norm = 0.0
            for p in ppo_runner.alg.actor_critic.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** 0.5
            
            print(f"\n[{iteration}/{max_iterations}] Steps: {total_steps/1e6:.2f}M")
            print(f"  Curriculum: K={curriculum_params['K']}, ε={curriculum_params['epsilon']:.3f}, τ={curriculum_params['tau']:.2f}")
            print(f"  Phase: {curriculum_params.get('phase', 0)}, Focus task: {env.task_names[curriculum_params.get('focus_task', 0)]}")
            print(f"  Learning: LR={current_lr:.2e}, grad_norm={grad_norm:.4f}")
            print(f"  Task Dist: reach={task_counts.get('reach', 0)}, button={task_counts.get('button', 0)}, "
                  f"cabinet={task_counts.get('cabinet', 0)}, ball={task_counts.get('ball', 0)}")
            print(f"             box={task_counts.get('box', 0)}, transfer={task_counts.get('transfer', 0)}, "
                  f"lift={task_counts.get('lift', 0)}, carry={task_counts.get('carry', 0)}")
            
            # EPISODE REWARDS (cumulative like single-task)
            # Keys from get_task_stats(): Episode/rew_<task>
            print(f"  Episode Rewards (avg):")
            print(f"    reach={task_stats.get('Episode/rew_reach', 0):.1f}, "
                  f"button={task_stats.get('Episode/rew_button', 0):.1f}, "
                  f"cabinet={task_stats.get('Episode/rew_cabinet', 0):.1f}, "
                  f"ball={task_stats.get('Episode/rew_ball', 0):.1f}")
            print(f"    box={task_stats.get('Episode/rew_box', 0):.1f}, "
                  f"transfer={task_stats.get('Episode/rew_transfer', 0):.1f}, "
                  f"lift={task_stats.get('Episode/rew_lift', 0):.1f}, "
                  f"carry={task_stats.get('Episode/rew_carry', 0):.1f}")
            
            # METRICS (error values - lower is better)
            # Keys: Metric/{task}_{metric} from task_metrics[task_{task}_{metric}]
            print(f"  Metrics (errors):")
            print(f"    reach_wrist={task_stats.get('Metric/reach_wrist_error', -1):.3f}, "
                  f"button_wrist={task_stats.get('Metric/button_wrist_error', -1):.3f}, "
                  f"cabinet_wrist={task_stats.get('Metric/cabinet_wrist_error', -1):.3f}, "
                  f"ball_torso={task_stats.get('Metric/ball_torso_error', -1):.3f}")
            
            # Log to wandb with curriculum params
            if HAS_WANDB and args.wandb:
                wandb_log = {
                    'iteration': iteration,
                    'total_steps': total_steps,
                    # Curriculum
                    'Curriculum/K': curriculum_params['K'],
                    'Curriculum/epsilon': curriculum_params['epsilon'],
                    'Curriculum/tau': curriculum_params['tau'],
                    'Curriculum/c_ent_skill': curriculum_params['c_ent_skill'],
                    'Curriculum/phase': curriculum_params.get('phase', 0),
                    'Curriculum/focus_task': curriculum_params.get('focus_task', 0),
                    # Learning
                    'Learning/lr': current_lr,
                    'Learning/grad_norm': grad_norm,
                    # Task stats (Episode/rew_*, Metric/*)
                    **task_stats,
                    # Task distribution (all 8 tasks)
                    **{f'TaskDist/{k}': v for k, v in task_counts.items()}
                }
                wandb.log(wandb_log)
    
    print("\n✅ Training Complete!")


if __name__ == '__main__':
    args = get_args()
    train_hrl(args)