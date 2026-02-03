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
            
            # Get active tasks from curriculum
            active_tasks = curriculum_params.get('active_tasks', [0])
            focus_task = curriculum_params.get('focus_task', 0)
            
            print(f"\n[{iteration}/{max_iterations}] Steps: {total_steps/1e6:.2f}M")
            print(f"  Curriculum: K={curriculum_params['K']}, ε={curriculum_params['epsilon']:.3f}, τ={curriculum_params['tau']:.2f}")
            print(f"  Phase: {curriculum_params.get('phase', 0)}, Focus: {env.task_names[focus_task]}, Active: {[env.task_names[i] for i in active_tasks]}")
            print(f"  Learning: LR={current_lr:.2e}, grad_norm={grad_norm:.4f}")
            
            # Task Distribution (only show active tasks with count > 0)
            dist_parts = [f"{name}={task_counts.get(name, 0)}" for name in env.task_names if task_counts.get(name, 0) > 0]
            print(f"  Task Dist: {', '.join(dist_parts)}")
            
            # Episode Rewards (only active tasks)
            rew_parts = [f"{env.task_names[i]}={task_stats.get(f'Episode/rew_{env.task_names[i]}', 0):.1f}" for i in active_tasks]
            print(f"  Rewards: {', '.join(rew_parts)}")
            
            # Metrics/Errors (only active tasks - show primary metric for each)
            # Each task has different metrics: reach→wrist, button→wrist, cabinet→door, ball→torso, box→box
            task_primary_metrics = {
                'reach': 'wrist_error',
                'button': 'wrist_error', 
                'cabinet': 'door_error',
                'ball': 'torso_error',
                'box': 'box_error',
                'lift': 'box_error',
                'transfer': 'box_error',
                'carry': 'box_error',
            }
            metric_parts = []
            for task_idx in active_tasks:
                task_name = env.task_names[task_idx]
                metric_name = task_primary_metrics.get(task_name, 'error')
                metric_key = f'Metric/{task_name}_{metric_name}'
                value = task_stats.get(metric_key, -1)
                if value != -1:
                    metric_parts.append(f"{task_name}={value:.3f}")
            if metric_parts:
                print(f"  Errors: {', '.join(metric_parts)}")
            
            # Progress rewards (only active tasks)
            progress_parts = []
            for task_idx in active_tasks:
                task_name = env.task_names[task_idx]
                progress_key = f'Metric/{task_name}_progress'
                value = task_stats.get(progress_key, None)
                if value is not None:
                    sign = "+" if value > 0 else ""
                    progress_parts.append(f"{task_name}={sign}{value:.3f}")
            if progress_parts:
                print(f"  Progress: {', '.join(progress_parts)}")
            
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
                    'Curriculum/num_active_tasks': len(active_tasks),
                    # Learning
                    'Learning/lr': current_lr,
                    'Learning/grad_norm': grad_norm,
                    # All task stats (Episode/rew_*, Metric/*, TaskDist/*)
                    **task_stats,
                }
                wandb.log(wandb_log)
    
    print("\n✅ Training Complete!")


if __name__ == '__main__':
    args = get_args()
    train_hrl(args)