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

# Import HRL policy - Use SIMPLIFIED version
from rsl_rl.modules.actor_critic_hrl_simple import ActorCriticHRLSimple, create_hrl_policy_simple

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
    
    # Create PPO runner
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
    )
    
    # Create HRL policy - SIMPLIFIED PPO-correct version
    device = args.rl_device
    policy = create_hrl_policy_simple(train_cfg, env.num_envs, device)
    
    # Override policy in runner
    ppo_runner.alg.actor_critic = policy
    ppo_runner.alg.optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    ppo_runner.alg.learning_rate = 3e-4  # Sync with optimizer for correct W&B logging
    
    # DEBUG: Print checkpoint load status
    print(f"\n>>> CHECKPOINT STATUS <<<")
    print(f"Current learning iteration: {ppo_runner.current_learning_iteration}")
    print(f"Log dir: {ppo_runner.log_dir}")
    print(f"This is a FRESH training session (iter=0)" if ppo_runner.current_learning_iteration == 0 
          else f"⚠️ RESUMED from iteration {ppo_runner.current_learning_iteration}")
    print(f">>> END CHECKPOINT STATUS <<<\n")
    
    # Training loop with curriculum
    # FIXED: Use STEPS not iterations for schedule
    # With 4096 envs × 24 steps = 98304 steps/iter
    # Phase 1 = 5M steps = ~51 iters, Phase 2 = 5M+ steps
    max_iterations = train_cfg.runner.max_iterations
    if max_iterations_override > 0:
        max_iterations = max_iterations_override
    
    steps_per_iter = env.num_envs * train_cfg.runner.num_steps_per_env
    phase1_steps = 4_915_200_000  # 50000 iters × 98,304 steps/iter = 4.9152B steps
    phase1_iters = phase1_steps // steps_per_iter  # 50000 iters
    
    print(f"\n{'='*60}")
    print("HIERARCHICAL RL TRAINING")
    print(f"{'='*60}")
    print(f"Envs: {env.num_envs}")
    print(f"Steps/iter: {steps_per_iter}")
    print(f"Tasks: reach, button, cabinet, ball, box, transfer, lift, carry (8 tasks)")
    print(f"Skills: walk, reach, squat, step")
    print(f"Phase 1 (Frozen): 0 - {phase1_iters} iters (~4.9B steps)")
    print(f"Phase 2 (Unfrozen): {phase1_iters}+ iters (~4.9B steps)")
    print(f"Max iterations: {max_iterations} (~9.8B steps total)")
    print(f"{'='*60}\n")
    
    for iteration in range(max_iterations):
        # Update curriculum based on STEPS
        total_steps = iteration * steps_per_iter
        policy.set_training_step(total_steps)
        
        # Logging phase transition
        if iteration == phase1_iters:
            print(f"\n>>> PHASE 2: Residual UNFROZEN (clip=±0.05) at {total_steps/1e6:.1f}M steps\n")
        
        # Training step
        ppo_runner.learn(num_learning_iterations=1, init_at_random_ep_len=(iteration == 0))
        
        # Logging every 10 iters (more frequent for debugging)
        if iteration % 10 == 0:
            phase = "FROZEN" if iteration < phase1_iters else "UNFROZEN"
            
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
            
            print(f"\n[{iteration}/{max_iterations}] Steps: {total_steps/1e6:.2f}M, Phase: {phase}")
            print(f"  Learning: LR={current_lr:.2e}, grad_norm={grad_norm:.4f}")
            print(f"  Task Dist: reach={task_counts.get('reach', 0)}, button={task_counts.get('button', 0)}, "
                  f"cabinet={task_counts.get('cabinet', 0)}, ball={task_counts.get('ball', 0)}")
            print(f"             box={task_counts.get('box', 0)}, transfer={task_counts.get('transfer', 0)}, "
                  f"lift={task_counts.get('lift', 0)}, carry={task_counts.get('carry', 0)}")
            
            # EPISODE REWARDS (cumulative like single-task)
            print(f"  Episode Rewards:")
            print(f"    reach={task_stats.get('episode_reward/reach', 0):.1f}, "
                  f"button={task_stats.get('episode_reward/button', 0):.1f}, "
                  f"cabinet={task_stats.get('episode_reward/cabinet', 0):.1f}, "
                  f"ball={task_stats.get('episode_reward/ball', 0):.1f}")
            print(f"    box={task_stats.get('episode_reward/box', 0):.1f}, "
                  f"transfer={task_stats.get('episode_reward/transfer', 0):.1f}, "
                  f"lift={task_stats.get('episode_reward/lift', 0):.1f}, "
                  f"carry={task_stats.get('episode_reward/carry', 0):.1f}")
            
            # STEP REWARDS (instantaneous)
            print(f"  Step Rewards:")
            print(f"    reach={task_stats.get('step_reward/reach', 0):.3f}, "
                  f"button={task_stats.get('step_reward/button', 0):.3f}, "
                  f"cabinet={task_stats.get('step_reward/cabinet', 0):.3f}, "
                  f"ball={task_stats.get('step_reward/ball', 0):.3f}")
            print(f"    box={task_stats.get('step_reward/box', 0):.3f}, "
                  f"transfer={task_stats.get('step_reward/transfer', 0):.3f}, "
                  f"lift={task_stats.get('step_reward/lift', 0):.3f}, "
                  f"carry={task_stats.get('step_reward/carry', 0):.3f}")
            
            # Log to wandb
            if HAS_WANDB and args.wandb:
                wandb_log = {
                    'iteration': iteration,
                    'total_steps': total_steps,
                    'phase': 1 if iteration < phase1_iters else 2,
                    'residual_clip': policy.residual_clip,
                    'learning/lr': current_lr,
                    'learning/grad_norm': grad_norm,
                    **task_stats,
                    **{f'task_count/{k}': v for k, v in task_counts.items()}
                }
                wandb.log(wandb_log)
    
    print("\n✅ Training Complete!")


if __name__ == '__main__':
    args = get_args()
    train_hrl(args)
