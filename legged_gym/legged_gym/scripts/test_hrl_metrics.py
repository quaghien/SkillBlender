#!/usr/bin/env python
"""Test script to verify HRL reward metrics are computed correctly"""

from isaacgym import gymapi, gymtorch
import torch
import sys
import os

# Add paths BEFORE importing legged_gym
sys.path.insert(0, '/home/crl/hienhq/SkillBlender/legged_gym')
sys.path.insert(0, '/home/crl/hienhq/SkillBlender/rsl_rl')
os.environ['WANDB_MODE'] = 'disabled'

def test_metrics():
    """Test that metrics are computed correctly"""
    # Import after path setup
    from legged_gym.envs.h1.h1_hrl.h1_hrl import H1HRLEnv, H1HRLCfg, H1HRLCfgPPO
    from legged_gym.utils.task_registry import task_registry
    
    # Register task if not already registered
    task_registry.register("h1_hrl", H1HRLEnv, H1HRLCfg(), H1HRLCfgPPO(), 
                          path='/home/crl/hienhq/SkillBlender/legged_gym/logs/h1_hrl')
    
    # Create minimal args
    class Args:
        task = 'h1_hrl'
        sim_device = 'cuda:0'
        rl_device = 'cuda:0'
        graphics_device_id = 0
        headless = True
        pipeline = 'gpu'
        physics_engine = gymapi.SIM_PHYSX
        num_threads = 0
        subscenes = 0
        num_envs = 256  # Small for testing
        use_gpu = True
        use_gpu_pipeline = True
    
    args = Args()
    
    # Modify config for small test
    cfg = H1HRLCfg()
    cfg.env.num_envs = 256
    
    print("Creating environment...")
    env, _ = task_registry.make_env(name=args.task, args=args)
    
    print(f"\n=== Environment Info ===")
    print(f"num_envs: {env.num_envs}")
    print(f"num_bodies: {env.num_bodies}")
    print(f"wrist_indices: {env.wrist_indices}")
    print(f"torso_indices: {env.torso_indices}")
    print(f"elbow_indices: {env.elbow_indices}")
    
    # Run a few steps
    print("\n=== Running simulation ===")
    action = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    
    for step in range(5):
        obs, rew, dones, infos, priv_obs = env.step(action)
        
        # Check metrics
        stats = env.get_task_stats()
        
        print(f"\nStep {step}:")
        print(f"  Rewards: min={rew.min():.3f}, max={rew.max():.3f}, mean={rew.mean():.3f}")
        
        # Print metrics
        for key, val in stats.items():
            if 'Metric' in key:
                expected_range = "ERROR!" if val > 10 else "OK"
                print(f"  {key}: {val:.4f} [{expected_range}]")
    
    # Detailed check of wrist positions
    print("\n=== Wrist Position Check ===")
    
    # Get reach task envs
    reach_mask = (env.task_ids == 0)
    n_reach = reach_mask.sum().item()
    print(f"Number of reach envs: {n_reach}")
    
    if n_reach > 0:
        # Get wrist positions from rigid_state
        wrist_pos = env.rigid_state[reach_mask][:, env.wrist_indices, :3]
        print(f"wrist_pos shape: {wrist_pos.shape}")
        print(f"wrist_pos[0]: {wrist_pos[0]}")
        
        # Get target from goal_value
        target = env.goal_value[reach_mask, :6].reshape(-1, 2, 3)
        print(f"target shape: {target.shape}")
        print(f"target[0]: {target[0]}")
        
        # Compute error
        diff = wrist_pos - target
        diff_flat = torch.flatten(diff, start_dim=1)
        error = torch.mean(torch.abs(diff_flat), dim=1)
        print(f"error range: [{error.min():.4f}, {error.max():.4f}]")
        print(f"error mean: {error.mean():.4f}")
        
        if error.mean() > 10:
            print("\n⚠️  ERROR: wrist_error is too large!")
            print("Possible causes:")
            print("1. wrist_indices point to wrong bodies")
            print("2. rigid_state not refreshed properly")
            print("3. goal_value not in robot frame")
    
    # Check box task
    print("\n=== Box Position Check ===")
    box_mask = (env.task_ids == 4)  # Task 4 = box
    n_box = box_mask.sum().item()
    print(f"Number of box envs: {n_box}")
    
    if n_box > 0:
        box_pos = env.box_pos[box_mask]
        box_target = env.box_target[box_mask]
        box_error = torch.mean(torch.abs(box_pos - box_target), dim=1)
        print(f"box_error range: [{box_error.min():.4f}, {box_error.max():.4f}]")
        print(f"box_error mean: {box_error.mean():.4f}")


if __name__ == '__main__':
    test_metrics()
