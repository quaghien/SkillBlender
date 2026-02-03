#!/usr/bin/env python3
"""
Play script for Hierarchical RL (HRL) trained models.

Usage:
    # With visualization (GUI) + recording
    python legged_gym/scripts/play_hrl.py \
        --task h1_hrl \
        --experiment_name h1_hrl \
        --load_run hrl_v8.1 \
        --checkpoint -1 \
        --sim_device cuda:0 \
        --rl_device cuda:0 \
        --visualize
    
    # Headless mode (no GUI) + recording video
    python legged_gym/scripts/play_hrl.py \
        --task h1_hrl \
        --experiment_name h1_hrl \
        --load_run hrl_v8.1 \
        --checkpoint -1 \
        --sim_device cuda:0 \
        --rl_device cuda:0 \
        --headless \
        --record

Key features:
    1. Uses ActorCriticHRL with pretrained low-level skills (walking, reaching, squatting, stepping)
    2. Blends 4 pretrained policies based on high-level learned weights
    3. Supports all 8 tasks: reach, button, cabinet, ball, box, transfer, lift, carry
    4. Can record video in headless mode with --record flag
"""

import os
import sys
import math
import argparse

# Parse --record flag before isaacgym import
RECORD_VIDEO = '--record' in sys.argv
if RECORD_VIDEO:
    sys.argv.remove('--record')

# IMPORTANT: Import isaacgym BEFORE torch
import isaacgym
from isaacgym import gymapi
from isaacgym import gymutil

import numpy as np
import torch
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, Logger
from legged_gym.utils.helpers import class_to_dict

# Import HRL policy (V2 with hold skill)
from rsl_rl.modules import ActorCriticHRL


# Display settings
H, W = 480, 640
EXPORT_POLICY = False
EGO_CENTRIC = False


def visualize_hrl_task(env, task_id, task_name):
    """Visualize task-specific information in viewer."""
    if not hasattr(env, 'viewer') or env.viewer is None:
        return
    
    env.gym.clear_lines(env.viewer)
    
    # Create wireframe geometries
    sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
    sphere_pose = gymapi.Transform(r=sphere_rot)
    axes_geom = gymutil.AxesGeometry(0.15)
    yellow_geom = gymutil.WireframeSphereGeometry(0.05, 12, 12, sphere_pose, color=(1, 1, 0))
    red_geom = gymutil.WireframeSphereGeometry(0.05, 12, 12, sphere_pose, color=(1, 0, 0))
    green_geom = gymutil.WireframeSphereGeometry(0.08, 12, 12, sphere_pose, color=(0, 1, 0))
    
    # Visualize based on task type
    for i in range(env.num_envs):
        env_task = env.task_ids[i].item() if hasattr(env, 'task_ids') else task_id
        
        if env_task == 0:  # Reach
            if hasattr(env, 'ref_wrist_pos'):
                ref_wrist_pos = env.ref_wrist_pos[i]
                for j in range(2):
                    pos = ref_wrist_pos[j, :3]
                    transform = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), gymapi.Quat())
                    gymutil.draw_lines(yellow_geom, env.gym, env.viewer, env.envs[i], transform)
                    
        elif env_task == 1:  # Button
            if hasattr(env, 'button_goal_pos'):
                pos = env.button_goal_pos[i, :3]
                transform = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), gymapi.Quat())
                gymutil.draw_lines(red_geom, env.gym, env.viewer, env.envs[i], transform)
                
        elif env_task == 2:  # Cabinet
            # Cabinet doesn't need special visualization
            pass
            
        elif env_task == 3:  # Ball
            if hasattr(env, 'goal_pos'):
                pos = env.goal_pos[i, :3]
                transform = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), gymapi.Quat())
                large_red = gymutil.WireframeSphereGeometry(0.2, 12, 12, sphere_pose, color=(1, 0, 0))
                gymutil.draw_lines(large_red, env.gym, env.viewer, env.envs[i], transform)
                
        elif env_task in [4, 5, 6, 7]:  # Box, Transfer, Lift, Carry
            if hasattr(env, 'box_goal_pos'):
                pos = env.box_goal_pos[i]
                if len(pos.shape) == 0 or pos.shape[0] == 1:
                    pos = pos.unsqueeze(0) if len(pos.shape) == 0 else pos
                pos = pos[:3] if len(pos) >= 3 else pos
                transform = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2] if len(pos) > 2 else 0.5), gymapi.Quat())
                gymutil.draw_lines(green_geom, env.gym, env.viewer, env.envs[i], transform)


def load_hrl_policy(log_dir, device, env_cfg, train_cfg, args, checkpoint=-1):
    """Load HRL policy from checkpoint.
    
    This creates an ActorCriticHRL which:
    1. Loads 4 pretrained low-level policies (walking, reaching, squatting, stepping)
    2. Creates a high-level policy that outputs commands + blend weights
    3. Loads the high-level weights from checkpoint
    """
    
    # Get policy config
    policy_cfg = class_to_dict(train_cfg.policy) if hasattr(train_cfg, 'policy') else {}
    
    # Create HRL policy (this loads pretrained low-level skills)
    policy = ActorCriticHRL(
        num_actor_obs=env_cfg.env.num_observations,
        num_critic_obs=env_cfg.env.num_privileged_obs,
        num_actions=env_cfg.env.num_actions,
        device=device,
        args=args,
        **policy_cfg
    )
    
    # Find checkpoint file
    if checkpoint == -1:
        # Find latest checkpoint
        models = [f for f in os.listdir(log_dir) if f.startswith('model_') and f.endswith('.pt')]
        if not models:
            raise FileNotFoundError(f"No model checkpoints found in {log_dir}")
        # Sort by iteration number
        models.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        checkpoint_file = models[-1]
    else:
        checkpoint_file = f"model_{checkpoint}.pt"
    
    checkpoint_path = os.path.join(log_dir, checkpoint_file)
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load weights
    loaded = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in loaded:
        state_dict = loaded['model_state_dict']
    elif 'actor_critic_state_dict' in loaded:
        state_dict = loaded['actor_critic_state_dict']
    else:
        state_dict = loaded
    
    # Only load high-level policy weights (actor, critic, std)
    # Low-level policies are already loaded from their own checkpoints
    model_dict = policy.state_dict()
    
    # Filter to only load high-level params
    high_level_params = {}
    skipped_params = []
    
    for k, v in state_dict.items():
        # Skip low-level policy params (policy_list.*)
        if k.startswith('policy_list'):
            skipped_params.append(k)
            continue
        
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                high_level_params[k] = v
            else:
                print(f"  Shape mismatch: {k} ({v.shape} vs {model_dict[k].shape})")
                skipped_params.append(k)
        else:
            skipped_params.append(k)
    
    # Update model dict
    model_dict.update(high_level_params)
    policy.load_state_dict(model_dict, strict=False)
    
    print(f"✅ Loaded {len(high_level_params)} high-level parameters")
    print(f"  Skipped {len(skipped_params)} parameters (low-level policies loaded separately)")
    
    policy.eval()
    return policy, checkpoint_file


def get_camera_pose(task_name):
    """Get camera position based on task."""
    if not EGO_CENTRIC:
        if task_name in ['button', 'ball', 'cabinet']:
            camera_offset = gymapi.Vec3(-1, -2, 1)
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(45))
        else:
            camera_offset = gymapi.Vec3(1, -1, 1)
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135))
    else:
        camera_offset = gymapi.Vec3(0.1, 0, 0.9)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(45))
    return gymapi.Transform(camera_offset, camera_rotation)


def override_env_cfg(env_cfg, args):
    """Override env config for playback."""
    print(f'====> URDF file: {env_cfg.asset.file}')
    
    # Reduce num_envs for visualization/recording
    if args.visualize:
        default_num_envs = 4
    elif RECORD_VIDEO:
        default_num_envs = 1  # Only 1 env needed for recording
    else:
        default_num_envs = 16
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, default_num_envs)
    
    # Extend episode length for better visualization
    env_cfg.env.episode_length_s = 30
    
    # Disable noise and randomization for cleaner visualization
    env_cfg.terrain.num_rows = 3
    env_cfg.terrain.num_cols = 3
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    
    return env_cfg


@torch.no_grad()
def play_hrl(args):
    """Main play function for HRL policy."""
    
    # Determine if we should record
    should_record = RECORD_VIDEO or args.visualize
    
    # Get configs from registered task (NOT from checkpoint folder)
    # This avoids requiring *_config.py in checkpoint folder
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # Override for playback
    env_cfg = override_env_cfg(env_cfg, args)
    
    # Create environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # Build log directory path
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.experiment_name)
    
    if args.load_run == -1:
        # Find latest run
        runs = sorted([d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))])
        if not runs:
            raise FileNotFoundError(f"No runs found in {log_root}")
        load_run = runs[-1]
    else:
        load_run = args.load_run
    
    log_dir = os.path.join(log_root, load_run)
    print(f"Loading from: {log_dir}")
    
    # Load HRL policy (pass args for loading pretrained skills)
    device = env.device
    policy, checkpoint_name = load_hrl_policy(log_dir, device, env_cfg, train_cfg, args, args.checkpoint)
    policy.to(device)
    
    # Set to eval mode (no dropout, batchnorm in eval, etc.)
    policy.eval()
    
    model_name = f'{load_run}_{checkpoint_name.replace(".pt", "")}'
    print(f"Model name: {model_name}")
    
    # Setup recording
    robot_index = 0
    frame_path = None
    video = None
    cam = None
    
    if should_record:
        frame_path = os.path.join(log_root, 'exported', 'frames')
        os.makedirs(frame_path, exist_ok=True)
        
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = W
        camera_properties.height = H
        cam = env.gym.create_camera_sensor(env.envs[robot_index], camera_properties)
        
        # Attach camera
        camera_pose = get_camera_pose('reach')  # Default camera
        actor_handle = env.gym.get_actor_handle(env.envs[robot_index], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[robot_index], actor_handle, 0)
        env.gym.attach_camera_to_body(
            cam,
            env.envs[robot_index],
            body_handle,
            camera_pose,
            gymapi.FOLLOW_POSITION if not EGO_CENTRIC else gymapi.FOLLOW_TRANSFORM
        )
        print(f"✅ Recording enabled. Videos will be saved to: {frame_path}")
    else:
        print("⚠️ Recording disabled. Use --visualize or --record to enable video recording.")
    
    # Task names for logging
    task_names = ['reach', 'button', 'cabinet', 'ball', 'box', 'transfer', 'lift', 'carry']
    
    # Rollout settings - test all 8 tasks, 200 steps each
    steps_per_task = 200
    N_rollouts = 1
    max_steps = steps_per_task * len(task_names)  # 200 * 8 = 1600 total
    
    print(f"\n{'='*60}")
    print("HRL PLAYBACK")
    print(f"{'='*60}")
    print(f"Num envs: {env.num_envs}")
    print(f"Steps per task: {steps_per_task}")
    print(f"Total steps: {max_steps}")
    print(f"Tasks: {task_names}")
    print(f"N rollouts: {N_rollouts}")
    print(f"{'='*60}\n")
    
    # Stats tracking
    all_stats = {name: {'success': 0, 'total': 0} for name in task_names}
    
    for i_rollout in range(N_rollouts):
        print(f"\n====> Rollout {i_rollout+1}/{N_rollouts}")
        
        # Setup video
        if should_record and frame_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            filename_mp4 = f'hrl_{model_name}_{i_rollout}.mp4'
            # Calculate fps: record every 2nd frame at 50Hz = 25fps video
            video_fps = 25.0
            video = cv2.VideoWriter(os.path.join(frame_path, filename_mp4), fourcc, video_fps, (W, H))
            if not video.isOpened():
                print(f"❌ Failed to open video writer!")
            else:
                print(f"✅ Video writer opened: {filename_mp4} at {video_fps} fps")
        
        # Logger
        logger = Logger(env.dt)
        
        # Store original states
        if hasattr(env, 'root_states'):
            env.ori_root_states = env.root_states.clone()
        
        # Track task statistics
        task_rewards = {name: 0.0 for name in task_names}
        task_steps = {name: 0 for name in task_names}
        
        for i in tqdm(range(max_steps), desc=f"Rollout {i_rollout+1}"):
            # Cycle through all 8 tasks, each for steps_per_task steps
            current_task_id = i // steps_per_task  # 0-7
            current_task_name = task_names[current_task_id]
            
            # Set task_ids in environment
            if hasattr(env, 'task_ids'):
                env.task_ids[:] = current_task_id
            
            # Print task change
            if i % steps_per_task == 0:
                print(f"\n  >>> Switching to Task {current_task_id}: {current_task_name.upper()} (steps {i}-{i+steps_per_task-1})")
            
            # Visualize task
            if args.visualize:
                visualize_hrl_task(env, current_task_id, current_task_name)
            
            # Get action from HRL policy (deterministic inference)
            action = policy.act_inference(obs.detach())
            
            # Get skill info for logging
            info = {
                'skill_exec': policy.current_skill if hasattr(policy, 'current_skill') and policy.current_skill is not None else torch.zeros(env.num_envs, dtype=torch.long, device=device),
            }
            
            # Step environment
            obs, _, rews, dones, infos = env.step(action.detach())
            
            # Check robot stability
            base_height = env.root_states[robot_index, 2].item()
            base_pitch = env.base_euler_xyz[robot_index, 1].item()
            base_roll = env.base_euler_xyz[robot_index, 0].item()
            
            # Flag if robot is falling
            is_falling = base_height < 0.3 or abs(base_pitch) > 0.8 or abs(base_roll) > 0.8
            if is_falling and i % 50 == 0:
                print(f"  ⚠️ Robot falling at step {i}! H={base_height:.2f} P={base_pitch:.2f} R={base_roll:.2f}")
            
            # Track rewards per current task
            task_rewards[current_task_name] += rews.mean().item()
            task_steps[current_task_name] += 1
            
            # Record frame
            if should_record and video is not None and cam is not None:
                if i % 2 == 0:  # Every 2nd frame = 25 fps recording at 50Hz sim (ghi hết)
                    env.gym.fetch_results(env.sim, True)
                    env.gym.step_graphics(env.sim)
                    env.gym.render_all_camera_sensors(env.sim)
                    img = env.gym.get_camera_image(env.sim, env.envs[robot_index], cam, gymapi.IMAGE_COLOR)
                    img = np.reshape(img, (H, W, 4))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    video.write(img[..., :3])
            
            # Log states
            logger.log_states({
                'dof_pos_target': action[robot_index, :].cpu().numpy() * env.cfg.control.action_scale,
                'dof_pos': env.dof_pos[robot_index, :].cpu().numpy(),
                'dof_vel': env.dof_vel[robot_index, :].cpu().numpy(),
                'dof_torque': env.torques[robot_index, :].cpu().numpy(),
                'base_vel_x': env.base_lin_vel[robot_index, 0].cpu().numpy(),
                'base_vel_y': env.base_lin_vel[robot_index, 1].cpu().numpy(),
                'base_vel_z': env.base_lin_vel[robot_index, 2].cpu().numpy(),
                'base_vel_yaw': env.base_ang_vel[robot_index, 2].cpu().numpy(),
                'base_roll': env.base_euler_xyz[robot_index, 0].cpu().numpy(),
                'base_pitch': env.base_euler_xyz[robot_index, 1].cpu().numpy(),
                'base_height': env.root_states[robot_index, 2].cpu().numpy(),
            })
            
            # Log rewards
            if infos.get("episode"):
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes > 0:
                    logger.log_rewards(infos["episode"], num_episodes)
            
            # Print skill selection info periodically
            if i % 50 == 0:
                skill_names = ['walk', 'reach', 'squat', 'step']
                
                # Get dominant skill
                if hasattr(policy, 'current_skill') and policy.current_skill is not None:
                    skill_id = policy.current_skill[robot_index].item()
                    skill_name = skill_names[skill_id] if skill_id < len(skill_names) else 'unknown'
                else:
                    skill_name = 'unknown'
                
                # Get blend weights if available
                gating_str = ""
                if hasattr(policy, 'blend_weights') and policy.blend_weights is not None:
                    # blend_weights: [B, num_skills, num_dofs]
                    weights = policy.blend_weights[robot_index]  # [num_skills, num_dofs]
                    avg_weights = weights.mean(dim=-1).cpu().numpy()  # [num_skills]
                    gating_str = f" | Blend: W={avg_weights[0]:.2f} R={avg_weights[1]:.2f} S={avg_weights[2]:.2f} T={avg_weights[3]:.2f}"
                
                print(f"  Step {i}: Task={current_task_name}, Skill={skill_name}{gating_str}")
        
        # Print rollout stats
        print(f"\n--- Rollout {i_rollout+1} Stats ---")
        for tname in task_names:
            if task_steps[tname] > 0:
                avg_rew = task_rewards[tname] / task_steps[tname]
                print(f"  {tname}: avg_reward={avg_rew:.3f} ({task_steps[tname]} steps)")
        
        # Cleanup
        if video is not None:
            # Flush and close video properly
            for _ in range(5):  # Write a few more frames to ensure buffer flush
                video.write(np.zeros((H, W, 3), dtype=np.uint8))
            video.release()
            print(f"✅ Video saved and released")
            video = None
        
        logger.print_rewards()
        
        # Save plot
        if should_record and frame_path:
            try:
                fig = logger._plot()
                filename_png = f'hrl_{model_name}_{i_rollout}.png'
                fig.savefig(os.path.join(frame_path, filename_png))
                plt.close(fig)
            except Exception as e:
                print(f"Could not save plot: {e}")
        
        del logger
    
    print(f"\n{'='*60}")
    print("PLAYBACK COMPLETE")
    if should_record and frame_path:
        print(f"Videos saved to: {frame_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    args = get_args(test=True)
    play_hrl(args)