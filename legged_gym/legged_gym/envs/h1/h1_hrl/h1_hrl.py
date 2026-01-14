# SPDX-License-Identifier: BSD-3-Clause
# H1 HRL Meta-Environment for 8 manipulation tasks
# Unified environment for: reach, button, cabinet, ball, box, transfer, lift, carry

from isaacgym import gymapi
from isaacgym.torch_utils import *
import torch
import numpy as np

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


# ============================================================================
# CONFIGURATION
# ============================================================================

class H1HRLCfg(LeggedRobotCfg):
    """Configuration for 8-task HRL meta-environment"""
    
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_actions = 19  # H1 DOFs
        frame_stack = 1
        c_frame_stack = 3
        command_dim = 3
        num_single_obs = 3 * num_actions + 6 + command_dim  # 69 = 57 + 6 + 6 (same as h1_walking)
        num_observations = 105  # State(69) + Goal(14) + Mask(14) + TaskID(8)
        single_num_privileged_obs = 4 * num_actions + 25  # 101
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)  # 303
        episode_length_s = 20  # 20 second episodes
        use_ref_actions = False
        
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False
        
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.0]  # Standing height
        default_joint_angles = {
            # Legs
            'left_hip_yaw_joint': 0.0,
            'left_hip_roll_joint': 0.0,
            'left_hip_pitch_joint': -0.4,
            'left_knee_joint': 0.8,
            'left_ankle_joint': -0.4,
            'right_hip_yaw_joint': 0.0,
            'right_hip_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.4,
            'right_knee_joint': 0.8,
            'right_ankle_joint': -0.4,
            # Torso
            'torso_joint': 0.0,
            # Arms
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 0.0,
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 0.0,
        }
        
    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 150.0}
        damping = {'joint': 5.0}
        action_scale = 0.25
        decimation = 20  # 50 Hz control
        
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1_wrist.urdf'
        name = 'h1'
        foot_name = 'ankle'
        knee_name = 'knee'
        elbow_name = 'elbow'
        wrist_name = 'wrist'  # Note: wrist links have no collision, use elbow as proxy
        torso_name = 'torso'
        terminate_after_contacts_on = ['pelvis', 'torso', 'shoulder', 'elbow']
        penalize_contacts_on = ['hip', 'knee']
        self_collisions = 0
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False
    
    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1
        up_axis = 1  # z
        
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.1
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2
            
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1.0, 1.0]
        push_robots = False
        push_interval_s = 5
        max_push_vel_xy = 0.5
        max_push_ang_vel = 0.5
        dynamic_randomization = 0.0
    
    class rewards(LeggedRobotCfg.rewards):
        # Disable all base class rewards - we use custom task rewards
        class scales:
            pass  # Empty - all rewards handled in compute_reward()
        
        only_positive_rewards = False
        base_height_target = 0.98


class H1HRLCfgPPO(LeggedRobotCfgPPO):
    """PPO configuration for HRL training"""
    seed = 5  # Same as author's single-task configs (5 for reproducibility, -1 for random)
    runner_class_name = 'OnPolicyRunner'
    
    class policy:
        init_noise_std = 0.5
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        
    class algorithm:
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 3.0e-4  # Fixed LR for HRL stability
        schedule = 'fixed'  # IMPORTANT: disable adaptive to prevent LR explosion
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01  # Not used with fixed schedule
        max_grad_norm = 1.0
        
    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 100000
        save_interval = 1000
        experiment_name = 'h1_hrl'
        run_name = ''
        resume = False
        resume_path = ''
        load_run = -1
        checkpoint = -1


# ============================================================================
# ENVIRONMENT
# ============================================================================

class H1HRLEnv(LeggedRobot):
    """
    Hierarchical RL Meta-Environment for 8 manipulation tasks.
    
    Tasks (0-7):
        0: reach    - Move wrists to target positions
        1: button   - Press button with left hand
        2: cabinet  - Close cabinet door
        3: ball     - Kick ball to goal
        4: box      - Push box to target
        5: transfer - Transfer box between tables
        6: lift     - Lift box to target height
        7: carry    - Pick up and carry box to goal
        
    Observation (105D):
        - State: 69D (base pos/vel/ori, joint pos/vel, etc.)
        - Goal: 14D (task-specific target)
        - Mask: 14D (which goal dims are active)
        - TaskID: 8D (one-hot encoding)
        
    Rewards:
        Each task has specific reward following original single-task implementations.
        All use exp(-4 * error) pattern with task-specific scales.
    """
    
    cfg: H1HRLCfg
    
    # Task names for logging
    task_names = ['reach', 'button', 'cabinet', 'ball', 'box', 'transfer', 'lift', 'carry']
    
    def __init__(self, cfg: H1HRLCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg = cfg
        
        # CRITICAL: Override num_obs for 8-task setup (State 69 + Goal 14 + Mask 14 + TaskID 8 = 105)
        self.num_obs = cfg.env.num_observations
        
        # Task assignment per env (after super init so self.device exists)
        self.task_ids = torch.randint(0, 8, (self.num_envs,), device=self.device)
        
        # Goal storage
        self.goal_value = torch.zeros(self.num_envs, 14, device=self.device)
        self.goal_mask = torch.zeros(self.num_envs, 14, device=self.device)
        self.task_onehot = torch.zeros(self.num_envs, 8, device=self.device)
        
        # Box states (for box/transfer/lift/carry)
        self.box_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.box_target = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Button position (for button task)
        self.button_pos = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Right arm default dof positions for button task (indices 15-18: right shoulder/elbow)
        self.right_arm_indices = slice(15, 19)  # right_shoulder_pitch to right_elbow
        
        # Cabinet door angle (for cabinet task)
        self.door_angle = torch.zeros(self.num_envs, device=self.device)
        self.door_target = torch.zeros(self.num_envs, device=self.device)
        
        # Ball states (for ball task)
        self.ball_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.ball_target = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Note: wrist_indices set in _create_envs()
        
        # === CRITIC HISTORY FOR FRAME STACKING ===
        from collections import deque
        single_priv_obs_dim = self.cfg.env.single_num_privileged_obs  # 101
        self.critic_history = deque(maxlen=self.cfg.env.c_frame_stack)  # 3 frames
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_history.append(torch.zeros(
                self.num_envs, single_priv_obs_dim, device=self.device))
        
        # === TASK-SPECIFIC REWARD TRACKING ===
        self.task_episode_rewards = {name: torch.zeros(self.num_envs, device=self.device) 
                                     for name in self.task_names}
        self.task_episode_lengths = {name: torch.zeros(self.num_envs, device=self.device)
                                     for name in self.task_names}
        # Aggregated stats (updated on episode end)
        self.task_avg_rewards = {name: 0.0 for name in self.task_names}
        self.task_episode_counts = {name: 0 for name in self.task_names}
        self.task_reward_components = {name: {'total': 0.0} for name in self.task_names}
        
        # Sample initial goals
        self._sample_goals(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
    
    def create_sim(self):
        """Create simulation with plane terrain"""
        self.up_axis_idx = 2  # z-up
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, 
            self.physics_engine, self.sim_params
        )
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type is not None:
            raise ValueError(f"Terrain mesh type {mesh_type} not supported for HRL env")
        self._create_envs()
    
    def _create_envs(self):
        """Create envs and setup body indices"""
        super()._create_envs()
        
        # Get elbow indices (used as wrist proxy - wrist links have no collision/inertia in URDF)
        elbow_names = [s for s in self.body_names if self.cfg.asset.elbow_name in s]
        self.elbow_indices = torch.zeros(len(elbow_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(elbow_names):
            self.elbow_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name)
        
        # Use elbow as wrist proxy (wrist links not in rigid_body_state due to no collision mesh)
        # This matches the actual physical end of the arm that can be tracked
        self.wrist_indices = self.elbow_indices
        print(f"[HRL] Using elbow as wrist proxy. wrist_indices: {self.wrist_indices}")
        
        # Get torso indices
        torso_names = [s for s in self.body_names if self.cfg.asset.torso_name in s]
        self.torso_indices = torch.zeros(len(torso_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(torso_names):
            self.torso_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name)
    
    def _sample_goals(self, env_ids):
        """Sample new goals for specified environments"""
        n = len(env_ids)
        
        # Resample tasks
        self.task_ids[env_ids] = torch.randint(0, 8, (n,), device=self.device)
        
        # Reset buffers
        self.goal_value[env_ids] = 0
        self.goal_mask[env_ids] = 0
        self.task_onehot[env_ids] = 0
        
        for i, env_id in enumerate(env_ids):
            task_id = self.task_ids[env_id].item()
            self.task_onehot[env_id, task_id] = 1
            
            if task_id == 0:  # Reach: 6D wrist target (world coordinates)
                # Sample reachable wrist positions (typical arm workspace)
                # Left wrist: x=[0.2, 0.6], y=[0.1, 0.5], z=[0.7, 1.3]
                # Right wrist: x=[0.2, 0.6], y=[-0.5, -0.1], z=[0.7, 1.3]
                target = torch.zeros(6, device=self.device)
                # Left wrist target
                target[0] = 0.2 + torch.rand(1, device=self.device).item() * 0.4  # x: 0.2-0.6
                target[1] = 0.1 + torch.rand(1, device=self.device).item() * 0.4  # y: 0.1-0.5
                target[2] = 0.7 + torch.rand(1, device=self.device).item() * 0.6  # z: 0.7-1.3
                # Right wrist target
                target[3] = 0.2 + torch.rand(1, device=self.device).item() * 0.4  # x: 0.2-0.6
                target[4] = -0.5 + torch.rand(1, device=self.device).item() * 0.4  # y: -0.5--0.1
                target[5] = 0.7 + torch.rand(1, device=self.device).item() * 0.6  # z: 0.7-1.3
                self.goal_value[env_id, :6] = target
                self.goal_mask[env_id, :6] = 1
                
            elif task_id == 1:  # Button: 3D button position (reachable by left arm)
                # Button in reachable workspace: x=[0.3, 0.6], y=[0.1, 0.4], z=[0.8, 1.2]
                button_pos = torch.zeros(3, device=self.device)
                button_pos[0] = 0.3 + torch.rand(1, device=self.device).item() * 0.3  # x: 0.3-0.6
                button_pos[1] = 0.1 + torch.rand(1, device=self.device).item() * 0.3  # y: 0.1-0.4
                button_pos[2] = 0.8 + torch.rand(1, device=self.device).item() * 0.4  # z: 0.8-1.2
                self.button_pos[env_id] = button_pos
                self.goal_value[env_id, :3] = button_pos
                self.goal_mask[env_id, :3] = 1
                
            elif task_id == 2:  # Cabinet: close door (target angle = 0)
                # Cabinet handle in reachable workspace: x=[0.4, 0.7], y=[0.0, 0.3], z=[0.8, 1.1]
                self.door_angle[env_id] = 1.0  # Start open
                self.door_target[env_id] = 0.0  # Target closed
                # Store handle position for reward calculation
                handle_x = 0.4 + torch.rand(1, device=self.device).item() * 0.3
                handle_y = 0.0 + torch.rand(1, device=self.device).item() * 0.3
                handle_z = 0.8 + torch.rand(1, device=self.device).item() * 0.3
                self.goal_value[env_id, 0] = handle_x
                self.goal_value[env_id, 1] = handle_y
                self.goal_value[env_id, 2] = handle_z
                self.goal_value[env_id, 3] = 0.0  # target door angle
                self.goal_mask[env_id, :4] = 1
                
            elif task_id == 3:  # Ball: kick ball to goal
                ball_start = torch.tensor([0.8, 0, 0.2], device=self.device)
                ball_start[1] += (torch.rand(1, device=self.device) * 0.6 - 0.3).item()  # y: -0.3 to 0.3
                self.ball_pos[env_id] = ball_start
                
                goal_pos = torch.tensor([5.0, 0, 0.25], device=self.device)
                goal_pos[1] += (torch.rand(1, device=self.device) * 4.0 - 2.0).item()  # y: -2 to 2
                self.ball_target[env_id] = goal_pos
                self.goal_value[env_id, :3] = goal_pos
                self.goal_mask[env_id, :3] = 1
                
            elif task_id in [4, 5, 6, 7]:  # Box/Transfer/Lift/Carry: 3D box target
                target = torch.zeros(3, device=self.device)
                target[0] = (torch.rand(1, device=self.device) * 1.5 - 0.5).item()  # x: -0.5 to 1.0
                target[1] = (torch.rand(1, device=self.device) * 1.2 - 0.6).item()  # y: -0.6 to 0.6
                target[2] = (0.4 + torch.rand(1, device=self.device) * 0.4).item()  # z: 0.4 to 0.8
                
                self.goal_value[env_id, :3] = target
                self.goal_mask[env_id, :3] = 1
                self.box_target[env_id] = target
                
                # Init box position
                self.box_pos[env_id] = torch.tensor([0.7, 0, 0.3], device=self.device)
    
    def compute_observations(self):
        """
        Observation: State (69) + Goal (14) + Mask (14) + TaskID (8) = 105
        Privileged Obs: 303 = 3 frames × 101 dims per frame
        """
        # State (69 dims) - matches h1_walking structure
        state = torch.cat([
            self.base_lin_vel * 2.0,                    # 3
            self.base_ang_vel * 0.25,                   # 3
            self.projected_gravity,                      # 3
            self.commands[:, :3] * torch.tensor([2.0, 2.0, 0.25], device=self.device),  # 3
            (self.dof_pos - self.default_dof_pos) * 1.0,  # 19
            self.dof_vel * 0.05,                         # 19
            self.actions,                                # 19
        ], dim=-1)  # Total: 69
        
        # Full observation (for actor)
        self.obs_buf = torch.cat([
            state,                  # 69
            self.goal_value,        # 14
            self.goal_mask,         # 14
            self.task_onehot,       # 8
        ], dim=-1)  # Total: 105
        
        # Privileged observation (101 dims per frame, matching single_num_privileged_obs)
        # Structure: commands(5) + dof_pos(19) + dof_vel(19) + actions(19) + diff(19) + base(9) + extras(11) = 101
        cmd_5d = torch.zeros(self.num_envs, 5, device=self.device)
        cmd_5d[:, :3] = self.commands[:, :3] * torch.tensor([2.0, 2.0, 0.25], device=self.device)
        
        priv_single = torch.cat([
            cmd_5d,                                      # 5 (command_input)
            (self.dof_pos - self.default_dof_pos) * 1.0,  # 19
            self.dof_vel * 0.05,                         # 19
            self.actions,                                # 19
            self.dof_pos - self.default_dof_pos,         # 19 (diff as placeholder)
            self.base_lin_vel * 2.0,                     # 3
            self.base_ang_vel * 0.25,                    # 3
            self.projected_gravity,                       # 3
            torch.zeros(self.num_envs, 2, device=self.device),  # push force (2)
            torch.zeros(self.num_envs, 3, device=self.device),  # push torque (3)
            torch.ones(self.num_envs, 1, device=self.device),   # friction (1)
            torch.ones(self.num_envs, 1, device=self.device) * 0.5,  # mass (1)
            torch.ones(self.num_envs, 2, device=self.device),   # stance mask (2)
            torch.ones(self.num_envs, 2, device=self.device),   # contact mask (2)
        ], dim=-1)  # Total: 5+19+19+19+19+3+3+3+2+3+1+1+2+2 = 101
        
        # Update critic history with current privileged obs
        self.critic_history.append(priv_single)
        
        # Stack frames for critic (3 × 101 = 303)
        self.privileged_obs_buf = torch.cat(
            [self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)
    
    def compute_reward(self):
        """Compute task-specific rewards using ORIGINAL formulas from single-task envs"""
        self.rew_buf[:] = 0.0
        
        for task_id in range(8):
            mask = (self.task_ids == task_id)
            if not mask.any():
                continue
                
            if task_id == 0:  # Reach
                task_rew = self._reward_reach(mask)
            elif task_id == 1:  # Button
                task_rew = self._reward_button(mask)
            elif task_id == 2:  # Cabinet
                task_rew = self._reward_cabinet(mask)
            elif task_id == 3:  # Ball
                task_rew = self._reward_ball(mask)
            else:  # Box tasks (4-7)
                task_rew = self._reward_box_task(mask, task_id)
            
            self.rew_buf[mask] = task_rew
            
            # Track per-task rewards
            self.task_episode_rewards[self.task_names[task_id]][mask] += task_rew
            self.task_episode_lengths[self.task_names[task_id]][mask] += 1
        
        # Store reward components for logging
        for task_id in range(8):
            mask = (self.task_ids == task_id)
            if mask.any():
                task_name = self.task_names[task_id]
                self.task_reward_components[task_name]['total'] = self.rew_buf[mask].mean().item()
    
    def _reward_reach(self, mask):
        """Reach reward: wrist position to target
        
        Original: scale=5, decay=-4, formula: exp(-4 * error)
        BALANCED: scale down by 0.003 (reach ~0.03 -> ~10 with factor 333)
        Returns reward ONLY for masked envs (shape matches mask.sum())
        """
        # Get wrist positions (2 wrists × 3 dimensions = 6)
        wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :3]  # [N_masked, 2, 3]
        wrist_pos = wrist_pos.reshape(mask.sum(), 6)  # [N_masked, 6]
        target = self.goal_value[mask, :6]  # [N_masked, 6]
        
        # Mean absolute error - ORIGINAL uses decay=-4
        error = torch.mean(torch.abs(wrist_pos - target), dim=-1)  # [N_masked]
        raw_reward = 5.0 * torch.exp(-4.0 * error)  # scale=5, decay=-4 (ORIGINAL)
        return raw_reward * 333.0  # BALANCE: 0.03 * 333 = ~10
    
    def _reward_button(self, mask):
        """Button press reward: left wrist to button + right arm default
        
        Original: wrist_button_distance=5, right_arm_default=0.5, decay=-4
        BALANCED: scale down by 0.357 (button ~28 -> ~10)
        Returns reward ONLY for masked envs
        """
        # Left wrist to button (scale=5, decay=-4)
        left_wrist_pos = self.rigid_state[mask][:, self.wrist_indices[0], :3]  # [N_masked, 3]
        button_pos = self.button_pos[mask]  # [N_masked, 3]
        wrist_error = torch.mean(torch.abs(left_wrist_pos - button_pos), dim=-1)
        rew_wrist = 5.0 * torch.exp(-4.0 * wrist_error)  # scale=5, decay=-4
        
        # Right arm default position (scale=0.5, decay=-4)
        # Note: default_dof_pos is [1, num_dof], need to broadcast
        right_arm_dof = self.dof_pos[mask][:, self.right_arm_indices]
        right_arm_default = self.default_dof_pos[0, self.right_arm_indices]  # [4] - broadcast
        arm_error = torch.mean(torch.abs(right_arm_dof - right_arm_default), dim=-1)
        rew_arm = 0.5 * torch.exp(-4.0 * arm_error)  # scale=0.5, decay=-4
        
        raw_reward = rew_wrist + rew_arm
        return raw_reward * 0.357  # BALANCE: 28 * 0.357 = ~10
    
    def _reward_cabinet(self, mask):
        """Cabinet task reward: both wrists to handle + door angle
        
        Original: wrist_arti_obj_distance=5, arti_obj_dof=5, decay=-4
        BALANCED: scale up by 1.82 (cabinet ~5.5 -> ~10)
        Uses BOTH wrists (2×3=6 dims) like original
        Returns reward ONLY for masked envs
        """
        # Both wrists to door handle (scale=5, decay=-4)
        wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :3]  # [N_masked, 2, 3]
        handle_pos = self.goal_value[mask, :3]  # Handle position [N_masked, 3]
        # Expand handle_pos to match wrist shape
        wrist_handle_diff = wrist_pos - handle_pos.unsqueeze(1)  # [N_masked, 2, 3]
        wrist_error = torch.mean(torch.abs(wrist_handle_diff.reshape(mask.sum(), 6)), dim=-1)
        rew_wrist = 5.0 * torch.exp(-4.0 * wrist_error)  # scale=5, decay=-4
        
        # Door angle to target (scale=5, decay=-4)
        angle_error = torch.abs(self.door_angle[mask] - self.door_target[mask])
        rew_door = 5.0 * torch.exp(-4.0 * angle_error)  # scale=5, decay=-4
        
        raw_reward = rew_wrist + rew_door
        return raw_reward * 1.82  # BALANCE: 5.5 * 1.82 = ~10
    
    def _reward_ball(self, mask):
        """Ball kick reward: torso to ball + ball to goal
        
        Original: torso_pos=1 (decay=-4), ball_pos=5 (decay=-1)
        BALANCED: scale down by 0.182 (ball ~55 -> ~10)
        NOTE: torso reward uses ORIGINAL ball position (where ball started)
        Returns reward ONLY for masked envs
        """
        # Torso to ORIGINAL ball position (xy only, scale=1, decay=-4)
        torso_pos = self.rigid_state[mask][:, self.torso_indices[0], :2]  # [N_masked, 2]
        # Use ball_target as proxy for original ball position direction
        ori_ball_xy = self.ball_pos[mask, :2]  # Current ball pos as approximation
        torso_error = torch.mean(torch.abs(torso_pos - ori_ball_xy), dim=-1)
        rew_torso = 1.0 * torch.exp(-4.0 * torso_error)  # scale=1, decay=-4
        
        # Ball to goal (xyz, scale=5, decay=-1)
        ball_pos = self.ball_pos[mask]  # [N_masked, 3]
        goal_pos = self.ball_target[mask]  # [N_masked, 3]
        ball_error = torch.mean(torch.abs(ball_pos - goal_pos), dim=-1)
        rew_ball = 5.0 * torch.exp(-1.0 * ball_error)  # scale=5, decay=-1 (ORIGINAL)
        
        raw_reward = rew_torso + rew_ball
        return raw_reward * 0.182  # BALANCE: 55 * 0.182 = ~10
    
    def _reward_box_task(self, mask, task_id):
        """Box manipulation reward: box position to target + wrist proximity
        
        Original scales:
        - box: box_pos=5, wrist_box_distance=5 (both wrists)
        - transfer: box_pos=5, wrist_box_distance=1 (both wrists)
        - lift: box_pos=5 (z-only), wrist_box_distance=5 (both wrists)
        - carry: box_pos=5, wrist_box_distance=5 (both wrists)
        All use decay=-4
        
        BALANCED:
        - box: ~82 * 0.122 = ~10
        - transfer: ~80 * 0.125 = ~10
        - lift: ~105 * 0.095 = ~10
        - carry: ~85 * 0.118 = ~10
        
        Returns reward ONLY for masked envs (shape matches mask.sum())
        """
        box_pos = self.box_pos[mask]        # [N_masked, 3]
        target = self.box_target[mask]      # [N_masked, 3]
        
        # For lift task, only check z-axis for box_pos
        if task_id == 6:  # Lift
            box_error = torch.abs(box_pos[:, 2] - target[:, 2])  # z-only
        else:
            box_error = torch.mean(torch.abs(box_pos - target), dim=-1)  # all axes
        
        rew_box = 5.0 * torch.exp(-4.0 * box_error)  # scale=5, decay=-4 (ORIGINAL)
        
        # BOTH wrists to box proximity (matches original wrist_box_distance)
        wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :3]  # [N_masked, 2, 3]
        box_pos_expanded = box_pos.unsqueeze(1)  # [N_masked, 1, 3]
        wrist_box_diff = wrist_pos - box_pos_expanded  # [N_masked, 2, 3]
        wrist_error = torch.mean(torch.abs(wrist_box_diff.reshape(mask.sum(), 6)), dim=-1)
        
        # Scale: transfer=1, others=5 (ORIGINAL)
        wrist_scale = 1.0 if task_id == 5 else 5.0
        rew_grasp = wrist_scale * torch.exp(-4.0 * wrist_error)  # decay=-4
        
        raw_reward = rew_box + rew_grasp
        
        # BALANCE factors per task_id
        balance_factors = {
            4: 0.122,  # box: 82 * 0.122 = ~10
            5: 0.125,  # transfer: 80 * 0.125 = ~10
            6: 0.095,  # lift: 105 * 0.095 = ~10
            7: 0.118,  # carry: 85 * 0.118 = ~10
        }
        return raw_reward * balance_factors[task_id]
    
    def reset_idx(self, env_ids):
        """Reset specified environments with per-task stats logging"""
        
        # === AGGREGATE TASK STATS BEFORE RESET ===
        for env_id in env_ids:
            task_id = self.task_ids[env_id].item()
            task_name = self.task_names[task_id]
            
            # Get episode reward for this task
            ep_reward = self.task_episode_rewards[task_name][env_id].item()
            ep_length = self.task_episode_lengths[task_name][env_id].item()
            
            if ep_length > 0:
                # Update running average reward
                n = self.task_episode_counts[task_name]
                old_avg = self.task_avg_rewards[task_name]
                self.task_avg_rewards[task_name] = (old_avg * n + ep_reward) / (n + 1)
                self.task_episode_counts[task_name] += 1
        
        # Reset episode trackers for these envs
        for name in self.task_names:
            self.task_episode_rewards[name][env_ids] = 0.0
            self.task_episode_lengths[name][env_ids] = 0.0
        
        # Call parent reset
        super().reset_idx(env_ids)
        
        # Sample new goals for reset envs
        self._sample_goals(env_ids)
    
    def get_task_rewards(self):
        """Get current average rewards per task for logging"""
        return {name: self.task_avg_rewards[name] for name in self.task_names}
    
    def get_task_stats(self):
        """Get task statistics for logging (called by train_hrl.py)
        
        Returns dict with keys like 'episode_reward/reach', 'step_reward/reach', etc.
        """
        stats = {}
        
        # Episode rewards (cumulative)
        for name in self.task_names:
            stats[f'episode_reward/{name}'] = self.task_avg_rewards[name]
        
        # Step rewards (current instantaneous from task_reward_components)
        for name in self.task_names:
            if name in self.task_reward_components:
                stats[f'step_reward/{name}'] = self.task_reward_components[name].get('total', 0.0)
            else:
                stats[f'step_reward/{name}'] = 0.0
        
        # Raw data for advanced logging
        stats['rewards'] = self.task_avg_rewards.copy()
        stats['counts'] = self.task_episode_counts.copy()
        
        return stats
    
    def step(self, actions):
        """Step the environment with logging"""
        obs, rew, dones, infos, privileged_obs = super().step(actions)
        
        # infos is a dict from base class, add our task-specific info
        if isinstance(infos, dict):
            infos['task_rewards'] = self.get_task_rewards()
            infos['task_counts'] = self.task_episode_counts.copy()
        
        return obs, rew, dones, infos, privileged_obs
    
    def _post_physics_step_callback(self):
        """Called after physics step - update simulated objects"""
        # Update ball position (simple physics simulation)
        ball_mask = (self.task_ids == 3)
        if ball_mask.any():
            # Ball moves based on robot proximity (simplified)
            wrist_pos = self.rigid_state[ball_mask][:, self.wrist_indices[0], :3]
            ball_dist = torch.norm(wrist_pos - self.ball_pos[ball_mask], dim=-1)
            push_mask = ball_dist < 0.3  # If wrist close to ball
            if push_mask.any():
                push_dir = self.ball_pos[ball_mask] - wrist_pos
                push_dir = push_dir / (torch.norm(push_dir, dim=-1, keepdim=True) + 1e-6)
                self.ball_pos[ball_mask] += push_dir * 0.05 * push_mask.unsqueeze(-1).float()
        
        # Update door angle (simplified)
        cabinet_mask = (self.task_ids == 2)
        if cabinet_mask.any():
            # Door closes when wrist is near handle
            wrist_pos = self.rigid_state[cabinet_mask][:, self.wrist_indices[0], :3]
            handle_pos = self.goal_value[cabinet_mask, :3]
            handle_dist = torch.norm(wrist_pos - handle_pos, dim=-1)
            close_mask = handle_dist < 0.2
            if close_mask.any():
                self.door_angle[cabinet_mask] -= 0.02 * close_mask.float()
                self.door_angle[cabinet_mask] = torch.clamp(self.door_angle[cabinet_mask], 0, 1)
