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
        num_envs = 8192  # 8x reduced, 1024 per task
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
        # PD Drive parameters (matching original single-task configs)
        stiffness = {'hip_yaw': 200,
                     'hip_roll': 200,
                     'hip_pitch': 200,
                     'knee': 300,
                     'ankle': 40,
                     'torso': 300,
                     'shoulder': 100,
                     'elbow': 100,
                     }  # [N*m/rad]
        damping = {'hip_yaw': 5,
                   'hip_roll': 5,
                   'hip_pitch': 5,
                   'knee': 6,
                   'ankle': 2,
                   'torso': 6,
                   'shoulder': 2,
                   'elbow': 2,
                   }  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 10  # 100 Hz control (MUST match pretrained skills!)
        
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/h1/urdf/h1_wrist.urdf'
        name = 'h1'
        foot_name = 'ankle'
        knee_name = 'knee'
        elbow_name = 'elbow'
        wrist_name = 'wrist'  # Note: wrist links have no collision, use elbow as proxy
        torso_name = 'torso'
        terminate_after_contacts_on = ['pelvis', 'torso', 'shoulder', 'elbow']
        penalize_contacts_on = ['hip', 'knee', 'pelvis', 'torso', 'shoulder', 'elbow']  # Match original
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
    """PPO configuration for HRL training with pretrained low-level skills"""
    seed = 5  # Same as author's single-task configs (5 for reproducibility, -1 for random)
    runner_class_name = 'OnPolicyRunnerHRL'  # HRL runner
    
    class policy:
        init_noise_std = 0.5
        # High-level network architecture - LARGER for multi-task complexity
        # Input: 105 (state + goal + mask + task_id)
        # Output: 26 (4 skill logits + 22 command dims)
        actor_hidden_dims = [512, 256, 128]  # ~400k params, same as original single-task
        critic_hidden_dims = [768, 256, 128]  # Larger for 303D privileged obs
        activation = 'elu'
        
        # Low-level skill parameters
        num_skills = 4
        frame_stack = 1
        command_dim = 3  # Original env command dim (will be replaced per skill)
        num_dofs = 19
        hold_steps = 5  # Hold skill for 5 steps (reduces jitter)
        
        # Pretrained skill configs with per-dimension command ranges
        # Format: 'low_high': ([low_per_dim], [high_per_dim])
        skill_dict = {
            'h1_walking': {
                # command_dim=3: [lin_vel_x, lin_vel_y, ang_vel_yaw]
                # Output clamp to [-1, 1] then scale by [2.0, 2.0, 1.0]
                # Final range: vel_x ∈ [-2, 2], vel_y ∈ [-2, 2], yaw ∈ [-1, 1]
                'experiment_name': 'h1_walking',
                'load_run': '0000_best',
                'checkpoint': -1,
                'low_high': (-1.0, 1.0)  # Clamp before scale
            },
            'h1_reaching': {
                # command_dim=14: [l_wrist_xyz(3), l_wrist_quat(4), r_wrist_xyz(3), r_wrist_quat(4)]
                'experiment_name': 'h1_reaching',
                'load_run': '0000_best',
                'checkpoint': -1,
                'low_high': (
                    [-0.10, -0.10, -0.25, -1, -1, -1, -1, -0.10, -0.25, -0.25, -1, -1, -1, -1],
                    [0.25, 0.25, 0.25, 1, 1, 1, 1, 0.25, 0.10, 0.25, 1, 1, 1, 1]
                )
            },
            'h1_squatting': {
                # command_dim=1: [root_height]
                'experiment_name': 'h1_squatting',
                'load_run': '0000_best',
                'checkpoint': -1,
                'low_high': (
                    [0.2],   # min height
                    [1.1]    # max height
                )
            },
            'h1_stepping': {
                # command_dim=4: [l_foot_x, l_foot_y, r_foot_x, r_foot_y]
                'experiment_name': 'h1_stepping',
                'load_run': '0000_best',
                'checkpoint': -1,
                'low_high': (
                    [-0.25, -0.25, -0.25, -0.25],
                    [0.25, 0.25, 0.25, 0.25]
                )
            },
        }
        
    class algorithm:
        # Standard PPO parameters
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01  # Action entropy
        num_learning_epochs = 5
        num_mini_batches = 64
        learning_rate = 1e-4
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
        
        # Simple curriculum (linear decay, no stages)
        c_ent_skill = 0.05    # Skill entropy for diversity (constant)
        total_iterations = 8000  # 8 tasks × 1000 iters each
        K_start = 5
        K_end = 5
        epsilon_start = 0.18
        epsilon_end = 0.0
        tau_start = 2.0
        tau_end = 1.0
        
    class runner:
        policy_class_name = 'ActorCriticHRL'  # HRL policy
        algorithm_class_name = 'PPO_HRL'      # HRL algorithm
        num_steps_per_env = 60
        max_iterations = 8000  # 8 tasks × 1000 iters each
        save_interval = 500
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
    # Task order: Easy → Medium → Hard (from SkillBlender paper)
    # Easy:   reach, button, cabinet
    # Medium: ball, box, lift  
    # Hard:   transfer, carry
    task_names = ['reach', 'button', 'cabinet', 'ball', 'box', 'lift', 'transfer', 'carry']
    
    def __init__(self, cfg: H1HRLCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg = cfg
        
        # CRITICAL: Override num_obs for 8-task setup (State 69 + Goal 14 + Mask 14 + TaskID 8 = 105)
        self.num_obs = cfg.env.num_observations
        
        # Task assignment per env - BALANCED SAMPLING
        # Each task gets equal number of envs (num_envs / 8)
        self.task_ids = self._init_balanced_tasks()
        
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
        self.ball_ori_pos = torch.zeros(self.num_envs, 3, device=self.device)  # ORIGINAL ball position (for torso reward)
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
        
        # Get elbow indices
        elbow_names = [s for s in self.body_names if self.cfg.asset.elbow_name in s]
        self.elbow_indices = torch.zeros(len(elbow_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(elbow_names):
            self.elbow_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name)
        
        # Get wrist indices - fallback to elbow if wrist not found
        # (H1 URDF has wrist as fixed joints without collision, so they may not appear as rigid bodies)
        wrist_names = [s for s in self.body_names if self.cfg.asset.wrist_name in s]
        if len(wrist_names) > 0:
            self.wrist_indices = torch.zeros(len(wrist_names), dtype=torch.long, device=self.device)
            for i, name in enumerate(wrist_names):
                self.wrist_indices[i] = self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], name)
            print(f"[HRL] Using REAL wrist indices: {self.wrist_indices}")
        else:
            # Fallback to elbow indices
            self.wrist_indices = self.elbow_indices.clone()
            print(f"[HRL] Wrist not found, using elbow indices as proxy: {self.wrist_indices}")
        
        # Get torso indices
        torso_names = [s for s in self.body_names if self.cfg.asset.torso_name in s]
        self.torso_indices = torch.zeros(len(torso_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(torso_names):
            self.torso_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name)
    
    def set_task_weights(self, weights):
        """Set task sampling weights from curriculum.
        
        Called by runner each iteration to update task distribution.
        Args:
            weights: list of 8 weights (sum = 1.0)
        """
        self.task_weights = torch.tensor(weights, dtype=torch.float, device=self.device)
        # Redistribute envs to match new weights
        self._redistribute_tasks()
    
    def _redistribute_tasks(self):
        """Redistribute all envs to match current task_weights."""
        # Calculate target counts per task
        target_counts = (self.task_weights * self.num_envs).long()
        # Handle rounding errors by adding remainder to focus task
        remainder = self.num_envs - target_counts.sum()
        if remainder > 0:
            focus_task = torch.argmax(self.task_weights)
            target_counts[focus_task] += remainder
        
        # Build new task_ids tensor
        new_task_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        start_idx = 0
        for task_id in range(8):
            count = target_counts[task_id].item()
            if count > 0:
                new_task_ids[start_idx:start_idx + count] = task_id
                start_idx += count
        
        # Shuffle to avoid spatial correlation
        perm = torch.randperm(self.num_envs, device=self.device)
        new_task_ids = new_task_ids[perm]
        
        # Update task assignments
        self.task_ids = new_task_ids
        
        # CRITICAL: Resample goals for new task distribution!
        self._sample_goals(torch.arange(self.num_envs, device=self.device))
        
        # Log the new distribution
        counts = [(self.task_ids == i).sum().item() for i in range(8)]
        print(f"[HRL] Task redistribution: {counts} envs per task")
    
    def _init_balanced_tasks(self):
        """Initialize tasks for phase 0: 100% task 0 (reach)
        
        Curriculum will call set_task_weights() to change distribution later.
        """
        # Phase 0: All envs start with task 0
        self.task_weights = torch.zeros(8, dtype=torch.float, device=self.device)
        self.task_weights[0] = 1.0  # 100% task 0
        
        task_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # All envs = task 0
        print(f"[HRL] Phase 0 init: All {self.num_envs} envs → task 0 (reach)")
        return task_ids
    
    def _sample_balanced_tasks(self, env_ids):
        """Sample tasks to MAINTAIN balanced distribution across all envs.
        
        Strategy: Assign tasks to prioritize under-represented tasks.
        This prevents task collapse when some tasks have higher mortality.
        """
        n = len(env_ids)
        
        # Get current task counts (excluding envs being reset)
        remaining_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        remaining_mask[env_ids] = False
        current_counts = torch.zeros(8, device=self.device)
        for task_id in range(8):
            current_counts[task_id] = ((self.task_ids == task_id) & remaining_mask).sum()
        
        # Target counts after reset
        target_per_task = self.num_envs // 8
        
        # Calculate how many envs each task needs to reach target
        deficit = target_per_task - current_counts
        deficit = torch.clamp(deficit, min=0)  # Only consider under-represented tasks
        
        # Assign tasks proportionally to deficit
        new_task_ids = torch.zeros(n, dtype=torch.long, device=self.device)
        if deficit.sum() > 0:
            # Weighted sampling based on deficit
            weights = deficit / deficit.sum()
            for i in range(n):
                # Sample task weighted by how much it's under-represented
                task_probs = weights.clone()
                # Add small uniform probability to avoid complete starvation
                task_probs = 0.9 * task_probs + 0.1 / 8
                new_task_ids[i] = torch.multinomial(task_probs, 1).item()
                # Update weights to reflect this assignment
                deficit[new_task_ids[i]] -= 1
                deficit = torch.clamp(deficit, min=0)
                if deficit.sum() > 0:
                    weights = deficit / deficit.sum()
                else:
                    weights = torch.ones(8, device=self.device) / 8
        else:
            # All tasks at or above target, random assignment
            new_task_ids = torch.randint(0, 8, (n,), device=self.device)
        
        return new_task_ids
    
    def _sample_goals(self, env_ids):
        """Sample new goals for specified environments.
        
        NOTE: Task assignment is FIXED at init. Here we only sample NEW GOALS
        for the same task. This maintains balanced task distribution.
        """
        n = len(env_ids)
        
        # DO NOT resample tasks! Keep original task assignment.
        # self.task_ids[env_ids] = self._sample_balanced_tasks(env_ids)  # REMOVED
        
        # Reset goal buffers only
        self.goal_value[env_ids] = 0
        self.goal_mask[env_ids] = 0
        self.task_onehot[env_ids] = 0
        
        for i, env_id in enumerate(env_ids):
            task_id = self.task_ids[env_id].item()  # Keep original task!
            self.task_onehot[env_id, task_id] = 1
            
            # Get robot base position in world frame for coordinate conversion
            # root_states: [num_envs, 13] = [pos(3), quat(4), lin_vel(3), ang_vel(3)]
            robot_base_pos = self.root_states[env_id, :3]  # [x, y, z] in world frame
            
            if task_id == 0:  # Reach: 6D wrist target (WORLD coordinates)
                # Sample reachable wrist positions in ROBOT FRAME, then convert to world
                # Robot frame: x=forward, y=left, z=up (relative to robot base)
                # Left wrist: x=[0.2, 0.6], y=[0.1, 0.5], z=[0.7, 1.3] (robot frame)
                # Right wrist: x=[0.2, 0.6], y=[-0.5, -0.1], z=[0.7, 1.3] (robot frame)
                target_local = torch.zeros(6, device=self.device)
                # Left wrist target (robot frame)
                target_local[0] = 0.2 + torch.rand(1, device=self.device).item() * 0.4  # x: 0.2-0.6
                target_local[1] = 0.1 + torch.rand(1, device=self.device).item() * 0.4  # y: 0.1-0.5
                target_local[2] = 0.7 + torch.rand(1, device=self.device).item() * 0.6  # z: 0.7-1.3
                # Right wrist target (robot frame)
                target_local[3] = 0.2 + torch.rand(1, device=self.device).item() * 0.4  # x: 0.2-0.6
                target_local[4] = -0.5 + torch.rand(1, device=self.device).item() * 0.4  # y: -0.5--0.1
                target_local[5] = 0.7 + torch.rand(1, device=self.device).item() * 0.6  # z: 0.7-1.3
                
                # Convert to world frame: add robot base position
                target_world = target_local.clone()
                target_world[0:3] += robot_base_pos  # Left wrist
                target_world[3:6] += robot_base_pos  # Right wrist
                
                self.goal_value[env_id, :6] = target_world
                self.goal_mask[env_id, :6] = 1
                
            elif task_id == 1:  # Button: 3D button position (WORLD coordinates)
                # Button in reachable workspace (robot frame): x=[0.3, 0.6], y=[0.1, 0.4], z=[0.8, 1.2]
                button_local = torch.zeros(3, device=self.device)
                button_local[0] = 0.3 + torch.rand(1, device=self.device).item() * 0.3  # x: 0.3-0.6
                button_local[1] = 0.1 + torch.rand(1, device=self.device).item() * 0.3  # y: 0.1-0.4
                button_local[2] = 0.8 + torch.rand(1, device=self.device).item() * 0.4  # z: 0.8-1.2
                
                # Convert to world frame
                button_world = button_local + robot_base_pos
                self.button_pos[env_id] = button_world
                self.goal_value[env_id, :3] = button_world
                self.goal_mask[env_id, :3] = 1
                
            elif task_id == 2:  # Cabinet: close door (target angle = 0)
                # Cabinet handle in reachable workspace (robot frame): x=[0.4, 0.7], y=[0.0, 0.3], z=[0.8, 1.1]
                self.door_angle[env_id] = 1.0  # Start open
                self.door_target[env_id] = 0.0  # Target closed
                
                # Handle position in robot frame
                handle_local = torch.zeros(3, device=self.device)
                handle_local[0] = 0.4 + torch.rand(1, device=self.device).item() * 0.3
                handle_local[1] = 0.0 + torch.rand(1, device=self.device).item() * 0.3
                handle_local[2] = 0.8 + torch.rand(1, device=self.device).item() * 0.3
                
                # Convert to world frame
                handle_world = handle_local + robot_base_pos
                self.goal_value[env_id, 0] = handle_world[0]
                self.goal_value[env_id, 1] = handle_world[1]
                self.goal_value[env_id, 2] = handle_world[2]
                self.goal_value[env_id, 3] = 0.0  # target door angle
                self.goal_mask[env_id, :4] = 1
                
            elif task_id == 3:  # Ball: kick ball to goal
                # Ball position relative to robot, then convert to world
                ball_local = torch.tensor([0.8, 0, 0.2], device=self.device)
                ball_local[1] += (torch.rand(1, device=self.device) * 0.6 - 0.3).item()  # y: -0.3 to 0.3
                
                ball_world = ball_local + robot_base_pos
                self.ball_pos[env_id] = ball_world
                self.ball_ori_pos[env_id] = ball_world.clone()  # Store ORIGINAL position for torso reward
                
                # Goal is further away (world coordinates)
                goal_offset = torch.tensor([5.0, 0, 0.05], device=self.device)  # z=0.05 (on ground)
                goal_offset[1] += (torch.rand(1, device=self.device) * 4.0 - 2.0).item()  # y: -2 to 2
                goal_world = goal_offset + robot_base_pos
                self.ball_target[env_id] = goal_world
                self.goal_value[env_id, :3] = goal_world
                self.goal_mask[env_id, :3] = 1
                
            elif task_id in [4, 5, 6, 7]:  # Box/Transfer/Lift/Carry: 3D box target
                # Box target in robot frame
                target_local = torch.zeros(3, device=self.device)
                target_local[0] = (torch.rand(1, device=self.device) * 1.5 - 0.5).item()  # x: -0.5 to 1.0
                target_local[1] = (torch.rand(1, device=self.device) * 1.2 - 0.6).item()  # y: -0.6 to 0.6
                target_local[2] = (0.4 + torch.rand(1, device=self.device) * 0.4).item()  # z: 0.4 to 0.8
                
                # Convert to world frame
                target_world = target_local + robot_base_pos
                self.goal_value[env_id, :3] = target_world
                self.goal_mask[env_id, :3] = 1
                self.box_target[env_id] = target_world
                
                # Init box position near robot
                box_local = torch.tensor([0.7, 0, 0.3], device=self.device)
                box_world = box_local + robot_base_pos
                self.box_pos[env_id] = box_world
    
    def compute_observations(self):
        """
        Observation: State (69) + Goal (14) + Mask (14) + TaskID (8) = 105
        Privileged Obs: 303 = 3 frames × 101 dims per frame
        
        NOTE: HRL uses a unified observation format. The high-level policy
        learns to interpret state + goal_value to generate skill commands.
        """
        # State (69 dims) - matches h1_walking structure
        # Command scale: [lin_vel=2.0, lin_vel=2.0, ang_vel=1.0] (NOT 0.25 for yaw!)
        state = torch.cat([
            self.base_lin_vel * 2.0,                    # 3
            self.base_ang_vel * 0.25,                   # 3
            self.projected_gravity,                      # 3
            self.commands[:, :3] * torch.tensor([2.0, 2.0, 1.0], device=self.device),  # 3 (FIXED: yaw scale = 1.0)
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
        cmd_5d[:, :3] = self.commands[:, :3] * torch.tensor([2.0, 2.0, 1.0], device=self.device)  # FIXED: yaw scale = 1.0
        
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
        """Compute task-specific rewards using ORIGINAL formulas + SHAPED REWARDS
        
        Structure for each task:
        1. BASE REWARD: Original exponential reward (matches single-task training)
        2. PROGRESS BONUS: Reward for reducing error vs previous step
        3. SUCCESS BONUS: Extra reward when error below threshold
        
        Metrics logged:
        - task_<name>_<error>: Raw error values for monitoring
        - task_<name>_progress: Progress (delta error) - positive = improving
        """
        self.rew_buf[:] = 0.0
        
        # Initialize metrics storage for this step
        if not hasattr(self, 'task_metrics'):
            self.task_metrics = {}
        
        # Initialize previous errors for progress tracking (first call)
        if not hasattr(self, 'prev_errors'):
            self.prev_errors = {
                'reach_wrist': torch.zeros(self.num_envs, device=self.device),
                'button_wrist': torch.zeros(self.num_envs, device=self.device),
                'cabinet_wrist': torch.zeros(self.num_envs, device=self.device),
                'cabinet_door': torch.zeros(self.num_envs, device=self.device),
                'ball_torso': torch.zeros(self.num_envs, device=self.device),
                'ball_goal': torch.zeros(self.num_envs, device=self.device),
                'box_box': torch.zeros(self.num_envs, device=self.device),
                'box_wrist': torch.zeros(self.num_envs, device=self.device),
            }
        
        # Reward shaping config (can be tuned)
        cfg_shaping = {
            'progress_scale': 1.0,      # Scale for progress bonus
        }
        
        # Task difficulty scales - Task khó cần reward cao hơn để học tốt
        # Easy: reach, button, cabinet → x1.0 (baseline)
        # Medium: ball, box, lift → x1.2-1.5
        # Hard: transfer, carry → x1.5-2.0
        TASK_DIFFICULTY_SCALES = {
            0: 1.0,   # reach (Easy)
            1: 1.0,   # button (Easy)
            2: 0.6,   # cabinet (Easy - giảm vì base=10 quá cao)
            3: 1.5,   # ball (Medium)
            4: 1.0,   # box (Medium - base=10 đã cao)
            5: 2.0,   # transfer (Hard - tăng vì base=6 quá thấp)
            6: 1.0,   # lift (Medium - base=10 đã cao)
            7: 1.3,   # carry (Hard)
        }
        
        for task_id in range(8):
            mask = (self.task_ids == task_id)
            if not mask.any():
                continue
            
            task_name = self.task_names[task_id]
                
            if task_id == 0:  # Reach
                task_rew, metrics, shaped_rew = self._reward_reach_shaped(mask, cfg_shaping)
            elif task_id == 1:  # Button
                task_rew, metrics, shaped_rew = self._reward_button_shaped(mask, cfg_shaping)
            elif task_id == 2:  # Cabinet
                task_rew, metrics, shaped_rew = self._reward_cabinet_shaped(mask, cfg_shaping)
            elif task_id == 3:  # Ball
                task_rew, metrics, shaped_rew = self._reward_ball_shaped(mask, cfg_shaping)
            else:  # Box tasks (4-7)
                task_rew, metrics, shaped_rew = self._reward_box_task_shaped(mask, task_id, cfg_shaping)
            
            # Total reward = (base + shaped) * difficulty_scale
            difficulty_scale = TASK_DIFFICULTY_SCALES[task_id]
            total_rew = (task_rew + shaped_rew) * difficulty_scale
            self.rew_buf[mask] = total_rew
            
            # Store metrics for this task
            for metric_name, value in metrics.items():
                self.task_metrics[f'task_{task_name}_{metric_name}'] = value
            
            # Track per-task rewards
            self.task_episode_rewards[task_name][mask] += total_rew
            self.task_episode_lengths[task_name][mask] += 1
        
        # Store reward components for logging
        for task_id in range(8):
            mask = (self.task_ids == task_id)
            if mask.any():
                task_name = self.task_names[task_id]
                self.task_reward_components[task_name]['total'] = self.rew_buf[mask].mean().item()
    
    # ===================== SHAPED REWARD FUNCTIONS =====================
    
    def _reward_reach_shaped(self, mask, cfg):
        """Reach reward with progress shaping"""
        # BASE: Original reward
        base_rew, metrics = self._reward_reach_with_metrics(mask)
        
        # Get current error
        wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :3]
        target = self.goal_value[mask, :6].reshape(-1, 2, 3)
        wrist_pos_diff = torch.flatten(wrist_pos - target, start_dim=1)
        curr_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)
        
        # PROGRESS: Reward for reducing error
        prev_error = self.prev_errors['reach_wrist'][mask]
        # Skip progress reward on first step (prev_error=0 means just reset)
        valid_prev = prev_error > 0  # Only compute progress if we have valid prev
        progress = torch.where(valid_prev, prev_error - curr_error, torch.zeros_like(curr_error))
        progress_rew = cfg['progress_scale'] * torch.clamp(progress, -0.5, 0.5)  # Clamp extremes
        
        # Update previous error
        self.prev_errors['reach_wrist'][mask] = curr_error.detach()
        
        # Total shaped reward (progress only)
        shaped_rew = progress_rew
        
        # Add metrics
        metrics['progress'] = progress.mean().item()
        
        return base_rew, metrics, shaped_rew
    
    def _reward_button_shaped(self, mask, cfg):
        """Button reward with progress shaping"""
        # BASE: Original reward
        base_rew, metrics = self._reward_button_with_metrics(mask)
        
        # Get current error (wrist to button)
        left_wrist_pos = self.rigid_state[mask][:, self.wrist_indices[0], :3]
        button_pos = self.button_pos[mask]
        wrist_button_diff = left_wrist_pos - button_pos
        curr_error = torch.mean(torch.abs(wrist_button_diff), dim=1)
        
        # PROGRESS: Reward for reducing error
        prev_error = self.prev_errors['button_wrist'][mask]
        # Skip progress reward on first step (prev_error=0 means just reset)
        valid_prev = prev_error > 0
        progress = torch.where(valid_prev, prev_error - curr_error, torch.zeros_like(curr_error))
        progress_rew = cfg['progress_scale'] * torch.clamp(progress, -0.5, 0.5)
        
        # Update previous error
        self.prev_errors['button_wrist'][mask] = curr_error.detach()
        
        # Total shaped reward (progress only)
        shaped_rew = progress_rew
        metrics['progress'] = progress.mean().item()
        
        return base_rew, metrics, shaped_rew
    
    def _reward_cabinet_shaped(self, mask, cfg):
        """Cabinet reward with progress shaping"""
        # BASE: Original reward
        base_rew, metrics = self._reward_cabinet_with_metrics(mask)
        
        # Get current errors
        wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :3]
        arti_obj_pos = self.goal_value[mask, :3]
        wrist_arti_obj_diff = torch.flatten(wrist_pos - arti_obj_pos.unsqueeze(1), start_dim=1)
        curr_wrist_error = torch.mean(torch.abs(wrist_arti_obj_diff), dim=1)
        
        door_diff = self.door_angle[mask] - self.door_target[mask]
        curr_door_error = torch.abs(door_diff)
        
        # PROGRESS: Combined wrist + door progress
        prev_wrist = self.prev_errors['cabinet_wrist'][mask]
        prev_door = self.prev_errors['cabinet_door'][mask]
        
        # Skip progress reward on first step (prev_error=0 means just reset)
        valid_prev = (prev_wrist > 0) | (prev_door > 0)
        progress_wrist = torch.where(valid_prev, prev_wrist - curr_wrist_error, torch.zeros_like(curr_wrist_error))
        progress_door = torch.where(valid_prev, prev_door - curr_door_error, torch.zeros_like(curr_door_error))
        
        # Weight wrist:door = 1:2 (door is harder)
        progress = progress_wrist + 2.0 * progress_door
        progress_rew = cfg['progress_scale'] * torch.clamp(progress, -0.5, 0.5)
        
        # Update previous errors
        self.prev_errors['cabinet_wrist'][mask] = curr_wrist_error.detach()
        self.prev_errors['cabinet_door'][mask] = curr_door_error.detach()
        
        # Total shaped reward (progress only)
        shaped_rew = progress_rew
        metrics['progress'] = progress.mean().item()
        
        return base_rew, metrics, shaped_rew
    
    def _reward_ball_shaped(self, mask, cfg):
        """Ball reward with progress shaping"""
        # BASE: Original reward
        base_rew, metrics = self._reward_ball_with_metrics(mask)
        
        # Get current errors
        torso_pos = self.rigid_state[mask][:, self.torso_indices[0], :3].squeeze(1)
        torso_ball_diff = self.ball_ori_pos[mask] - torso_pos
        curr_torso_error = torch.mean(torch.abs(torso_ball_diff[:, :2]), dim=1)  # XY only
        
        ball_goal_diff = self.ball_pos[mask] - self.ball_target[mask]
        curr_goal_error = torch.mean(torch.abs(ball_goal_diff), dim=1)
        
        # PROGRESS: Combined torso + ball_goal progress
        prev_torso = self.prev_errors['ball_torso'][mask]
        prev_goal = self.prev_errors['ball_goal'][mask]
        
        # Skip progress reward on first step (prev_error=0 means just reset)
        valid_prev = (prev_torso > 0) | (prev_goal > 0)
        progress_torso = torch.where(valid_prev, prev_torso - curr_torso_error, torch.zeros_like(curr_torso_error))
        progress_goal = torch.where(valid_prev, prev_goal - curr_goal_error, torch.zeros_like(curr_goal_error))
        
        # Weight torso:goal = 1:3 (goal is the objective)
        progress = progress_torso + 3.0 * progress_goal
        progress_rew = cfg['progress_scale'] * torch.clamp(progress, -0.5, 0.5)
        
        # Update previous errors
        self.prev_errors['ball_torso'][mask] = curr_torso_error.detach()
        self.prev_errors['ball_goal'][mask] = curr_goal_error.detach()
        
        # Total shaped reward (progress only)
        shaped_rew = progress_rew
        metrics['progress'] = progress.mean().item()
        
        return base_rew, metrics, shaped_rew
    
    def _reward_box_task_shaped(self, mask, task_id, cfg):
        """Box task reward with progress shaping"""
        # BASE: Original reward
        base_rew, metrics = self._reward_box_task_with_metrics(mask, task_id)
        
        # Get current errors
        box_pos = self.box_pos[mask]
        target = self.box_target[mask]
        box_pos_diff = box_pos - target
        curr_box_error = torch.mean(torch.abs(box_pos_diff), dim=1)
        
        wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :3]
        wrist_box_diff = torch.flatten(wrist_pos - box_pos.unsqueeze(1), start_dim=1)
        curr_wrist_error = torch.mean(torch.abs(wrist_box_diff), dim=1)
        
        # PROGRESS: Combined box + wrist progress
        prev_box = self.prev_errors['box_box'][mask]
        prev_wrist = self.prev_errors['box_wrist'][mask]
        
        # Skip progress reward on first step (prev_error=0 means just reset)
        valid_prev = (prev_box > 0) | (prev_wrist > 0)
        progress_box = torch.where(valid_prev, prev_box - curr_box_error, torch.zeros_like(curr_box_error))
        progress_wrist = torch.where(valid_prev, prev_wrist - curr_wrist_error, torch.zeros_like(curr_wrist_error))
        
        # Weight box:wrist varies by task
        # Box/Lift/Carry: box=2, wrist=1 (focus on moving box)
        # Transfer: box=1, wrist=1 (balanced)
        if task_id == 5:  # Transfer
            progress = progress_box + progress_wrist
        else:
            progress = 2.0 * progress_box + progress_wrist
        
        progress_rew = cfg['progress_scale'] * torch.clamp(progress, -0.5, 0.5)
        
        # Update previous errors
        self.prev_errors['box_box'][mask] = curr_box_error.detach()
        self.prev_errors['box_wrist'][mask] = curr_wrist_error.detach()
        
        # Total shaped reward (progress only)
        shaped_rew = progress_rew
        metrics['progress'] = progress.mean().item()
        
        return base_rew, metrics, shaped_rew
    
    def _reward_reach(self, mask):
        """Reach reward: wrist position to target (without metrics)"""
        rew, _ = self._reward_reach_with_metrics(mask)
        return rew
    
    def _reward_reach_with_metrics(self, mask):
        """Reach reward: wrist position to target
        
        MATCHES ORIGINAL h1_task_reach.py:
        - wrist_pos scale = 5 (from config)
        - decay = -4
        - Formula: 5 * exp(-4 * mean_abs_error)
        Returns: (reward, metrics_dict)
        """
        # Get wrist positions [N_masked, 2, 3] - MATCHES ORIGINAL
        wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :3]
        # Target from goal_value [N_masked, 6] (2 wrists × 3 dims)
        target = self.goal_value[mask, :6].reshape(-1, 2, 3)
        
        # Position diff [N_masked, 2, 3] then flatten to [N_masked, 6]
        wrist_pos_diff = wrist_pos - target
        wrist_pos_diff = torch.flatten(wrist_pos_diff, start_dim=1)  # [N_masked, 6]
        
        # Mean absolute error - EXACTLY like original
        wrist_pos_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)  # [N_masked]
        
        # Reward: scale=5, decay=-4 (from config rewards.scales.wrist_pos = 5)
        reward = 5.0 * torch.exp(-4.0 * wrist_pos_error)
        
        # Metrics
        metrics = {
            'wrist_error': wrist_pos_error.mean().item(),
        }
        
        return reward, metrics  # NO BALANCE FACTOR - matches original
    
    def _reward_button(self, mask):
        """Button press reward (without metrics)"""
        rew, _ = self._reward_button_with_metrics(mask)
        return rew
    
    def _reward_button_with_metrics(self, mask):
        """Button press reward: left wrist to button + right arm default
        
        MATCHES ORIGINAL h1_task_button.py:
        - wrist_button_distance scale = 5
        - right_arm_default scale = 0.5
        - decay = -4 for both
        Returns: (reward, metrics_dict)
        """
        # Left wrist to button (scale=5, decay=-4) - MATCHES ORIGINAL
        # Original: wrist_pos[:, 0, :3] - button_goal_pos[:, :3]
        left_wrist_pos = self.rigid_state[mask][:, self.wrist_indices[0], :3]  # [N_masked, 3]
        button_pos = self.button_pos[mask]  # [N_masked, 3]
        wrist_button_diff = left_wrist_pos - button_pos
        wrist_button_error = torch.mean(torch.abs(wrist_button_diff), dim=1)
        rew_wrist = 5.0 * torch.exp(-4.0 * wrist_button_error)  # scale=5
        
        # Right arm default position (scale=0.5, decay=-4) - MATCHES ORIGINAL
        # Original: joint_diff[:, 15:] (right_shoulder_pitch to end)
        right_shoulder_pitch_index = 15
        joint_diff = self.dof_pos[mask] - self.default_dof_pos
        right_arm_diff = joint_diff[:, right_shoulder_pitch_index:]  # joints 15-18 (4 joints)
        right_arm_error = torch.mean(torch.abs(right_arm_diff), dim=1)
        rew_arm = 0.5 * torch.exp(-4.0 * right_arm_error)  # scale=0.5
        
        # Total reward = wrist + arm (NO BALANCE FACTOR)
        reward = rew_wrist + rew_arm
        
        # Metrics
        metrics = {
            'wrist_error': wrist_button_error.mean().item(),
            'arm_error': right_arm_error.mean().item(),
        }
        
        return reward, metrics  # NO BALANCE FACTOR - matches original
    
    def _reward_cabinet(self, mask):
        """Cabinet task reward (without metrics)"""
        rew, _ = self._reward_cabinet_with_metrics(mask)
        return rew
    
    def _reward_cabinet_with_metrics(self, mask):
        """Cabinet task reward: both wrists to cabinet + door angle
        
        MATCHES ORIGINAL h1_task_cabinet.py:
        - wrist_arti_obj_distance scale = 5
        - arti_obj_dof scale = 5
        - decay = -4 for both
        Returns: (reward, metrics_dict)
        """
        # Both wrists to cabinet position (scale=5, decay=-4) - MATCHES ORIGINAL
        # Original: wrist_pos - arti_obj_pos.unsqueeze(1) then flatten to [N, 6]
        wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :3]  # [N_masked, 2, 3]
        arti_obj_pos = self.goal_value[mask, :3]  # Cabinet/handle position [N_masked, 3]
        wrist_arti_obj_diff = wrist_pos - arti_obj_pos.unsqueeze(1)  # [N_masked, 2, 3]
        wrist_arti_obj_diff = torch.flatten(wrist_arti_obj_diff, start_dim=1)  # [N_masked, 6]
        wrist_arti_obj_error = torch.mean(torch.abs(wrist_arti_obj_diff), dim=1)
        rew_wrist = 5.0 * torch.exp(-4.0 * wrist_arti_obj_error)  # scale=5
        
        # Door DOF to target (scale=5, decay=-4) - MATCHES ORIGINAL
        # Original: arti_obj_dof_state[:, :, 0] - arti_obj_dof_goal (2 DOFs)
        # Simplified: single door angle
        arti_obj_dof_diff = self.door_angle[mask] - self.door_target[mask]
        arti_obj_dof_error = torch.abs(arti_obj_dof_diff)
        rew_door = 5.0 * torch.exp(-4.0 * arti_obj_dof_error)  # scale=5
        
        # Total reward = wrist + door (NO BALANCE FACTOR)
        reward = rew_wrist + rew_door
        
        # Metrics
        metrics = {
            'wrist_error': wrist_arti_obj_error.mean().item(),
            'door_error': arti_obj_dof_error.mean().item(),
        }
        
        return reward, metrics  # NO BALANCE FACTOR - matches original
    
    def _reward_ball(self, mask):
        """Ball kick reward (without metrics)"""
        rew, _ = self._reward_ball_with_metrics(mask)
        return rew
    
    def _reward_ball_with_metrics(self, mask):
        """Ball kick reward: torso to original ball position + ball to goal
        
        MATCHES ORIGINAL h1_task_ball.py:
        - torso_pos scale = 1, decay = -4 (XY only)
        - ball_pos scale = 5, decay = -1 (XYZ, special slow decay!)
        Returns: (reward, metrics_dict)
        """
        # Torso to ORIGINAL ball position (XY only, scale=1, decay=-4) - MATCHES ORIGINAL
        # Original: ori_ball_pos - torso_pos, only [:2] for xy
        torso_pos = self.rigid_state[mask][:, self.torso_indices[0], :3].squeeze(1)  # [N_masked, 3]
        # ori_ball_pos is stored in ball_ori_pos (initial position)
        torso_ori_ball_pos_diff = self.ball_ori_pos[mask] - torso_pos
        torso_ori_ball_pos_diff = torso_ori_ball_pos_diff[:, :2]  # Only XY
        torso_ori_ball_pos_error = torch.mean(torch.abs(torso_ori_ball_pos_diff), dim=1)
        rew_torso = 1.0 * torch.exp(-4.0 * torso_ori_ball_pos_error)  # scale=1
        
        # Ball to goal (XYZ, scale=5, decay=-1) - MATCHES ORIGINAL (special decay!)
        # Original: ball_root_states[:, :3] - goal_pos
        ball_goal_diff = self.ball_pos[mask] - self.ball_target[mask]
        ball_goal_error = torch.mean(torch.abs(ball_goal_diff), dim=1)
        rew_ball = 5.0 * torch.exp(-1.0 * ball_goal_error)  # scale=5, decay=-1 (SPECIAL!)
        
        # Total reward = torso + ball (NO BALANCE FACTOR)
        reward = rew_torso + rew_ball
        
        # Metrics
        metrics = {
            'torso_error': torso_ori_ball_pos_error.mean().item(),
            'goal_error': ball_goal_error.mean().item(),
        }
        
        return reward, metrics  # NO BALANCE FACTOR - matches original
    
    def _reward_box_task(self, mask, task_id):
        """Box manipulation reward (without metrics)"""
        rew, _ = self._reward_box_task_with_metrics(mask, task_id)
        return rew
    
    def _reward_box_task_with_metrics(self, mask, task_id):
        """Box manipulation reward: box position to target + wrist to box distance
        
        MATCHES ORIGINAL h1_task_box/transfer/lift/carry.py:
        - box_pos scale = 5, decay = -4 (all tasks)
        - wrist_box_distance scale = 5 (box/lift/carry) or 1 (transfer), decay = -4
        Returns: (reward, metrics_dict)
        """
        # Box position to goal (scale=5, decay=-4) - MATCHES ORIGINAL
        # Original: box_root_states[:, :3] - box_goal_pos
        box_pos = self.box_pos[mask]        # [N_masked, 3]
        target = self.box_target[mask]      # [N_masked, 3]
        box_pos_diff = box_pos - target
        box_pos_error = torch.mean(torch.abs(box_pos_diff), dim=1)
        rew_box = 5.0 * torch.exp(-4.0 * box_pos_error)  # scale=5
        
        # Wrist to box distance (scale varies, decay=-4) - MATCHES ORIGINAL
        # Original: wrist_pos[:,:,:3] - box_pos.unsqueeze(1), flatten to [N, 6]
        wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :3]  # [N_masked, 2, 3]
        box_pos_expanded = box_pos.unsqueeze(1)  # [N_masked, 1, 3]
        wrist_box_diff = wrist_pos - box_pos_expanded  # [N_masked, 2, 3]
        wrist_pos_diff = torch.flatten(wrist_box_diff, start_dim=1)  # [N_masked, 6]
        wrist_box_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)
        
        # Scale: transfer=1, others=5 (from original configs)
        wrist_scale = 1.0 if task_id == 5 else 5.0
        rew_wrist = wrist_scale * torch.exp(-4.0 * wrist_box_error)
        
        # Total reward = box + wrist (NO BALANCE FACTOR)
        reward = rew_box + rew_wrist
        
        # Metrics
        metrics = {
            'box_error': box_pos_error.mean().item(),
            'wrist_error': wrist_box_error.mean().item(),
        }
        
        return reward, metrics  # NO BALANCE FACTOR - matches original
    
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
        
        # Reset prev_errors for progress tracking
        if hasattr(self, 'prev_errors'):
            for key in self.prev_errors:
                self.prev_errors[key][env_ids] = 0.0
        
        # Call parent reset
        super().reset_idx(env_ids)
        
        # Sample new goals for reset envs
        self._sample_goals(env_ids)
    
    def get_task_rewards(self):
        """Get current average rewards per task for logging"""
        return {name: self.task_avg_rewards[name] for name in self.task_names}
    
    def get_task_stats(self):
        """Get task statistics for logging
        
        Returns dict with:
        - Episode/rew_<task>: Episode reward per task (8 tasks)
        - Metric/<task>_<metric>: Error/progress metrics per task
        - TaskDist/<task>: Env count per task
        """
        stats = {}
        
        # Episode rewards per task (all 8 tasks, even if 0)
        for name in self.task_names:
            stats[f'Episode/rew_{name}'] = self.task_avg_rewards[name]
        
        # Task distribution (env counts)
        for i, name in enumerate(self.task_names):
            stats[f'TaskDist/{name}'] = (self.task_ids == i).sum().item()
        
        # Error metrics per task (from compute_reward)
        if hasattr(self, 'task_metrics'):
            for key, value in self.task_metrics.items():
                # key format: task_<taskname>_<metric> -> Metric/<taskname>_<metric>
                metric_key = key.replace('task_', '')
                stats[f'Metric/{metric_key}'] = value
        
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
