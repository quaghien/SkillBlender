================================================================================
COMMAND TABLE: Low-Level Skills & High-Level Tasks
================================================================================

LOW-LEVEL PRIMITIVE SKILLS
================================================================================

Skill      | Command Type           | Dim | Meaning                        | Range
-----------|------------------------|-----|--------------------------------|----------------------------------
Walking    | Velocity               | 3D  | vel_x, vel_y, ang_vel_yaw     | [-1,2], [-1,1], [-1,1] m/s,rad/s
Reaching   | End-effector error     | 14D | wrist_pos - ref_wrist (2h x7D) | N/A
Stepping   | Feet position          | 6D  | ref_feet_pos (2 feet x 3D)     | N/A
Squatting  | (None)                 | 0D  | Reward-driven (height)         | N/A

HIGH-LEVEL LOCO-MANIPULATION TASKS
================================================================================

Task       | Command Type           | Dim | Meaning                        | Success Condition
-----------|------------------------|-----|--------------------------------|-------------------
Task_Ball  | Object error           | 3D  | ball_pos - goal_pos            | |error| < 0.5m
Task_Box   | Object error           | 3D  | box_pos - goal_pos             | |error| < threshold
Task_Button| Distance               | 3D  | wrist_pos - button_pos         | Reach & press
Task_Reach | End-effector error     | 14D | wrist_pos - ref_wrist (2h x7D) | Both reach target
Task_Lift  | Height error           | 1D  | object_height - target_height  | At target height
Task_Carry | Object error           | 3D  | object_pos - goal_pos          | At goal position
Task_Trans | Transfer error         | 3D  | object_pos - goal_pos          | A -> B transfer
Task_Cabi  | Distance               | 3D  | wrist_pos - handle_pos         | Open/close door

================================================================================
DIMENSION SUMMARY
================================================================================

3D Commands:  Walking, Task_Ball, Task_Box, Task_Button, Task_Carry, Task_Transfer, Task_Cabinet
6D Commands:  Stepping
14D Commands: Reaching, Task_Reach
1D Commands:  Task_Lift
0D Commands:  Squatting

================================================================================
KEY DIFFERENCES
================================================================================

LOW-LEVEL (Walking):
- Input: VELOCITY COMMAND [vel_x, vel_y, ang_vel_yaw]
- Range: Random sampled per episode
- Meaning: "Go this fast in this direction"
- Success: Track velocity while maintaining gait

HIGH-LEVEL (Task_*):
- Input: GOAL ERROR [current_state - goal_state] (RELATIVE)
- Range: Goal position sampled per episode
- Meaning: "Object is X meters away from goal"
- Success: Minimize error to 0

CRITICAL: High-level tasks have lin_vel = [0, 0] → Robot self-determines speed!

================================================================================
COMMAND RANGES EXAMPLES
================================================================================

H1_WALKING (velocity-based):
  lin_vel_x = [-1.0, 2.0]     m/s
  lin_vel_y = [-1.0, 1.0]     m/s
  ang_vel_yaw = [-1.0, 1.0]   rad/s

H1_TASK_BALL (error-based):
  lin_vel_x = [0, 0]          (ZERO velocity!)
  lin_vel_y = [0, 0]          (ZERO velocity!)
  ang_vel_yaw = [0, 0]        (ZERO velocity!)
  goal_x = [5.0, 5.0]         m (fixed)
  goal_y = [-2.0, 2.0]        m (random)
  goal_z = [0, 0.5]           m (random)
  threshold = 0.5             m (success)

================================================================================
SIM-TO-REAL TRANSFER
================================================================================

LOW-LEVEL (Walking):
  Sim: "Go 1.0 m/s forward"
  Real: Send velocity command to robot
  Transfer: DIRECT & EASY

HIGH-LEVEL (Task_Ball):
  Sim: error = [ball_x - goal_x, ball_y - goal_y, ball_z - goal_z]
  Real: Detect ball + define goal → compute error (SAME!)
  Transfer: EASY (relative signals)
  No absolute world coordinates needed!

================================================================================
