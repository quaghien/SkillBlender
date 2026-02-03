# HRL Reward Structure Documentation

## Overview

This document describes the reward structure for the HRL (Hierarchical Reinforcement Learning) 
meta-environment that trains a high-level policy to select and command 4 pretrained low-level skills.

## Reward Components

Each task has **3 reward components**:

### 1. BASE Reward (Position-based)
- **Formula**: `scale * exp(-4 * error)`
- **Purpose**: Rewards being close to target (absolute distance)
- **Signal Type**: Continuous, position-based

### 2. PROGRESS Reward (Velocity-based) 
- **Formula**: `progress_scale * clamp(prev_error - curr_error, -0.5, 0.5)`
- **Purpose**: Rewards **improvement** (moving toward target)
- **Signal Type**: Temporal delta (velocity of improvement)
- **Note**: Skipped on first step after reset (when `prev_error=0`)

### 3. SUCCESS Reward (Goal-based)
- **Formula**: `success_bonus * (error < threshold)`
- **Purpose**: Discrete bonus for achieving goal
- **Signal Type**: Binary/sparse

## Why NOT Redundant?

| Component | Information Used | Analogy |
|-----------|------------------|---------|
| BASE | Current error | Position |
| PROGRESS | Error delta | Velocity |
| SUCCESS | Threshold crossing | Goal achievement |

These are **complementary signals**:
- BASE: "How close am I?" (absolute)
- PROGRESS: "Am I getting closer?" (relative change)  
- SUCCESS: "Did I succeed?" (binary)

---

## Task-Specific Metrics

### Task 0: REACH
**Goal**: Move both wrists to target positions

| Metric | Formula | Expected Range |
|--------|---------|----------------|
| `wrist_error` | mean(abs(wrist_pos - target)) over 6D | 0.0-1.0m |

**Rewards**:
- Base: `5.0 * exp(-4 * wrist_error)`
- Progress: `1.0 * clamp(delta_wrist, -0.5, 0.5)`
- Success: `2.0 if wrist_error < 0.1`

---

### Task 1: BUTTON
**Goal**: Press button with left wrist, keep right arm at default

| Metric | Formula | Expected Range |
|--------|---------|----------------|
| `wrist_error` | mean(abs(left_wrist - button_pos)) over 3D | 0.0-0.5m |
| `arm_error` | mean(abs(right_arm_joints - default)) over 4 joints | 0.0-1.0 rad |

**Rewards**:
- Base wrist: `5.0 * exp(-4 * wrist_error)`
- Base arm: `0.5 * exp(-4 * arm_error)`
- Progress: `1.0 * clamp(delta_wrist, -0.5, 0.5)`
- Success: `2.0 if wrist_error < 0.1`

---

### Task 2: CABINET
**Goal**: Open cabinet door with wrists near handle

| Metric | Formula | Expected Range |
|--------|---------|----------------|
| `wrist_error` | mean(abs(wrists - handle_pos)) over 6D | 0.0-1.0m |
| `door_error` | abs(door_angle - target_angle) | 0.0-1.0 rad |

**Rewards**:
- Base wrist: `5.0 * exp(-4 * wrist_error)`
- Base door: `5.0 * exp(-4 * door_error)`
- Progress: `1.0 * clamp(delta_wrist + 2*delta_door, -0.5, 0.5)` (door weighted 2x)
- Success: `2.0 if wrist_error < 0.2 AND door_error < 0.1`

---

### Task 3: BALL
**Goal**: Walk to ball and kick it toward goal

| Metric | Formula | Expected Range |
|--------|---------|----------------|
| `torso_error` | mean(abs(torso_xy - ball_xy)) over 2D | 0.0-3.0m |
| `goal_error` | mean(abs(ball_pos - goal_pos)) over 3D | 0.0-5.0m |

**Rewards**:
- Base torso: `5.0 * exp(-4 * torso_error)`
- Base goal: `5.0 * exp(-4 * goal_error)`
- Progress: `1.0 * clamp(delta_torso + 3*delta_goal, -0.5, 0.5)` (goal weighted 3x)
- Success: `2.0 if goal_error < 0.3`

---

### Task 4-7: BOX, TRANSFER, LIFT, CARRY
**Goal**: Manipulate box to target position

| Metric | Formula | Expected Range |
|--------|---------|----------------|
| `box_error` | mean(abs(box_pos - target)) over 3D | 0.0-1.0m |
| `wrist_error` | mean(abs(wrists - box_pos)) over 6D | 0.0-1.0m |

**Rewards (all tasks)**:
- Base box: `5.0 * exp(-4 * box_error)`
- Base wrist: `5.0 * exp(-4 * wrist_error)` (or `1.0` for transfer)
- Progress: `1.0 * clamp(2*delta_box + delta_wrist, -0.5, 0.5)` (box weighted 2x)
- Success: `2.0 if box_error < 0.2`

**Task-specific wrist scale**:
- BOX (4): scale=5
- TRANSFER (5): scale=1 (less emphasis on wrist proximity)
- LIFT (6): scale=5
- CARRY (7): scale=5

---

## Reward Shaping Configuration

```python
cfg_shaping = {
    'progress_scale': 1.0,      # Scale for progress bonus
    'success_bonus': 2.0,       # Bonus when error < threshold
    'success_threshold': 0.1,   # Error threshold for success (varies by task)
}
```

## Implementation Notes

### First Step Handling
After reset, `prev_error = 0`. Without handling, this causes:
```
progress = 0 - curr_error = NEGATIVE
```

**Fix**: Skip progress reward when `prev_error == 0`:
```python
valid_prev = prev_error > 0
progress = torch.where(valid_prev, prev_error - curr_error, torch.zeros_like(curr_error))
```

### Metric Logging
Metrics are logged per-task to wandb:
- `Metric/{task}_wrist_error`: Wrist distance to target
- `Metric/{task}_box_error`: Box distance to target
- `Metric/{task}_door_error`: Door angle error
- etc.

**Expected values**:
| Metric | Expected Range | Notes |
|--------|---------------|-------|
| wrist_error | 0.3-0.6m | Distance to target |
| box_error | 0.3-0.6m | Box to target |
| torso_error | 0.3-0.6m | Robot to ball |
| goal_error | 1.5-2.5m | Ball to goal (larger) |
| door_error | 0.0-1.0 rad | Door angle |
| arm_error | 0.0-0.5 rad | Joint angle |

### Known Bug Fix (2026-02-02)
**Coordinate Frame Mismatch**: Goals were being set in robot-relative coordinates but
wrist positions come from `rigid_state` which uses world coordinates. When robots spawn
at different world positions (e.g., y=2.7m), this caused errors like ~15-250!

**Fix**: All targets in `_sample_goals()` now add `robot_base_pos = self.root_states[env_id, :3]`
to convert from robot-relative to world frame.

---

## Reward Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         compute_reward()                            │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
             ┌──────────┐   ┌──────────┐   ┌──────────┐
             │  Task 0  │   │  Task 1  │   │   ...    │
             │  Reach   │   │  Button  │   │          │
             └────┬─────┘   └────┬─────┘   └────┬─────┘
                  │              │              │
                  ▼              ▼              ▼
         ┌────────────────────────────────────────────┐
         │     _reward_{task}_shaped(mask, cfg)       │
         └────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌──────────┐       ┌──────────┐       ┌──────────┐
    │   BASE   │       │ PROGRESS │       │ SUCCESS  │
    │  reward  │       │  reward  │       │  reward  │
    └──────────┘       └──────────┘       └──────────┘
    5*exp(-4*e)     scale*clamp(Δe)      bonus*(e<th)
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                    ┌────────────────┐
                    │  total_reward  │
                    │ = base+shaped  │
                    └────────────────┘
```

## Summary

| Task | Primary Metric | Secondary Metric | Progress Weight |
|------|---------------|------------------|-----------------|
| Reach | wrist_error | - | 1:0 |
| Button | wrist_error | arm_error | 1:0 |
| Cabinet | wrist_error | door_error | 1:2 |
| Ball | torso_error | goal_error | 1:3 |
| Box/Lift/Carry | box_error | wrist_error | 2:1 |
| Transfer | box_error | wrist_error | 1:1 |
