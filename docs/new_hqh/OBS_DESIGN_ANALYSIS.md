# Phân Tích Obs Design - Single-Task vs HRL Unified

## 📋 Tổng Quan

Phân tích chi tiết cách tác giả thiết kế observation cho **8 single-task environments** và đề xuất unified obs design cho **HRL multi-task policy**.

**Đọc từ code thực tế (không suy đoán):**
- Task 0: h1_task_reach
- Task 1: h1_task_button  
- Task 2: h1_task_cabinet
- Task 3: h1_task_ball
- Task 4: h1_task_box
- Task 5: h1_task_transfer
- Task 6: h1_task_lift
- Task 7: h1_task_carry

---

## 🔍 Chi Tiết Obs Design Của Tác Giả

### **Common Pattern Across All 8 Tasks:**

```python
# === CHUNG CHO TẤT CẢ ===
obs_buf = torch.cat([
    task_error,              # ERROR (khác nhau theo task: 2-14 dims)
    q,                       # 19: (dof_pos - default) * obs_scales.dof_pos
    dq,                      # 19: dof_vel * obs_scales.dof_vel
    actions,                 # 19: last actions
    base_ang_vel * scale,    # 3:  ang_vel * obs_scales.ang_vel
    base_euler_xyz * scale,  # 3:  euler * obs_scales.quat
], dim=-1)

# ❌ KHÔNG CÓ:
# - command_input (COMMENTED OUT cho tất cả tasks)
# - base_lin_vel (KHÔNG có trong actor obs, CHỈ có trong critic)
```

### **Privileged Obs (Cho Critic) - Common Pattern:**

```python
privileged_obs_buf = torch.cat([
    # Task-specific (varies: 6-15 dims)
    [goal_pos, current_pos, error],  # Raw + computed
    
    # Robot state (same for all)
    (dof_pos - default_pd) * scale,  # 19
    dof_vel * scale,                 # 19
    actions,                         # 19
    base_lin_vel * scale,            # 3  ← ONLY in privileged!
    base_ang_vel * scale,            # 3
    base_euler_xyz * scale,          # 3
    
    # Domain randomization (same for all)
    rand_push_force[:, :2],          # 2
    rand_push_torque,                # 3
    env_frictions,                   # 1
    body_mass / 30.,                 # 1
    contact_mask,                    # 2
], dim=-1)
```

---

## 📊 Task-by-Task Breakdown

### **Task 0: Reach**

#### Obs (Actor):
```python
obs_buf = [
    diff_obs,          # 14: wrist_pos - ref_wrist_pos (2 wrists × 7)
    q,                 # 19
    dq,                # 19
    actions,           # 19
    base_ang_vel,      # 3
    base_euler_xyz,    # 3
]  # Total: 77 dims
```

#### Privileged Obs (Critic):
```python
privileged_obs_buf = [
    ref_wrist_pos_obs,  # 14: target positions
    wrist_pos_obs,      # 14: current positions
    (dof_pos - pd),     # 19
    dof_vel,            # 19
    actions,            # 19
    diff_obs,           # 14: ERROR
    base_lin_vel,       # 3  ← EXTRA
    base_ang_vel,       # 3
    base_euler_xyz,     # 3
    rand_push_force,    # 2
    rand_push_torque,   # 3
    env_frictions,      # 1
    body_mass,          # 1
    contact_mask,       # 2
]  # Total: 117 dims
```

**Key Points:**
- Error: 14 dims (2 wrists × 7: pos+quat)
- ❌ No command
- Privileged has RAW + ERROR

---

### **Task 1: Button**

#### Obs (Actor):
```python
obs_buf = [
    wrist_button_diff,  # 3: left_wrist - button_pos
    q,                  # 19
    dq,                 # 19
    actions,            # 19
    base_ang_vel,       # 3
    base_euler_xyz,     # 3
]  # Total: 66 dims
```

#### Privileged Obs (Critic):
```python
privileged_obs_buf = [
    button_goal_pos,    # 3: target
    wrist_pos,          # 3: current (left only)
    wrist_button_diff,  # 3: ERROR
    (dof_pos - pd),     # 19
    dof_vel,            # 19
    actions,            # 19
    base_lin_vel,       # 3
    base_ang_vel,       # 3
    base_euler_xyz,     # 3
    rand_push_force,    # 2
    rand_push_torque,   # 3
    env_frictions,      # 1
    body_mass,          # 1
    contact_mask,       # 2
]  # Total: 84 dims
```

**Key Points:**
- Error: 3 dims (left wrist - button)
- ❌ No command
- Only LEFT wrist used

---

### **Task 2: Cabinet**

#### Obs (Actor):
```python
obs_buf = [
    arti_obj_dof_diff,      # 2: door_angle - 0
    wrist_arti_obj_diff,    # 6: 2_wrists - handle
    q,                      # 19
    dq,                     # 19
    actions,                # 19
    base_ang_vel,           # 3
    base_euler_xyz,         # 3
]  # Total: 71 dims
```

#### Privileged Obs (Critic):
```python
privileged_obs_buf = [
    arti_obj_dof_diff,      # 2: door angle error
    wrist_arti_obj_diff,    # 6: wrist-handle error
    (dof_pos - pd),         # 19
    dof_vel,                # 19
    actions,                # 19
    base_lin_vel,           # 3
    base_ang_vel,           # 3
    base_euler_xyz,         # 3
    rand_push_force,        # 2
    rand_push_torque,       # 3
    env_frictions,          # 1
    body_mass,              # 1
    contact_mask,           # 2
]  # Total: 83 dims
```

**Key Points:**
- Error: 2+6=8 dims (door angle + 2 wrists to handle)
- ❌ No command
- Both wrists used

---

### **Task 3: Ball (FootballShoot)**

#### Obs (Actor):
```python
obs_buf = [
    ball_goal_diff,     # 3: ball - goal
    root_ball_diff,     # 3: torso - ball
    q,                  # 19
    dq,                 # 19
    actions,            # 19
    base_ang_vel,       # 3
    base_euler_xyz,     # 3
]  # Total: 69 dims
```

#### Privileged Obs (Critic):
```python
privileged_obs_buf = [
    goal_pos_obs,       # 3: goal position
    ball_pos_obs,       # 3: ball position
    torso_pos_obs,      # 3: torso position
    ball_goal_diff,     # 3: ball - goal ERROR
    root_ball_diff,     # 3: torso - ball ERROR
    (dof_pos - pd),     # 19
    dof_vel,            # 19
    actions,            # 19
    base_lin_vel,       # 3
    base_ang_vel,       # 3
    base_euler_xyz,     # 3
    rand_push_force,    # 2
    rand_push_torque,   # 3
    env_frictions,      # 1
    body_mass,          # 1
    contact_mask,       # 2
]  # Total: 90 dims
```

**Key Points:**
- Error: 6 dims (ball-goal + torso-ball)
- ❌ No command
- Torso XY only for approach

---

### **Task 4: Box (BoxPush)**

#### Obs (Actor):
```python
obs_buf = [
    diff_obs,           # 3: box - target
    wrist_box_diff,     # 6: 2_wrists - box
    q,                  # 19
    dq,                 # 19
    actions,            # 19
    base_ang_vel,       # 3
    base_euler_xyz,     # 3
]  # Total: 72 dims
```

#### Privileged Obs (Critic):
```python
privileged_obs_buf = [
    box_goal_pos_obs,   # 3: target
    box_pos_obs,        # 3: current box
    diff_obs,           # 3: box - target ERROR
    wrist_pos_obs,      # 6: wrist positions
    wrist_box_diff,     # 6: wrists - box ERROR
    (dof_pos - pd),     # 19
    dof_vel,            # 19
    actions,            # 19
    base_lin_vel,       # 3
    base_ang_vel,       # 3
    base_euler_xyz,     # 3
    rand_push_force,    # 2
    rand_push_torque,   # 3
    env_frictions,      # 1
    body_mass,          # 1
    contact_mask,       # 2
]  # Total: 93 dims
```

**Key Points:**
- Error: 9 dims (box-target + 2_wrists-box)
- ❌ No command
- Both wrists used

---

### **Task 5: Transfer (BoxTransfer)**

**GIỐNG HỆT Task 4 (Box)** - code y hệt

```python
# Same obs structure as Box (72 dims)
# Same privileged obs (93 dims)
```

---

### **Task 6: Lift (PackageLift)**

**GIỐNG HỆT Task 4 & 5** - code y hệt

```python
# Same obs structure as Box (72 dims)
# Same privileged obs (93 dims)
```

**Khác biệt:** Chỉ trong reward function (check z-axis only)

---

### **Task 7: Carry (PackageCarry)**

**GIỐNG HỆT Task 4, 5, 6** - code y hệt

```python
# Same obs structure as Box (72 dims)
# Same privileged obs (93 dims)
```

---

## 📈 Tổng Hợp Kích Thước Obs

| Task | Obs (Actor) | Privileged (Critic) | Error Dims | Error Type |
|------|-------------|---------------------|------------|------------|
| **0. Reach** | 77 | 117 | 14 | 2_wrists - target (pos+quat) |
| **1. Button** | 66 | 84 | 3 | left_wrist - button |
| **2. Cabinet** | 71 | 83 | 8 | door_angle(2) + 2_wrists-handle(6) |
| **3. Ball** | 69 | 90 | 6 | ball-goal(3) + torso-ball(3) |
| **4. Box** | 72 | 93 | 9 | box-target(3) + 2_wrists-box(6) |
| **5. Transfer** | 72 | 93 | 9 | box-target(3) + 2_wrists-box(6) |
| **6. Lift** | 72 | 93 | 9 | box-target(3) + 2_wrists-box(6) |
| **7. Carry** | 72 | 93 | 9 | box-target(3) + 2_wrists-box(6) |

**Range:** Obs 66-77 dims, Privileged 83-117 dims

---

## ✅ Common Patterns Tìm Thấy

### **1. Command KHÔNG được dùng**

```python
# TẤT CẢ 8 tasks đều comment out:
# self.command_input,  # 2 + 3 ← COMMENTED!
# self.command_input_wo_clock,  # 3 ← COMMENTED!
```

### **2. Base Linear Velocity CHỈ trong Privileged**

```python
# ❌ KHÔNG có trong obs_buf (actor)
# ✅ CÓ trong privileged_obs_buf (critic)
base_lin_vel * self.obs_scales.lin_vel  # 3
```

### **3. Obs Structure Pattern**

```python
# === ACTOR OBS ===
obs_buf = [
    task_error,        # Varies: 3-14 dims
    q,                 # 19
    dq,                # 19
    actions,           # 19
    base_ang_vel,      # 3
    base_euler_xyz,    # 3
]

# === CRITIC PRIVILEGED OBS ===
privileged_obs_buf = [
    # Task-specific (raw + error)
    [goal, current, error],  # Varies: 6-15 dims
    
    # Robot state
    (dof_pos - default_pd),  # 19
    dof_vel,                 # 19
    actions,                 # 19
    base_lin_vel,            # 3  ← EXTRA!
    base_ang_vel,            # 3
    base_euler_xyz,          # 3
    
    # Domain randomization
    rand_push_force,         # 2
    rand_push_torque,        # 3
    env_frictions,           # 1
    body_mass,               # 1
    contact_mask,            # 2
]
```

### **4. Frame Stacking**

```python
# Actor: frame_stack = 1 (single frame)
obs_buf_all = torch.stack([self.obs_history[i] for i in range(maxlen)], dim=1)
self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)

# Critic: c_frame_stack = 3 (3 frames)
self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(3)], dim=1)
```

---

## 🎯 Thiết Kế Unified Obs Cho HRL

### **Mục Tiêu:**
1. Single policy cho 8 tasks → **SAME obs size**
2. Match tác giả design → **ERROR encoding, NO command**
3. Privileged obs có thêm domain rand → **Asymmetric actor-critic**

---

### **✅ Unified Actor Obs Design (99 dims):**

```python
obs_buf = torch.cat([
    # === TASK ERROR (14 dims - padded) ===
    task_error,              # 14 dims (max: Reach 14, padded with 0s for others)
    
    # === ROBOT STATE (same for all) ===
    q,                       # 19: (dof_pos - default) * scale
    dq,                      # 19: dof_vel * scale
    actions,                 # 19: last actions
    base_ang_vel * 0.25,     # 3:  ang_vel scaled
    base_euler_xyz,          # 3:  projected_gravity (same as euler)
    
    # === CONDITIONING ===
    error_mask,              # 14: which dims of task_error are active
    task_onehot,             # 8:  task ID
], dim=-1)  # Total: 14 + 19 + 19 + 19 + 3 + 3 + 14 + 8 = 99 dims
```

**Design Rationale:**
- ✅ **ERROR encoding** (matches author's single-task design)
- ✅ **NO command** (author không dùng cho manipulation tasks)
- ✅ **NO base_lin_vel** (author chỉ có trong privileged obs)
- ✅ **Unified size** cho 8 tasks (pad task_error to max 14 dims)
- ✅ **Task conditioning** qua error_mask + task_onehot

---

### **✅ Unified Privileged Obs (Critic) Design:**

**Per-Frame Obs (113 dims):**

```python
# Single timestep privileged obs
priv_single = torch.cat([
    # === TASK ERROR (14 dims - same as actor) ===
    task_error,              # 14: computed errors (padded)
    
    # === TASK RAW OBSERVATIONS (15 dims max - padded) ===
    task_raw_goal,           # 15: raw goal/target values
    task_raw_current,        # 15: raw current state values
    
    # === ROBOT STATE ===
    (dof_pos - default_pd),  # 19: joint positions (centered)
    dof_vel * 0.05,          # 19: joint velocities (scaled)
    actions,                 # 19: last actions
    
    # === BASE STATE (EXTRA for critic) ===
    base_lin_vel * 2.0,      # 3: ← ONLY in privileged (not in actor)
    base_ang_vel * 0.25,     # 3
    base_euler_xyz,          # 3
    
    # === DOMAIN RANDOMIZATION ===
    rand_push_force[:, :2],  # 2: external force (x, y)
    rand_push_torque,        # 3: external torque
    env_frictions,           # 1: ground friction
    body_mass / 30.0,        # 1: mass scaling
    contact_feetfloor,       # 2: feet contact mask
    
    # === CONDITIONING ===
    error_mask,              # 14: which error dims are active
    task_onehot,             # 8: task ID one-hot
], dim=-1)
# Total: 14 + 15 + 15 + 19 + 19 + 19 + 3 + 3 + 3 + 2 + 3 + 1 + 1 + 2 + 14 + 8 = 141 dims
```

**Stacked Critic Obs (423 dims with 3-frame history):**

```python
# Stack 3 timesteps for temporal information
privileged_obs_buf = torch.cat([
    priv_single_t0,          # 141: current frame
    priv_single_t1,          # 141: t-1 frame
    priv_single_t2,          # 141: t-2 frame
], dim=-1)
# Total: 141 × 3 = 423 dims
```

**Critic Design Rationale:**
- ✅ **Asymmetric obs:** More info than actor for better value estimation
- ✅ **RAW + ERROR:** Both raw states and computed errors (like author)
- ✅ **base_lin_vel:** Included ONLY in critic (matches author)
- ✅ **Domain randomization:** 12 dims for robust critic (friction, mass, contact, external forces)
- ✅ **Temporal info:** 3-frame stacking for dynamics understanding
- ✅ **Unified size:** Same 141 dims per frame across all 8 tasks (padded)

**Task Raw Obs Encoding (15 dims max, padded):**

| Task | Raw Goal | Raw Current | Total Raw Dims |
|------|----------|-------------|----------------|
| Reach | wrist_target (7×2=14) | wrist_pos (7×2=14) | 28 → use 15+15 |
| Button | button_pos (3) | left_wrist_pos (3) | 6 |
| Cabinet | handle_pos (3) + 0° (2) | wrist_pos (6) + door_angle (2) | 13 |
| Ball | goal_pos (3) + ball_init (3) | ball_pos (3) + torso_pos (2) | 11 |
| Box/Transfer/Lift/Carry | box_target (3) | box_pos (3) + wrist_pos (6) | 12 |

**Lưu ý:** Reach task có 14 dims (2 wrists × 7) nên dùng 2 slots 15 dims là đủ

---

## 🔧 Implementation Details

### **A. Task Error Encoding (14 dims max)**

```python
def _compute_task_error(self):
    """Compute unified 14-dim error for all tasks (padded with zeros)"""
    error = torch.zeros(self.num_envs, 14, device=self.device)
    
    for task_id in range(8):
        mask = (self.task_ids == task_id)
        if not mask.any():
            continue
        
        if task_id == 0:  # Reach - 14 dims
            wrist = self.rigid_state[mask][:, self.wrist_indices, :7]
            target = ... # from goal storage
            error[mask, :14] = (wrist - target).reshape(-1, 14)
            
        elif task_id == 1:  # Button - 3 dims
            left_wrist = self.rigid_state[mask][:, self.wrist_indices[0], :3]
            button = ...
            error[mask, :3] = left_wrist - button
            # dims 3-13: padding (zeros)
            
        elif task_id == 2:  # Cabinet - 8 dims
            wrist = self.rigid_state[mask][:, self.wrist_indices, :3]
            handle = ...
            error[mask, :6] = (wrist - handle.unsqueeze(1)).reshape(-1, 6)
            error[mask, 6:8] = self.door_angle[mask] - 0.0
            # dims 8-13: padding
            
        elif task_id == 3:  # Ball - 6 dims
            torso = self.rigid_state[mask][:, self.torso_indices[0], :2]
            ball = self.ball_pos[mask, :2]
            error[mask, :2] = torso - ball
            error[mask, 2:5] = self.ball_pos[mask] - self.ball_target[mask]
            # dims 5-13: padding
            
        else:  # Box tasks (4-7) - 9 dims
            box = self.box_pos[mask]
            target = self.box_target[mask]
            wrist = self.rigid_state[mask][:, self.wrist_indices, :3]
            
            if task_id == 6:  # Lift: z-only
                error[mask, 0] = box[:, 2] - target[:, 2]
            else:
                error[mask, :3] = box - target
            
            error[mask, 3:9] = (wrist - box.unsqueeze(1)).reshape(-1, 6)
            # dims 9-13: padding
    
    return error
```

---

### **B. Task Raw Obs Encoding (15+15 dims for critic)**

```python
def _compute_task_raw_obs(self):
    """Compute raw goal and current state for critic (padded to 15 dims each)"""
    raw_goal = torch.zeros(self.num_envs, 15, device=self.device)
    raw_current = torch.zeros(self.num_envs, 15, device=self.device)
    
    for task_id in range(8):
        mask = (self.task_ids == task_id)
        if not mask.any():
            continue
        
        if task_id == 0:  # Reach - 14 dims (2 wrists × 7)
            # Goal
            wrist_target = ...  # shape: (N, 2, 7) - 2 wrists, pos(3)+quat(4)
            raw_goal[mask, :14] = wrist_target.reshape(-1, 14)
            # Current
            wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :7]
            raw_current[mask, :14] = wrist_pos.reshape(-1, 14)
            
        elif task_id == 1:  # Button - 3 dims
            raw_goal[mask, :3] = self.button_goal_pos[mask]
            raw_current[mask, :3] = self.rigid_state[mask][:, self.wrist_indices[0], :3]
            
        elif task_id == 2:  # Cabinet - 8 dims
            # Goal: handle_pos (3) + door_angle=0 (2)
            raw_goal[mask, :3] = self.handle_pos[mask]
            raw_goal[mask, 3:5] = 0.0  # target door angle
            # Current: wrist_pos (6) + door_angle (2)
            wrist = self.rigid_state[mask][:, self.wrist_indices, :3]
            raw_current[mask, :6] = wrist.reshape(-1, 6)
            raw_current[mask, 6:8] = self.door_angle[mask]
            
        elif task_id == 3:  # Ball - 8 dims
            # Goal: goal_pos (3) + ball_init_pos (3)
            raw_goal[mask, :3] = self.ball_goal_pos[mask]
            raw_goal[mask, 3:6] = self.ball_init_pos[mask]
            # Current: ball_pos (3) + torso_pos (2)
            raw_current[mask, :3] = self.ball_pos[mask]
            raw_current[mask, 3:5] = self.rigid_state[mask][:, self.torso_idx, :2]
            
        else:  # Box tasks (4-7) - 9 dims
            # Goal: box_target (3)
            raw_goal[mask, :3] = self.box_target[mask]
            # Current: box_pos (3) + wrist_pos (6)
            raw_current[mask, :3] = self.box_pos[mask]
            wrist = self.rigid_state[mask][:, self.wrist_indices, :3]
            raw_current[mask, 3:9] = wrist.reshape(-1, 6)
    
    return raw_goal, raw_current
```

---

### **C. Critic Obs Assembly (with 3-frame stacking)**

```python
def compute_privileged_observations(self):
    """Compute privileged obs for critic (141 dims per frame)"""
    
    # === COMPUTE COMPONENTS ===
    task_error = self._compute_task_error()           # 14
    raw_goal, raw_current = self._compute_task_raw_obs()  # 15 + 15
    
    # Robot state
    q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos  # 19
    dq = self.dof_vel * self.obs_scales.dof_vel                          # 19
    
    # Base state
    base_lin_vel = self.base_lin_vel * self.obs_scales.lin_vel           # 3
    base_ang_vel = self.base_ang_vel * self.obs_scales.ang_vel           # 3
    base_euler = self.base_euler_xyz                                     # 3
    
    # Domain randomization
    rand_push_f = self.rand_push_force[:, :2]                            # 2
    rand_push_t = self.rand_push_torque                                  # 3
    friction = self.env_frictions                                        # 1
    mass = self.body_mass / 30.0                                         # 1
    contact = self.contact_feetfloor                                     # 2
    
    # === ASSEMBLE SINGLE FRAME ===
    priv_single = torch.cat([
        task_error,              # 14
        raw_goal,                # 15
        raw_current,             # 15
        q,                       # 19
        dq,                      # 19
        self.actions,            # 19
        base_lin_vel,            # 3
        base_ang_vel,            # 3
        base_euler,              # 3
        rand_push_f,             # 2
        rand_push_t,             # 3
        friction,                # 1
        mass,                    # 1
        contact,                 # 2
        self.error_mask,         # 14
        self.task_onehot,        # 8
    ], dim=-1)  # Total: 141 dims
    
    # === FRAME STACKING (3 frames) ===
    self.critic_history.append(priv_single)
    if len(self.critic_history) > 3:
        self.critic_history.pop(0)
    
    # Pad with zeros if not enough history
    while len(self.critic_history) < 3:
        self.critic_history.insert(0, torch.zeros_like(priv_single))
    
    # Stack: [t-2, t-1, t-0]
    self.privileged_obs_buf = torch.cat([
        self.critic_history[0],  # oldest
        self.critic_history[1],
        self.critic_history[2],  # newest
    ], dim=-1)  # Total: 141 × 3 = 423 dims
    
    return self.privileged_obs_buf
```

---

## 📌 So Sánh: Code Hiện Tại vs Đề Xuất

### **Code Hiện Tại (HRL):**

```python
obs_buf = [
    base_lin_vel * 2.0,            # 3
    base_ang_vel * 0.25,           # 3
    projected_gravity,             # 3
    commands[:, :3] * [...],       # 3 ← Không dùng nhưng vẫn có
    (dof_pos - default),           # 19
    dof_vel * 0.05,                # 19
    actions,                       # 19
    goal_value,                    # 14 ← RAW target (PROBLEM!)
    goal_mask,                     # 14
    task_onehot,                   # 8
]  # 105 dims
```

**❌ Vấn Đề:**
1. `goal_value` = RAW targets, không phải ERROR
2. `commands` = [0,0,0] (placeholder không cần thiết)
3. Không match tác giả design

---

### **Thiết Kế Đề Xuất:**

```python
obs_buf = [
    base_lin_vel * 2.0,            # 3  (set = 0 or keep for consistency)
    base_ang_vel * 0.25,           # 3
    projected_gravity,             # 3
    torch.zeros(N, 3),             # 3  (command placeholder - không dùng)
    (dof_pos - default),           # 19
    dof_vel * 0.05,                # 19
    actions,                       # 19
    task_error,                    # 14 ← CHANGED: ERROR instead of target!
    error_mask,                    # 14
    task_onehot,                   # 8
]  # 105 dims (same size)
```

**✅ Cải Thiện:**
1. `task_error` = computed errors (matches tác giả)
2. Same obs size (105 dims)
3. Dễ học hơn (agent chỉ cần minimize error)

---

## 🎯 Recommendation

### **Thay Đổi Cần Làm:**

1. **Thay `goal_value` → `task_error`**
   - Compute error per task
   - Pad to 14 dims
   - Update `error_mask` accordingly

2. **Privileged Obs: Thêm Domain Rand**
   ```python
   priv_obs += [
       rand_push_force,
       rand_push_torque,
       env_frictions,
       body_mass,
       contact_mask,
   ]
   ```

3. **KHÔNG thêm command** (giữ zeros hoặc bỏ)

4. **Base Lin Vel:**
   - ❌ Bỏ khỏi actor obs (hoặc set = 0)
   - ✅ GIỮ trong privileged obs

---

## ✅ Final Design Summary

### **Obs Dimensions:**

| Component | Actor Obs | Privileged Obs (Critic) |
|-----------|-----------|-------------------------|
| **Task Error** | 14 dims (computed, padded) | 14 dims (same as actor) |
| **Task Raw Obs** | ❌ NO | 30 dims (goal 15 + current 15) |
| **Robot State** | 61 dims (q+dq+actions+ang+euler) | 61 dims (same components) |
| **Base Lin Vel** | ❌ NO | ✅ 3 dims (only in critic) |
| **Base Ang/Euler** | 6 dims | 6 dims |
| **Domain Rand** | ❌ NO | ✅ 12 dims (force+torque+friction+mass+contact) |
| **Conditioning** | 22 dims (mask 14 + ID 8) | 22 dims (same) |
| **Total (per frame)** | **99 dims** | **141 dims** |
| **With frame stack** | 99 × 1 = **99** | 141 × 3 = **423 dims** |

---

### **Key Design Principles:**

✅ **Actor Obs (99 dims):**
- **ERROR encoding** thay vì raw targets (matches tác giả)
- **NO command** (tác giả không dùng)
- **NO base_lin_vel** (tác giả chỉ có trong critic)
- **Unified size** cho 8 tasks (padding)
- **Single frame** (no history)

✅ **Critic Obs (423 dims = 141×3):**
- **Asymmetric**: More info than actor
- **RAW + ERROR**: Both for better value estimation
- **base_lin_vel**: Included (tác giả có)
- **Domain randomization**: 12 dims for robustness
- **3-frame stacking**: Temporal dynamics
- **Unified size**: Same 141 dims/frame for all tasks

---

### **Advantages vs Current HRL Code:**

| Aspect | Current (105 dims) | Proposed (99 dims) |
|--------|-------------------|-------------------|
| **Task Info** | `goal_value` (raw targets) | `task_error` (computed) |
| **Learning** | Harder (abs positioning) | Easier (relative error) |
| **Command** | zeros (placeholder) | Removed (cleaner) |
| **base_lin_vel** | Included (3 dims) | Removed (matches author) |
| **Match Author** | ❌ Different design | ✅ Same ERROR encoding |
| **Total** | 105 dims | 99 dims (6 dims saved) |

---

### **Advantages vs Current Critic Code:**

| Aspect | Current (312 dims) | Proposed (423 dims) |
|--------|-------------------|-------------------|
| **Task Raw Obs** | Only error | RAW + ERROR (both) |
| **Domain Rand** | Missing | ✅ 12 dims added |
| **base_lin_vel** | Missing? | ✅ 3 dims included |
| **Frame Stack** | 3 frames | 3 frames (same) |
| **Per-frame** | 104 dims | 141 dims (+37 dims) |
| **Total** | 312 dims | 423 dims |

**Trade-off:** +111 dims nhưng critic có thêm info quan trọng:
- Raw goal/current states giúp value estimation tốt hơn
- Domain randomization giúp critic robust với env variations
- base_lin_vel giúp critic hiểu robot velocity dynamics

---
