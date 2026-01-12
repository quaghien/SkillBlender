# 🔄 Early Termination & Agent Initialization

Tóm tắt cơ chế kết thúc episode sớm và khởi tạo robot ở các task.

---

## 🛑 Early Termination Conditions

### Locomotion Tasks (Walking, Reaching, etc.)

**Episode kết thúc sớm khi:**

```python
# 1. CONTACT VIOLATION - Chạm bộ phận không được phép
reset_buf = torch.any(
    torch.norm(contact_forces[:, termination_contact_indices, :], dim=-1) > 1.0,
    dim=1
)
# Chạm phạt: ['pelvis', 'torso', 'shoulder', 'elbow'] (mặc định)
# Nếu chạm → điểm bị trừ → có thể terminate

# 2. TIMEOUT - Hết thời gian episode
time_out_buf = episode_length_buf > max_episode_length
# max_episode_length = episode_length_s / dt (e.g., 24s / 0.001s = 24000 steps)
# Không có terminal penalty

# 3. Cuối cùng:
reset_buf |= time_out_buf  # Contact violation hoặc timeout đều reset
```

**Episode Length theo Task**:

| Task | Duration | Max Steps | Note |
|---|---|---|---|
| h1_walking | 24s | 24000 | Walking tự do |
| h1_reaching | 24s | 24000 | - |
| h1_squatting | - | - | - |
| h1_stepping | - | - | - |
| h1_task_transfer | 8s | 8000 | Short task |
| h1_task_lift | 8s | 8000 | Short task |
| h1_task_carry | 8s | 8000 | Short task |
| h1_task_reach | 24s | 24000 | Full skill combo |
| h1_task_ball | - | - | Plus: ball reaches goal → reset |

---

### Task-Specific Termination (task_ball)

```python
# Extra: Ball reaches goal zone
ball_pos = self.ball_root_states[:, :3]
goal_pos = self.goal_pos
ball_goal_dist = torch.norm(ball_pos - goal_pos, dim=1)

# Reset nếu bóng đến goal
reset_buf |= ball_goal_dist < self.cfg.commands.ranges.threshold
```

**→ Success condition!** (không phải failure)

---

## 🤖 Agent Initialization (Reset)

### Giai Đoạn Reset

```
reset_idx() được gọi → 3 bước:
    1. _reset_dofs() - Khớp
    2. _reset_root_states() - Vị trí & vận tốc chính
    3. _resample_commands() - Lệnh mục tiêu mới
    4. Reset buffers - Action history, etc.
```

---

### 1️⃣ Joint Positions (DOF) Reset

**Khởi tạo vị trí khớp**:

```python
# default_dof_pos từ config
# + Random noise: ±0.1 rad
dof_pos[env_ids] = default_dof_pos + torch_rand_float(-0.1, 0.1, shape)
dof_vel[env_ids] = 0.0  # Vận tốc khớp = 0
```

**Ví dụ (H1 Walking)**:
```python
default_joint_angles = {
    'left_hip_yaw_joint': 0.0,
    'left_hip_pitch_joint': -0.4,  # Squat position
    'left_knee_joint': 0.8,         # Knee bent
    'left_ankle_pitch_joint': -0.4,
    'right_hip_yaw_joint': 0.0,
    'right_hip_pitch_joint': -0.4,
    'right_knee_joint': 0.8,
    'right_ankle_pitch_joint': -0.4,
    'torso_joint': 0.0,
    'left_shoulder_pitch_joint': 0.0,  # Arm neutral
    'right_shoulder_pitch_joint': 0.0,
}
# Random ±0.1 rad thêm vào mỗi joint
```

---

### 2️⃣ Root Position (Base) Reset

**Khởi tạo vị trí thân chính**:

```python
# Base position
if custom_origins:  # Có terrain curriculum
    root_states[env_ids] = base_init_state
    root_states[env_ids, :3] += env_origins[env_ids]
    # XY random: ±1m quanh center
    root_states[env_ids, :2] += torch_rand_float(-1., 1., shape)
else:  # Plane terrain
    root_states[env_ids] = base_init_state
    # Center point
    root_states[env_ids, :3] += env_origins[env_ids]

# Base velocity
root_states[env_ids, 7:13] = 0  # Không có vận tốc ban đầu
# [7:10]: linear velocity (0, 0, 0)
# [10:13]: angular velocity (0, 0, 0)
```

**Default Base Position (config)**:
```python
class init_state:
    pos = [0.0, 0.0, 1.0]  # x, y, z [m]
    # z = 1.0m (chiều cao ban đầu để chân không chạm ground)
    rot = [0.0, 0.0, 0.0, 1.0]  # quaternion (neutral)
    lin_vel = [0.0, 0.0, 0.0]
    ang_vel = [0.0, 0.0, 0.0]
```

---

### 3️⃣ Command Reset

**Resample mục tiêu mới**:

```python
# Walking: Random velocity commands
commands[:, 0] = uniform(-1.0, 2.0)  # lin_vel_x
commands[:, 1] = uniform(-1.0, 1.0)  # lin_vel_y
commands[:, 2] = uniform(-1.0, 1.0)  # ang_vel_yaw
commands[:, 3] = uniform(-3.14, 3.14) # heading

# Task manipulation (transfer): Random object positions
commands[:, 0:3] = random_wrist_position()  # 3D wrist target
commands[:, 3:6] = random_box_position()    # 3D box target
commands[:, 6:9] = random_other_params()    # Extra params
```

---

### 4️⃣ Buffer Reset

**Clear history**:

```python
last_last_actions[env_ids] = 0.0
actions[env_ids] = 0.0
last_actions[env_ids] = 0.0
last_rigid_state[env_ids] = 0.0
last_dof_vel[env_ids] = 0.0
feet_air_time[env_ids] = 0.0
episode_length_buf[env_ids] = 0  # Counter reset to 0
reset_buf[env_ids] = 1  # Mark as reset
```

---

## 📊 Timeline Ví Dụ (Walking)

```
Time: 0s - Episode starts
├─ reset_idx() được gọi
│  ├─ DOF positions: default ±0.1
│  ├─ Base position: [0, 0, 1.0] + curriculum offset
│  ├─ Command: lin_vel_x=0.5, lin_vel_y=0.0, ang_vel_yaw=0.1
│  └─ Buffers cleared
│
├─ [0-24s] Episode running
│  ├─ Step 0-24000: Agent executes actions
│  ├─ Check termination each step:
│  │  ├─ Did robot touch forbidden body parts? → YES = terminate
│  │  ├─ Time > 24s? → YES = terminate
│  │  └─ NO = continue
│  └─ Accumulate rewards
│
├─ Time: 24s - Timeout reached
│  └─ reset_idx() → New episode starts
│
OR

├─ Time: 5.2s - Robot ngã
│  ├─ Contact with ['pelvis'] detected
│  ├─ reset_buf set to 1
│  └─ reset_idx() → New episode starts (5.2s < 24s)
```

---

## 🎯 Key Parameters

| Param | Value | Meaning |
|---|---|---|
| `episode_length_s` | 24s (loco), 8s (task) | Max duration |
| `dt` | 0.001s | Simulation timestep |
| `decimation` | 10 | Policy updates every 10 sim steps |
| `dof_init_noise` | ±0.1 rad | DOF position randomness |
| `base_init_pos` | [0, 0, 1.0] | Thân chính ban đầu |
| `termination_contacts` | ['pelvis', 'torso', 'shoulder', 'elbow'] | Forbidden body parts |

---

## 📌 Tuning Tips

**Nếu episode quá ngắn** (robot ngã nhanh):
- Kiểm tra `terminate_after_contacts_on` - quá nhiều body parts?
- Tăng `dof_init_noise` để agent học từ diverse states

**Nếu episode quá dài** (không converge):
- Giảm `episode_length_s`
- Hoặc tăng termination reward (-1.0 penalty)

**Nếu robot không reset properly**:
- Kiểm tra `curriculum` settings
- Kiểm tra `custom_origins` - terrain offset có tính?
