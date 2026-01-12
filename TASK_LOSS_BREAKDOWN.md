# 📊 Task Loss Breakdown - SkillBlender Training

Phân tích chi tiết loss/reward structure của các task HRL (không tính low-level skills) khi train SkillBlender.

---

## 🎯 Task Loss Formula

```
Total Loss = Clip( sum(reward_i * scale_i) , min=0 if only_positive_rewards=True )
           + termination_penalty (nếu episode kết thúc sớm)
```

**Key Points:**
- `only_positive_rewards = True` → Clip total reward ≥ 0 (tránh early termination)
- Mỗi step tính reward từ các thành phần
- Loss được accumulate qua episode (max 8000 steps cho task)

---

## 📋 Task-Specific Reward Scales

### 1️⃣ **task_transfer** - Chuyển Box

**Config**: `/h1_task_transfer/h1_task_transfer_config.py` (line 207-245)

**Active Rewards** (Scale ≠ 0):

| Component | Scale | Formula | Role |
|---|---|---|---|
| **box_pos** ⭐ | 5.0 | `exp(-4 * error)` | Primary: Box → goal |
| **wrist_box_distance** | 1.0 | `exp(-4 * error)` | Secondary: Tay gần box |

**Total Reward per step**:
```
R = 5.0 * box_pos(error)      [0 to 5]
  + 1.0 * wrist_distance(err) [0 to 1]
  ────────────────────────────
  Max ≈ 6.0 (if all perfect)
  
Clipped to [0, ∞) (only_positive_rewards = True)
```

**Inactive Rewards** (commented out):
```
joint_pos, wrist_pos, feet_clearance, feet_contact_number,
feet_air_time, foot_slip, feet_distance, knee_distance,
tracking_lin_vel, tracking_ang_vel, default_joint_pos, 
upper_body_pos, orientation, base_height, base_acc,
vel_mismatch_exp, low_speed, track_vel_hard, torques,
dof_vel, dof_acc, collision, action_smoothness
```

**Episode Total** (8 seconds = 8000 steps):
- Optimal: 8000 × 6.0 = 48,000
- Realistic: 8000 × 3-4 = 24,000-32,000
- Poor: 8000 × 0-1 = 0-8,000

---

### 2️⃣ **task_lift** - Nâng Box

**Config**: `/h1_task_lift/h1_task_lift_config.py` (line 197-245)

**Active Rewards**:

| Component | Scale | Formula | Role |
|---|---|---|---|
| **box_pos** ⭐ | 5.0 | `exp(-4 * error)` | Primary: Box nâng lên |
| **wrist_box_distance** ⭐ | 5.0 | `exp(-4 * error)` | Co-primary: Tay gần box |

**Total Reward per step**:
```
R = 5.0 * box_pos(error)        [0 to 5]
  + 5.0 * wrist_distance(error) [0 to 5]
  ────────────────────────────
  Max ≈ 10.0 (if all perfect)
```

**Key Difference from task_transfer**:
- ✅ `wrist_box_distance = 5.0` (vs 1.0) → Tay phải gần box hơn
- ✅ Box_pos tracks Z axis (height) không phải XY
- ❌ Không track velocity (chỉ position)

**Episode Total** (8 seconds = 8000 steps):
- Optimal: 8000 × 10.0 = 80,000
- Realistic: 8000 × 5-7 = 40,000-56,000
- Poor: 8000 × 0-2 = 0-16,000

---

### 3️⃣ **task_reach** - Cấp Tay

**Config**: `/h1_task_reach/h1_task_reach_config.py` (line 204-240)

**Active Rewards**:

| Component | Scale | Formula | Role |
|---|---|---|---|
| **wrist_pos** ⭐ | 5.0 | `exp(-4 * error)` | Primary: Tay → goal |

**Total Reward per step**:
```
R = 5.0 * wrist_pos(error) [0 to 5]
  ────────────────────────
  Max ≈ 5.0
```

**Simplest Task**:
- ✅ Chỉ track wrist position (không tracking box)
- ✅ Dễ nhất để converge
- ✅ 24 giây episode (vs 8s cho task khác)

**Episode Total** (24 seconds = 24000 steps):
- Optimal: 24000 × 5.0 = 120,000
- Realistic: 24000 × 3-4 = 72,000-96,000
- Poor: 24000 × 0-1 = 0-24,000

---

### 4️⃣ **task_carry** - Cầm Box Đi

**Config**: `/h1_task_carry/h1_task_carry_config.py` (line 199-237)

**Active Rewards** (Based on code):

| Component | Scale | Formula | Role |
|---|---|---|---|
| **box_pos** ⭐ | 5.0 | `exp(-4 * error)` | Primary: Box → goal XYZ |
| **wrist_box_distance** ⭐ | 5.0 | `exp(-4 * error)` | Co-primary: Tay cầm chặt |

**Total Reward per step**:
```
R = 5.0 * box_pos(error)        [0 to 5]
  + 5.0 * wrist_distance(error) [0 to 5]
  ────────────────────────────
  Max ≈ 10.0
```

**Similar to task_lift** nhưng:
- ✅ Box_pos tracks XYZ (chuyển động + nâng)
- ✅ Cần maintain grip khi đi (wrist_distance = 5.0)
- ❌ Có thể phức tạp hơn (cần walk + hold)

---

### Other Tasks (Chưa hoàn thiện)

**task_button, task_box, task_ball, task_cabinet**:
- Config có sẵn nhưng `reward.scales` **toàn bộ comment out**
- Không có active reward nào → **Loss = 0**
- Cần implement reward functions và uncomment scales

---

## 📈 Reward Component Details

### Primary Reward: `box_pos` (Scale = 5.0)

**Function**:
```python
def _reward_box_pos(self):
    box_pos_diff = self.box_root_states[:, :3] - self.box_goal_pos
    box_pos_error = torch.mean(torch.abs(box_pos_diff), dim=1)  # Mean error across XYZ
    return torch.exp(-4 * box_pos_error), box_pos_error
```

**Behavior**:
- Error = 0.0m → Reward = 1.0 (exp(0) = 1)
- Error = 0.1m → Reward ≈ 0.67 (exp(-0.4) ≈ 0.67)
- Error = 0.2m → Reward ≈ 0.45 (exp(-0.8) ≈ 0.45)
- Error = 0.5m → Reward ≈ 0.14 (exp(-2) ≈ 0.14)
- Error ≥ 1.0m → Reward ≈ 0 (exp(-4) ≈ 0)

**Effective Range**: 0-0.5m (reward ≥ 0.14)

---

### Secondary Reward: `wrist_box_distance` (Scale = 1.0 or 5.0)

**Function**:
```python
def _reward_wrist_box_distance(self):
    wrist_pos = self.rigid_state[:, self.wrist_indices, :3]  # [N, 2, 3] - 2 hands
    wrist_pos = wrist_pos.flatten()  # [N, 6]
    box_pos = self.box_root_states[:, :3]  # [N, 3]
    wrist_box_diff = torch.norm(wrist_pos - box_pos.unsqueeze(1))  # Distance
    return torch.exp(-4 * error), error
```

**Behavior**:
- Distance = 0.0m → Reward = 1.0
- Distance = 0.05m → Reward ≈ 0.82
- Distance = 0.1m → Reward ≈ 0.67
- Distance = 0.2m → Reward ≈ 0.45

**Role**:
- task_transfer: `scale=1.0` → "Nice-to-have" (tay gần box tốt)
- task_lift/carry: `scale=5.0` → "Must-have" (tay phải cầm chặt)

---

### Wrist Position Reward (Scale = 5.0)

**task_reach only**:
```python
def _reward_wrist_pos(self):
    wrist_pos_diff = wrist_pos - ref_wrist_pos  # [N, 2, 3]
    wrist_pos_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)
    return torch.exp(-4 * wrist_pos_error)
```

**Direct tracking of wrist to goal position**

---

## 🎯 Loss During Training

### Example: task_transfer Training

**Iteration 0** (random policy):
```
box_pos error ≈ 1.0m     → box_pos reward ≈ 0 × 5.0 = 0
wrist_distance ≈ 1.0m    → wrist reward ≈ 0 × 1.0 = 0
────────────────────────────
Step reward ≈ 0
Episode total ≈ 0
```

**Iteration 1000** (learning):
```
box_pos error ≈ 0.3m     → box_pos reward ≈ 0.30 × 5.0 = 1.5
wrist_distance ≈ 0.2m    → wrist reward ≈ 0.45 × 1.0 = 0.45
────────────────────────────
Step reward ≈ 1.95
Episode total ≈ 1.95 × 8000 ≈ 15,600
```

**Iteration 10000** (convergence):
```
box_pos error ≈ 0.05m    → box_pos reward ≈ 0.82 × 5.0 = 4.1
wrist_distance ≈ 0.05m   → wrist reward ≈ 0.82 × 1.0 = 0.82
────────────────────────────
Step reward ≈ 4.92
Episode total ≈ 4.92 × 8000 ≈ 39,360
```

---

## 📊 Comparison Table

| Task | Episode Length | Num Rewards | Max Scale | Expected Convergence | Difficulty |
|---|---|---|---|---|---|
| **task_reach** | 24s | 1 | 5.0 | Easiest | ⭐ |
| **task_transfer** | 8s | 2 | 5.0+1.0=6.0 | Easy | ⭐⭐ |
| **task_lift** | 8s | 2 | 5.0+5.0=10.0 | Medium | ⭐⭐⭐ |
| **task_carry** | 8s | 2 | 5.0+5.0=10.0 | Hard | ⭐⭐⭐⭐ |

---

## 🔧 Tuning Loss

### Nếu loss quá nhỏ (không converge):

```python
# Tăng scale của primary reward
class scales:
    box_pos = 10.0  # (từ 5.0) → Enforce box position tracking mạnh hơn
    wrist_box_distance = 2.0  # (từ 1.0 or 5.0) → Enforce grip mạnh hơn
```

### Nếu loss quá lớn (unstable):

```python
# Giảm scale
class scales:
    box_pos = 2.0  # (từ 5.0)
    wrist_box_distance = 0.5  # (từ 1.0)
```

### Nếu agent fail (ngã):

```python
# Thêm stability rewards
class scales:
    box_pos = 5.0
    wrist_box_distance = 1.0
    orientation = 1.0  # Giữ thân thẳng
    base_height = 0.2  # Giữ hông ở độ cao
    default_joint_pos = 0.5  # Giữ từ thế chuẩn
```

---

## 🎯 Key Insights

| Aspect | Finding |
|---|---|
| **Simplest Task** | task_reach (wrist_pos only, 5.0 scale) |
| **Most Constrained** | task_lift/carry (10.0 combined scale, hard to balance) |
| **Least Stable** | task_transfer (asymmetric scales 5.0+1.0) |
| **Best for Learning** | task_reach (dễ xem progress) |
| **Best for Complex Skill** | task_carry (nhiều constraint = learning signal) |

---

## 🚀 Recommended Loss Config for Better Convergence

```python
# Current (aggressive):
class scales:
    box_pos = 5.0
    wrist_box_distance = 1.0  # vs 5.0

# Proposed (balanced):
class scales:
    # Primary objective
    box_pos = 5.0
    # Secondary objective (penalty weight)
    wrist_box_distance = 2.0  # Increase to 2.0 for more stability
    # Add stability rewards
    orientation = 0.5
    base_height = 0.1
```

Với config này:
- ✅ Max scale = 7.6 (moderate)
- ✅ Clear priority (box > wrist > stability)
- ✅ Learning signal rõ ràng
- ✅ Không quá aggressive → stable convergence
