# 🎮 Task Manipulation Rewards - HRL Tasks

Các task manipulation sử dụng **Hierarchical Reinforcement Learning (HRL)** - kết hợp low-level locomotion skills với high-level object manipulation.

---

## 📋 Quick Summary

| Task | Primary Reward | Scale | Goal | Skills Used |
|---|---|---|---|---|
| **task_reach** | wrist_pos | 5.0 | Tay cấp vật thể | walking, reaching |
| **task_transfer** | box_pos | 5.0 | Chuyển box từ A → B | walking, reaching |
| **task_lift** | box_pos + wrist_distance | 5.0 + 5.0 | Nâng box cao | reaching, squatting |
| **task_carry** | box_pos + wrist_distance | 5.0 + 5.0 | Cầm box đi | walking, reaching |
| **task_button** | - | - | Bấm nút | reaching |
| **task_box** | - | - | Đẩy box | walking, reaching |
| **task_ball** | - | - | Chơi bóng | walking, reaching |
| **task_cabinet** | - | - | Mở tủ | reaching |

---

## 🎯 Chi Tiết Các Task

### 1️⃣ **task_reach** - Cấp Tay Mục Tiêu

**Goal**: Tay robot tiếp cận object tại vị trí random

**Reward Config**:
```python
class scales:
    wrist_pos = 5  # ⭐ Primary: Vị trí cổ tay → mục tiêu
    # All others commented out (not used)
```

**Xác suất thành công**: Robot phải cấp được cổ tay đến object
**Episode**: 24 giây
**Command**: Random wrist position targets (3D)

---

### 2️⃣ **task_transfer** - Chuyển Box

**Goal**: Cầm box từ vị trí A, chuyển đến vị trí B

**Reward Config**:
```python
class scales:
    box_pos = 5              # ⭐ Primary: Vị trí box → mục tiêu
    wrist_box_distance = 1   # Secondary: Cổ tay gần box
```

**Xác suất thành công**: 
- Box phải ở vị trí mục tiêu
- Cổ tay phải gần box để "cầm" được

**Episode**: 8 giây (ngắn, vì focus vào 1 động tác)
**Skill Hierarchy**:
- Dùng `h1_walking` để di chuyển
- Dùng `h1_reaching` để cầm box

---

### 3️⃣ **task_lift** - Nâng Box

**Goal**: Nâng box từ sàn lên cao (Z > 0.3m-0.6m)

**Reward Config**:
```python
class scales:
    box_pos = 5              # ⭐ Primary: Vị trí box cao → mục tiêu (Z axis)
    wrist_box_distance = 5   # ⭐ Co-primary: Cổ tay phải gần box
```

**Xác suất thành công**: 
- Box phải ở độ cao 0.3-0.6m
- Tay phải cầm chặt (distance nhỏ)

**Episode**: 8 giây
**Skill Hierarchy**:
- Dùng `h1_reaching` để cầm box
- Dùng `h1_squatting` để nâng cao (hông lên)

**Khác biệt với transfer**: 
- transfer = chuyển XY
- lift = nâng Z (theo phương thẳng đứng)

---

### 4️⃣ **task_carry** - Cầm Box Đi Bộ

**Goal**: Cầm box, di chuyển đến vị trí mục tiêu

**Reward Config**:
```python
class scales:
    box_pos = 5              # ⭐ Primary: Vị trí box XYZ → mục tiêu
    wrist_box_distance = 5   # ⭐ Co-primary: Cổ tay gần box
```

**Xác suất thành công**: 
- Box phải đi tới vị trí XYZ
- Tay phải cầm chặt lúc di chuyển

**Episode**: 8 giây
**Command**: Box target position (XYZ random)
**Skill Hierarchy**:
- Dùng `h1_walking` để di chuyển
- Dùng `h1_reaching` để cầm box
- Dùng `h1_squatting` để điều chỉnh độ cao

**Khác biệt với transfer**: 
- transfer = chuyển ngang (XY)
- carry = chuyển + cầm (XYZ + force control)

---

### 5️⃣ **task_button** - Bấm Nút

**Goal**: Robot cấp được tay tới nút và bấm

**Status**: Config có sẵn nhưng reward scale comment out (chưa hoàn thiện)

**Dự kiến Reward**:
```python
# Có thể dùng:
wrist_pos = 5          # Vị trí cổ tay
# + Contact force reward (force > threshold)
```

---

### 6️⃣ **task_box** - Đẩy Box

**Goal**: Robot đẩy box sang một hướng

**Status**: Config có sẵn nhưng reward scale comment out (chưa hoàn thiện)

**Dự kiến Reward**:
```python
# Có thể dùng:
box_pos = 5            # Vị trí box sau khi đẩy
# + Contact with hand reward
```

---

### 7️⃣ **task_ball** - Chơi Bóng

**Goal**: Robot chơi với bóng (kick, catch, throw)

**Status**: Config có sẵn nhưng reward scale comment out (chưa hoàn thiện)

---

### 8️⃣ **task_cabinet** - Mở Tủ

**Goal**: Mở cửa tủ

**Status**: Config có sẵn nhưng reward scale comment out (chưa hoàn thiện)

---

## 🔗 Hierarchical Skill Structure

Mỗi task manipulation kết hợp multiple low-level skills:

```
task_reach
├── h1_walking (di chuyển)
└── h1_reaching (cấp tay)

task_transfer
├── h1_walking (di chuyển)
└── h1_reaching (cầm + chuyển box)

task_lift
├── h1_reaching (cầm box)
└── h1_squatting (nâng lên)

task_carry
├── h1_walking (cầm box đi)
├── h1_reaching (cầm chặt)
└── h1_squatting (điều chỉnh độ cao)
```

**Cách hoạt động**:
- High-level policy (task level) quyết định **KHI NÀO** dùng skill nào
- Low-level policy (skill level) quyết định **LÀM CÁCH NÀO** thực hiện skill
- Ví dụ: Để carry, robot phải:
  1. **Walking skill**: Quyết định từng bước đi
  2. **Reaching skill**: Quyết định vị trí tay cầm box
  3. **Squatting skill**: Quyết định độ cao cơ thể

---

## ⚙️ Tuning Guidelines

### Khi thay đổi reward scale:

1. **Primary goal (5.0)**: Chỉ có 1-2 primary rewards
   - Ví dụ: `box_pos = 5` → robot focus nâng box cao
   
2. **Secondary goal (1.0-5.0)**: Constraint để làm primary goal
   - Ví dụ: `wrist_box_distance = 5` → tay phải gần box
   
3. **Penalty (âm)**: Tránh hành động không mong muốn
   - Mặc định: torques, dof_vel, dof_acc (comment out)

### Ví dụ tuning:

```python
# Hiện tại (lift):
box_pos = 5                # Nâng box cao
wrist_box_distance = 5     # Tay gần box

# Nếu robot quên cầm box (rơi box):
# → Tăng wrist_box_distance lên 10

# Nếu robot cầm nhưng không nâng cao:
# → Tăng box_pos lên 10
# → Hoặc thêm box_velocity reward
```

---

## 📊 Episode Config

| Task | Episode Length | Num Envs | Command Dim | Action Dim |
|---|---|---|---|---|
| All task_* | 8 sec | 4096 | 9 (8 task) | 19 (arm + leg) |
| (vs reach) | 24 sec | 4096 | 14 (full) | 19 |

**Lưu ý**:
- `command_dim = 9`: Task command (wrist position hoặc box position)
- Episode ngắn (8s) vì mỗi task là 1 action ngắn (chuyển, nâng, etc.)
- H1 Wrist URDF: 19 DOF (15 leg + 4 arm cho wrist)

---

## 🚀 Implementation Notes

**HRL Architecture**:
- `ActorCriticHierarchical` policy (line 306 config)
- Mỗi skill có riêng Actor-Critic network
- Task-level policy aggregates skills output

**Skill Loading** (line 270-295 config):
```python
skill_dict = {
    'h1_walking': {
        "experiment_name": "h1_walking",
        "load_run": "0000_best",  # Best checkpoint
        "checkpoint": -1,          # Auto find latest
        "low_high": (-2, 2)        # Output scale
    },
    'h1_reaching': {...},
    'h1_squatting': {...},
    # h1_stepping: commented out (not needed)
}
```

Để thêm task mới:
1. Tạo folder `h1_task_XXX/`
2. Tạo config với `num_actions = 19`, `command_dim = 9`
3. Thêm reward scales cho primary goal
4. Define skill_dict với suitable low-level skills
5. Train với `--num_envs 4` hoặc 4096 tùy GPU

