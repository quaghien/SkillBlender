# Reward Functions Analysis - 8 Tasks Comparison

## 📋 Tổng Quan

Tài liệu này so sánh **reward functions gốc của tác giả** (trong các file single-task) với **reward functions của HRL meta-environment** (trong `h1_hrl.py`).

---

## 🎯 Training Flow (Lệnh chạy)

```bash
python legged_gym/scripts/train.py --task h1_hrl --headless \
    --wandb hrlv1_project \
    --run_name v3
```

### Code Flow Chi Tiết

```
train.py
│
├── args = get_args()                           # Parse arguments
│
├── wandb.init(project=args.wandb, name=args.run_name)
│
├── env, env_cfg = task_registry.make_env(name="h1_hrl")
│   └── H1HRLEnv(H1HRLCfg, ...)               # From legged_gym/envs/__init__.py line 167
│       ├── num_envs = 4096
│       ├── num_obs = 105  (State 69 + Goal 14 + Mask 14 + TaskID 8)
│       ├── num_privileged_obs = 303  (3 × 101)
│       └── compute_reward()                    # Uses task-specific rewards
│
├── ppo_runner, train_cfg = task_registry.make_alg_runner(env, "h1_hrl")
│   └── OnPolicyRunnerHRL(env, H1HRLCfgPPO)
│       ├── actor_critic = ActorCriticHRL(num_obs=105, ...)
│       │   ├── Shared Encoder (105 → 256 → 256 → 256)
│       │   ├── Skill Head (256 → 128 → 4 skills)
│       │   ├── Command Head (256+4 → 256 → 128 → 14 commands)
│       │   └── Low-Level Skills (pretrained: walking, reaching, squatting, stepping)
│       │
│       └── alg = PPO_HRL(actor_critic, ...)
│           └── CurriculumController(stage1_end=20k, total=100k)
│               ├── Stage 1: K=10, ε=0.18, τ=2.0 (explore skills)
│               └── Stage 2: K→5, ε→0, τ→1.0 (refine commands)
│
└── ppo_runner.learn(num_learning_iterations=100000)
    │
    └── Training Loop (each iteration):
        ├── env.step(actions)
        │   └── compute_reward()               # Task-specific rewards
        │       ├── task_id=0: _reward_reach_with_metrics()
        │       ├── task_id=1: _reward_button_with_metrics()
        │       ├── task_id=2: _reward_cabinet_with_metrics()
        │       ├── task_id=3: _reward_ball_with_metrics()
        │       └── task_id=4-7: _reward_box_task_with_metrics()
        │
        ├── alg.update()                        # PPO update
        │   └── curriculum.update(iteration)   # Update K, ε, τ, entropy coef
        │
        └── wandb.log({
                "reward/total", "reward/per_task",
                "skill/histogram", "skill/switch_rate",
                "curriculum/K", "curriculum/epsilon", "curriculum/tau"
            })
```

### Key Files
| File | Purpose |
|------|---------|
| [train.py](../../legged_gym/legged_gym/scripts/train.py) | Entry point |
| [h1_hrl.py](../../legged_gym/legged_gym/envs/h1/h1_hrl/h1_hrl.py) | HRL Environment + 8 task rewards |
| [on_policy_runner_hrl.py](../../rsl_rl/rsl_rl/runners/on_policy_runner_hrl.py) | HRL training orchestrator |
| [ppo_hrl.py](../../rsl_rl/rsl_rl/algorithms/ppo_hrl.py) | HRL PPO + Curriculum |
| [actor_critic_hrl.py](../../rsl_rl/rsl_rl/modules/actor_critic_hrl.py) | HRL policy network |

### Curriculum Learning (2-Stage)

| Stage | Iterations | K | ε | τ | c_ent_skill | lr_cmd |
|-------|------------|---|---|---|-------------|--------|
| **1** | 0 → 20k | 10 | 0.18 | 2.0 | 0.02 | 0.2× |
| **2** | 20k → 100k | 10→5 | 0.18→0 | 2.0→1.0 | 0.02→0.005 | 1.0× |

- **K**: Option duration (giữ skill bao nhiêu bước)
- **ε**: Exploration rate (random skill selection)
- **τ**: Temperature (softmax sharpness)
- **c_ent_skill**: Skill entropy bonus coefficient

---

## 🔍 8 Task Reward Functions Comparison

### **Task 0: REACH**

#### Original (h1_task_reach.py + config)
```python
# Config scale: wrist_pos = 5

def _reward_wrist_pos(self):
    wrist_pos = self.rigid_state[:, self.wrist_indices, :7]  # [num_envs, 2, 7]
    wrist_pos_diff = wrist_pos[:,:,:3] - self.ref_wrist_pos[:,:,:3]  # position only
    wrist_pos_diff = torch.flatten(wrist_pos_diff, start_dim=1)  # [num_envs, 6]
    wrist_pos_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)
    return torch.exp(-4 * wrist_pos_error), wrist_pos_error

# Formula: scale × exp(-4 × error)
# Total: 5 × exp(-4 × mean_abs_error)
```

#### HRL Version (h1_hrl.py)
```python
def _reward_reach_with_metrics(self, mask):
    wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :3]  # [N_masked, 2, 3]
    wrist_pos = wrist_pos.reshape(mask.sum(), 6)  # [N_masked, 6]
    target = self.goal_value[mask, :6]  # [N_masked, 6]
    
    error = torch.mean(torch.abs(wrist_pos - target), dim=-1)  # [N_masked]
    raw_reward = 5.0 * torch.exp(-4.0 * error)  # scale=5, decay=-4
    
    return raw_reward * 120.0, metrics  # BALANCE factor
```

| Aspect | Original | HRL |
|--------|----------|-----|
| Scale | 5 | 5 × 120.0 = 600 |
| Decay | -4 | -4 |
| Target | ref_wrist_pos (from trajectory) | goal_value[:6] (sampled) |
| Dims | 6 (2 wrists × 3 pos) | 6 (2 wrists × 3 pos) |
**🇻🇳 Giải thích khác biệt (Reach):**
- **Nguồn mục tiêu**: Gốc dùng `ref_wrist_pos` từ trajectory được record sẵn, HRL dùng `goal_value[:6]` được sample ngẫu nhiên trong không gian 3D
- **Balance factor rất lớn (×120)**: Vì task reach chỉ có 1 reward term (wrist_pos), reward magnitude nhỏ hơn các task khác có 2 terms → cần scale lên nhiều để cân bằng với các task khác trong multi-task learning
- **Công thức hoàn toàn giống**: Cùng scale=5, decay=-4, cùng tính mean absolute error của 6 dims (2 wrists × 3 pos xyz)
---

### **Task 1: BUTTON**

#### Original (h1_task_button.py + config)
```python
# Config scales: wrist_button_distance = 5, right_arm_default = 0.5

def _reward_wrist_button_distance(self):
    wrist_pos = self.rigid_state[:, self.wrist_indices, :7]  # two hands
    wrist_pos = wrist_pos[:, 0, :3]  # LEFT hand only
    button_goal_pos = self.button_goal_pos[:, :3]
    wrist_button_diff = wrist_pos - button_goal_pos
    wrist_button_error = torch.mean(torch.abs(wrist_button_diff), dim=1)
    return torch.exp(-4 * wrist_button_error), wrist_button_error

def _reward_right_arm_default(self):
    right_shoulder_pitch_index = 15
    joint_diff = self.dof_pos - self.default_joint_pd_target
    right_arm_diff = joint_diff[:, right_shoulder_pitch_index:]  # indices 15-18
    right_arm_error = torch.mean(torch.abs(right_arm_diff), dim=1)
    return torch.exp(-4 * right_arm_error), right_arm_error

# Total: 5 × exp(-4 × wrist_error) + 0.5 × exp(-4 × arm_error)
```

#### HRL Version (h1_hrl.py)
```python
def _reward_button_with_metrics(self, mask):
    # Left wrist to button
    left_wrist_pos = self.rigid_state[mask][:, self.wrist_indices[0], :3]
    button_pos = self.button_pos[mask]
    wrist_error = torch.mean(torch.abs(left_wrist_pos - button_pos), dim=-1)
    rew_wrist = 5.0 * torch.exp(-4.0 * wrist_error)  # scale=5
    
    # Right arm default position
    right_arm_dof = self.dof_pos[mask][:, self.right_arm_indices]
    right_arm_default = self.default_dof_pos[0, self.right_arm_indices]
    arm_error = torch.mean(torch.abs(right_arm_dof - right_arm_default), dim=-1)
    rew_arm = 0.5 * torch.exp(-4.0 * arm_error)  # scale=0.5
    
    raw_reward = rew_wrist + rew_arm
    return raw_reward * 0.167, metrics  # BALANCE factor
```

| Aspect | Original | HRL |
|--------|----------|-----|
| Wrist Scale | 5 | 5 |
| Arm Scale | 0.5 | 0.5 |
| Balance | None | × 0.167 |
| Hand | Left only | Left only |

**🇻🇳 Giải thích khác biệt (Button):**
- **2 thành phần reward**: (1) Khoảng cách tay trái → nút bấm, (2) Giữ tay phải ở vị trí mặc định
- **Tại sao chỉ tay trái?**: Nút bấm được đặt bên trái robot, tay phải cần giữ yên để không va chạm
- **Arm scale nhỏ (0.5)**: Penalty nhẹ cho tay phải lệch vị trí, không ảnh hưởng nhiều đến tổng reward
- **Balance factor nhỏ (×0.167)**: Task button có 2 terms nên reward magnitude lớn hơn reach → giảm xuống

---

### **Task 2: CABINET**

#### Original (h1_task_cabinet.py + config)
```python
# Config scales: wrist_arti_obj_distance = 5, arti_obj_dof = 5

def _reward_wrist_arti_obj_distance(self):
    wrist_pos = self.rigid_state[:, self.wrist_indices, :3]  # [num_envs, 2, 3]
    arti_obj_pos = self.arti_obj_root_states[:, :3]  # cabinet position
    wrist_arti_obj_diff = wrist_pos - arti_obj_pos.unsqueeze(1)  # [num_envs, 2, 3]
    wrist_arti_obj_diff = torch.flatten(wrist_arti_obj_diff, start_dim=1)  # [num_envs, 6]
    wrist_arti_obj_error = torch.mean(torch.abs(wrist_arti_obj_diff), dim=1)
    return torch.exp(-4 * wrist_arti_obj_error), wrist_arti_obj_error

def _reward_arti_obj_dof(self):
    arti_obj_dof_diff = self.arti_obj_dof_state[:, :, 0] - self.arti_obj_dof_goal  # [num_envs, 2]
    arti_obj_dof_error = torch.mean(torch.abs(arti_obj_dof_diff), dim=1)
    return torch.exp(-4 * arti_obj_dof_error), arti_obj_dof_error

# Total: 5 × exp(-4 × wrist_error) + 5 × exp(-4 × door_error)
```

#### HRL Version (h1_hrl.py)
```python
def _reward_cabinet_with_metrics(self, mask):
    # BOTH wrists to door handle
    wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :3]  # [N, 2, 3]
    handle_pos = self.goal_value[mask, :3]  # Handle position
    wrist_handle_diff = wrist_pos - handle_pos.unsqueeze(1)  # [N, 2, 3]
    wrist_error = torch.mean(torch.abs(wrist_handle_diff.reshape(mask.sum(), 6)), dim=-1)
    rew_wrist = 5.0 * torch.exp(-4.0 * wrist_error)  # scale=5
    
    # Door angle to target
    angle_error = torch.abs(self.door_angle[mask] - self.door_target[mask])
    rew_door = 5.0 * torch.exp(-4.0 * angle_error)  # scale=5
    
    raw_reward = rew_wrist + rew_door
    return raw_reward * 0.728, metrics  # BALANCE factor
```

| Aspect | Original | HRL |
|--------|----------|-----|
| Wrist Scale | 5 | 5 |
| Door Scale | 5 | 5 |
| Balance | None | × 0.728 |
| Wrists | BOTH (2×3=6 dims) | BOTH (2×3=6 dims) |

**🇻🇳 Giải thích khác biệt (Cabinet):**
- **2 thành phần reward**: (1) Hai tay tiếp cận tay cầm tủ, (2) Góc mở cửa tủ đạt target
- **Dùng CẢ 2 TAY**: Mở tủ cần cả 2 tay để cầm và kéo, khác với button chỉ cần 1 tay
- **Door angle**: Gốc dùng `arti_obj_dof_state` từ articulated object, HRL dùng `door_angle` được track riêng
- **Balance factor khá lớn (×0.728)**: Cabinet có 2 terms nhưng door angle khó đạt hơn wrist position → cần reward cao hơn

---

### **Task 3: BALL**

#### Original (h1_task_ball.py + config)
```python
# Config scales: torso_pos = 1, ball_pos = 5

def _reward_torso_pos(self):
    torso_pos = self.rigid_state[:, self.torso_indices, :3].squeeze(1)  # [envs, 3]
    torso_ori_ball_pos_diff = self.ori_ball_pos - torso_pos
    torso_ori_ball_pos_diff = torso_ori_ball_pos_diff[:, :2]  # ONLY xy
    torso_ori_ball_pos_error = torch.mean(torch.abs(torso_ori_ball_pos_diff), dim=1)
    return torch.exp(-4 * torso_ori_ball_pos_error), torso_ori_ball_pos_error

def _reward_ball_pos(self):
    ball_goal_diff = self.ball_root_states[:, :3] - self.goal_pos
    ball_goal_error = torch.mean(torch.abs(ball_goal_diff), dim=1)
    return torch.exp(-1 * ball_goal_error), ball_goal_error  # NOTE: decay=-1

# Total: 1 × exp(-4 × torso_error_xy) + 5 × exp(-1 × ball_error_xyz)
```

#### HRL Version (h1_hrl.py)
```python
def _reward_ball_with_metrics(self, mask):
    # Torso to ORIGINAL ball position (xy only)
    torso_pos = self.rigid_state[mask][:, self.torso_indices[0], :2]  # [N, 2]
    ori_ball_xy = self.ball_pos[mask, :2]  # approximation
    torso_error = torch.mean(torch.abs(torso_pos - ori_ball_xy), dim=-1)
    rew_torso = 1.0 * torch.exp(-4.0 * torso_error)  # scale=1, decay=-4
    
    # Ball to goal (xyz)
    ball_pos = self.ball_pos[mask]  # [N, 3]
    goal_pos = self.ball_target[mask]  # [N, 3]
    ball_error = torch.mean(torch.abs(ball_pos - goal_pos), dim=-1)
    rew_ball = 5.0 * torch.exp(-1.0 * ball_error)  # scale=5, decay=-1
    
    raw_reward = rew_torso + rew_ball
    return raw_reward * 0.091, metrics  # BALANCE factor
```

| Aspect | Original | HRL |
|--------|----------|-----|
| Torso Scale | 1 | 1 |
| Ball Scale | 5 | 5 |
| Torso Decay | -4 | -4 |
| Ball Decay | **-1** | **-1** (special!) |
| Balance | None | × 0.091 |

**🇻🇳 Giải thích khác biệt (Ball):**
- **ĐẶC BIỆT - Decay khác nhau**: Ball dùng decay=-1 thay vì -4 thông thường
  - Decay=-1 → reward giảm chậm hơn khi error tăng → khuyến khích exploration
  - Lý do: Đá bóng khó kiểm soát chính xác, cần cho phép sai số lớn hơn
- **2 thành phần reward**: (1) Torso tiến về vị trí ban đầu của bóng (chỉ xy), (2) Bóng đến vị trí goal (xyz)
- **Torso chỉ dùng XY**: Robot cần di chuyển ngang để đến gần bóng, không cần quan tâm chiều cao z
- **Balance factor nhỏ (×0.091)**: Ball task có potential reward lớn (decay chậm) → cần giảm xuống

---

### **Task 4: BOX**

#### Original (h1_task_box.py + config)
```python
# Config scales: box_pos = 5, wrist_box_distance = 5

def _reward_box_pos(self):
    box_pos_diff = self.box_root_states[:, :3] - self.box_goal_pos
    box_pos_error = torch.mean(torch.abs(box_pos_diff), dim=1)
    return torch.exp(-4 * box_pos_error), box_pos_error

def _reward_wrist_box_distance(self):
    wrist_pos = self.rigid_state[:, self.wrist_indices, :7]  # [num_envs, 2, 7]
    wrist_pos = wrist_pos[:,:,:3]  # position only [num_envs, 2, 3]
    box_pos = self.box_root_states[:, :3]  # [num_envs, 3]
    wrist_box_diff = wrist_pos - box_pos.unsqueeze(1)  # [num_envs, 2, 3]
    wrist_pos_diff = torch.flatten(wrist_box_diff, start_dim=1)  # [num_envs, 6]
    wrist_box_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)
    return torch.exp(-4 * wrist_box_error), wrist_box_error

# Total: 5 × exp(-4 × box_error) + 5 × exp(-4 × wrist_error)
```

#### HRL Version (h1_hrl.py)
```python
def _reward_box_task_with_metrics(self, mask, task_id=4):  # box
    box_pos = self.box_pos[mask]
    target = self.box_target[mask]
    
    box_error = torch.mean(torch.abs(box_pos - target), dim=-1)  # all axes
    rew_box = 5.0 * torch.exp(-4.0 * box_error)  # scale=5
    
    # BOTH wrists to box
    wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :3]  # [N, 2, 3]
    box_pos_expanded = box_pos.unsqueeze(1)  # [N, 1, 3]
    wrist_box_diff = wrist_pos - box_pos_expanded  # [N, 2, 3]
    wrist_error = torch.mean(torch.abs(wrist_box_diff.reshape(mask.sum(), 6)), dim=-1)
    
    wrist_scale = 5.0  # box task
    rew_grasp = wrist_scale * torch.exp(-4.0 * wrist_error)  # scale=5
    
    raw_reward = rew_box + rew_grasp
    return raw_reward * 0.061, metrics  # BALANCE factor for box
```

| Aspect | Original | HRL |
|--------|----------|-----|
| Box Scale | 5 | 5 |
| Wrist Scale | 5 | 5 |
| Balance | None | × 0.061 |
| Wrists | BOTH | BOTH |

**🇻🇳 Giải thích khác biệt (Box):**
- **Task cơ bản nhất trong 4 box tasks**: Đơn giản là đưa hộp từ vị trí A đến vị trí B
- **2 thành phần reward**: (1) Hộp đến vị trí goal, (2) Hai tay cầm sát hộp
- **Cả 2 tay phải cầm hộp**: Box nặng cần dùng 2 tay để nâng và mang
- **Balance factor nhỏ nhất (×0.061)**: Box có tổng reward lớn từ 2 terms đều scale=5 → cần giảm nhiều
- **Error tính cả XYZ**: Khác với lift chỉ quan tâm Z, box quan tâm vị trí 3D đầy đủ

---

### **Task 5: TRANSFER**

#### Original (h1_task_transfer.py + config)
```python
# Config scales: box_pos = 5, wrist_box_distance = 1 (NOTE: different from box!)

# Same reward functions as box task
# Total: 5 × exp(-4 × box_error) + 1 × exp(-4 × wrist_error)
```

#### HRL Version (h1_hrl.py)
```python
def _reward_box_task_with_metrics(self, mask, task_id=5):  # transfer
    # ... same as box ...
    wrist_scale = 1.0  # transfer task has lower wrist scale
    rew_grasp = wrist_scale * torch.exp(-4.0 * wrist_error)  # scale=1
    
    raw_reward = rew_box + rew_grasp
    return raw_reward * 0.08125, metrics  # BALANCE factor for transfer
```

| Aspect | Original | HRL |
|--------|----------|-----|
| Box Scale | 5 | 5 |
| Wrist Scale | **1** | **1** (different!) |
| Balance | None | × 0.08125 |

**🇻🇳 Giải thích khác biệt (Transfer):**
- **Wrist scale giảm từ 5 xuống 1**: Đây là điểm khác biệt quan trọng nhất!
  - Transfer = chuyển hộp giữa các vị trí xa nhau
  - Ưu tiên hộp đến đích (scale=5) hơn là tay cầm chặt (scale=1)
  - Cho phép "ném" hoặc "đẩy" hộp thay vì luôn phải cầm sát
- **Balance factor hơi lớn hơn box (×0.08125)**: Vì wrist term giảm, tổng reward nhỏ hơn → cần scale lên
- **Chiến lược học khác**: Robot có thể học cách đẩy/lăn hộp thay vì nâng và bê

---

### **Task 6: LIFT**

#### Original (h1_task_lift.py + config)
```python
# Config scales: box_pos = 5, wrist_box_distance = 5

def _reward_box_pos(self):
    # NOTE: For lift task, the box_pos_z range is [0.3, 0.6] - z-axis focused
    box_pos_diff = self.box_root_states[:, :3] - self.box_goal_pos
    box_pos_error = torch.mean(torch.abs(box_pos_diff), dim=1)
    return torch.exp(-4 * box_pos_error), box_pos_error

# Total: 5 × exp(-4 × box_error) + 5 × exp(-4 × wrist_error)
# But goal is z-axis only in practice
```

#### HRL Version (h1_hrl.py)
```python
def _reward_box_task_with_metrics(self, mask, task_id=6):  # lift
    box_pos = self.box_pos[mask]
    target = self.box_target[mask]
    
    # For lift task, only check z-axis for box_pos
    box_error = torch.abs(box_pos[:, 2] - target[:, 2])  # Z-ONLY
    rew_box = 5.0 * torch.exp(-4.0 * box_error)  # scale=5
    
    # Wrists same as other box tasks
    wrist_scale = 5.0
    rew_grasp = wrist_scale * torch.exp(-4.0 * wrist_error)  # scale=5
    
    raw_reward = rew_box + rew_grasp
    return raw_reward * 0.0475, metrics  # BALANCE factor for lift
```

| Aspect | Original | HRL |
|--------|----------|-----|
| Box Scale | 5 | 5 |
| Wrist Scale | 5 | 5 |
| Box Error | XYZ | **Z-only** |
| Balance | None | × 0.0475 |

**🇻🇳 Giải thích khác biệt (Lift):**
- **HRL chỉ dùng Z-axis cho box error**: Đây là đơn giản hóa quan trọng!
  - Lift = nâng hộp lên độ cao nhất định
  - Không quan tâm hộp ở vị trí XY nào, chỉ cần đạt chiều cao Z
  - Giúp robot tập trung vào việc "nâng" thay vì "di chuyển + nâng"
- **Gốc dùng XYZ**: Tác giả gốc yêu cầu hộp phải đạt vị trí 3D chính xác
- **Balance factor nhỏ nhất (×0.0475)**: Lift dễ đạt reward cao (chỉ 1D) → cần giảm nhiều
- **Wrist vẫn dùng scale=5**: Cần cầm chắc hộp khi nâng, không được rơi

---

### **Task 7: CARRY**

#### Original (h1_task_carry.py + config)
```python
# Config scales: box_pos = 5, wrist_box_distance = 5

# Same reward functions as box task, but with different goal ranges:
# box_pos_x = [0.3, 1.0], box_pos_y = [-0.3, 0.3], box_pos_z = [0.3, 0.6]

# Total: 5 × exp(-4 × box_error) + 5 × exp(-4 × wrist_error)
```

#### HRL Version (h1_hrl.py)
```python
def _reward_box_task_with_metrics(self, mask, task_id=7):  # carry
    # Same as box task
    wrist_scale = 5.0
    
    raw_reward = rew_box + rew_grasp
    return raw_reward * 0.0767, metrics  # BALANCE factor for carry
```

| Aspect | Original | HRL |
|--------|----------|-----|
| Box Scale | 5 | 5 |
| Wrist Scale | 5 | 5 |
| Balance | None | × 0.0767 |

**🇻🇳 Giải thích khác biệt (Carry):**
- **Công thức giống hệt Box task**: Cùng scale, cùng decay, cùng cách tính error
- **Khác biệt nằm ở GOAL RANGE**:
  - Box: goal gần robot hơn, phạm vi nhỏ
  - Carry: goal xa hơn, phạm vi rộng hơn (x: 0.3→1.0, y: -0.3→0.3, z: 0.3→0.6)
- **Carry = Box + di chuyển xa**: Robot cần bê hộp đi quãng đường dài hơn
- **Balance factor lớn hơn box (×0.0767 vs ×0.061)**: Carry khó hơn → reward cao hơn để khuyến khích
- **Kỹ năng cần**: Walking + grasping + stability khi di chuyển

---

## 📊 Summary Table

| Task | Original Formula | HRL Balance Factor | Decay |
|------|------------------|-------------------|-------|
| **Reach** | 5 × exp(-4 × wrist_err) | × 120.0 | -4 |
| **Button** | 5 × exp(-4 × wrist_err) + 0.5 × exp(-4 × arm_err) | × 0.167 | -4 |
| **Cabinet** | 5 × exp(-4 × wrist_err) + 5 × exp(-4 × door_err) | × 0.728 | -4 |
| **Ball** | 1 × exp(-4 × torso_err) + 5 × exp(**-1** × ball_err) | × 0.091 | -4 / **-1** |
| **Box** | 5 × exp(-4 × box_err) + 5 × exp(-4 × wrist_err) | × 0.061 | -4 |
| **Transfer** | 5 × exp(-4 × box_err) + **1** × exp(-4 × wrist_err) | × 0.08125 | -4 |
| **Lift** | 5 × exp(-4 × **z_err**) + 5 × exp(-4 × wrist_err) | × 0.0475 | -4 |
| **Carry** | 5 × exp(-4 × box_err) + 5 × exp(-4 × wrist_err) | × 0.0767 | -4 |

---

## � PHÂN TÍCH CHI TIẾT CODE - TẠI SAO HRL KHÓ HỌC HƠN?

### 1. **REACH TASK - SO SÁNH CODE TỪNG DÒNG**

#### 🟢 Original (h1_task_reach.py) - DỄ HỌC
```python
# === NGUỒN MỤC TIÊU ===
# ref_wrist_pos được tính từ trajectory record sẵn + vị trí ban đầu
self.ori_wrist_pos = self.rigid_state[:, self.wrist_indices, :7].clone()  # [num_envs, 2, 7]
self.ref_wrist_pos = self.target_wp[self.target_wp_i, self.target_wp_j] + self.ori_wrist_pos

# === WRIST INDICES ===  
wrist_names = [s for s in self.body_names if "wrist" in s]  # Tìm link "wrist" THẬT
self.wrist_indices[i] = gym.find_actor_rigid_body_handle(..., wrist_names[i])

# === REWARD ===
def _reward_wrist_pos(self):
    wrist_pos = self.rigid_state[:, self.wrist_indices, :7]  # Shape: [N, 2, 7] - pos+quat
    wrist_pos_diff = wrist_pos[:,:,:3] - self.ref_wrist_pos[:,:,:3]  # Chỉ lấy position [:3]
    wrist_pos_diff = torch.flatten(wrist_pos_diff, start_dim=1)  # [N, 6]
    wrist_pos_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)  # [N]
    return torch.exp(-4 * wrist_pos_error), wrist_pos_error  # scale=1 (config=5)
```

#### 🔴 HRL (h1_hrl.py) - KHÓ HỌC HƠN
```python
# === NGUỒN MỤC TIÊU - VẤN ĐỀ 1 ===
# goal_value được sample NGẪU NHIÊN trong không gian 3D cố định
target[0] = 0.2 + torch.rand(1).item() * 0.4  # x: 0.2-0.6 (KHÔNG PHỤ THUỘC robot pose!)
target[1] = 0.1 + torch.rand(1).item() * 0.4  # y: 0.1-0.5
target[2] = 0.7 + torch.rand(1).item() * 0.6  # z: 0.7-1.3
self.goal_value[env_id, :6] = target  # ĐÂY LÀ TỌA ĐỘ WORLD TUYỆT ĐỐI!

# === WRIST INDICES - VẤN ĐỀ 2 ===
self.wrist_indices = self.elbow_indices  # DÙNG ELBOW THAY VÌ WRIST!
# Lý do: wrist links không có collision mesh → dùng elbow làm proxy
# Vấn đề: Elbow cách wrist ~0.3m → error luôn có offset!

# === REWARD - VẤN ĐỀ 3 ===
def _reward_reach_with_metrics(self, mask):
    wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :3]  # [N_masked, 2, 3]
    wrist_pos = wrist_pos.reshape(mask.sum(), 6)  # Flatten khác cách!
    target = self.goal_value[mask, :6]  # goal_value là TỌA ĐỘ TUYỆT ĐỐI
    
    error = torch.mean(torch.abs(wrist_pos - target), dim=-1)
    raw_reward = 5.0 * torch.exp(-4.0 * error)
    return raw_reward * 120.0, metrics  # Balance factor cực lớn!
```

#### ❌ LỖI CHÍNH CỦA HRL REACH:
| Vấn đề | Original | HRL | Tại sao HRL sai? |
|--------|----------|-----|------------------|
| **Goal source** | `ref_wrist_pos` = trajectory + `ori_wrist_pos` | `goal_value` = random trong world frame | Original: goal TƯƠNG ĐỐI với pose ban đầu → luôn reachable. HRL: goal TUYỆT ĐỐI → có thể unreachable nếu robot đứng xa |
| **Wrist link** | `wrist_indices` = link "wrist" thật | `elbow_indices` = link "elbow" | Elbow cách wrist ~0.3m → error không bao giờ = 0 |
| **Balance factor** | None (reward ~0-5) | ×120 | Phải scale lên 120 lần vì reward magnitude nhỏ → gradients cực lớn, unstable |

---

### 2. **BUTTON TASK - SO SÁNH CODE TỪNG DÒNG**

#### 🟢 Original (h1_task_button.py) - DỄ HỌC
```python
# === NGUỒN MỤC TIÊU ===
# button_goal_pos từ object thật trong simulation
self.button_goal_pos = self.button_actor_states[:, :3]  # VỊ TRÍ THẬT của button

# === REWARD ===
def _reward_wrist_button_distance(self):
    wrist_pos = self.rigid_state[:, self.wrist_indices, :7]
    wrist_pos = wrist_pos[:, 0, :3]  # LEFT hand only, position [:3]
    button_goal_pos = self.button_goal_pos[:, :3]  # VỊ TRÍ THẬT
    wrist_button_diff = wrist_pos - button_goal_pos
    wrist_button_error = torch.mean(torch.abs(wrist_button_diff), dim=1)
    return torch.exp(-4 * wrist_button_error), wrist_button_error

def _reward_right_arm_default(self):
    right_shoulder_pitch_index = 15  # INDEX CHÍNH XÁC
    joint_diff = self.dof_pos - self.default_joint_pd_target  # default_joint_pd_target = target
    right_arm_diff = joint_diff[:, right_shoulder_pitch_index:]  # joints 15-18 (4 joints)
    right_arm_error = torch.mean(torch.abs(right_arm_diff), dim=1)
    return torch.exp(-4 * right_arm_error), right_arm_error
```

#### 🔴 HRL (h1_hrl.py) - KHÓ HỌC HƠN
```python
# === NGUỒN MỤC TIÊU - VẤN ĐỀ 1 ===
# button_pos được sample NGẪU NHIÊN (không có object thật!)
button_pos[0] = 0.3 + torch.rand(1).item() * 0.3  # x: 0.3-0.6
button_pos[1] = 0.1 + torch.rand(1).item() * 0.3  # y: 0.1-0.4
button_pos[2] = 0.8 + torch.rand(1).item() * 0.4  # z: 0.8-1.2
self.button_pos[env_id] = button_pos  # KHÔNG CÓ BUTTON THẬT!

# === REWARD - VẤN ĐỀ 2 ===
def _reward_button_with_metrics(self, mask):
    left_wrist_pos = self.rigid_state[mask][:, self.wrist_indices[0], :3]  # ELBOW!
    button_pos = self.button_pos[mask]  # VỊ TRÍ ẢO
    wrist_error = torch.mean(torch.abs(left_wrist_pos - button_pos), dim=-1)
    rew_wrist = 5.0 * torch.exp(-4.0 * wrist_error)
    
    # Right arm default - VẤN ĐỀ 3
    right_arm_dof = self.dof_pos[mask][:, self.right_arm_indices]  # Phải define self.right_arm_indices
    right_arm_default = self.default_dof_pos[0, self.right_arm_indices]  # default_dof_pos khác default_joint_pd_target?
    arm_error = torch.mean(torch.abs(right_arm_dof - right_arm_default), dim=-1)
    rew_arm = 0.5 * torch.exp(-4.0 * arm_error)
    
    return (rew_wrist + rew_arm) * 0.167, metrics
```

#### ❌ LỖI CHÍNH CỦA HRL BUTTON:
| Vấn đề | Original | HRL | Tại sao HRL sai? |
|--------|----------|-----|------------------|
| **Button source** | `button_actor_states` từ object simulation | `button_pos` random | Không có object thật → không có feedback từ physics |
| **Wrist link** | `wrist_indices[0]` = wrist trái | `self.wrist_indices[0]` = ELBOW trái | Elbow cách button thật xa hơn wrist |
| **Right arm ref** | `default_joint_pd_target` | `default_dof_pos` | Có thể khác nhau nếu config khác |

---

### 3. **CABINET TASK - SO SÁNH CODE TỪNG DÒNG**

#### 🟢 Original (h1_task_cabinet.py) - DỄ HỌC
```python
# === NGUỒN MỤC TIÊU ===
# arti_obj_root_states từ articulated object THẬT trong simulation
arti_obj_pos = self.arti_obj_root_states[:, :3]  # Vị trí cabinet THẬT
arti_obj_dof_state = self.arti_obj_dof_state[:, :, 0]  # Góc cửa THẬT từ physics

# === REWARD ===
def _reward_wrist_arti_obj_distance(self):
    wrist_pos = self.rigid_state[:, self.wrist_indices, :3]  # [N, 2, 3] - WRIST THẬT
    arti_obj_pos = self.arti_obj_root_states[:, :3]  # VỊ TRÍ THẬT
    wrist_arti_obj_diff = wrist_pos - arti_obj_pos.unsqueeze(1)  # broadcast
    wrist_arti_obj_diff = torch.flatten(wrist_arti_obj_diff, start_dim=1)  # [N, 6]
    wrist_arti_obj_error = torch.mean(torch.abs(wrist_arti_obj_diff), dim=1)
    return torch.exp(-4 * wrist_arti_obj_error), wrist_arti_obj_error

def _reward_arti_obj_dof(self):
    arti_obj_dof_diff = self.arti_obj_dof_state[:, :, 0] - self.arti_obj_dof_goal  # GÓC THẬT - GOAL
    arti_obj_dof_error = torch.mean(torch.abs(arti_obj_dof_diff), dim=1)
    return torch.exp(-4 * arti_obj_dof_error), arti_obj_dof_error
```

#### 🔴 HRL (h1_hrl.py) - KHÓ HỌC HƠN
```python
# === NGUỒN MỤC TIÊU - VẤN ĐỀ 1 ===
# door_angle và handle_pos được SET thủ công (không có object!)
self.door_angle[env_id] = 1.0  # GIẢ ĐỊNH bắt đầu mở
self.door_target[env_id] = 0.0  # GIẢ ĐỊNH target đóng
handle_x = 0.4 + torch.rand(1).item() * 0.3  # VỊ TRÍ RANDOM
self.goal_value[env_id, :3] = [handle_x, handle_y, handle_z]

# === REWARD - VẤN ĐỀ 2 ===
def _reward_cabinet_with_metrics(self, mask):
    wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :3]  # ELBOW!
    handle_pos = self.goal_value[mask, :3]  # VỊ TRÍ ẢO
    wrist_handle_diff = wrist_pos - handle_pos.unsqueeze(1)
    wrist_error = torch.mean(torch.abs(wrist_handle_diff.reshape(mask.sum(), 6)), dim=-1)
    rew_wrist = 5.0 * torch.exp(-4.0 * wrist_error)
    
    # Door angle - VẤN ĐỀ 3: KHÔNG CẬP NHẬT!
    angle_error = torch.abs(self.door_angle[mask] - self.door_target[mask])  # door_angle KHÔNG đổi!
    rew_door = 5.0 * torch.exp(-4.0 * angle_error)  # Luôn = exp(-4 * 1.0) = 0.018!
    
    return (rew_wrist + rew_door) * 0.728, metrics
```

#### ❌ LỖI NGHIÊM TRỌNG CỦA HRL CABINET:
| Vấn đề | Original | HRL | Tại sao HRL sai? |
|--------|----------|-----|------------------|
| **Door angle** | `arti_obj_dof_state` từ physics | `self.door_angle` = hằng số 1.0 | **BUG**: door_angle KHÔNG ĐƯỢC CẬP NHẬT → rew_door luôn = 0.018 |
| **Handle pos** | `arti_obj_root_states` thật | `goal_value[:3]` random | Không có object → không có interaction |
| **Wrist link** | Wrist thật | Elbow proxy | Sai vị trí ~0.3m |

---

### 4. **BALL TASK - SO SÁNH CODE TỪNG DÒNG**

#### 🟢 Original (h1_task_ball.py) - DỄ HỌC
```python
# === NGUỒN MỤC TIÊU ===
self.ori_ball_pos = self.ball_root_states[:, :3].clone()  # Vị trí ban đầu THẬT
self.ball_root_states = ...  # Cập nhật từ physics mỗi step

# === REWARD ===
def _reward_torso_pos(self):
    torso_pos = self.rigid_state[:, self.torso_indices, :3].squeeze(1)  # [N, 3]
    torso_ori_ball_pos_diff = self.ori_ball_pos - torso_pos  # ori_ball_pos = VỊ TRÍ BAN ĐẦU THẬT
    torso_ori_ball_pos_diff = torso_ori_ball_pos_diff[:, :2]  # Chỉ XY
    torso_ori_ball_pos_error = torch.mean(torch.abs(torso_ori_ball_pos_diff), dim=1)
    return torch.exp(-4 * torso_ori_ball_pos_error), torso_ori_ball_pos_error

def _reward_ball_pos(self):
    ball_goal_diff = self.ball_root_states[:, :3] - self.goal_pos  # ball_root_states = VỊ TRÍ HIỆN TẠI THẬT
    ball_goal_error = torch.mean(torch.abs(ball_goal_diff), dim=1)
    return torch.exp(-1 * ball_goal_error), ball_goal_error  # decay=-1
```

#### 🔴 HRL (h1_hrl.py) - KHÓ HỌC HƠN
```python
# === NGUỒN MỤC TIÊU - VẤN ĐỀ 1 ===
ball_start = torch.tensor([0.8, 0, 0.2])  # VỊ TRÍ CỐ ĐỊNH
self.ball_pos[env_id] = ball_start  # KHÔNG CẬP NHẬT SAU ĐÓ!
self.ball_target[env_id] = goal_pos  # Target random

# === REWARD - VẤN ĐỀ 2 ===
def _reward_ball_with_metrics(self, mask):
    torso_pos = self.rigid_state[mask][:, self.torso_indices[0], :2]  # OK
    ori_ball_xy = self.ball_pos[mask, :2]  # ball_pos KHÔNG ĐỔI!
    torso_error = torch.mean(torch.abs(torso_pos - ori_ball_xy), dim=-1)
    rew_torso = 1.0 * torch.exp(-4.0 * torso_error)
    
    ball_pos = self.ball_pos[mask]  # BUG: self.ball_pos = vị trí ban đầu, KHÔNG PHẢI vị trí hiện tại!
    goal_pos = self.ball_target[mask]
    ball_error = torch.mean(torch.abs(ball_pos - goal_pos), dim=-1)
    rew_ball = 5.0 * torch.exp(-1.0 * ball_error)  # ball_error KHÔNG ĐỔI → reward cố định!
    
    return (rew_torso + rew_ball) * 0.091, metrics
```

#### ❌ LỖI NGHIÊM TRỌNG CỦA HRL BALL:
| Vấn đề | Original | HRL | Tại sao HRL sai? |
|--------|----------|-----|------------------|
| **Ball position** | `ball_root_states` cập nhật từ physics | `self.ball_pos` = hằng số | **BUG**: ball_pos KHÔNG ĐƯỢC CẬP NHẬT → rew_ball không đổi theo action! |
| **Torso ref** | `ori_ball_pos` = vị trí ban đầu thật | `ball_pos` cố định | Không phản ánh ball đã di chuyển |

---

### 5. **BOX/TRANSFER/LIFT/CARRY - SO SÁNH CODE TỪNG DÒNG**

#### 🟢 Original (h1_task_box.py) - DỄ HỌC
```python
# === NGUỒN MỤC TIÊU ===
self.box_root_states = ...  # Cập nhật từ physics mỗi step
self.box_goal_pos = ...  # Target position

# === REWARD ===
def _reward_box_pos(self):
    box_pos_diff = self.box_root_states[:, :3] - self.box_goal_pos  # box_root_states = VỊ TRÍ THẬT
    box_pos_error = torch.mean(torch.abs(box_pos_diff), dim=1)
    return torch.exp(-4 * box_pos_error), box_pos_error

def _reward_wrist_box_distance(self):
    wrist_pos = self.rigid_state[:, self.wrist_indices, :7][:,:,:3]  # WRIST THẬT
    box_pos = self.box_root_states[:, :3]  # VỊ TRÍ HỘP THẬT
    wrist_box_diff = wrist_pos - box_pos.unsqueeze(1)
    wrist_pos_diff = torch.flatten(wrist_box_diff, start_dim=1)
    wrist_box_error = torch.mean(torch.abs(wrist_pos_diff), dim=1)
    return torch.exp(-4 * wrist_box_error), wrist_box_error
```

#### 🔴 HRL (h1_hrl.py) - KHÓ HỌC HƠN
```python
# === NGUỒN MỤC TIÊU - VẤN ĐỀ 1 ===
self.box_pos[env_id] = torch.tensor([0.7, 0, 0.3])  # VỊ TRÍ CỐ ĐỊNH!
self.box_target[env_id] = target  # Target random

# === REWARD - VẤN ĐỀ 2 ===
def _reward_box_task_with_metrics(self, mask, task_id):
    box_pos = self.box_pos[mask]  # BUG: self.box_pos = vị trí ban đầu, KHÔNG CẬP NHẬT!
    target = self.box_target[mask]
    
    if task_id == 6:  # Lift - chỉ z
        box_error = torch.abs(box_pos[:, 2] - target[:, 2])
    else:
        box_error = torch.mean(torch.abs(box_pos - target), dim=-1)
    rew_box = 5.0 * torch.exp(-4.0 * box_error)  # box_error KHÔNG ĐỔI!
    
    wrist_pos = self.rigid_state[mask][:, self.wrist_indices, :3]  # ELBOW!
    box_pos_expanded = box_pos.unsqueeze(1)  # box_pos cố định
    wrist_box_diff = wrist_pos - box_pos_expanded
    wrist_error = torch.mean(torch.abs(wrist_box_diff.reshape(mask.sum(), 6)), dim=-1)
    rew_grasp = wrist_scale * torch.exp(-4.0 * wrist_error)
    
    return raw_reward * balance_factors[task_id], metrics
```

#### ❌ LỖI NGHIÊM TRỌNG CỦA HRL BOX TASKS:
| Vấn đề | Original | HRL | Tại sao HRL sai? |
|--------|----------|-----|------------------|
| **Box position** | `box_root_states` cập nhật từ physics | `self.box_pos` = hằng số [0.7, 0, 0.3] | **BUG**: box_pos KHÔNG ĐƯỢC CẬP NHẬT → rew_box không thay đổi! |
| **Wrist link** | Wrist thật | Elbow proxy | Sai vị trí ~0.3m |
| **Grasp feedback** | Có contact/force từ physics | Không có | Robot không biết có đang cầm hộp hay không |

---

## 🔴 TỔNG KẾT: TẠI SAO HRL KHÓ HỌC HƠN?

### 1. **BUG NGHIÊM TRỌNG - OBJECT STATES KHÔNG CẬP NHẬT**
```python
# HRL KHÔNG CẬP NHẬT CÁC BIẾN NÀY SAU KHI RESET:
self.box_pos[env_id] = [0.7, 0, 0.3]    # CỐ ĐỊNH!
self.ball_pos[env_id] = [0.8, 0, 0.2]   # CỐ ĐỊNH!
self.door_angle[env_id] = 1.0            # CỐ ĐỊNH!

# → Reward cho box/ball/cabinet KHÔNG PHỤ THUỘC VÀO ACTION!
# → Robot không nhận được feedback từ hành động của mình
# → KHÔNG THỂ HỌC!
```

### 2. **DÙNG ELBOW THAY VÌ WRIST**
```python
# Original: wrist_indices = link "wrist" thật
# HRL: self.wrist_indices = self.elbow_indices

# Khoảng cách elbow-wrist ~0.3m
# → Error luôn có offset 0.3m
# → Reward exp(-4 * 0.3) = 0.3 (chỉ còn 30% so với optimal)
```

### 3. **GOAL TUYỆT ĐỐI VS TƯƠNG ĐỐI**
```python
# Original (Reach): goal = trajectory offset + ori_wrist_pos
#   → Goal TƯƠNG ĐỐI với pose ban đầu → luôn reachable

# HRL: goal = random trong [0.2-0.6, 0.1-0.5, 0.7-1.3]
#   → Goal TUYỆT ĐỐI trong world frame
#   → Có thể unreachable nếu robot đứng ở vị trí khác
```

### 4. **BALANCE FACTORS KHÔNG HỢP LÝ**
```python
# Reach: × 120.0 (quá lớn!)
# Button: × 0.167
# Cabinet: × 0.728
# Ball: × 0.091
# Box: × 0.061

# Variance quá lớn (120 vs 0.061 = 1967 lần!)
# → Multi-task learning khó cân bằng
```

### 5. **KHÔNG CÓ OBJECT THẬT TRONG SIMULATION**
- Original có button/box/ball/cabinet thật trong physics
- HRL chỉ có goal positions ẢO
- → Không có contact forces, friction, gravity feedback
- → Robot không học được interaction thực sự

---

## ✅ KHUYẾN NGHỊ SỬA HRL

### 1. **Cập nhật object states mỗi step**
```python
def post_physics_step(self):
    # ... existing code ...
    # CẬP NHẬT OBJECT STATES TỪ PHYSICS
    if hasattr(self, 'box_actor_indices'):
        self.box_pos = self.root_states[self.box_actor_indices, :3]
    if hasattr(self, 'ball_actor_indices'):
        self.ball_pos = self.ball_root_states[:, :3]
    if hasattr(self, 'cabinet_actor_indices'):
        self.door_angle = self.arti_obj_dof_state[:, 0, 0]
```

### 2. **Sử dụng wrist thật hoặc forward kinematics**
```python
# Nếu wrist không có collision, tính vị trí từ elbow + offset
wrist_offset = torch.tensor([0.0, 0.0, -0.3])  # Adjust based on URDF
wrist_pos = elbow_pos + wrist_offset  # Forward kinematics
```

### 3. **Normalize balance factors**
```python
# Đưa tất cả về range [0.5, 2.0]
balance_factors = {
    'reach': 1.0,   # baseline
    'button': 0.9,
    'cabinet': 1.1,
    'ball': 0.8,
    'box': 1.0,
    'transfer': 1.2,
    'lift': 0.7,
    'carry': 1.3,
}
```

### 4. **Thêm objects thật vào simulation**
```python
# Spawn actual objects với physics
box_asset = gym.load_asset(sim, asset_root, "box.urdf", ...)
gym.create_actor(env, box_asset, box_pose, "box", ...)
```

---

## �🔑 Key Differences

### 1. **Goal Source**
- **Original**: Uses `ref_wrist_pos`, `button_goal_pos`, `arti_obj_root_states`, etc. from actual simulation objects
- **HRL**: Uses simplified `goal_value` tensor (14D) with task-specific masking

### 2. **Balance Factors**
- **Original**: No balance factors, each task has its own total reward magnitude
- **HRL**: Applies balance factors to normalize rewards across tasks (for multi-task learning)

### 3. **Special Cases**
- **Ball task**: Uses `decay=-1` for ball position (slower decay = more gradual reward)
- **Lift task**: HRL uses z-only box error, original uses full XYZ
- **Transfer task**: Uses `wrist_scale=1` instead of 5

### 4. **Wrist Definition**
- **Original**: Uses `wrist_indices` from actual wrist links
- **HRL**: Uses `elbow_indices` as proxy (wrist links have no collision mesh in URDF)

---

## 📝 Notes

1. **All tasks use exponential reward**: `reward = scale × exp(decay × error)`
2. **Standard decay is -4** except Ball task ball_pos uses -1
3. **HRL balance factors** are tuned to give roughly equal reward magnitudes across tasks
4. **Metrics tracking**: HRL version tracks errors for logging (e.g., `task_reach_wrist_error`)
