# HRL (Hierarchical Reinforcement Learning) - COMPLETE TECHNICAL DOCUMENTATION

**Version:** v4.2 PPO-Correct + Command Clamping  
**Last Updated:** 2026-01-13  
**File:** `rsl_rl/rsl_rl/modules/actor_critic_hrl_simple.py`

---

## QUICK REFERENCE

| Parameter | Value |
|-----------|-------|
| **Tasks** | 8 (reach, button, cabinet, ball, box, transfer, lift, carry) |
| **Skills** | 4 (walk, reach, squat, step) |
| **Actor Input** | 105D (State 69 + Goal 14 + Mask 14 + TaskID 8) |
| **Critic Input** | 303D (3 frames × 101D) |
| **Command Output** | 14D (split into 4 skill slices) |
| **Episode Length** | 1000 steps (20s × 50Hz) |
| **Learning Rate** | 3e-4 (fixed) |

---

## 1. LOW-LEVEL SKILL COMMAND SYSTEM

### 1.1 Tổng quan

High-level policy output 14D command vector. Mỗi skill chỉ sử dụng một **slice** của vector này:

```
Command Vector (14D):
┌─────────────────────────────────────────────────────────────┐
│ [0:3]      │ [3:9]           │ [9:11]   │ [11:14]          │
│ Walk (3D)  │ Reach (6D)      │ Squat(2D)│ Step (3D)        │
│ vx,vy,ω    │ l_xyz, r_xyz    │ h, hold  │ feet_xy delta    │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Chi tiết Command của từng Skill

#### 🚶 SKILL 0: WALKING (3D) - slice [0:3]

| Index | Name | Min | Max | Unit | Mô tả |
|-------|------|-----|-----|------|-------|
| 0 | `vx` | -1.0 | 1.0 | m/s | Vận tốc tịnh tiến theo trục X (trước/sau) |
| 1 | `vy` | -1.0 | 1.0 | m/s | Vận tốc tịnh tiến theo trục Y (trái/phải) |
| 2 | `omega` | -1.0 | 1.0 | rad/s | Vận tốc góc quay (xoay quanh trục Z) |

**Nguồn gốc range:** `h1_walking_config.py` lines 167-169
```python
class ranges:
    lin_vel_x = [-1.0, 2.0]  # Config says 2.0 but only stable at [-1, 1]
    lin_vel_y = [-1.0, 1.0]
    ang_vel_yaw = [-1.0, 1.0]
```

**⚠️ LƯU Ý:** Config gốc cho phép `vx` tới 2.0 m/s nhưng thực tế robot **CHỈ ỔN ĐỊNH** trong khoảng `[-1, 1]` m/s. Vượt quá sẽ gây ngã!

---

#### 🤚 SKILL 1: REACHING (6D) - slice [3:9]

| Index | Name | Min | Max | Unit | Mô tả |
|-------|------|-----|-----|------|-------|
| 3 | `l_wrist_x` | -0.10 | 0.25 | m | Delta X tay trái (so với default pos) |
| 4 | `l_wrist_y` | -0.10 | 0.25 | m | Delta Y tay trái |
| 5 | `l_wrist_z` | -0.25 | 0.25 | m | Delta Z tay trái |
| 6 | `r_wrist_x` | -0.10 | 0.25 | m | Delta X tay phải |
| 7 | `r_wrist_y` | -0.25 | 0.10 | m | Delta Y tay phải (asymmetric!) |
| 8 | `r_wrist_z` | -0.25 | 0.25 | m | Delta Z tay phải |

**Nguồn gốc range:** `h1_reaching_config.py` lines 174-181
```python
class ranges:
    wrist_max_radius = 0.25
    l_wrist_pos_x = [-0.10, 0.25]
    l_wrist_pos_y = [-0.10, 0.25]
    l_wrist_pos_z = [-0.25, 0.25]
    r_wrist_pos_x = [-0.10, 0.25]
    r_wrist_pos_y = [-0.25, 0.10]  # NOTE: Asymmetric!
    r_wrist_pos_z = [-0.25, 0.25]
```

**⚠️ LƯU Ý:**
- Đây là **DELTA position** so với default wrist position, KHÔNG phải absolute position
- `r_wrist_y` có range **bất đối xứng** [-0.25, 0.10] do kinematics tay phải
- `sample_wp()` trong `utils/human.py` sample points trong sphere rồi filter theo ranges

---

#### 🧎 SKILL 2: SQUATTING (2D) - slice [9:11]

| Index | Name | Min | Max | Unit | Mô tả |
|-------|------|-----|-----|------|-------|
| 9 | `root_height` | 0.2 | 1.1 | m | Chiều cao target của root (pelvis) |
| 10 | `placeholder` | 0.0 | 1.0 | - | Không sử dụng (padding) |

**Nguồn gốc range:** `h1_squatting_config.py` lines 172-176
```python
class ranges:
    root_height_std = 0.2
    min_root_height = 0.2   # Squat thấp nhất
    max_root_height = 1.1   # Đứng cao nhất
```

**⚠️ LƯU Ý:**
- Default standing height là ~0.89m (`base_height_target`)
- `root_height < 0.2m` sẽ gây collapse
- `root_height > 1.1m` không khả thi về kinematics

---

#### 🦶 SKILL 3: STEPPING (3D) - slice [11:14]

| Index | Name | Min | Max | Unit | Mô tả |
|-------|------|-----|-----|------|-------|
| 11 | `l_foot_dx` | -0.25 | 0.25 | m | Delta X chân trái |
| 12 | `l_foot_dy` | -0.25 | 0.25 | m | Delta Y chân trái |
| 13 | `r_foot_dx` | -0.25 | 0.25 | m | Delta X chân phải |

**Nguồn gốc range:** `h1_stepping_config.py` line 174
```python
class ranges:
    feet_max_radius = 0.25  # Maximum step distance
```

**⚠️ LƯU Ý:**
- Đây là **DELTA position** so với current foot position
- `sample_fp()` trong `utils/human.py` sample với constraint: một chân đứng yên, chân kia di chuyển
- Radius = 0.25m là khoảng cách bước tối đa an toàn

---

### 1.3 Command Ranges trong Code

```python
# File: rsl_rl/rsl_rl/modules/actor_critic_hrl_simple.py

SKILL_CMD_SLICES = {
    0: slice(0, 3),    # Walk: [vx, vy, omega]
    1: slice(3, 9),    # Reach: [l_xyz, r_xyz]
    2: slice(9, 11),   # Squat: [height, placeholder]
    3: slice(11, 14),  # Step: [l_xy, r_x]
}

COMMAND_RANGES = {
    0: {  # Walking
        'min': torch.tensor([-1.0, -1.0, -1.0]),
        'max': torch.tensor([1.0, 1.0, 1.0]),
    },
    1: {  # Reaching
        'min': torch.tensor([-0.10, -0.10, -0.25, -0.10, -0.25, -0.25]),
        'max': torch.tensor([0.25, 0.25, 0.25, 0.25, 0.10, 0.25]),
    },
    2: {  # Squatting
        'min': torch.tensor([0.2, 0.0]),
        'max': torch.tensor([1.1, 1.0]),
    },
    3: {  # Stepping
        'min': torch.tensor([-0.25, -0.25, -0.25]),
        'max': torch.tensor([0.25, 0.25, 0.25]),
    },
}
```

---

## 2. COMMAND CLAMPING MECHANISM

### 2.1 Tại sao cần Clamp?

Low-level skills được **pre-trained** với command ranges cụ thể. Nếu high-level policy generate commands **ngoài ranges này**:
- Robot sẽ **ngã** hoặc có hành vi không xác định
- Training sẽ **diverge** do unexpected rewards

### 2.2 Implementation

```python
def _clamp_command_to_valid_ranges(self, command_full, skill_exec):
    """
    Clamp command values to valid ranges per skill.
    
    Args:
        command_full: [batch, 14] raw sampled command
        skill_exec: [batch] skill ID being executed
    
    Returns:
        command_clamped: [batch, 14] clamped command
    """
    command_clamped = command_full.clone()
    
    for skill_id in range(self.num_skills):
        mask = (skill_exec == skill_id)
        if not mask.any():
            continue
        
        cmd_slice = self.SKILL_CMD_SLICES[skill_id]
        ranges = self._cmd_ranges_device[skill_id]
        
        # Clamp only the relevant slice
        cmd_vals = command_clamped[mask, cmd_slice]
        cmd_clamped = torch.clamp(cmd_vals, ranges['min'], ranges['max'])
        command_clamped[mask, cmd_slice] = cmd_clamped
    
    return command_clamped
```

### 2.3 Flow trong Forward Pass

```
CommandHead Output (14D mean + std)
         ↓
Sample from Normal(mean, std)  →  command_full_raw (14D)
         ↓
_clamp_command_to_valid_ranges()  →  command_full (14D, clamped)
         ↓
Extract slice per skill  →  command_used (3-6D)
         ↓
Execute low-level skill with command_used
```

### 2.4 Log Prob với Clamped Command

**Quan trọng:** PPO cần tính log_prob của **what was SAMPLED**, không phải what was executed.

```python
# In forward():
info = {
    'command_full': command_full,         # Clamped - for execution
    'command_full_raw': command_full_raw, # Raw - for log_prob
    ...
}

# In get_actions_log_prob():
command_for_logprob = self.last_info['command_full_raw']  # Use RAW
log_prob_cmd = dist.log_prob(command_for_logprob[slice]).sum()
```

---

## 3. CÁCH TÙY CHỈNH COMMAND RANGES

### 3.1 Thay đổi Range cho một Skill

**Ví dụ:** Giảm walking speed range xuống [-0.5, 0.5]:

```python
# In actor_critic_hrl_simple.py

COMMAND_RANGES = {
    0: {  # Walking - MODIFIED
        'min': torch.tensor([-0.5, -0.5, -0.5]),  # Slower
        'max': torch.tensor([0.5, 0.5, 0.5]),     # Slower
    },
    # ... other skills unchanged
}
```

### 3.2 Thêm Skill mới

**Step 1:** Define slice trong `SKILL_CMD_SLICES`:
```python
SKILL_CMD_SLICES = {
    ...
    4: slice(14, 17),  # New skill: 3D command
}
```

**Step 2:** Define ranges trong `COMMAND_RANGES`:
```python
COMMAND_RANGES = {
    ...
    4: {
        'min': torch.tensor([...]),
        'max': torch.tensor([...]),
    },
}
```

**Step 3:** Tăng `command_dim` trong `__init__`:
```python
command_dim=17,  # Was 14, now 17
```

**Step 4:** Update `num_skills`:
```python
num_skills=5,  # Was 4
```

### 3.3 Asymmetric Ranges

Một số skills cần asymmetric ranges (vd: reaching). Cách handle:

```python
# Reaching: right wrist Y range is [-0.25, 0.10] NOT [-0.25, 0.25]
1: {
    'min': torch.tensor([-0.10, -0.10, -0.25, -0.10, -0.25, -0.25]),
    'max': torch.tensor([0.25, 0.25, 0.25, 0.25, 0.10, 0.25]),
    #                                          ↑ r_wrist_y max = 0.10
},
```

### 3.4 Dynamic Ranges (Curriculum)

Nếu muốn ranges thay đổi theo training progress:

```python
def set_training_step(self, step):
    # Curriculum for walking speed
    if step < 1_000_000_000:  # First 1B steps
        self.COMMAND_RANGES[0]['max'][0] = 0.5  # Max vx = 0.5
    else:
        self.COMMAND_RANGES[0]['max'][0] = 1.0  # Max vx = 1.0
    
    # Re-initialize device tensors
    self._cmd_ranges_device[0] = {
        'min': self.COMMAND_RANGES[0]['min'].to(self.device),
        'max': self.COMMAND_RANGES[0]['max'].to(self.device),
    }
```

---

## 4. POLICY ARCHITECTURE

### 4.1 Network Structure

```
Actor Observation (105D)
         ↓
┌─────────────────────────────┐
│     Actor Backbone          │
│  Linear(105, 512) + ELU     │
│  Linear(512, 256) + ELU     │
│  Linear(256, 128) + ELU     │
└─────────────────────────────┘
         ↓
      Features (128D)
         ↓
   ┌─────┴─────┬─────────────┐
   ↓           ↓             ↓
┌──────┐  ┌──────────┐  ┌──────────┐
│Gating│  │ Command  │  │ Residual │
│(4D)  │  │ (14D×2)  │  │ (19D×2)  │
└──────┘  └──────────┘  └──────────┘
   ↓           ↓             ↓
Skill ID   Mean+Std      Mean+Std
```

### 4.2 Output Heads

#### GatingHead
```python
class GatingHead(nn.Module):
    def __init__(self, feature_dim=128, num_skills=4):
        self.fc = nn.Linear(feature_dim, num_skills)
    
    def forward(self, features):
        logits = self.fc(features)
        probs = F.softmax(logits, dim=-1)
        return logits, probs  # Categorical distribution
```

#### CommandHead (với Log-Std Clamping)
```python
class CommandHead(nn.Module):
    LOG_STD_MIN = -5.0  # std >= 0.0067
    LOG_STD_MAX = 2.0   # std <= 7.39
    
    def __init__(self, feature_dim=128, command_dim=14):
        self.fc_mean = nn.Linear(feature_dim, command_dim)
        self.fc_log_std = nn.Linear(feature_dim, command_dim)
    
    def forward(self, features):
        mean = self.fc_mean(features)
        log_std = torch.clamp(self.fc_log_std(features), 
                              self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std
```

#### ResidualHead (init near zero)
```python
class ResidualHead(nn.Module):
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 0.0   # std <= 1.0 (residual should be small)
    
    def __init__(self, feature_dim=128, action_dim=19):
        self.fc_mean = nn.Linear(feature_dim, action_dim)
        self.fc_log_std = nn.Linear(feature_dim, action_dim)
        
        # Init near zero for curriculum
        nn.init.uniform_(self.fc_mean.weight, -1e-4, 1e-4)
        nn.init.zeros_(self.fc_mean.bias)
        nn.init.constant_(self.fc_log_std.bias, -2.0)  # std ~ 0.135
```

---

## 5. FORWARD PASS (CHI TIẾT)

```python
def forward(self, obs):
    batch_size = obs.shape[0]
    is_training = (batch_size == self.num_envs)
    
    # 1. Extract features
    features = self.actor_backbone(obs)
    
    # 2. Gating - sample skill
    gating_logits, gating_probs = self.gating_head(features)
    gating_dist = Categorical(probs=gating_probs)
    
    # 3. Hold time logic (keep skill for 3 steps)
    if is_training:
        sample_mask = self.hold_time.should_sample()
        skill_new = gating_dist.sample()
        skill_exec, is_held = self.hold_time.update(skill_new, sample_mask)
    else:
        skill_exec = gating_dist.sample()
        is_held = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
    
    # 4. Command - sample all 14D
    cmd_mean, cmd_std = self.command_head(features)
    cmd_dist = Normal(cmd_mean, cmd_std)
    command_full_raw = cmd_dist.sample()
    
    # 5. CLAMP command to valid ranges ← CRITICAL!
    command_full = self._clamp_command_to_valid_ranges(command_full_raw, skill_exec)
    
    # 6. Execute single skill (hard option)
    base_action = self._execute_single_skill(obs, command_full, skill_exec)
    
    # 7. Residual
    res_mean, res_std = self.residual_head(features)
    res_dist = Normal(res_mean, res_std)
    residual_raw = res_dist.sample()
    
    if self.residual_clip > 0:  # Phase 2
        residual = torch.clamp(residual_raw, -self.residual_clip, self.residual_clip)
    else:  # Phase 1 (frozen)
        residual = torch.zeros_like(residual_raw)
    
    # 8. Final action
    action_exec = torch.clamp(base_action + residual, -1.0, 1.0)
    
    # 9. Store info for log_prob
    info = {
        'skill_exec': skill_exec,
        'is_held': is_held,
        'command_full': command_full,
        'command_full_raw': command_full_raw,  # For log_prob!
        'residual_raw': residual_raw,
        'gating_probs': gating_probs,
    }
    
    return action_exec, None, info
```

---

## 6. PPO LOG PROBABILITY

### 6.1 Correct Formula

```python
def get_log_prob(self, skill_exec, command_full_raw, residual_raw, is_held):
    """
    PPO-correct log_prob computation.
    
    Key principles:
    1. Gating: 0 if held (don't penalize past decision)
    2. Command: only dims in used slice
    3. Residual: all 19D
    """
    batch_size = skill_exec.shape[0]
    
    # 1. Gating
    log_prob_gating = self.last_gating_dist.log_prob(skill_exec)
    log_prob_gating = log_prob_gating * (~is_held).float()  # Zero if held
    
    # 2. Command (ONLY used slice!)
    log_prob_cmd = torch.zeros(batch_size, device=self.device)
    for skill_id in range(self.num_skills):
        mask = (skill_exec == skill_id)
        if not mask.any():
            continue
        
        cmd_slice = self.SKILL_CMD_SLICES[skill_id]
        cmd_used = command_full_raw[mask, cmd_slice]  # Use RAW for log_prob
        mean_slice = self.last_command_dist.mean[mask, cmd_slice]
        std_slice = self.last_command_dist.stddev[mask, cmd_slice]
        
        log_prob_slice = Normal(mean_slice, std_slice).log_prob(cmd_used).sum(dim=-1)
        log_prob_cmd[mask] = log_prob_slice
    
    # 3. Residual
    log_prob_res = self.last_residual_dist.log_prob(residual_raw).sum(dim=-1)
    
    total = log_prob_gating + log_prob_cmd + log_prob_res
    return torch.clamp(total, -100, 100)  # Prevent extreme values
```

### 6.2 Tại sao dùng `command_full_raw`?

| Scenario | What to use |
|----------|-------------|
| **Execute action** | `command_full` (clamped) |
| **Compute log_prob** | `command_full_raw` (sampled) |

PPO cần biết probability của **what was sampled**, không phải what was executed.

---

## 7. TRAINING CONFIG

### 7.1 Environment
```python
class H1HRLCfg:
    class env:
        num_envs = 4096
        num_actions = 19
        num_observations = 105
        num_privileged_obs = 303
        episode_length_s = 20
    
    class control:
        decimation = 20  # 50 Hz control
        action_scale = 0.25
```

### 7.2 Algorithm
```python
class algorithm:
    learning_rate = 3e-4
    schedule = 'fixed'        # NOT adaptive!
    num_learning_epochs = 5
    num_mini_batches = 4
    clip_param = 0.2
    entropy_coef = 0.01
    value_loss_coef = 1.0
    max_grad_norm = 1.0
    gamma = 0.99
    lam = 0.95
```

### 7.3 Training Phases
| Phase | Iterations | Residual Clip |
|-------|-----------|---------------|
| 1 | 0 - 49,999 | 0 (frozen) |
| 2 | 50,000+ | ±0.05 |

---

## 8. FILES

```
SkillBlender/
├── rsl_rl/rsl_rl/modules/
│   └── actor_critic_hrl_simple.py   ← Main policy file
│
├── legged_gym/legged_gym/
│   ├── envs/h1/h1_hrl/
│   │   └── h1_hrl.py                ← Environment
│   ├── envs/h1/h1_walking/
│   │   └── h1_walking_config.py     ← Walking ranges
│   ├── envs/h1/h1_reaching/
│   │   └── h1_reaching_config.py    ← Reaching ranges
│   ├── envs/h1/h1_squatting/
│   │   └── h1_squatting_config.py   ← Squatting ranges
│   └── envs/h1/h1_stepping/
│       └── h1_stepping_config.py    ← Stepping ranges
│
└── HRL_COMPLETE_DOCUMENTATION.md    ← This file
```

---

## 9. TROUBLESHOOTING

### Robot ngã ngay khi training?
→ Command vượt quá valid range. Check clamp logic đã được apply chưa.

### Loss spike to 1e10?
→ Check W&B `Debug/ratio_max`. Nếu > 10, log_prob computation có vấn đề.

### Skill không được chọn đều?
→ Check `Debug/gating_probs` hoặc tăng `entropy_coef`.

### Wrist position không đúng?
→ Reaching command là DELTA position, không phải absolute. Check offset trong env.

---

## 10. TRAINING COMMAND

```bash
cd /home/crl/hienhq/SkillBlender/legged_gym

python legged_gym/scripts/train_hrl.py \
    --task h1_hrl \
    --run_name hrl_v4_clamped \
    --num_envs 8192 \
    --max_iterations 100000 \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --headless \
    --wandb hrl_v4
```

---

**Document Status:** Complete  
**Code Version:** v4.2 (PPO-Correct + Command Clamping)
