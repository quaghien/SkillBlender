# 📊 Metrics & Logging - So sánh Original vs HRL

## 🔴 Tổng quan: HRL đang log DƯ và THỪA rất nhiều!

| Aspect | Original (1 task) | HRL (8 tasks) | Vấn đề |
|--------|-------------------|---------------|--------|
| **Số metrics** | ~15-20 | ~80-100+ | **5x nhiều hơn** |
| **Cấu trúc** | Đơn giản, rõ ràng | Lặp lại, chồng chéo | Khó đọc |
| **Hữu ích** | Cao | Thấp (nhiều noise) | Tốn bandwidth |

---

## 📋 ORIGINAL METRICS (on_policy_runner.py)

### 1. Episode Rewards (từ `extras["episode"]`)
```python
# Được tính trong legged_robot.py reset_idx()
self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key]) / max_episode_length_s

# WandB key: Episode/rew_<reward_name>
# Ví dụ cho h1_task_reach:
Episode/rew_wrist_pos           # Scale=5, reward cho vị trí wrist
```

### 2. Episode Metrics (từ `extras["episode_metrics"]`)
```python
# Được tính trong legged_robot.py compute_reward()
if isinstance(rew_func_return, tuple):
    unscaled_rew, metric = rew_func_return
    self.episode_metrics[name] = metric.mean().item()

# WandB key: Metric/<reward_name>
# Ví dụ:
Metric/wrist_pos               # Error trung bình (meters)
```

### 3. Loss Metrics
```python
Loss/value_function            # Value loss
Loss/surrogate                 # Policy loss
Loss/entropy                   # Total entropy
Loss/learning_rate             # Current LR
```

### 4. Debug Metrics (stability monitoring)
```python
Debug/ratio_mean               # Importance sampling ratio mean
Debug/ratio_max                # Max ratio (clip indicator)
Debug/kl_approx                # Approximate KL divergence
Debug/clipfrac                 # Fraction of clipped gradients
Debug/adv_mean, adv_std, adv_max  # Advantage statistics
```

### 5. Training Metrics
```python
Train/mean_reward              # Mean episode reward
Train/mean_episode_length      # Mean episode length
Train/dones                    # Done rate
```

### 6. Performance Metrics
```python
Perf/total_fps                 # Steps per second
Perf/collection_time           # Data collection time
Perf/learning_time             # PPO update time
```

### 7. Std Metrics
```python
Std/mean_std                   # Mean action noise std
Std/std_dim_0 ~ Std/std_dim_18 # Per-dimension std (19 joints)
```

### 📝 Tổng Original: ~15 + 19 (std) = ~34 metrics

---

## 🔴 HRL METRICS (on_policy_runner_hrl.py)

### 1. Tất cả Original metrics ở trên +

### 2. Episode Rewards per Task (THỪA!)
```python
# Từ get_task_stats() - Log 2 lần mỗi task!
episode_reward/reach           # Cumulative episode reward
episode_reward/button
episode_reward/cabinet
episode_reward/ball
episode_reward/box
episode_reward/transfer
episode_reward/lift
episode_reward/carry           # = 8 metrics

step_reward/reach              # Instantaneous step reward
step_reward/button
...                            # = 8 metrics nữa

# VẤN ĐỀ: episode_reward vs step_reward chồng chéo với Episode/rew_*
```

### 3. TaskMetric per Task (THỪA!)
```python
# Từ get_task_stats() - mỗi task có 1-2 error metrics
TaskMetric/task_reach_wrist_error
TaskMetric/task_button_wrist_error
TaskMetric/task_button_arm_error
TaskMetric/task_cabinet_wrist_error
TaskMetric/task_cabinet_door_error
TaskMetric/task_ball_torso_error
TaskMetric/task_ball_goal_error
TaskMetric/task_box_box_error
TaskMetric/task_box_wrist_error
TaskMetric/task_transfer_box_error
TaskMetric/task_transfer_wrist_error
TaskMetric/task_lift_box_error
TaskMetric/task_lift_wrist_error
TaskMetric/task_carry_box_error
TaskMetric/task_carry_wrist_error  # = 15 metrics

# VẤN ĐỀ: Chồng chéo với Metric/<name>
```

### 4. Entropy Breakdown (HRL-specific, HỮU ÍCH)
```python
Entropy/total                  # Total entropy
Entropy/skill                  # Skill selection entropy
Entropy/command                # Command entropy
Entropy/action                 # Low-level action entropy
# = 4 metrics (OK)
```

### 5. Curriculum Parameters (HRL-specific, HỮU ÍCH)
```python
Curriculum/stage               # 1 hoặc 2
Curriculum/K                   # Option duration
Curriculum/epsilon             # Exploration rate
Curriculum/tau                 # Temperature
Curriculum/c_ent_skill         # Skill entropy coefficient
Curriculum/lr_cmd_ratio        # Command LR ratio
# = 6 metrics (OK)
```

### 6. Skill Histogram (HRL-specific, THỪA format!)
```python
Skill/histogram_reach          # Usage count
Skill/histogram_button
Skill/histogram_cabinet
Skill/histogram_ball
Skill/histogram_box
Skill/histogram_transfer
Skill/histogram_lift
Skill/histogram_carry          # = 8 metrics
Skill/switch_rate              # Actual switch rate
Skill/expected_switch_rate     # Target switch rate
# = 10 metrics (có thể gộp thành 1 histogram)
```

### 7. SkillReward per Skill (THỪA!)
```python
SkillReward/reach
SkillReward/button
SkillReward/cabinet
SkillReward/ball
SkillReward/box
SkillReward/transfer
SkillReward/lift
SkillReward/carry              # = 8 metrics

# VẤN ĐỀ: Giống hệt step_reward/* và Episode/rew_*
```

### 8. Task Count (từ train_hrl.py)
```python
task_count/reach
task_count/button
task_count/cabinet
task_count/ball
task_count/box
task_count/transfer
task_count/lift
task_count/carry               # = 8 metrics
```

### 📝 Tổng HRL: ~34 + 8 + 8 + 15 + 4 + 6 + 10 + 8 + 8 = **~101 metrics!**

---

## 🔴 PHÂN TÍCH VẤN ĐỀ

### 1. **CHỒNG CHÉO NGHIÊM TRỌNG**

| Thông tin | Original key | HRL key 1 | HRL key 2 | HRL key 3 |
|-----------|--------------|-----------|-----------|-----------|
| Reach reward | `Episode/rew_wrist_pos` | `episode_reward/reach` | `step_reward/reach` | `SkillReward/reach` |
| Reach error | `Metric/wrist_pos` | `TaskMetric/task_reach_wrist_error` | - | - |

**→ Cùng 1 thông tin được log 3-4 lần!**

### 2. **NAMING KHÔNG NHẤT QUÁN**

```python
# Original style:
Episode/rew_<name>
Metric/<name>

# HRL thêm nhiều style:
episode_reward/<task>          # Không có prefix!
step_reward/<task>
TaskMetric/task_<task>_<metric>  # Prefix khác!
SkillReward/<skill>
Skill/histogram_<skill>
```

### 3. **THÔNG TIN KHÔNG CẦN THIẾT**

```python
# Per-dimension std (19 values mỗi iteration!)
Std/std_dim_0 ~ Std/std_dim_18

# → Chỉ cần Std/mean_std là đủ
```

---

## ✅ KHUYẾN NGHỊ: METRICS CẦN GIỮ

### 🟢 Giữ lại (Essential):
```python
# === LOSS ===
Loss/value_function
Loss/surrogate
Loss/learning_rate

# === TRAINING ===
Train/mean_reward
Train/mean_episode_length

# === HRL-SPECIFIC ===
Entropy/skill                  # Quan trọng cho HRL
Curriculum/stage
Curriculum/K
Curriculum/epsilon
Skill/switch_rate

# === PER-TASK (chọn 1 trong 3 style) ===
# Recommend: dùng Episode/rew_<task> style
Episode/rew_reach
Episode/rew_button
Episode/rew_cabinet
Episode/rew_ball
Episode/rew_box
Episode/rew_transfer
Episode/rew_lift
Episode/rew_carry

# === ERROR METRICS (cho debug) ===
Metric/reach_wrist_error       # Chỉ khi cần debug
Metric/button_wrist_error
...
```

### 🔴 Xóa bỏ (Redundant):
```python
# CHỒNG CHÉO VỚI Episode/rew_*
episode_reward/<task>          # XÓA
step_reward/<task>             # XÓA
SkillReward/<skill>            # XÓA

# CHỒNG CHÉO VỚI Metric/*
TaskMetric/task_*_*_error      # XÓA hoặc rename

# KHÔNG CẦN THIẾT
Std/std_dim_*                  # XÓA (giữ mean_std)
Skill/histogram_*              # Gộp thành 1 metric
task_count/*                   # Có thể tính từ histogram
Skill/expected_switch_rate     # Tính được từ K
```

---

## 📝 CODE CHANGES ĐỀ XUẤT

### 1. Sửa `on_policy_runner_hrl.py`:

```python
def log(self, locs, width=80, pad=35):
    wandb_dict = {}
    
    # === ESSENTIAL METRICS ONLY ===
    wandb_dict['Loss/value_function'] = locs['mean_value_loss']
    wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
    wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
    
    # HRL entropy (only skill entropy matters)
    wandb_dict['Entropy/skill'] = self.alg.mean_entropy_skill
    
    # Curriculum (only key params)
    wandb_dict['Curriculum/stage'] = curriculum_params['stage']
    wandb_dict['Curriculum/K'] = curriculum_params['K']
    wandb_dict['Curriculum/epsilon'] = curriculum_params['epsilon']
    
    # Skill switch rate
    wandb_dict['Skill/switch_rate'] = self.alg.actor_critic.get_skill_switch_rate()
    
    # Per-task rewards (single style!)
    if hasattr(self.env, 'get_task_stats'):
        task_stats = self.env.get_task_stats()
        for task in ['reach', 'button', 'cabinet', 'ball', 'box', 'transfer', 'lift', 'carry']:
            wandb_dict[f'Episode/rew_{task}'] = task_stats.get(f'episode_reward/{task}', 0)
    
    # Training
    if len(locs['rewbuffer']) > 0:
        wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
        wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
    
    # Performance
    wandb_dict['Perf/total_fps'] = fps
    
    # Std (only mean)
    wandb_dict['Std/mean_std'] = std.mean()
    
    wandb.log(wandb_dict, step=locs['it'])
```

### 2. Sửa `h1_hrl.py get_task_stats()`:

```python
def get_task_stats(self):
    """Get task statistics - SIMPLIFIED version"""
    stats = {}
    
    # Only episode rewards (matches Original format)
    for name in self.task_names:
        stats[f'episode_reward/{name}'] = self.task_avg_rewards[name]
    
    # Raw data (không log lên wandb)
    stats['_raw_rewards'] = self.task_avg_rewards.copy()
    stats['_raw_counts'] = self.task_episode_counts.copy()
    
    return stats
```

---

## 📊 METRICS COUNT COMPARISON

| Category | Original | HRL Current | HRL Optimized |
|----------|----------|-------------|---------------|
| Loss | 4 | 4 | 4 |
| Training | 3 | 3 | 3 |
| Performance | 3 | 3 | 3 |
| Debug | 6 | 6 | 0 (optional) |
| Std | 20 | 20 | 1 |
| Episode rewards | 1-2 | 8×3=24 | 8 |
| Metrics | 1-2 | 15 | 0 (optional) |
| HRL Curriculum | 0 | 6 | 3 |
| HRL Skill | 0 | 10 | 1 |
| Task count | 0 | 8 | 0 |
| **TOTAL** | **~35** | **~101** | **~23** |

**→ Giảm từ 101 xuống 23 metrics (77% reduction)!**

---

## 🎯 WANDB DASHBOARD ĐỀ XUẤT

### Panel 1: Training Overview
- Train/mean_reward (line)
- Loss/value_function + Loss/surrogate (dual axis)

### Panel 2: Per-Task Performance
- Episode/rew_reach, button, cabinet, ball, box, transfer, lift, carry (8 lines)

### Panel 3: HRL Dynamics
- Curriculum/stage (step)
- Curriculum/K, epsilon (dual axis)
- Skill/switch_rate (line)
- Entropy/skill (line)

### Panel 4: Performance
- Perf/total_fps (line)
- Train/mean_episode_length (line)

---

## 📝 TÓM TẮT

| Vấn đề | Mô tả | Giải pháp |
|--------|-------|-----------|
| **Chồng chéo** | Cùng thông tin log 3-4 lần | Chọn 1 style duy nhất |
| **Quá nhiều metrics** | 101 metrics mỗi iteration | Giảm xuống ~23 |
| **Naming không nhất quán** | 4-5 kiểu prefix khác nhau | Dùng Original style |
| **Std per-dim** | 19 metrics không cần thiết | Chỉ log mean_std |
| **Task count** | Có thể tính từ histogram | Xóa bỏ |
