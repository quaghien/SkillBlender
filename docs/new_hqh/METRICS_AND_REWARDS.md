# METRICS VÀ REWARDS - HRL TRAINING V7.1

**Tài liệu:** Giải thích chi tiết các metric và reward của từng task  
**Phiên bản:** v7.1  
**Ngày cập nhật:** 2026-01-14

---

## 📊 PHẦN 1: GIẢI THÍCH CHUNG

### Metrics là gì?
- **Metrics** = Các chỉ số đo lường hiệu suất của robot
- Được log vào **W&B** (Weights & Biases) để theo dõi training
- Mỗi task có riêng metrics để debug và evaluate

### Rewards là gì?
- **Rewards** = Điểm thưởng mà agent nhận khi hoàn thành task
- Dùng để guide training → agent học tối ưu action
- Reward scale khác nhau per task → cân bằng learning

### Tại sao cần Curriculum Scaling?
- **Easy tasks** (×0.8): Reward thấp hơn → agent không "lười"
- **Hard tasks** (×1.3): Reward cao hơn → agent focus học
- **Kết quả:** Learning balance, không có task domination

---

## 📈 PHẦN 2: METRICS CHI TIẾT

### Các Loại Metrics

| Category | Metric | Ý Nghĩa | Range | Tốt là |
|----------|--------|--------|-------|--------|
| **Loss** | `Loss/entropy` | Độ explore của policy | 20~50 | ~27 |
| | `Loss/surrogate` | PPO surrogate loss | -0.1~0.1 | ~0 |
| | `Loss/value` | Critic value loss | 0~1000 | <100 |
| | `Loss/total` | Total loss = surrogate + value + entropy | - | ~0 |
| **Performance** | `Perf/success_rate` | % task thành công | 0~1 | >0.5 |
| | `Perf/avg_reward_per_step` | Trung bình reward/step | - | >1 |
| | `Perf/episode_length` | Số step trong episode | 500~1000 | ~1000 |
| **Debug** | `Debug/gating_probs_max` | Max probability skill được chọn | 0~1 | <0.7 |
| | `Debug/ratio_max` | Max ratio (new_policy / old_policy) | 0~100 | 1~2 |
| | `Debug/ratio_mean` | Mean ratio | - | ~1.0 |

### Task-Specific Metrics (TaskMetric/)

Mỗi task có format: `TaskMetric/task_{taskname}_{metric}`

#### 🎯 REACH Task Metrics

| Metric | Ý Nghĩa | Unit | Tốt là |
|--------|---------|------|--------|
| `task_reach_wrist_error` | Khoảng cách wrist đến target | m | <0.1 |
| `task_reach_wrist_error_l` | Error tay trái | m | <0.15 |
| `task_reach_wrist_error_r` | Error tay phải | m | <0.15 |
| `task_reach_reward_raw` | Raw reward trước scaling | - | >0 |

#### 🔘 BUTTON Task Metrics

| Metric | Ý Nghĩa | Unit | Tốt là |
|--------|---------|------|--------|
| `task_button_wrist_error` | Khoảng cách wrist đến button | m | <0.05 |
| `task_button_arm_error` | Sai lệch arm pose so với target | - | <1.0 |
| `task_button_contact_force` | Lực tác động lên button | N | >10 |
| `task_button_pressed` | % button được ấn | % | >0.5 |

#### 🚪 CABINET Task Metrics

| Metric | Ý Nghĩa | Unit | Tốt là |
|--------|---------|------|--------|
| `task_cabinet_wrist_error` | Khoảng cách đến handle | m | <0.1 |
| `task_cabinet_door_angle` | Góc mở cửa cabinet | deg | >30 |
| `task_cabinet_force_applied` | Lực kéo | N | >10 |
| `task_cabinet_pull_success` | % lần kéo thành công | % | >0.5 |

#### ⚽ BALL Task Metrics

| Metric | Ý Nghĩa | Unit | Tốt là |
|--------|---------|------|--------|
| `task_ball_goal_error` | Khoảng cách ball đến goal | m | <0.2 |
| `task_ball_kick_success` | % lần đá thành công | % | >0.3 |
| `task_ball_avg_speed` | Tốc độ trung bình ball | m/s | >0.5 |
| `task_ball_travel_distance` | Quãng đường ball đi được | m | >1.0 |

#### 📦 BOX Task Metrics

| Metric | Ý Nghĩa | Unit | Tốt là |
|--------|---------|------|--------|
| `task_box_goal_error` | Khoảng cách box đến goal | m | <0.2 |
| `task_box_grasp_success` | % lần nắm thành công | % | >0.3 |
| `task_box_distance_to_pick` | Khoảng cách wrist đến box | m | <0.2 |
| `task_box_placement_error` | Sai lệch khi đặt box | m | <0.1 |

#### 🔄 TRANSFER Task Metrics

| Metric | Ý Nghĩa | Unit | Tốt là |
|--------|---------|------|--------|
| `task_transfer_source_error` | Khoảng cách đến source object | m | <0.15 |
| `task_transfer_dest_error` | Khoảng cách đến destination | m | <0.2 |
| `task_transfer_pick_success` | % lần pick thành công | % | >0.2 |
| `task_transfer_place_success` | % lần place thành công | % | >0.2 |

#### 🫀 LIFT Task Metrics

| Metric | Ý Nghĩa | Unit | Tốt là |
|--------|---------|------|--------|
| `task_lift_object_error` | Khoảng cách đến object | m | <0.1 |
| `task_lift_height_error` | Sai lệch chiều cao nâng | m | <0.1 |
| `task_lift_success` | % lần nâng thành công | % | >0.3 |
| `task_lift_hold_time` | Thời gian giữ object | s | >2.0 |

#### 👜 CARRY Task Metrics

| Metric | Ý Nghĩa | Unit | Tốt là |
|--------|---------|------|--------|
| `task_carry_goal_error` | Khoảng cách đến goal | m | <0.3 |
| `task_carry_object_height` | Chiều cao object được nâng | m | >0.3 |
| `task_carry_stability` | Độ ổn định nâng object | - | >0.8 |
| `task_carry_distance_traveled` | Quãng đường đi được | m | >1.0 |

---

## 💰 PHẦN 3: REWARD CHI TIẾT

### Reward Formula Chung

```
total_reward = raw_reward × scale × curriculum_factor

Ví dụ:
Reach task:
  raw_reward = 1 - distance_to_target
  scale = 120.0
  curriculum_factor = 0.8 (easy task)
  total = (1 - 0.1) × 120.0 × 0.8 = 86.4
```

### Reward Scale của Từng Task (v7.1)

#### 🟦 EASY TASKS (Curriculum ×0.8)

| Task | Raw Reward | Scale | Curriculum | Final Scale | Target/Step | Giải thích |
|------|-----------|-------|-----------|-------------|------------|-----------|
| **Reach** | 1 - dist | 150.0 | ×0.8 | **120.0** | ~4 | Tay robot tới vị trí target. Easy vì có 2 tay, feedback rõ ràng |
| **Button** | 1 - dist | 0.178 | ×0.8 | **0.1428** | ~4 | Ấn nút cabinet. Easy vì target nhỏ, reward dense |
| **Cabinet** | 1 - dist | 0.91 | ×0.8 | **0.728** | ~4 | Kéo mở cửa cabinet. Khó hơn reach nhưng vẫn easy tier |

#### 🟨 MEDIUM TASKS (Curriculum ×1.0)

| Task | Raw Reward | Scale | Curriculum | Final Scale | Target/Step | Giải thích |
|------|-----------|-------|-----------|-------------|------------|-----------|
| **Ball** | 1 - dist | 0.091 | ×1.0 | **0.091** | ~5 | Đá banh tới goal. Trung bình vì cần coordination + movement |
| **Box** | 1 - dist | 0.061 | ×0.8 | **0.0488** | ~4 | Nắm + di chuyển hộp. Phức tạp hơn reach |
| **Lift** | 1 - height | 0.0475 | ×1.0 | **0.0475** | ~5 | Nâng object. Trung bình vì cần grasp + balance |

#### 🟥 HARD TASKS (Curriculum ×1.3)

| Task | Raw Reward | Scale | Curriculum | Final Scale | Target/Step | Giải thích |
|------|-----------|-------|-----------|-------------|------------|-----------|
| **Transfer** | 1 - dist | 0.0625 | ×1.3 | **0.08125** | ~6.5 | Pick + move + place. Hard vì 3 sub-tasks sequentially |
| **Carry** | 1 - dist | 0.059 | ×1.3 | **0.0767** | ~6.5 | Nâng + di chuyển xa. Hard vì maintain balance + distance |

### Tại Sao Curriculum Scaling?

```
Scenario 1: KHÔNG Curriculum (uniform scale = 1.0)
┌──────────────────────────────────────────────────┐
│ Easy tasks:    reward/step = 5 ← Quá cao!       │
│ Medium tasks:  reward/step = 5                   │
│ Hard tasks:    reward/step = 5 ← Quá thấp!      │
└──────────────────────────────────────────────────┘
❌ Result: Agent học easy tasks quá tốt → ignore hard tasks

Scenario 2: VỚI Curriculum (Easy ×0.8, Hard ×1.3)
┌──────────────────────────────────────────────────┐
│ Easy tasks:    reward/step = 4   ← Giảm, agent không lười
│ Medium tasks:  reward/step = 5   ← Cân bằng
│ Hard tasks:    reward/step = 6.5 ← Tăng, agent focus
└──────────────────────────────────────────────────┘
✅ Result: Learning balance, mỗi task được chú ý
```

### Reward Range Analysis

| Category | Min | Max | Avg per Episode | Giải thích |
|----------|-----|-----|-----------------|-----------|
| **Easy Tasks** | 0 | 600 | ~200 | Scale nhỏ nhưng easy → agent thường succeed |
| **Medium Tasks** | 0 | 50 | ~20 | Khó hơn → reward thấp hơn |
| **Hard Tasks** | 0 | 65 | ~25 | Khó nhất nhưng scale cao nhất |

---

## 🎓 PHẦN 4: CÁCH ĐỌC W&B LOGS

### Các Metric Group trong W&B

```
📊 W&B Dashboard Structure:
├── 📈 Loss/
│   ├── entropy          → Nên ổn định ~27
│   ├── surrogate        → Nên quanh 0
│   ├── value            → Nên <100
│   └── total
│
├── 📊 Perf/
│   ├── success_rate     → % task thành công
│   ├── avg_reward_per_step
│   └── episode_length
│
├── 🐛 Debug/
│   ├── gating_probs_max → Skill diversity
│   ├── ratio_max        → PPO ratio health
│   └── ratio_mean
│
├── 🎯 TaskMetric/
│   ├── task_reach_*     → Reach metrics
│   ├── task_button_*    → Button metrics
│   ├── task_cabinet_*   → Cabinet metrics
│   ├── task_ball_*      → Ball metrics
│   ├── task_box_*       → Box metrics
│   ├── task_transfer_*  → Transfer metrics
│   ├── task_lift_*      → Lift metrics
│   └── task_carry_*     → Carry metrics
```

### Cách Đọc Từng Metric

#### 1️⃣ Loss/entropy
```
❌ BAD:  entropy > 60 → Policy quá random, không converge
⚠️ OK:   entropy = 40~50
✅ GOOD: entropy = 20~30 ← Target!
```

#### 2️⃣ Loss/surrogate
```
❌ BAD:  surrogate > 1 hoặc < -1 → PPO blows up
⚠️ OK:   surrogate = -0.1 ~ 0.1
✅ GOOD: surrogate ≈ 0 ← Bình thường
```

#### 3️⃣ Debug/ratio_max
```
❌ BAD:  ratio_max > 10 → Log prob computation sai
⚠️ OK:   ratio_max = 2~5
✅ GOOD: ratio_max = 1.2~1.5 ← Target!
```

#### 4️⃣ TaskMetric/task_{name}_*
```
Reach task ví dụ:
┌─────────────────────────────────────┐
│ task_reach_wrist_error = 0.05 m     │ ← Error giảm = learning tốt
│ task_reach_reward_raw = 0.95        │ ← Raw reward cao = task dễ
└─────────────────────────────────────┘

Trajectory Tốt:
Step 0k:    wrist_error = 0.50 m
Step 10k:   wrist_error = 0.30 m ← Improving ✅
Step 20k:   wrist_error = 0.10 m ← Getting better ✅
Step 50k:   wrist_error = 0.05 m ← Converged ✅
```

---

## 📝 PHẦN 5: BẢNG TỔNG HỢP TẤT CẢ TASKS

### Tổng Hợp 8 Tasks

| # | Task | Difficulty | Scale | Curriculum | Target Reward/Step | Loại | Giải thích |
|---|------|-----------|-------|-----------|------------------|------|-----------|
| 1 | Reach | Easy | 150.0 | ×0.8 | **120.0** (~4) | Position | Tay tới vị trí |
| 2 | Button | Easy | 0.178 | ×0.8 | **0.1428** (~4) | Action | Ấn nút |
| 3 | Cabinet | Easy | 0.91 | ×0.8 | **0.728** (~4) | Action | Kéo cửa |
| 4 | Ball | Medium | 0.091 | ×1.0 | **0.091** (~5) | Manipulation | Đá banh |
| 5 | Box | Medium | 0.061 | ×0.8 | **0.0488** (~4) | Manipulation | Di chuyển hộp |
| 6 | Lift | Medium | 0.0475 | ×1.0 | **0.0475** (~5) | Manipulation | Nâng object |
| 7 | Transfer | Hard | 0.0625 | ×1.3 | **0.08125** (~6.5) | Complex | Pick → move → place |
| 8 | Carry | Hard | 0.059 | ×1.3 | **0.0767** (~6.5) | Complex | Nâng + di chuyển |

### Phân Tích Khó - Dễ

```
EASY (×0.8):
├── Reach       ← Feedback rõ, 2 tay, target fixed
├── Button      ← Target nhỏ, reward dense, feedback rõ
└── Cabinet     ← Có reference frame, hành động đơn giản

MEDIUM (×1.0):
├── Ball        ← Cần coordination, movement, timing
├── Box         ← Grasp phức tạp, path planning
└── Lift        ← Cân bằng, gripper control

HARD (×1.3):
├── Transfer    ← Sequential: pick→move→place (3 phases)
└── Carry       ← Long distance, balance, stability
```

---

## 🚀 PHẦN 6: HƯỚNG DẪN MONITORING

### Khi Training, Cần Theo Dõi

```
Phase 1 (Iterations 0-20k): Residual Frozen (clip=0)
├── ✅ entropy: 20~30
├── ✅ surrogate: ~0
├── ✅ value_loss: <500
└── ✅ task reward: tăng từ từ

Phase 2 (Iterations 20k-100k): Residual Unfrozen (clip=±0.05)
├── ✅ entropy: vẫn ~27 (không spike!)
├── ✅ ratio_max: 1.2~1.5
├── ✅ task reward: tiếp tục tăng
└── ✅ success_rate: >0.5 cho mỗi task
```

### Red Flags ⚠️

| Symptom | Nguyên nhân | Fix |
|---------|-----------|-----|
| Entropy tăng > 60 | LOG_STD quá cao | Check CommandHead/ResidualHead |
| Surrogate loss > 1 | Wrong log_prob | Use `get_actions_log_prob_hrl()` |
| Ratio_max > 10 | HRL info mismatch | Store skill/command/residual |
| All tasks fail | Reward scale quá nhỏ | Tăng scale ×2 |
| Some tasks 0% success | Reward quá skew | Apply curriculum scaling |
| Episode length drop | Robot ngã liên tục | Check command clamping |

---

## 📞 PHẦN 7: QUICK REFERENCE

### Metric Targets v7.1

```python
# Training Health
Loss/entropy = ~27 ← Target
Loss/surrogate = ~0
Loss/value < 100
Loss/total < 100

# Task Success
Reach: wrist_error < 0.1m
Button: button_pressed > 50%
Cabinet: door_angle > 30°
Ball: goal_error < 0.2m
Box: goal_error < 0.2m
Transfer: pick+place success > 20%
Lift: height_error < 0.1m
Carry: goal_error < 0.3m

# PPO Health
Debug/ratio_max = 1.2~1.5
Debug/gating_probs_max < 0.7
Perf/episode_length ~1000
```

### Command Summary

```bash
# Run training v7.1
cd /home/crl/hienhq/SkillBlender/legged_gym

python legged_gym/scripts/train_hrl.py \
    --task h1_hrl \
    --run_name hrl_v7.1 \
    --num_envs 4096 \
    --max_iterations 100000 \
    --sim_device cuda:0 \
    --rl_device cuda:0 \
    --headless \
    --wandb hrl_v7
```

---

**Document Status:** Complete v7.1  
**Bản dịch:** Tiếng Việt (Vietnamese)  
**Số trang:** 8  
**Last Updated:** 2026-01-14
