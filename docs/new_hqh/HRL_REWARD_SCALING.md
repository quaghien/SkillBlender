# HRL Reward Structure - Phân tích & Scale

## 🎯 Cấu trúc Reward Mới

Mỗi task có **3 thành phần** reward:

```
Total = Base + Progress + Success
```

| Thành phần | Công thức | Ý nghĩa |
|------------|-----------|---------|
| **Base** | `scale * exp(-4 * error)` | Reward gốc - thưởng khi gần target |
| **Progress** | `1.0 * clamp(prev - curr, -0.5, 0.5)` | Thưởng khi error giảm |
| **Success** | `2.0 * (error < 0.1)` | Bonus khi đạt threshold |

---

## 📊 Phân tích Reward Gốc (từ single-task configs)

### Độ khó theo Paper:

| Mức độ | Tasks |
|--------|-------|
| **Easy** | FarReach, ButtonPress, CabinetClose |
| **Medium** | FootballShoot, BoxPush, PackageLift |
| **Hard** | BoxTransfer, PackageCarry |

### Reward Scales Gốc:

| Task | Reward Components | Scales | Max Base | Max Total* |
|------|-------------------|--------|----------|------------|
| **reach** | wrist_pos | 5 | **5.0** | ~7.5 |
| **button** | wrist_button + right_arm | 5 + 0.5 | **5.5** | ~8.0 |
| **cabinet** | wrist_arti + arti_dof | 5 + 5 | **10.0** | ~12.5 |
| **ball** | torso_pos + ball_pos | 1 + 5 | **6.0** | ~8.5 |
| **box** | box_pos + wrist_box | 5 + 5 | **10.0** | ~12.5 |
| **transfer** | box_pos + wrist_box | 5 + **1** | **6.0** | ~8.5 |
| **lift** | box_pos + wrist_box | 5 + 5 | **10.0** | ~12.5 |
| **carry** | box_pos + wrist_box | 5 + 5 | **10.0** | ~12.5 |

*Max Total = Base + Progress(0.5) + Success(2.0)

---

## ⚠️ Vấn đề hiện tại

**Reward KHÔNG phù hợp với độ khó!**

| Task | Độ khó | Max Base | Vấn đề |
|------|--------|----------|--------|
| reach | Easy | 5.0 | ✅ OK |
| button | Easy | 5.5 | ✅ OK |
| cabinet | Easy | **10.0** | ⚠️ Quá cao cho Easy! |
| ball | Medium | 6.0 | ✅ OK |
| box | Medium | 10.0 | ✅ OK |
| lift | Medium | 10.0 | ✅ OK |
| transfer | Hard | **6.0** | ⚠️ Quá thấp cho Hard! |
| carry | Hard | 10.0 | ✅ OK |

---

## 🔧 Đề xuất Scale theo 3 Mức Độ

### Mục tiêu:
- **Easy**: Max ~6-8 (baseline)
- **Medium**: Max ~10-12 (+50%)
- **Hard**: Max ~14-16 (+100%)

### Task Scale Multipliers:

```python
TASK_DIFFICULTY_SCALE = {
    # EASY (baseline x1.0)
    'reach': 1.0,      # 5.0 → 5.0
    'button': 1.0,     # 5.5 → 5.5
    'cabinet': 0.6,    # 10.0 → 6.0 (giảm!)
    
    # MEDIUM (x1.5)
    'ball': 1.5,       # 6.0 → 9.0
    'box': 1.0,        # 10.0 → 10.0 (đã cao)
    'lift': 1.0,       # 10.0 → 10.0 (đã cao)
    
    # HARD (x2.0)
    'transfer': 2.0,   # 6.0 → 12.0 (tăng!)
    'carry': 1.3,      # 10.0 → 13.0
}
```

### Kết quả sau Scale:

| Task | Độ khó | Gốc | Scale | Mới | Target Range |
|------|--------|-----|-------|-----|--------------|
| reach | Easy | 5.0 | 1.0 | **5.0** | 5-8 ✅ |
| button | Easy | 5.5 | 1.0 | **5.5** | 5-8 ✅ |
| cabinet | Easy | 10.0 | 0.6 | **6.0** | 5-8 ✅ |
| ball | Medium | 6.0 | 1.5 | **9.0** | 9-12 ✅ |
| box | Medium | 10.0 | 1.0 | **10.0** | 9-12 ✅ |
| lift | Medium | 10.0 | 1.0 | **10.0** | 9-12 ✅ |
| transfer | Hard | 6.0 | 2.0 | **12.0** | 12-16 ✅ |
| carry | Hard | 10.0 | 1.3 | **13.0** | 12-16 ✅ |

---

## ✅ Cách implement

Thêm vào `compute_reward()`:

```python
# Task difficulty scales
TASK_SCALES = {
    0: 1.0,   # reach (Easy)
    1: 1.0,   # button (Easy)
    2: 0.6,   # cabinet (Easy - giảm)
    3: 1.5,   # ball (Medium)
    4: 1.0,   # box (Medium)
    5: 2.0,   # transfer (Hard - tăng)
    6: 1.0,   # lift (Medium)
    7: 1.3,   # carry (Hard)
}

# Apply scale khi tính reward
total_rew = (base_rew + shaped_rew) * TASK_SCALES[task_id]
```

---

## 📝 Tóm tắt

1. **Reward gốc** từ configs không phù hợp độ khó thực tế
2. **Cabinet** (Easy) có reward quá cao (10.0)
3. **Transfer** (Hard) có reward quá thấp (6.0)
4. Cần **scale theo độ khó** để agent học tốt hơn
5. Task khó cần reward cao hơn → motivation học nhiều hơn
