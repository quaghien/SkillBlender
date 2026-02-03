# HRL Training - Quick Start

## 🚀 Train

```bash
cd /home/crl/hienhq/SkillBlender/legged_gym

# Train mới
python legged_gym/scripts/train_hrl.py --task h1_hrl --headless

# Với WandB
python legged_gym/scripts/train_hrl.py --task h1_hrl --headless --wandb

# Resume
python legged_gym/scripts/train_hrl.py --task h1_hrl --headless --resume
```

## 🎮 Play

```bash
python legged_gym/scripts/play_hrl.py --task h1_hrl --load_run <run_name>
```

---

## ⚙️ Config (h1_hrl.py)

| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| `num_envs` | 16384 | Giảm nếu OOM |
| `num_skills` | 4 | walk, reach, squat, step |
| `num_tasks` | 8 | Incremental training |
| `max_iterations` | 8000 | 1000/task × 8 |

## 📊 Curriculum (Single Stage - Linear Decay)

| Param | Start → End | Ghi chú |
|-------|-------------|---------|
| K | 5 | Cố định (option duration) |
| ε | 0.18 → 0 | Exploration decay |
| τ | 2.0 → 1.0 | Temperature decay |
| c_ent_skill | 0.05 | Skill entropy (constant) |

## 🎯 Incremental Task Training

| Phase | Iter | Focus Task (70%) | Old Tasks (30%) |
|-------|------|------------------|-----------------|
| 0 | 0-999 | reach | - |
| 1 | 1000-1999 | button | reach |
| 2 | 2000-2999 | cabinet | reach, button |
| 3 | 3000-3999 | ball | reach, button, cabinet |
| 4 | 4000-4999 | box | +ball |
| 5 | 5000-5999 | lift | +box |
| 6 | 6000-6999 | transfer | +lift |
| 7 | 7000-7999 | carry | all |

**Task Order (Easy→Hard):** reach, button, cabinet, ball, box, lift, transfer, carry

## 📁 Files chính

| File | Mô tả |
|------|-------|
| `envs/h1/h1_hrl/h1_hrl.py` | Env + Config |
| `rsl_rl/modules/actor_critic_hrl_v2.py` | Skill-Aware Policy |
| `rsl_rl/algorithms/ppo_hrl.py` | PPO + Curriculum |

## 📈 WandB Metrics

- `Curriculum/phase`, `Curriculum/focus_task` - Training progress
- `Episode/rew_*` - Per-task rewards (8 tasks)
- `TaskDist/*` - Env distribution per task
- `Entropy/skill` - Skill selection diversity

## ⚠️ Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| CUDA OOM | Giảm `num_envs` trong h1_hrl.py |
| Skill Collapse | Check `Entropy/skill` > 1.0 |
| Low Reward | Check phase đúng task chưa |
