# HRL Training - Quick Start

## 🚀 Train

```bash
cd /home/crl/hienhq/SkillBlender/legged_gym

# Train mới
python legged_gym/scripts/train_hrl.py --task h1_hrl --headless

# Với WandB
python legged_gym/scripts/train_hrl.py --task h1_hrl --headless --wandb hrlv1 --run_name exp1

# Custom envs/iterations
python legged_gym/scripts/train_hrl.py --task h1_hrl --headless --num_envs 256 --max_iterations 1000

# Resume
python legged_gym/scripts/train_hrl.py --task h1_hrl --headless --resume
```

## 🎮 Play

```bash
python legged_gym/scripts/play_hrl.py --task h1_hrl --load_run <run_name>
```

---

## ⚙️ Tham số từ Code

### 🔒 Cố định (KHÔNG đổi - từ h1_hrl.py config)

| Tham số | Giá trị | File |
|---------|---------|------|
| `num_skills` | 4 | policy.num_skills |
| `num_tasks` | 8 | hardcoded |
| `num_actions` | 19 | env.num_actions |
| `num_observations` | 105 | env.num_observations |
| `num_privileged_obs` | 303 | env.num_privileged_obs (3×101) |
| `decimation` | 10 | control.decimation (100Hz) |
| `dt` | 0.001 | sim.dt (1000Hz) |
| `command_dims` | [3,14,1,4] | skill_dict (total=22) |

### ✅ Tuỳ chỉnh qua CLI args

| Tham số | Mặc định | Ghi chú |
|---------|----------|---------|
| `--num_envs` | **16384** | Giảm nếu OOM (4096, 8192) |
| `--max_iterations` | **2000** | Test: 100-500, Full: 2000 |

### 🎚️ Tuỳ chỉnh trong code (h1_hrl.py)

| Tham số | Mặc định | Vị trí |
|---------|----------|--------|
| `learning_rate` | 1e-4 | algorithm.learning_rate |
| `episode_length_s` | 20s | env.episode_length_s |
| `hold_steps` | 5 | policy.hold_steps |
| `num_steps_per_env` | 60 | runner.num_steps_per_env |
| `save_interval` | 500 | runner.save_interval |

### 📊 Curriculum (ppo_hrl.py)

| Tham số | Stage 1 | Stage 2 |
|---------|---------|---------|
| Iterations | 0 → 600 | 600 → 2000 |
| K (hold steps) | 10 | 10 → 5 |
| ε (exploration) | 0.18 | 0.18 → 0 |
| τ (temperature) | 2.0 | 2.0 → 1.0 |
| lr_cmd_ratio | 0.2 | 1.0 |

### 🎯 Task Difficulty Scales (h1_hrl.py compute_reward)

| Task | Scale | Độ khó |
|------|-------|--------|
| reach | 1.0 | Easy |
| button | 1.0 | Easy |
| cabinet | 0.6 | Easy |
| ball | 1.5 | Medium |
| box | 1.0 | Medium |
| transfer | 2.0 | Hard |
| lift | 1.0 | Medium |
| carry | 1.3 | Hard |

---

## 📁 Files chính

| File | Mô tả |
|------|-------|
| `scripts/train_hrl.py` | Script train |
| `scripts/play_hrl.py` | Script test |
| `envs/h1/h1_hrl/h1_hrl.py` | Env + Config + Rewards |
| `rsl_rl/modules/actor_critic_hrl_v2.py` | HRL Policy |
| `rsl_rl/algorithms/ppo_hrl.py` | PPO + Curriculum |

## ⚠️ Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| **Import Error** | `cd rsl_rl && pip install -e . && cd ../legged_gym && pip install -e .` |
| **CUDA OOM** | `--num_envs 2048` hoặc sửa `num_envs` trong h1_hrl.py |
| **Skill Collapse** | Tăng `epsilon_start = 0.25`, `tau_start = 2.5` |
| **Switch rate sai** | Kiểm tra K value trong curriculum |
| **Reward không tăng** | Kiểm tra task reward functions |
