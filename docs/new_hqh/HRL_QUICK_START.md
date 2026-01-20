# HRL Training - Quick Start

## 🚀 Lệnh Chạy

```bash
cd /home/crl/hienhq/SkillBlender/legged_gym

# Train mới (mặc định 4096 envs, 100k iters)
python legged_gym/scripts/train.py --task h1_hrl --headless

# Train với WandB project + run name
python legged_gym/scripts/train.py --task h1_hrl --headless \
    --wandb hrlv1_project \
    --run_name v3

# Train với số envs khác (giảm nếu OOM)
python legged_gym/scripts/train.py --task h1_hrl --headless --num_envs 2048

# Train với max iterations khác
python legged_gym/scripts/train.py --task h1_hrl --headless --max_iterations 50000

# Train với seed khác
python legged_gym/scripts/train.py --task h1_hrl --headless --seed 42

# Train trên GPU cụ thể
CUDA_VISIBLE_DEVICES=1 python legged_gym/scripts/train.py --task h1_hrl --headless

# Resume từ checkpoint gần nhất
python legged_gym/scripts/train.py --task h1_hrl --headless --resume

# Resume từ checkpoint cụ thể
python legged_gym/scripts/train.py --task h1_hrl --headless --resume --load_run <run_name> --checkpoint <iter>

# Test policy (có GUI)
python legged_gym/scripts/play.py --task h1_hrl --load_run <run_name> --checkpoint <iter>

# Test không GUI
python legged_gym/scripts/play.py --task h1_hrl --load_run <run_name> --checkpoint <iter> --headless
```

---

## ⚙️ Config Trong Code

### **Curriculum (ppo_hrl.py hoặc h1_hrl.py)**
```python
class algorithm:
    stage1_end = 20000        # Stage 1 kết thúc (explore skills)
    total_iterations = 100000 # Tổng số iterations
    
    K_start = 10              # Option duration ban đầu
    K_end = 5                 # Option duration cuối
    
    epsilon_start = 0.18      # Exploration ε ban đầu
    epsilon_end = 0.0         # ε cuối (greedy)
    
    tau_start = 2.0           # Temperature τ ban đầu (soft sampling)
    tau_end = 1.0             # τ cuối (sharper)
    
    c_ent_skill = 0.02        # Skill entropy bonus
    entropy_coef = 0.001      # Action entropy bonus
    
    learning_rate = 1e-5      # Base learning rate
    lr_cmd_ratio_stage1 = 0.2 # Command LR = 0.2 × base (stage 1)
```

### **Network (h1_hrl.py)**
```python
class policy:
    num_skills = 4                      # Số skills
    command_dim = 14                    # Command dimension
    encoder_hidden_dims = [256]         # Encoder layers
    skill_hidden_dims = [128]           # Skill head layers
    command_hidden_dims = [256, 128]    # Command head layers
    low_hidden_dims = [256, 128]        # Low policy layers
    critic_hidden_dims = [512, 256, 128]# Critic layers
```

### **Environment (h1_hrl.py)**
```python
class env:
    num_envs = 4096           # Số parallel envs (giảm nếu OOM)
    episode_length_s = 20     # Episode length (seconds)
    num_observations = 105    # Actor obs dim
    num_privileged_obs = 303  # Critic obs dim (3 frames × 101)
```

---

## 📁 Files HRL

| File | Chức năng |
|------|-----------|
| `rsl_rl/modules/actor_critic_hrl.py` | 3-level network: Skill→Command→Action |
| `rsl_rl/algorithms/ppo_hrl.py` | PPO + Curriculum (2 stages) |
| `rsl_rl/runners/on_policy_runner_hrl.py` | Training loop + HRL logging |
| `legged_gym/envs/h1/h1_hrl/h1_hrl.py` | Environment 8 tasks + Config |

---

## 📊 Curriculum Timeline

```
Iter 0 ━━━━━━━━━━━━━━ 20k ━━━━━━━━━━━━━━━━━━━━━━━ 100k
      │               │                            │
   Start         Stage 1→2                      End

Stage 1 (explore): K=10, ε=0.18, τ=2.0, high skill entropy
Stage 2 (refine):  K→5, ε→0, τ→1.0, converging
```

---

## 📈 Metrics Quan Trọng (WandB)

| Metric | Stage 1 Target | Stage 2 Target |
|--------|----------------|----------------|
| `Skill/histogram_skill_*` | ~25% mỗi skill | Specialization |
| `Skill/switch_rate` | ~0.1 (1/K=10) | ~0.2 (1/K=5) |
| `Entropy/skill` | >1.0 | 0.5-0.8 |
| `Train/mean_reward` | Tăng dần | Converge cao |

---

## ⚠️ Troubleshooting

| Vấn đề | Giải pháp |
|--------|-----------|
| **Import Error** | `cd rsl_rl && pip install -e . && cd ../legged_gym && pip install -e .` |
| **CUDA OOM** | `--num_envs 2048` hoặc sửa `num_envs` trong h1_hrl.py |
| **Skill Collapse** | Tăng `epsilon_start = 0.25`, `tau_start = 2.5` |
| **Switch rate sai** | Kiểm tra K value trong curriculum |
| **Reward không tăng** | Kiểm tra task reward functions |
