# HRL Training Plan - Hierarchical Skill-Command Architecture

## 📋 Tổng Quan

Thiết kế HRL 2-level hierarchy cho 8 manipulation tasks với curriculum learning 2 stages.

**Architecture:**
```
Observation (99 dims)
    ↓
[HEAD 1: Skill Selection (Gating)]
    ↓
Skill ID (discrete, 4-8 skills) → Onehot(4-8 dims)
    ↓
[HEAD 2: Command Generation]
    ↓
Command (subgoal: 6-14 dims tùy task)
    ↓
[LOW POLICY: Motor Control]
    ↓
Actions (19 DOFs)
```

**Key Features:**
- **Option Duration**: Giữ skill K bước, không resample liên tục
- **2-Stage Curriculum**: Stage 1 explore skills, Stage 2 fine-tune commands
- **Separate Entropy**: Skill exploration cao, command precision cao
- **Asymmetric Learning Rates**: Command học chậm hơn trong stage 1

---

## 🎯 Network Architecture

### **1. Shared Encoder**

**Input:** 99 dims actor observation
```
task_error (14) + q (19) + dq (19) + actions (19) + 
base_ang_vel (3) + base_euler (3) + error_mask (14) + task_onehot (8)
```

**Output:** Feature embedding (256 dims)

**Structure:**
- Layer 1: Linear(99 → 256) + ELU
- Layer 2: Linear(256 → 256) + ELU
- Layer 3: Linear(256 → 256) + ELU

**Purpose:** Shared representation cho cả skill selection và value estimation

---

### **2. HEAD 1: Skill Selection (Gating Network)**

**Input:** Feature embedding (256 dims)

**Output:** Skill logits (4-8 dims, tùy thiết kế)

**Structure:**
- Linear(256 → 128) + ELU
- Linear(128 → num_skills)
- No activation (raw logits)

**Skill Space Design (4 skills cho đơn giản):**
- Skill 0: Approach (di chuyển gần object)
- Skill 1: Grasp/Contact (manipulation tinh tế)
- Skill 2: Lift/Push (tác động vật lý)
- Skill 3: Stabilize (giữ cân bằng, điều chỉnh)

**Sampling Strategy:**
- Stage 1: Mixture sampling với high exploration
- Stage 2: Greedy/softmax với low temperature
- Option duration: K bước giữ nguyên skill

**Outputs:**
- `skill_logits`: Raw logits (4 dims)
- `skill_id`: Sampled discrete skill (scalar int)
- `skill_onehot`: One-hot encoding (4 dims)
- `skill_logprob`: Log probability cho PPO loss

---

### **3. HEAD 2: Command Generation Network**

**Input:** Feature embedding (256) + skill_onehot (4) = 260 dims

**Output:** Command vector (14 dims max, padded)

**Structure:**
- Layer 1: Linear(260 → 256) + ELU
- Layer 2: Linear(256 → 128) + ELU
- Layer 3: Linear(128 → 14)
- Activation: Tanh (normalized output)

**Command Space Design (14 dims max, task-dependent):**

| Task | Command Dims | Command Meaning |
|------|--------------|-----------------|
| Reach | 6 | left_wrist_target (3) + right_wrist_target (3) |
| Button | 3 | target_position (3) for left wrist |
| Cabinet | 4 | handle_target (3) + door_opening (1) |
| Ball | 6 | ball_target (3) + torso_approach (3) |
| Box/Transfer/Lift/Carry | 6 | box_target (3) + wrist_approach (3) |

**Command Scaling:**
- Tanh output → [-1, 1]
- Scale to task-specific ranges (ví dụ: position ±0.5m)
- Mask unused dims (padding) bằng command_mask

**Outputs:**
- `command_raw`: Raw tanh output (14 dims)
- `command`: Scaled command (14 dims)
- `command_mask`: Which dims are active (14 dims)

---

### **4. LOW POLICY: Motor Control Network**

**Input:** Command (14) + Robot State (61) = 75 dims

**Robot State Components:**
```
q (19) + dq (19) + actions (19) + base_ang_vel (3) + base_euler (3)
```

**Output:** Joint actions (19 dims)

**Structure:**
- Layer 1: Linear(75 → 256) + ELU
- Layer 2: Linear(256 → 256) + ELU
- Layer 3: Linear(256 → 128) + ELU
- Output: Linear(128 → 19)
- Activation: Tanh → scale to action range

**Purpose:** Pure motor controller
- Nhận command = "what to do"
- Translate thành joint-level actions = "how to do"
- Không nhận task_id, error_mask (abstracted away)

---

### **5. Critic Network (Value Estimation)**

**Input:** Privileged obs (423 dims = 141 × 3 frames)

**Output:** Value estimate (1 scalar)

**Structure:**
- Layer 1: Linear(423 → 256) + ELU
- Layer 2: Linear(256 → 256) + ELU
- Layer 3: Linear(256 → 128) + ELU
- Output: Linear(128 → 1)

**Privileged Info:**
```
Per frame (141 dims):
  task_error (14) + raw_goal (15) + raw_current (15) +
  q (19) + dq (19) + actions (19) +
  base_lin_vel (3) + base_ang_vel (3) + base_euler (3) +
  rand_push_force (2) + rand_push_torque (3) +
  friction (1) + mass (1) + contact (2) +
  error_mask (14) + task_onehot (8)

Stacked: 141 × 3 = 423 dims
```

**Purpose:** Better value estimation với full state info

---

## 🎓 Two-Stage Curriculum Learning

### **Stage 1: Skill Exploration & Stabilization**

**Duration:** 0% → 40% of total training steps (ví dụ: 0-40M steps nếu train 100M)

**Goal:** 
- Gating học phân biệt skills (which skill for which situation)
- Command học rough mapping (skill → approximate subgoal)
- Tránh collapse vào 1 skill

---

#### **1.1. Option Duration**

**K = 10 steps**

**Mechanism:**
- Sample skill từ gating network mỗi K=10 bước
- Giữ nguyên `skill_id` và `skill_onehot` trong K bước
- Chỉ resample khi:
  - Đủ K bước
  - Episode kết thúc (reset)
  - Manual override (nếu có)

**Implementation Logic:**
```
# Pseudo-logic (không phải code)
Initialize:
  step_counter = 0
  current_skill_id = None
  current_skill_onehot = None

Every timestep:
  IF step_counter == 0 OR step_counter >= K:
    # Resample skill
    skill_logits = SkillHead(obs)
    skill_id = Sample(skill_logits, exploration_strategy)
    skill_onehot = OneHot(skill_id, num_skills)
    step_counter = 0
  
  # Use current skill
  command = CommandHead(obs, skill_onehot)
  action = LowPolicy(command, robot_state)
  
  step_counter += 1
```

**Benefits:**
- Giảm jitter (skill không đổi mỗi step)
- Command head nhận consistent onehot trong K bước
- Gating vẫn học qua logprob (không cần gradient qua onehot)

**Logging:**
- `skill_switch_rate`: Frequency of skill changes (should be ~1/K)
- `skill_hold_duration`: Actual K measured (check if logic đúng)

---

#### **1.2. Mixture Sampling Strategy**

**Formula:**
```
p_mix = (1 - ε) * softmax(logits / τ) + ε * Uniform(num_skills)
```

**Parameters:**
- ε (epsilon) = 0.15 - 0.20: Uniform exploration weight
- τ (tau) = 1.5 - 2.0: Temperature (higher = more random)

**Purpose:**
- Softmax component: Exploit learned preference
- Uniform component: Force explore all skills
- Temperature: Soften distribution (avoid premature convergence)

**Sampling Process:**
```
1. Compute skill_logits from gating network
2. Softmax: p_softmax = softmax(skill_logits / τ)
3. Uniform: p_uniform = [1/N, 1/N, ..., 1/N]
4. Mix: p_mix = (1-ε) * p_softmax + ε * p_uniform
5. Sample: skill_id ~ Categorical(p_mix)
6. Compute logprob: logprob = log(p_mix[skill_id]) for PPO
```

**Example (4 skills, ε=0.2, τ=2.0):**
```
logits = [2.0, 1.0, -0.5, 0.3]
softmax(logits/2.0) = [0.42, 0.26, 0.12, 0.20]
p_uniform = [0.25, 0.25, 0.25, 0.25]
p_mix = 0.8*[0.42,0.26,0.12,0.20] + 0.2*[0.25,0.25,0.25,0.25]
      = [0.386, 0.258, 0.146, 0.210]

→ Skill 2 (low logit) vẫn có 14.6% chance được chọn
```

**Rationale:**
- Tránh gating collapse vào 1-2 skills sớm
- Cho command head data về ALL skills
- Entropy_skill cao → nhiều khám phá

---

#### **1.3. Separated Entropy Regularization**

**Skill Entropy (HIGH):**
```
H_skill = -∑ p(skill) * log(p(skill))
```
- Coefficient: `c_ent_skill = 0.02` (gấp 10× command)
- Target: Keep skill distribution spread out
- Monitor: Should be close to log(num_skills) initially

**Command Entropy (LOW):**
```
H_command = -∑ p(action|skill) * log(p(action|skill))
```
- Coefficient: `c_ent_cmd = 0.002` (10× nhỏ hơn skill)
- Target: Command precise cho mỗi skill
- Monitor: Should decrease as learning progresses

**Loss Composition:**
```
L_total = L_ppo + c_ent_skill * H_skill + c_ent_cmd * H_command
```

**Rationale:**
- Skill head: Cần explore → high entropy bonus
- Command head: Cần precision → low entropy bonus
- Asymmetric regularization = different learning behaviors

**Monitoring:**
```
Log every N steps:
  - entropy_skill: Current H_skill
  - entropy_command: Current H_command
  - entropy_ratio: H_skill / H_command (should be ~10×)
```

---

#### **1.4. Learning Rate: Command Learns Slower**

**Skill Network LR:**
```
lr_skill = lr_base = 3e-4
```

**Command Network LR:**
```
lr_cmd = 0.2 × lr_base = 6e-5
```

**Low Policy LR:**
```
lr_low = lr_base = 3e-4
```

**Critic LR:**
```
lr_critic = lr_base = 3e-4
```

**Rationale:**
- Gating thay đổi nhanh → command cần học chậm để tránh overfitting
- Command chỉ học rough mapping trong stage 1
- Low policy học bình thường (motor control ổn định)

**Alternative Implementation:**
- Thay vì separate LR, có thể dùng:
  - Loss weight: `L_cmd = 0.2 × MSE(command, target)`
  - Gradient clipping riêng cho command head
  - Fewer update epochs cho command (ví dụ: 2 epochs vs 10 epochs)

---

#### **1.5. Stage 1 Monitoring & Success Criteria**

**Key Metrics:**

1. **Skill Histogram**
   - Plot: Frequency of each skill being selected
   - Good: All skills > 15% usage
   - Bad: 1 skill > 70% (collapse)

2. **Skill Switch Rate**
   - Metric: Số lần skill change / total steps
   - Expected: ~1/K = 1/10 = 0.1 switches/step
   - Too high: K không work, logic sai
   - Too low: Skill stuck, không explore

3. **Error Slice by Skill**
   - Compute: Mean task_error cho mỗi skill
   - Good: Different skills → different error patterns
   - Example:
     ```
     Skill 0 (Approach): High position error, low orientation error
     Skill 1 (Grasp): Low position error, medium orientation error
     Skill 2 (Lift): Medium position error, high Z-axis error
     Skill 3 (Stabilize): Low overall error
     ```
   - Bad: All skills có same error distribution

4. **Entropy Tracking**
   - entropy_skill: Should stay high (~log(4) = 1.39 for 4 skills)
   - entropy_command: Can be medium (learning precision)
   - entropy_ratio: Should be ~10×

5. **Reward Progress**
   - Task success rate: May increase slowly (chưa phải focus)
   - Episode return: Gradual improvement
   - Command loss: Should decrease (command converging)

**Exit Criteria (Move to Stage 2):**
- Training steps ≥ 40% of total
- Skill histogram: All skills > 10% usage
- Skill switch rate: ~0.08-0.12 (gần 1/K)
- Entropy_skill: > 0.8 × log(num_skills)

**If NOT Meeting Criteria:**
- Increase ε to 0.25-0.3 (more uniform)
- Increase τ to 2.5-3.0 (softer softmax)
- Increase c_ent_skill to 0.03
- Extend stage 1 to 50-60% of training

---

### **Stage 2: Command Refinement & Task Mastery**

**Duration:** 40% → 100% of total training steps (ví dụ: 40M-100M steps)

**Goal:**
- Gating converge to optimal skill selection
- Command learn precise subgoals
- Low policy execute accurately
- Maximize task success rate

---

#### **2.1. Reduced Option Duration**

**K = 5 steps**

**Rationale:**
- Skills đã ổn định → có thể switch nhanh hơn
- Faster reaction to environment changes
- More responsive to task dynamics

**Transition:**
- Linear anneal K from 10 → 5 over first 10% of stage 2
- Monitor skill_switch_rate: Should increase to ~0.2 switches/step

---

#### **2.2. Anneal Exploration Parameters**

**Epsilon Annealing:**
```
ε(t) = ε_start × (1 - t/T_stage2)
```
- Start: ε = 0.15-0.20 (từ stage 1)
- End: ε = 0.0 (pure policy, no uniform)
- Schedule: Linear or exponential decay

**Temperature Annealing:**
```
τ(t) = 1.0 + (τ_start - 1.0) × (1 - t/T_stage2)
```
- Start: τ = 1.5-2.0
- End: τ = 1.0 (standard softmax)
- Schedule: Linear decay

**Sampling Transition:**
```
Stage 1 end:   p_mix = 0.8*softmax(logits/2.0) + 0.2*Uniform
Stage 2 start: p_mix = 0.8*softmax(logits/2.0) + 0.2*Uniform
Midpoint:      p_mix = 0.9*softmax(logits/1.5) + 0.1*Uniform
Stage 2 end:   p_mix = 1.0*softmax(logits/1.0) + 0.0*Uniform
```

**Monitoring:**
- `epsilon_current`: Track decay
- `tau_current`: Track decay
- `skill_distribution_entropy`: Should decrease naturally

---

#### **2.3. Entropy Regularization Adjustment**

**Skill Entropy: Gradual Decrease**
```
c_ent_skill(t) = c_ent_skill_start × decay_factor(t)
```
- Start: 0.02
- End: 0.005 (giảm 4×)
- Schedule: Exponential decay with half-life at 70% of stage 2

**Command Entropy: Keep Low**
```
c_ent_cmd = 0.002 (constant)
```
- No change: Command luôn cần precision
- May even decrease to 0.001 near end

**Rationale:**
- Gating đã explore đủ → converge to good policy
- Command cần maintain precision → keep low entropy
- Natural entropy decrease from exploration decay

**Target Values at End:**
- H_skill: ~0.5-0.8 (concentrated but not collapsed)
- H_command: ~0.3-0.5 (precise)

---

#### **2.4. Learning Rate: Command Normal Speed**

**All Networks Equal LR:**
```
lr_skill = lr_cmd = lr_low = lr_critic = 3e-4
```

**Optional: LR Decay at End**
```
lr(t) = lr_base × 0.5 if t > 90% of training
```

**Rationale:**
- Skills ổn định → command có thể học nhanh
- Fine-tuning toàn bộ architecture
- Standard PPO learning rate

---

#### **2.5. Stage 2 Monitoring & Success Criteria**

**Key Metrics:**

1. **Skill Histogram Evolution**
   - Expected: Distribution narrowing
   - Example transition:
     ```
     Stage 1 end: [28%, 24%, 26%, 22%] - balanced
     Stage 2 mid: [35%, 30%, 20%, 15%] - preference forming
     Stage 2 end: [45%, 30%, 15%, 10%] - clear specialization
     ```
   - Per-task specialization: Different tasks may prefer different skills

2. **Skill Switch Rate**
   - Expected: ~0.2 switches/step (K=5)
   - Too high (>0.3): Skill unstable, may need increase K
   - Too low (<0.15): Skill stuck, check logic

3. **Error Slice Mean & Std by Skill**
   - Mean error per skill: Should DECREASE
   - Std error per skill: Should DECREASE
   - Cross-skill variance: Should INCREASE (specialization)
   - Example:
     ```
     Skill 0: mean=0.15, std=0.08 → specialized for far approach
     Skill 1: mean=0.05, std=0.03 → specialized for fine manipulation
     ```

4. **Command Precision**
   - MSE(command, optimal_command): Decrease
   - Command variance per skill: Decrease (less noisy)
   - Command-to-action correlation: Increase

5. **Task Success Rate**
   - Primary metric: Success rate per task
   - Expected: Rapid improvement in stage 2
   - Goal: 70-90% success rate by end

6. **Entropy Tracking**
   - entropy_skill: Decreasing (0.8 → 0.5)
   - entropy_command: Stable low (0.3-0.5)
   - entropy_ratio: Decreasing (10× → 1-2×)

**Exit Criteria (Training Complete):**
- Training steps = 100%
- Task success rate > target threshold (70-90%)
- Skill distribution converged (std of histogram < 0.1)
- Command loss plateaued
- Value loss plateaued

---

## 📊 Comprehensive Logging Strategy

### **Every 1000 Steps (High Frequency)**

1. **Scalar Metrics:**
   - `train/reward_mean`: Average episode reward
   - `train/reward_std`: Reward variance
   - `train/episode_length`: Average episode length
   - `train/success_rate`: Task success rate
   - `train/value_loss`: Critic loss
   - `train/policy_loss`: Actor loss total
   - `train/entropy_skill`: Skill selection entropy
   - `train/entropy_command`: Command generation entropy
   - `train/entropy_ratio`: H_skill / H_command
   - `train/skill_switch_rate`: Frequency of skill changes
   - `train/epsilon_current`: Current exploration ε
   - `train/tau_current`: Current temperature τ
   - `train/K_current`: Current option duration

2. **Learning Rate Tracking:**
   - `train/lr_skill`: Gating network LR
   - `train/lr_command`: Command network LR
   - `train/lr_low`: Low policy LR
   - `train/lr_critic`: Critic LR

3. **Gradient Norms:**
   - `train/grad_norm_skill`: Gating gradient magnitude
   - `train/grad_norm_command`: Command gradient magnitude
   - `train/grad_norm_low`: Low policy gradient magnitude
   - `train/grad_norm_critic`: Critic gradient magnitude

---

### **Every 5000 Steps (Medium Frequency)**

1. **Skill Histogram (Bar Chart):**
   ```
   X-axis: Skill ID [0, 1, 2, 3]
   Y-axis: Frequency (%)
   Title: "Skill Selection Distribution"
   ```
   - Compute từ rollout buffer
   - Show per-task breakdown if needed

2. **Error Decomposition by Skill (Table):**
   ```
   | Skill | Mean Error | Std Error | Count | % Usage |
   |-------|-----------|-----------|-------|---------|
   | 0     | 0.245     | 0.123     | 4521  | 28%     |
   | 1     | 0.189     | 0.098     | 3892  | 24%     |
   | 2     | 0.301     | 0.156     | 4234  | 26%     |
   | 3     | 0.267     | 0.134     | 3553  | 22%     |
   ```

3. **Command Statistics:**
   - `train/command_mean`: Mean command values (per dim)
   - `train/command_std`: Command variance (per dim)
   - `train/command_magnitude`: L2 norm of commands

4. **Per-Task Metrics:**
   - `task_0_reach/success_rate`
   - `task_0_reach/avg_reward`
   - `task_0_reach/avg_error`
   - ... (repeat for 8 tasks)

---

### **Every 20000 Steps (Low Frequency)**

1. **Skill Transition Matrix (Heatmap):**
   ```
   From\To | Skill 0 | Skill 1 | Skill 2 | Skill 3 |
   --------|---------|---------|---------|---------|
   Skill 0 | 0.85    | 0.08    | 0.05    | 0.02    |
   Skill 1 | 0.10    | 0.80    | 0.07    | 0.03    |
   Skill 2 | 0.06    | 0.09    | 0.78    | 0.07    |
   Skill 3 | 0.03    | 0.05    | 0.12    | 0.80    |
   ```
   - Diagonal high = skill持續性 good
   - Off-diagonal = transition patterns

2. **Command Visualization (Scatter Plot):**
   - Plot command dims colored by skill
   - Show clustering patterns
   - Detect skill specialization

3. **Value Prediction Accuracy:**
   - MSE(predicted_value, actual_return)
   - Per-skill value accuracy
   - Advantage estimation quality

4. **Checkpoint Save:**
   - Save model weights
   - Save optimizer states
   - Save curriculum stage info
   - Save replay buffer (if using)

---

### **Stage Transition Logging (Special Events)**

**When Entering Stage 2:**
```
LOG: "========== STAGE 1 → STAGE 2 TRANSITION =========="
LOG: "Step: 40,000,000"
LOG: "Stage 1 Summary:"
  - Skill histogram: [27%, 25%, 26%, 22%]
  - Switch rate: 0.102
  - Entropy_skill: 1.35 (target: >1.11)
  - Success rate: 34%
LOG: "Stage 2 Config:"
  - K: 10 → 5 (anneal over 10% steps)
  - ε: 0.18 → 0.0 (linear decay)
  - τ: 2.0 → 1.0 (linear decay)
  - c_ent_skill: 0.02 → 0.005 (exp decay)
  - lr_cmd: 6e-5 → 3e-4
LOG: "=================================================="
```

**Every 10% of Stage 2:**
```
LOG: "Stage 2 Progress: 50% (step 70M/100M)"
  - ε: 0.09 (50% decayed)
  - τ: 1.5 (50% decayed)
  - K: 5 (fully transitioned)
  - Skill entropy: 0.85 (decreasing)
  - Success rate: 68% (improving)
```

---

## 🔧 Implementation Considerations

### **1. Rollout Buffer Structure**

**Standard PPO Buffer + HRL Extensions:**

Per timestep storage:
```
- obs: (99,) actor observation
- privileged_obs: (423,) critic observation
- action: (19,) low-level actions
- reward: scalar
- done: bool
- value: scalar (critic prediction)
- logprob: scalar (total policy logprob)

HRL additions:
- skill_id: int (which skill selected)
- skill_onehot: (4,) one-hot encoding
- skill_logprob: scalar (gating logprob)
- command: (14,) command vector
- command_logprob: scalar (command logprob)
- step_in_option: int (0 to K-1)
- task_error: (14,) for monitoring
```

**Buffer Size:**
- PPO typically: 2048 steps × 4096 envs = 8.4M transitions
- With HRL: Same size, just more fields
- Memory: ~2GB for full buffer with HRL fields

---

### **2. Gradient Flow Considerations**

**Important: Onehot Gradient Blocking**

```
Execution flow:
  skill_logits = SkillHead(obs)
  skill_id = Sample(skill_logits)
  skill_onehot = OneHot(skill_id)  ← Discrete, no gradient
  command = CommandHead(obs, skill_onehot)
  action = LowPolicy(command, robot_state)

Gradient flow:
  L_policy → command → CommandHead → [obs]
                                    → [skill_onehot] ✗ BLOCKED
                                    
  L_policy_skill → skill_logprob → SkillHead → [obs] ✓ FLOWS
```

**Key Point:**
- Gating learns via REINFORCE (logprob gradient)
- Command learns via standard backprop
- NO gradient flows from command to skill_onehot
- This is CORRECT and INTENDED

---

### **3. K-Step Option Duration Implementation**

**Per-Env Counters (Vectorized):**

```
Maintain state:
  current_skill_id: [B] tensor of ints
  current_skill_onehot: [B, num_skills] tensor
  step_counter: [B] tensor of ints

Every step:
  # Check which envs need resample
  need_resample = (step_counter >= K) | done_mask
  
  # Resample for those envs
  IF any(need_resample):
    new_skill_logits = SkillHead(obs[need_resample])
    new_skill_id = Sample(new_skill_logits, exploration)
    new_skill_onehot = OneHot(new_skill_id)
    
    current_skill_id[need_resample] = new_skill_id
    current_skill_onehot[need_resample] = new_skill_onehot
    step_counter[need_resample] = 0
  
  # Use current skill for ALL envs
  command = CommandHead(obs, current_skill_onehot)
  
  # Increment counter
  step_counter += 1
```

**Reset Handling:**
```
When env[i] resets:
  step_counter[i] = K  # Force resample on next step
  current_skill_id[i] = -1  # Invalid marker
  current_skill_onehot[i] = 0  # Zero vector
```

---

### **4. Mixture Sampling Implementation**

**Numerically Stable Version:**

```
Algorithm:
  1. Compute logits: z = SkillHead(obs)  # [B, num_skills]
  
  2. Softmax with temperature:
     p_softmax = exp((z - max(z)) / τ) / sum(exp((z - max(z)) / τ))
     
  3. Uniform:
     p_uniform = 1 / num_skills
     
  4. Mix:
     p_mix = (1 - ε) * p_softmax + ε * p_uniform
     
  5. Sample:
     skill_id = Categorical(p_mix).sample()
     
  6. Compute logprob:
     logprob = log(p_mix[range(B), skill_id])
```

**Edge Cases:**
- When ε=0, τ=1: Standard softmax sampling
- When ε=1: Pure uniform (ignore network)
- Numerical stability: Use log-sum-exp trick

---

### **5. Loss Function Composition**

**Total Loss:**
```
L_total = L_ppo_low + L_ppo_skill + L_ppo_command + L_value + L_ent

Where:
  L_ppo_low: PPO loss for low policy
    = -min(ratio * A, clip(ratio) * A)
    ratio = π_new(a|cmd) / π_old(a|cmd)
    
  L_ppo_skill: PPO loss for skill selection
    = -min(ratio_skill * A, clip(ratio_skill) * A)
    ratio_skill = π_new(skill|obs) / π_old(skill|obs)
    
  L_ppo_command: PPO loss for command generation
    = -min(ratio_cmd * A, clip(ratio_cmd) * A)
    ratio_cmd = π_new(cmd|obs, skill) / π_old(cmd|obs, skill)
    
  L_value: Value function MSE
    = MSE(V(obs), returns)
    
  L_ent: Entropy regularization
    = -c_ent_skill * H(π_skill) - c_ent_cmd * H(π_cmd)
```

**Advantage Sharing:**
- Use SAME advantage A for all 3 policy losses
- Advantage computed from critic: A = returns - V(obs)
- This is key: All components optimize same return

---

### **6. Command Space Normalization**

**Tanh Output → Task Range:**

```
For each task:
  command_raw = tanh(CommandHead(obs, skill))  # [-1, 1]
  
  Task-specific scaling:
    Reach: target = command_raw * 0.5 + current_pos
           (±0.5m workspace)
           
    Button: target = command_raw * [0.3, 0.3, 0.2] + button_pos
            (small offset from button)
            
    Cabinet: handle = command_raw[:3] * 0.2 + handle_init
             door = command_raw[3] * 1.57  # ±90°
             
    Ball: ball_target = command_raw[:3] * 2.0 + [5, 0, 0.1]
          (±2m around goal area)
          
    Box: box_target = command_raw[:3] * 1.0 + box_init
```

**Clamping:**
- Apply workspace limits after scaling
- Prevent commands outside valid range
- Log violation rate for debugging

---

### **7. Debugging & Sanity Checks**

**Every N Steps, Assert:**

1. **Skill ID Validity:**
   - All skill_id ∈ [0, num_skills)
   - No NaN or invalid values

2. **Option Duration:**
   - step_counter ∈ [0, K]
   - Switch rate ≈ 1/K (within 20%)

3. **Probability Validity:**
   - All logprobs are finite (not NaN, not -inf)
   - sum(p_mix) = 1.0 (within numerical tolerance)

4. **Gradient Health:**
   - No gradient explosion (norm < 10.0)
   - No gradient vanishing (norm > 1e-6)
   - All parameters updating (mean param change > 0)

5. **Command Range:**
   - Command values within expected range
   - Command_mask correctly applied
   - No NaN in commands

6. **Entropy Bounds:**
   - 0 ≤ H_skill ≤ log(num_skills)
   - 0 ≤ H_command ≤ log(action_space_size)

---

## 🎯 Expected Learning Curves

### **Stage 1 (0-40% steps):**

**Skill Entropy:** 
- Start: ~1.39 (uniform over 4 skills)
- End: ~1.2-1.3 (slight preference forming)
- Shape: Slow decrease

**Command Entropy:**
- Start: ~1.5 (high variance)
- End: ~0.8 (converging)
- Shape: Steady decrease

**Success Rate:**
- Start: ~5-10%
- End: ~30-40%
- Shape: Slow linear increase

**Skill Histogram:**
- Start: [25%, 25%, 25%, 25%] uniform
- End: [28%, 24%, 26%, 22%] near-uniform
- Shape: Stay balanced

**Reward:**
- Start: Negative or very low
- End: Positive, moderate
- Shape: Noisy but upward

---

### **Stage 2 (40-100% steps):**

**Skill Entropy:**
- Start: ~1.2
- End: ~0.5-0.8
- Shape: Exponential decay

**Command Entropy:**
- Start: ~0.8
- End: ~0.3-0.5
- Shape: Slow decrease, plateau

**Success Rate:**
- Start: ~30-40%
- End: ~70-90%
- Shape: Rapid increase, then plateau

**Skill Histogram:**
- Start: [28%, 24%, 26%, 22%]
- End: [45%, 30%, 15%, 10%] or similar specialization
- Shape: Diverging distribution

**Reward:**
- Start: Moderate
- End: High
- Shape: Steep increase, then plateau

---

## 🚨 Common Failure Modes & Solutions

### **Failure 1: Skill Collapse**

**Symptom:**
- One skill > 80% usage by 20% of training
- Other skills rarely selected
- Success rate stagnates low

**Diagnosis:**
- Check skill histogram
- Check entropy_skill (should be > 1.0)
- Check ε and τ (may be too low)

**Solutions:**
1. Increase ε to 0.3-0.4
2. Increase τ to 2.5-3.0
3. Increase c_ent_skill to 0.03-0.05
4. Extend stage 1 duration
5. Add skill-specific rewards (encourage using all skills)

---

### **Failure 2: Skill Oscillation**

**Symptom:**
- Switch rate >> 1/K (e.g., 0.5 when expecting 0.1)
- Skills change every step despite K=10
- Commands unstable, high variance

**Diagnosis:**
- Check step_counter logic
- Check reset handling
- Check need_resample mask

**Solutions:**
1. Debug option duration implementation
2. Add logging: Print step_counter, need_resample
3. Check done_mask handling
4. Verify skill_id persistence

---

### **Failure 3: Command Ignores Skill**

**Symptom:**
- Command output same for all skills
- No skill specialization in error patterns
- Skill histogram uniform but success rate low

**Diagnosis:**
- Check gradient flow to CommandHead
- Check skill_onehot input
- Check if CommandHead uses skill_onehot

**Solutions:**
1. Verify skill_onehot concatenation
2. Increase command network capacity
3. Add skill-conditional batch norm
4. Increase lr_cmd temporarily
5. Add auxiliary loss: Predict skill from command

---

### **Failure 4: Low Policy Doesn't Follow Commands**

**Symptom:**
- Commands look reasonable
- Actions don't correlate with commands
- Success rate stays low despite good skills

**Diagnosis:**
- Check command → action correlation
- Check low policy input
- Check command scaling

**Solutions:**
1. Pre-train low policy on random commands
2. Increase lr_low
3. Add command tracking reward
4. Check command range validity
5. Simplify command space

---

### **Failure 5: Value Overestimation**

**Symptom:**
- Value loss very high
- Predicted value >> actual returns
- Training unstable, high variance

**Diagnosis:**
- Check returns computation
- Check advantage normalization
- Check critic architecture

**Solutions:**
1. Clip value loss
2. Normalize advantages
3. Reduce lr_critic
4. Add value clipping
5. Use Huber loss instead of MSE

---

## 📝 Summary & Quick Reference

### **Stage 1 Checklist (0-40%):**

- [x] K = 10
- [x] ε = 0.15-0.20
- [x] τ = 1.5-2.0
- [x] c_ent_skill = 0.02
- [x] c_ent_cmd = 0.002
- [x] lr_cmd = 0.2 × lr_base
- [x] Monitor: Skill histogram balanced
- [x] Monitor: Switch rate ≈ 0.1
- [x] Monitor: Entropy_skill > 1.0

### **Stage 2 Checklist (40-100%):**

- [x] K = 5
- [x] ε: 0.2 → 0.0 (linear)
- [x] τ: 2.0 → 1.0 (linear)
- [x] c_ent_skill: 0.02 → 0.005 (exp)
- [x] lr_cmd = lr_base
- [x] Monitor: Success rate increasing
- [x] Monitor: Skill specialization
- [x] Monitor: Error decreasing

### **Critical Logs (Every 1000 steps):**

1. entropy_skill
2. entropy_command
3. skill_switch_rate
4. success_rate
5. skill_histogram (every 5000)

### **Red Flags:**

- Skill collapse: 1 skill > 70%
- Switch rate >> 1/K: Logic bug
- Entropy_skill < 0.5 in stage 1: Need more exploration
- Success rate < 20% at 40%: Extend stage 1
- NaN in any loss: Numerical instability

---

## 🎓 Next Steps

After this plan is approved, implementation will proceed in this order:

1. **Network Architecture Code** (actor với 3 heads + critic)
2. **Option Duration Logic** (K-step skill holding)
3. **Mixture Sampling** (ε-τ exploration)
4. **Loss Functions** (3 PPO losses + entropy)
5. **Logging Infrastructure** (WandB integration)
6. **Curriculum Controller** (stage transition logic)
7. **Training Loop** (PPO với HRL modifications)
8. **Debugging Tools** (sanity checks, visualization)

Mỗi bước sẽ có code chi tiết + test cases.
