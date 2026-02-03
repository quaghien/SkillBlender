# Hold Skill & Policy Update Analysis

## 📋 Tóm tắt vấn đề

| Aspect | Hiện tại (actor_critic_hrl.py) | Cần thiết (HRL chuẩn) |
|--------|-------------------------------|----------------------|
| **Skill Selection** | SOFT BLEND (softmax weights) | HARD OPTION (1 skill) |
| **Hold Logic** | ❌ KHÔNG CÓ (sample mỗi step) | ✅ Giữ K steps |
| **K Parameter** | Lưu nhưng KHÔNG DÙNG | Dùng cho option duration |
| **PPO log_prob** | Standard (không xét held) | log_prob=0 khi held |
| **Blend vs Option** | Blend TẤT CẢ skills | Chỉ 1 skill |

---

## 🔴 BUG 1: K, epsilon, tau KHÔNG ĐƯỢC SỬ DỤNG

### Code hiện tại (actor_critic_hrl.py)

```python
# Line 165-170: Lưu params nhưng KHÔNG dùng
self.K = 10  # Option duration (will be updated)
self.epsilon = 0.18
self.tau = 2.0

# Line 401-404: Update params từ curriculum
def update_curriculum_params(self, K, epsilon, tau):
    """Update curriculum parameters (for compatibility)"""
    self.K = K        # ← Lưu K
    self.epsilon = epsilon  # ← Lưu epsilon
    self.tau = tau    # ← Lưu tau
    # BUG: Không làm gì với các params này!
```

### Trong _actor() - KHÔNG có logic sử dụng K, epsilon, tau

```python
def _actor(self, observations):
    raw_mean = self.actor(observations)  # [B, num_output]
    
    # Split thành commands và weights
    masks = []
    for i in range(self.num_skills):
        mask = mask_to_low_level_policies[:, i*self.num_dofs:(i+1)*self.num_dofs]
        masks.append(mask)
    masks = torch.stack(masks, dim=1)
    masks = torch.softmax(masks, dim=1)  # ← Luôn softmax, KHÔNG xét K
    
    # Blend TẤT CẢ skills
    for i in range(self.num_skills):
        weighted_action = action_i * masks[:, i]  # ← Blend mọi skill
        means.append(weighted_action)
    
    actions_mean = sum(means)  # ← Sum of weighted actions
    
    # current_skill chỉ để logging
    avg_weights = masks.mean(dim=-1)
    self.current_skill = avg_weights.argmax(dim=-1)  # ← Chỉ argmax cho log!
```

### Vấn đề

**K không được dùng** → Robot KHÔNG giữ skill, chọn lại mỗi step
- Gây jitter (giật) giữa các skills
- Không có temporal abstraction (HRL mất ý nghĩa)
- Variance cao → học chậm

**epsilon không được dùng** → Không có exploration bonus
- Không đổi skill random theo epsilon
- Greedy 100% theo softmax

**tau không được dùng** → Temperature không ảnh hưởng
- Softmax temperature cố định (=1.0)
- Không thể điều chỉnh độ "sharp" của skill selection

---

## 🔴 BUG 2: SOFT BLEND thay vì HARD OPTION

### Code hiện tại

```python
# _actor() luôn blend TẤT CẢ skills:
for i in range(self.num_skills):
    action_i = self.policy_list[i](obs_for_skill_i)
    weighted_action = action_i * masks[:, i]  # weight từ softmax
    means.append(weighted_action)

actions_mean = sum(means)  # Blend = sum of weighted actions
```

### Vấn đề

**Soft blend KHÔNG phải HRL chuẩn:**
- HRL chuẩn: Chọn 1 skill, giữ K steps, rồi mới chọn lại
- Hiện tại: Blend tất cả skills mỗi step

**Hậu quả:**
1. **Actions conflicting**: Walk + Reach + Squat + Step cùng lúc → actions triệt tiêu nhau
2. **Robot ngã**: Legs muốn walk nhưng bị squat kéo xuống
3. **Không temporal coherence**: Skill switching jitter liên tục

---

## 🔴 BUG 3: PPO log_prob KHÔNG ĐÚNG cho HRL

### Code hiện tại

```python
# ppo_hrl.py update():
actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
# ↑ Standard log_prob cho continuous actions
```

```python
# actor_critic_hrl.py:
def get_actions_log_prob(self, actions):
    return self.distribution.log_prob(actions).sum(dim=-1)
    # ↑ Chỉ tính log_prob của final action, không xét skill selection
```

### Vấn đề

**HRL cần 3 log_probs:**
1. `log_prob_gating`: P(skill | obs) - Categorical
2. `log_prob_command`: P(command | obs, skill) - Normal
3. `log_prob_action`: P(action | obs, skill, command) - Normal (từ low-level)

**Hiện tại chỉ có:**
- `log_prob` của blended action distribution

**Khi skill được HELD (giữ từ step trước):**
- PPO KHÔNG nên penalize quyết định cũ
- `log_prob_gating` phải = 0 khi held

---

## ✅ Code đúng: actor_critic_hrl_simple.py (KHÔNG DÙNG)

### HoldTimeController

```python
class HoldTimeController:
    """
    Controls skill holding for K steps.
    PPO-correct: tracks which skills are newly sampled vs held.
    """
    def __init__(self, hold_steps=3, num_envs=4096, device='cuda'):
        self.hold_steps = hold_steps  # ← K
        self.current_skill = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.steps_remaining = torch.zeros(num_envs, dtype=torch.long, device=device)
    
    def should_sample(self):
        """Returns mask of envs that need new skill sample"""
        return self.steps_remaining <= 0
    
    def update(self, new_skill, sample_mask):
        """Update skill states"""
        # Only update where we're actually sampling
        self.current_skill = torch.where(sample_mask, new_skill, self.current_skill)
        
        # Reset timer where sampled, decrement otherwise
        self.steps_remaining = torch.where(
            sample_mask,
            torch.full_like(self.steps_remaining, self.hold_steps),
            self.steps_remaining - 1
        )
        
        is_held = ~sample_mask  # ← Track which are held
        return self.current_skill, is_held
```

### Forward với Hold Logic

```python
def forward(self, obs):
    # 1. Gating
    gating_logits, gating_probs = self.gating_head(features)
    gating_dist = Categorical(probs=gating_probs)
    
    # 2. Hold time logic
    if is_training:
        sample_mask = self.hold_time.should_sample()  # ← Check K
        skill_new = gating_dist.sample()
        skill_exec, is_held = self.hold_time.update(skill_new, sample_mask)  # ← Hold K steps
    else:
        skill_exec = gating_dist.sample()
        is_held = torch.zeros(batch_size, dtype=torch.bool)
    
    # 3. Execute SINGLE skill (HARD OPTION - no blend)
    base_action = self._execute_single_skill(obs, command_full, skill_exec)
    # ↑ Chỉ 1 skill, không blend!
```

### PPO log_prob với Held

```python
def get_log_prob(self, skill_exec, command_full, residual_raw, is_held):
    """PPO-CORRECT log_prob"""
    # 1. Gating log_prob (0 if held)
    log_prob_gating = self.last_gating_dist.log_prob(skill_exec)
    log_prob_gating = log_prob_gating * (~is_held).float()  # ← Zero out held!
    
    # 2. Command log_prob (only used slice)
    log_prob_cmd = torch.zeros(batch_size, device=self.device)
    for skill_id in range(self.num_skills):
        mask = (skill_exec == skill_id)
        if not mask.any():
            continue
        cmd_slice = self.SKILL_CMD_SLICES[skill_id]
        cmd_used = command_full[mask, cmd_slice]
        log_prob_slice = Normal(mean_slice, std_slice).log_prob(cmd_used).sum(dim=-1)
        log_prob_cmd[mask] = log_prob_slice
    
    # 3. Residual log_prob
    log_prob_res = self.last_residual_dist.log_prob(residual_raw).sum(dim=-1)
    
    return log_prob_gating + log_prob_cmd + log_prob_res
```

---

## 🔧 Giải pháp

### Option 1: Sử dụng ActorCriticHRLSimple (Khuyến nghị)

Thay đổi config:
```python
# h1_hrl.py:
policy_class_name = 'ActorCriticHRLSimple'  # Thay vì 'ActorCriticHRL'
```

Cần import:
```python
# on_policy_runner_hrl.py:
from rsl_rl.modules import ActorCriticHRLSimple
```

**Pros:**
- Đã có hold logic đúng
- PPO-correct log_prob
- Hard option (không blend)

**Cons:**
- Cần verify pretrained skills loading
- Cần adjust command slices

### Option 2: Fix ActorCriticHRL để dùng K, epsilon, tau

**Cần thêm vào _actor():**

```python
def _actor(self, observations):
    raw_mean = self.actor(observations)
    
    # Parse weights
    masks = torch.stack(masks, dim=1)
    
    # Apply temperature (tau) to softmax
    masks = torch.softmax(masks / self.tau, dim=1)  # ← Dùng tau!
    
    # epsilon-greedy exploration
    if self.training:
        random_mask = torch.rand(masks.shape[0], device=self.device) < self.epsilon
        random_skill = torch.randint(0, self.num_skills, (masks.shape[0],), device=self.device)
        # One-hot for random skill
        random_weights = F.one_hot(random_skill, self.num_skills).unsqueeze(-1)
        random_weights = random_weights.expand(-1, -1, self.num_dofs).float()
        masks = torch.where(random_mask.view(-1,1,1), random_weights, masks)
    
    # Hold logic (cần tracking state)
    if self.step_in_option is None:
        self.init_option_state(observations.shape[0])
    
    # Check if need to re-sample
    needs_resample = self.step_in_option >= self.K  # ← Dùng K!
    if needs_resample.any():
        # Re-sample skills for those envs
        self.step_in_option[needs_resample] = 0
    
    self.step_in_option += 1
    
    # ... rest of blending
```

**Cần fix PPO:**

```python
# ppo_hrl.py update():
# Store is_held trong transition
# Tính log_prob riêng cho gating, command, action
# Zero out log_prob_gating khi is_held
```

---

## 📊 So sánh Soft Blend vs Hard Option

| Aspect | Soft Blend (hiện tại) | Hard Option (cần) |
|--------|----------------------|-------------------|
| Skill/step | Blend 4 skills | 1 skill |
| Actions | Sum weighted | Single skill |
| K steps | N/A | Giữ skill K steps |
| Jitter | Cao (switch mỗi step) | Thấp (hold K steps) |
| PPO | Standard log_prob | Separate gating/cmd/action |
| Credit assignment | Khó (blend nào tốt?) | Dễ (1 skill = 1 reward) |
| Training | Chậm, unstable | Nhanh, stable |

---

## 🚨 Kết luận

**Bugs nghiêm trọng trong ActorCriticHRL:**

1. ❌ K, epsilon, tau KHÔNG ĐƯỢC DÙNG (chỉ lưu)
2. ❌ Soft blend thay vì hard option
3. ❌ PPO log_prob không xét held status

**Khuyến nghị:**
- **Chuyển sang ActorCriticHRLSimple** vì đã có logic đúng
- Hoặc fix ActorCriticHRL theo Option 2

---

## 📁 Files liên quan

| File | Vấn đề |
|------|--------|
| `actor_critic_hrl.py` | ❌ K,ε,τ không dùng, soft blend |
| `actor_critic_hrl_simple.py` | ✅ Có hold logic đúng (không dùng) |
| `ppo_hrl.py` | ⚠️ CurriculumController update K,ε,τ nhưng không được dùng |
| `on_policy_runner_hrl.py` | ⚠️ Track skill nhưng không có hold state |
| `h1_hrl.py` | Config dùng 'ActorCriticHRL' |
