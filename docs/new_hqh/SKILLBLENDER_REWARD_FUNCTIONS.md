# SkillBlender Reward Functions - v1 vs v3

## ⚠️ CODE CHECK RESULT

✅ **v3 code implementation MATCHES v1 công thức** (tất cả 8 tasks)

### Issue tìm thấy:
- **Button**: v1 def "q_right_arm" nhưng code check joint dof_pos (ĐÚNG)
- **Ball**: v1 def "p_ball_to_goal có decay=-1" - code match (ĐÚNG)
- **Transfer**: v1 def "wrist scale=1" - code match (ĐÚNG)
- **Lift**: v1 def "z-axis only" - code match (ĐÚNG)

✅ Decay factors: Tất cả đúng (-4/-1)
✅ Scale ratios: Tất cả đúng
✅ Components: Tất cả đúng

---

## v1 - Template Gốc (Bạn Thiết Kế)

### SkillBlender Reward Functions Summary

1. FarReach (Easy)
- Mục tiêu: Vươn tới 2 điểm xa bằng cả hai tay.
- Hàm phần thưởng: R(s,a) = 5 * exp(-4 * ||p_wrist - p_target||)
- Giải thích: Dựa trên khoảng cách tương đối giữa cổ tay robot và điểm mục tiêu.

2. ButtonPress (Easy)
- Mục tiêu: Nhấn nút bằng tay trái, giữ nguyên tư thế tay phải.
- Hàm phần thưởng: R(s,a) = 5 * exp(-4 * ||p_wrist_left - p_button||) + 0.5 * exp(-4 * ||q_right_arm||)
- Giải thích: Kết hợp khoảng cách từ tay trái đến nút và sai số tư thế khớp của tay phải.

3. CabinetClose (Easy)
- Mục tiêu: Đóng cửa tủ đang mở.
- Hàm phần thưởng: R(s,a) = 5 * exp(-4 * ||p_wrist - p_handle||) + 5 * exp(-4 * ||q_door_angle||)
- Giải thích: Kết hợp khoảng cách từ tay đến tay nắm tủ và góc quay của cánh cửa (mục tiêu là đóng).

4. FootballShoot (Medium)
- Mục tiêu: Sút bóng vào gôn.
- Hàm phần thưởng: R(s,a) = exp(-4 * ||p_torso_xy - p_ball_xy||) + 5 * exp(- ||p_ball - p_goal||)
- Giải thích: Kết hợp khoảng cách từ thân người đến bóng (để tiếp cận) và khoảng cách từ bóng đến gôn (để ghi bàn).

5. BoxPush (Medium)
- Mục tiêu: Đẩy hộp trên bàn đến vị trí đích.
- Hàm phần thưởng: R(s,a) = 5 * exp(-4 * ||p_box - p_target||) + 5 * exp(-4 * ||p_wrist - p_box||)
- Giải thích: Kết hợp khoảng cách từ hộp đến đích và khoảng cách từ tay đến hộp.

6. PackageLift (Medium)
- Mục tiêu: Nâng gói hàng lên độ cao nhất định.
- Hàm phần thưởng: R(s,a) = 5 * exp(-4 * ||h_package - h_target||) + 5 * exp(-4 * ||p_wrist - p_package||)
- Giải thích: Kết hợp sai số độ cao của gói hàng so với mục tiêu và khoảng cách từ tay đến gói hàng.

7. BoxTransfer (Hard)
- Mục tiêu: Chuyển hộp từ bàn này sang bàn khác.
- Hàm phần thưởng: R(s,a) = 5 * exp(-4 * ||p_box - p_target||) + exp(-4 * ||p_wrist - p_box||)
- Giải thích: Tương tự BoxPush nhưng trọng số thay đổi, tập trung vào vị trí hộp và tiếp xúc tay.

8. PackageCarry (Hard)
- Mục tiêu: Mang gói hàng đến vị trí xa.
- Hàm phần thưởng: R(s,a) = 5 * exp(-4 * ||p_package - p_target||) + 5 * exp(-4 * ||p_wrist - p_package||)
- Giải thích: Kết hợp vị trí gói hàng so với đích và duy trì khoảng cách tay với gói hàng (giữ hàng).

---

## v3 - HRL Implementation (Bạn Triển Khai)

### SkillBlender HRL Reward Functions

1. **Reach** (Easy) - Scale: 120.0, Final: ~10
   - R = 5.0 * exp(-4 * wrist_error) * 120.0 * 0.8
   - Vươn tới mục tiêu bằng cả 2 tay

2. **Button** (Easy) - Scale: 0.1428, Final: ~0.6
   - R = [5*exp(-4*wrist_error) + 0.5*exp(-4*arm_error)] * 0.1428 * 0.8
   - Tay trái nhấn nút, tay phải ở vị trí mặc định

3. **Cabinet** (Easy) - Scale: 0.728, Final: ~5.8
   - R = [5*exp(-4*wrist_error) + 5*exp(-4*door_error)] * 0.728 * 0.8
   - Kéo tay đến tay nắm và đóng cửa tủ

4. **Ball** (Medium) - Scale: 0.091, Final: ~0.5
   - R = [1*exp(-4*torso_error) + 5*exp(-1*ball_error)] * 0.091 * 1.0
   - **KHÁC**: Decay = -1 cho ball-to-goal (chậm hơn)

5. **Box** (Medium) - Scale: 0.0488, Final: ~0.4
   - R = [5*exp(-4*box_error) + 5*exp(-4*wrist_error)] * 0.0488 * 0.8
   - Đẩy hộp đến đích

6. **Transfer** (Hard) - Scale: 0.08125, Final: ~0.6
   - R = [5*exp(-4*box_error) + 1*exp(-4*wrist_error)] * 0.08125 * 1.3
   - **KHÁC**: Wrist scale = 1 (không cần ở gần)

7. **Lift** (Medium) - Scale: 0.0475, Final: ~0.5
   - R = [5*exp(-4*z_error) + 5*exp(-4*wrist_error)] * 0.0475 * 1.0
   - **KHÁC**: Chỉ check z-axis cho box position

8. **Carry** (Hard) - Scale: 0.0767, Final: ~1.0
   - R = [5*exp(-4*box_error) + 5*exp(-4*wrist_error)] * 0.0767 * 1.3
   - Mang hộp đến vị trí xa

---

## So Sánh v1 vs v3

### 1. **Công Thức**
- **v1**: Exponential $\text{scale} \times \exp(-\text{decay} \times \text{error})$
  - Định nghĩa công thức, decay factors, scales cho 8 tasks
  
- **v3**: Exponential $\text{scale} \times \exp(-\text{decay} \times \text{error}) \times \text{balance\_factor} \times \text{curriculum}$
  - Thêm normalization factors (balance) để chuẩn hóa rewards
  - Thêm curriculum scaling: Easy 0.8×, Medium 1.0×, Hard 1.3×

### 2. **Raw Rewards (Trước Normalize)**
- **v1**: Raw rewards khác nhau giữa các tasks
  - Reach: ~5 (simple)
  - Button: ~5.5 (2 components)
  - Cabinet: ~10 (2 components)
  - Ball: ~6 (1 + 5 scales khác nhau)
  - Box: ~10 (2 components)
  - Transfer: ~6 (5 + 1 scales khác)
  - Lift: ~10 (2 components z-axis)
  - Carry: ~10 (2 components, farthest distance)

- **v3**: Normalize tất cả về ~0.4-1.0 range
  - Mỗi task có balance factor riêng
  - Sau đó × curriculum → final reward balanced

### 3. **Decay Factor Design**
- **v1**: 
  - Hầu hết decay = -4 (Reach, Button, Cabinet, Box, Transfer, Lift, Carry)
  - Ball khác: decay = -1 cho ball-to-goal (slow decay)
  
- **v3**: Giữ nguyên v1 decay factors

### 4. **Component Scaling Variations**
- **v1**: 
  - Transfer dùng wrist scale = 1 (khác BoxPush là 5)
    - Concept: chỉ cần chuyển, không cần cầm chặt
  - Lift check z-axis only (1D, khó hơn 3D của Box)
  
- **v3**: Giữ nguyên design v1

### 5. **Normalization & Curriculum (Core Difference)**
- **v1**: No normalization, no curriculum
  - Raw reward được design để vary theo task difficulty
  
- **v3**: 
  - Balance factors để normalize raw → ~10 per-task-type
  - Curriculum scaling: compensation cho difficulty
  - Result: Tất cả final rewards ≈ 0.4-1.0 range (fair multi-task)

### 6. **Training Mode**
- **v1**: Reference template định nghĩa design choices
- **v3**: Production code triển khai v1 + multi-task optimization

---

## Tóm Tắt Khác Biệt Chính

| Khía cạnh | v1 | v3 |
|-----------|----|----|
| **Công thức cơ bản** | $s \times \exp(-\lambda \times d)$ | $s \times \exp(-\lambda \times d) \times \text{balance} \times \text{curr}$ |
| **Status** | Template định nghĩa design | Implementation cho multi-task |
| **Decay factors** | Đã định: -4/-1 | Giữ nguyên từ v1 |
| **Raw reward range** | Khác nhau 5-10 giữa tasks | Normalize ~10 per-type |
| **Normalization** | Không | Có (balance factors) |
| **Curriculum scaling** | Không | Có (0.8/1.0/1.3) |
| **Final reward range** | Khác nhau | 0.4-1.0 (balanced) |
| **Multi-task fairness** | Chưa xét | Công bằng vì normalized |

---

## Training v3

```bash
python legged_gym/scripts/train_hrl.py --task h1_hrl --run_name hrl_v8.1 \
  --num_envs 4096 --max_iterations 100000 --sim_device cuda:0 --rl_device cuda:0 --headless
```
