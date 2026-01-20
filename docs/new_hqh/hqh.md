```
python legged_gym/scripts/train.py --task <task_name> --wandb <wandb_name> --run_name <run_name> --headless --sim_device cuda:<i> --rl_device cuda:<i>
```

```
python legged_gym/scripts/train.py --task h1_walking --wandb hqh_h1_walking --run_name h1_walking --headless --sim_device cuda:0 --rl_device cuda:0 --num_envs 4
```

```
python legged_gym/scripts/train.py --task h1_walking --wandb hqh_24_12_walking --run_name h1_walking --sim_device cuda:0 --rl_device cuda:0  --num_envs 4
```

python legged_gym/scripts/train.py --task h1_task_reach --run_name h1_task_reach --headless --sim_device cuda:0 --rl_device cuda:0

python legged_gym/scripts/train.py --task h1_task_button --run_name h1_task_button --headless --sim_device cuda:0 --rl_device cuda:0

python legged_gym/scripts/train.py --task h1_task_carry --run_name h1_task_carry --headless --sim_device cuda:0 --rl_device cuda:0

python legged_gym/scripts/play.py --task h1_reaching  --experiment_name h1_reaching  --load_run 0000_best --checkpoint -1 --sim_device cuda:0 --rl_device cuda:0 

python legged_gym/scripts/play.py --task h1_walking  --experiment_name h1_walking  --load_run 0000_best --checkpoint -1 --sim_device cuda:0 --rl_device cuda:0  --visualize

python legged_gym/scripts/play.py --task h1_task_carry  --experiment_name h1_task_carry  --load_run 0000_best --checkpoint -1 --sim_device cuda:0 --rl_device cuda:0  --visualize

# HRL Training & Play Commands
# ============================

# Train HRL (8 tasks: reach, button, cabinet, ball, box, transfer, lift, carry)
python legged_gym/scripts/train_hrl.py --task h1_hrl --run_name hrl_v8.1 --num_envs 4096 --max_iterations 100000 --sim_device cuda:0 --rl_device cuda:0 --headless --wandb hrl_v8

# Play HRL with GUI + video recording
python legged_gym/scripts/play_hrl.py --task h1_hrl --experiment_name h1_hrl --load_run v3 --checkpoint -1 --sim_device cuda:0 --rl_device cuda:0 --visualize

# Play HRL headless mode + video recording (no GUI, for remote server)
python legged_gym/scripts/play_hrl.py --task h1_hrl --experiment_name h1_hrl --load_run v3 --checkpoint 1000 --sim_device cuda:0 --rl_device cuda:0 --headless --record

# Play HRL with specific checkpoint (e.g., model_11755.pt)
python legged_gym/scripts/play_hrl.py --task h1_hrl --experiment_name h1_hrl --load_run hrl_v8.1 --checkpoint 11755 --sim_device cuda:0 --rl_device cuda:0 --visualize

# Play HRL headless + no recording (fast, just for testing)
python legged_gym/scripts/play_hrl.py --task h1_hrl --experiment_name h1_hrl --load_run hrl_v8.1 --checkpoint -1 --sim_device cuda:0 --rl_device cuda:0 --headless

h1_task_ball

<task_name>:
h1_reaching, h1_squatting, h1_stepping, h1_task_ball, h1_task_box, h1_task_button, h1_task_cabinet, h1_task_carry, h1_task_lift, h1_task_reach, h1_task_transfer, h1_walking

<wandb_name>: "type": str, "default": "h1_walking"
<run_name>: e.g. 0001_test. This is used to save the ckpt and log files.
--headless :run in headless mode (no GUI), no visualize 3D