# Pupper Sim MJX
MJX/Brax training workflow for DJI Pupper v2, split from the main `puppersim` package.

## System setup
### Operating system requirements
* Mac
* Linux
* Windows (untested, not recommended)

### Python requirements
MJX path requires Python 3.10+.

## Quick setup (Python 3.10+)
From the repository root:
```bash
conda create -n rl_pupper_mjx python=3.10 -y
conda activate rl_pupper_mjx
pip install -e .
pip install "jax[cpu]" brax mujoco flax orbax-checkpoint tyro imageio imageio-ffmpeg wandb
```

## Training
From the repository root run:
```bash
python puppersimMJX/pupper_train_ppo_brax.py --env-name pupper_v2 --backend mjx
```

### Simple Forward (50M steps)
```bash
python puppersimMJX/pupper_train_ppo_brax.py \
  --env-name pupper_v2 \
  --backend mjx \
  --profile puppersimMJX/tasks/simple_forward/config/training_profiles/simple_forward_locomotion.json \
  --num-timesteps 50000000
```

### Command Locomotion (300M steps)
```bash
python puppersimMJX/pupper_train_ppo_brax.py \
  --env-name pupper_v2 \
  --backend mjx \
  --profile puppersimMJX/tasks/cc_locomotion/config/training_profiles/command_locomotion.json \
  --num-timesteps 300000000
```
For command locomotion, ideally train for at least `500M+` steps for stronger tracking stability.

### Training profiles
The trainer supports modular profiles so reward/task setups can be switched without changing code:
```bash
python puppersimMJX/pupper_train_ppo_brax.py \
  --env-name pupper_v2 \
  --backend mjx \
  --profile puppersimMJX/tasks/simple_forward/config/training_profiles/simple_forward_locomotion.json
```

Available starter profiles:
* `puppersimMJX/tasks/simple_forward/config/training_profiles/simple_forward_locomotion.json`
* `puppersimMJX/tasks/simple_forward/config/training_profiles/simple_forward_command_policy.json`
* `puppersimMJX/tasks/cc_locomotion/config/training_profiles/command_locomotion.json`
* `puppersimMJX/tasks/cc_locomotion/config/training_profiles/navigation_controller_template.json`
* `puppersimMJX/tasks/navigation/config/training_profiles/sparse_navigation.json`
* `puppersimMJX/tasks/navigation/config/training_profiles/sparse_navigation_nav_controller.json`
* `puppersimMJX/tasks/navigation/config/training_profiles/sparse_collect_spheres.json`

Notes:
* A profile can define `env_kwargs` and optional randomization defaults.
* CLI args still override profile values (for example `--env-kwargs` or randomization flags).
* For navigation/controller experiments, duplicate the template profile and tune reward scales there.
* Reward selection is plug-in based: set `reward_module` + `reward_config` in `env_kwargs`.
* A reward module only needs to expose `build_reward(config)` and return `(reward, terms)` from the built callable.
* The sparse navigation profile uses hierarchical control: PPO outputs high-level `[vx, vy, yaw]` commands and a frozen low-level locomotion bundle maps these to motor targets.

Example env kwargs snippet:
```json
{
  "reward_module": "puppersimMJX.tasks.simple_forward.reward",
  "reward_requires_command": false,
  "reward_config": {
    "weight": 1.0,
    "clip_velocity": 0.0025,
    "forward_axis": "y_neg"
  }
}
```

### MJCF utility
Fix URDF mesh paths / MuJoCo tags:
```bash
python puppersimMJX/fix_urdf.py \
  --urdf_path puppersim/data/pupper_v2a.urdf \
  --output_path puppersim/data/pupper_v2a.fixed.urdf
```

Create stable MJX XML from a fixed URDF-converted XML:
```bash
python puppersimMJX/create_mujoco_xml.py
```

Add front camera to create `puppersim/data/pupper_v2_final_stable_cam.xml`:
```bash
python puppersimMJX/add_camera_to_xml.py \
  --xml-path puppersim/data/pupper_v2_final_stable.xml \
  --output puppersim/data/pupper_v2_final_stable_cam.xml \
  --force
```

Add a red goal marker site to create `puppersim/data/pupper_v2_final_stable_cam_goal.xml`:
```bash
python puppersimMJX/add_goal_marker_to_xml.py --force
```

## Run On Robot (After `deploy_to_robot.sh`)
From your laptop (repo root), deploy to Pi and open a shell there:
```bash
./deploy_to_robot.sh
```

On the Pi shell (`/home/pi/puppersim_deploy`):

1. Run policy on real robot (+ optional PiCam window when `DISPLAY` is available):
```bash
python puppersimMJX/play_policy_robot.py \
  --bundle-dir puppersimMJX/pretrained_policies/cc_locomotion/policy_bundle
```

Keyboard controls in `play_policy_robot.py`:
* `A/D`: x command
* `W/S`: y command
* `Q/E`: yaw command
* `R`: zero command
* `X`: exit

### Troubleshooting
<details>
<summary>Click to expand</summary>

* **ImportError for Brax/MJX packages**: ensure Python is 3.10+ and reinstall the MJX dependencies above.
* **PyBullet hangs from other workflows**: close stale simulator sessions or restart your machine.

</details>
<br/>
