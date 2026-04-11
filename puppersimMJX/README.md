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
* `puppersimMJX/tasks/cc_locomotion/config/training_profiles/command_locomotion.json`
* `puppersimMJX/tasks/cc_locomotion/config/training_profiles/navigation_controller_template.json`

Notes:
* A profile can define `env_kwargs` and optional randomization defaults.
* CLI args still override profile values (for example `--env-kwargs` or randomization flags).
* For navigation/controller experiments, duplicate the template profile and tune reward scales there.
* Reward selection is plug-in based: set `reward_module` + `reward_config` in `env_kwargs`.
* A reward module only needs to expose `build_reward(config)` and return `(reward, terms)` from the built callable.

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

### Troubleshooting
<details>
<summary>Click to expand</summary>

* **ImportError for Brax/MJX packages**: ensure Python is 3.10+ and reinstall the MJX dependencies above.
* **PyBullet hangs from other workflows**: close stale simulator sessions or restart your machine.

</details>
<br/>
