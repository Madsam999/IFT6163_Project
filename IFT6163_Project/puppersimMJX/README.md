# Pupper Sim MJX

MJX/Brax training and policy playback workflow for DJI Pupper v2.

Run commands from the repository root.

## Setup

MJX requires Python 3.10+.

```bash
conda create -n rl_pupper_mjx python=3.10 -y
conda activate rl_pupper_mjx
pip install -e .
pip install "jax[cpu]" brax mujoco flax orbax-checkpoint tyro imageio imageio-ffmpeg wandb
```

## Assets And XML

MJX assets live in `puppersimMJX/assets`.

For anything related to MJX XML generation, cameras, AprilTag textures, or regenerating asset files, see:

```text
puppersimMJX/xml_generation/README.md
```

## Train No-Pixel Policies

These commands use the Brax PPO trainer with compact camera-like features (`camera_nopriv`), not rendered pixel observations.

### Single Target

Baseline:

```bash
python puppersimMJX/pupper_train_ppo_brax.py \
  --env-name pupper_v2 \
  --backend mjx \
  --profile puppersimMJX/tasks/apriltag_walls/config/training_profiles/apriltag_walls_camera_nopriv_hl_level2.json \
  --num-timesteps 30000000
```

ICM:

```bash
python puppersimMJX/pupper_train_ppo_brax.py \
  --env-name pupper_v2 \
  --backend mjx \
  --profile puppersimMJX/tasks/apriltag_walls/config/training_profiles/apriltag_walls_camera_nopriv_hl_level2.json \
  --num-timesteps 30000000 \
  --icm-enabled \
  --icm-reward-weight 0.005 \
  --icm-reward-weight-final 0.0
```

### Multi Target

Baseline:

```bash
python puppersimMJX/pupper_train_ppo_brax.py \
  --env-name pupper_v2 \
  --backend mjx \
  --profile puppersimMJX/tasks/apriltag_walls/config/training_profiles/apriltag_walls_camera_nopriv_hl_level3.json \
  --num-timesteps 100000000
```

ICM:

```bash
python puppersimMJX/pupper_train_ppo_brax.py \
  --env-name pupper_v2 \
  --backend mjx \
  --profile puppersimMJX/tasks/apriltag_walls/config/training_profiles/apriltag_walls_camera_nopriv_hl_level3.json \
  --num-timesteps 100000000 \
  --icm-enabled \
  --icm-reward-weight 0.005 \
  --icm-reward-weight-final 0.0
```

## Test Pretrained Policies In Simulation

The packaged policy shortcuts are defined in `puppersimMJX/play_policy_sim.py`.

Low-level command locomotion:

```bash
python puppersimMJX/play_policy_sim.py --policy cc_locomotion
```

Single-target baseline:

```bash
python puppersimMJX/play_policy_sim.py --policy single_target_baseline_seed1
```

Single-target ICM:

```bash
python puppersimMJX/play_policy_sim.py --policy single_target_icm_seed1
```

Multi-target baseline:

```bash
python puppersimMJX/play_policy_sim.py --policy multi_target_baseline_seed3
```

Multi-target ICM:

```bash
python puppersimMJX/play_policy_sim.py --policy multi_target_icm_seed3
```

To close automatically after a fixed rollout duration, add `--simend`, for example:

```bash
python puppersimMJX/play_policy_sim.py --policy multi_target_icm_seed3 --simend 30
```

## Test A Custom Exported Bundle

For a direct low-level policy:

```bash
python puppersimMJX/play_policy_sim.py \
  --bundle-dir puppersimMJX/pretrained_policies/cc_locomotion/policy_bundle \
  --env-kwargs puppersimMJX/tasks/cc_locomotion/config/pupper_brax_env_kwargs.command_locomotion.json
```

For a hierarchical AprilTag policy:

```bash
python puppersimMJX/play_policy_sim.py \
  --bundle-dir puppersimMJX/pretrained_policies/single_target_icm_seed1/policy_bundle \
  --env-kwargs puppersimMJX/tasks/apriltag_walls/config/pupper_brax_env_kwargs.apriltag_walls_camera_nopriv_hl_level2.json
```

## Run On Robot

Deploy from your laptop:

```bash
./deploy_to_robot.sh
```

On the Pi shell (`/home/pi/puppersim_deploy`), run the low-level locomotion bundle:

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

## Troubleshooting

* If Brax/MJX imports fail, confirm Python is 3.10+ and reinstall the MJX dependencies above.
* If the MuJoCo viewer fails to open over SSH, run locally or configure display forwarding.
