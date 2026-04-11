# Pupper Sim
Simulation and Reinforcement Learning for DJI Pupper v2 Robot

## System setup
### Operating system requirements
* Mac
* Linux
* Windows (untested, not recommended)

### Mac-only setup
Install xcode command line tools.
```bash
xcode-select --install
```
If you already have the tools installed you'll get an error saying so, which you can ignore.

### Conda setup
Install [miniconda](https://docs.conda.io/en/latest/miniconda.html), then
```
conda create --name rl_pupper python=3.7
conda activate rl_pupper
```

## Getting the code ready
```bash
mkdir -p ~/pupper_ws/src
cd ~/pupper_ws/src
git clone --recurse-submodules git@github.com:montrealrobotics/puppersim.git
cd puppersim
pip install -e .
```

You will also need to use this version of pybullet:
```bash
cd ~/pupper_ws/src
git clone https://github.com/montrealrobotics/bullet3.git
cd bullet3
pip install -e .
```

Then to verify the installation, run
```bash
python3 puppersim/pupper_example.py
```
You should see the PyBullet GUI pop up and see Pupper doing an exercise.

<details>
  <summary>Click for instructions if pupper_example.py is running slowly</summary>

  Stop `pupper_example.py`. Then run
  ```bash
  python3 puppersim/pupper_minimal_server.py
  ```
  then in a new terminal tab/window
  ```bash
  python3 puppersim/pupper_example.py --render=False
  ```
  This runs the visualizer GUI and simulator as two separate processes.
</details>
<br/>

## Training
From the outer puppersim folder run:
```bash
python puppersim/pupper_train_ppo_cont_action.py --seed 1 --env-id PupperGymEnv-v0 
--total-timesteps 5000 --save-model --capture_video
```
Depending on your computer specs, each training iteration will take around 1 - 5 seconds.

### MJX/Brax PPO training
You can also train Pupper v2 with Brax PPO on MJX backends:
```bash
pip install "jax[cpu]" brax mujoco flax orbax-checkpoint tyro
python puppersimMJX/pupper_train_ppo_brax.py --env-name pupper_v2 --backend mjx
```

Requires Python 3.10+ (MJX is not supported on Python 3.7).

The trainer uses local Pupper v2 assets from this repo only.
It auto-generates `puppersim/data/pupper_v2a_mjx.xml` from
`puppersim/data/pupper_v2a.urdf` if the MJCF file is missing.

If you want to generate/update the MJCF manually:
```bash
python puppersim/pupper_v2_urdf_to_mjcf.py \
  --urdf-path puppersim/data/pupper_v2a.urdf \
  --output-path puppersim/data/pupper_v2a_mjx.xml
```
`pupper_v2_urdf_to_mjcf.py` applies MJX-compatible geometry/contact settings by default.

### Troubleshooting
<details>
<summary>Click to expand</summary>

* **Pybullet hangs when starting training**. Possible issue: You have multiple suspended pybullet clients. Solution: Restart your computer. 
</details>
<br/>

### Protocol for saving policies
<details>
<summary>Click to expand</summary>

If you want to save a policy, create a folder within `puppersim/data` with the type of gait and date, eg `pretrained_trot_1_22_22`. From the `data` folder, copy the following files into the folder you just made.


* The `.npz` policy file you want, e.g. `lin_policy_plus_latest.npz`
* `log.txt`
* `params.json`

From `puppersim/config` also copy the `.gin` file you used to train the robot, e.g. `pupper_pmtg.gin` file into the folder you just made. When you run a policy on the robot, make sure your `pupper_robot_*_.gin` file matches the `pupper_pmtg.gin` file you saved.

Then add a `README.md` in the folder with a brief description of what you did, including your motivation for saving this policy. 
</details>
<br/>

## Test a policy
TODO


## Deployment
### Prerequisites
<details>
<summary>Linux</summary>

Set up Avahi (once per computer)
```
sudo apt install avahi-*
```
Run the following, you should see Pupper's IP address
```
avahi-resolve-host-name raspberrypi.local -4
```
Setup the zero password login for your pupper (once per computer) (original raspberry pi password: raspberry)
```
ssh-keygen
cat ~/.ssh/id_rsa.pub | ssh pi@`avahi-resolve-host-name raspberrypi.local -4 | awk '{print $2}'` 'mkdir .ssh/ && cat >> .ssh/authorized_keys'
```
</details>
<details>
<summary>Mac</summary>

Setup the zero password login for your pupper (only once per computer) (original raspberry pi password: raspberry)

Once per computer, run
```
ssh-keygen
cat ~/.ssh/id_rsa.pub | ssh pi@raspberrypi.local 'mkdir -p .ssh/ && cat >> .ssh/authorized_keys'
```
</details>
<br/>

### Run pretrained policy on Pupper
* Turn on the Pupper robot, wait for it to complete the calibration motion.
* Connect your laptop with the Pupper using an USB-C cable
* For legacy PyBullet policies (`lin_policy_plus_*.npz`), run:
```bash
./deploy_to_robot.sh python3 puppersim/puppersim/pupper_ars_run_policy.py --expert_policy_file=puppersim/data/lin_policy_plus_latest.npz --json_file=puppersim/data/params.json --run_on_robot
```

### Run Brax PPO locomotion policy on the real robot (simple flow)
This is the recommended path for checkpoints like:
`puppersim/data/pretrained_cc_locomotion/pupper_train_ppo_brax.params`

1. Export a policy bundle (only needed once per checkpoint):
```bash
python3 puppersimMJX/pupper_brax_export_policy_bundle.py \
  --params-path puppersim/data/pretrained_cc_locomotion/pupper_train_ppo_brax.params \
  --output-dir puppersim/data/pretrained_cc_locomotion
```

2. Deploy and run on robot:
```bash
./deploy_to_robot.sh python3 puppersimMJX/pupper_brax_run_policy_robot.py \
  --checkpoint-dir puppersim/data/pretrained_cc_locomotion \
  --seconds 60 \
  --command-x 0.35 \
  --command-y 0.0 \
  --command-yaw 0.0
```

Realtime command options:
* Fixed command (default): keep `--command-source fixed` (or omit).
* Keyboard control:
```bash
./deploy_to_robot.sh python3 puppersimMJX/pupper_brax_run_policy_robot.py \
  --checkpoint-dir puppersim/data/pretrained_cc_locomotion \
  --command-source keyboard \
  --keyboard-control-mode hold \
  --seconds 120
```
Controls: `W/S` -> `vx`, `A/D` -> `vy`, `Q/E` -> yaw, `SPACE` -> zero, `X` -> stop.

* HID joystick control (FrSky/BetaFPV, via `puppersim/JoystickInterface.py`):
```bash
./deploy_to_robot.sh python3 puppersimMJX/pupper_brax_run_policy_robot.py \
  --checkpoint-dir puppersim/data/pretrained_cc_locomotion \
  --command-source joystick \
  --seconds 120
```

Notes:
* The runner also supports `--auto-export` from `.params` if bundle files are missing.
* Start with small commands (`--command-x 0.2` to `0.35`) and short runs (`--seconds 15`) first.
* Control rate defaults to `--policy-dt 0.02` (50 Hz), matching the command-locomotion Brax setup.

## Simulating the heuristic controller
<details>
  <summary>Click to expand</summary>
  Navigate to the outer puppersim folder and run
  
  ```bash
  python3 puppersim/pupper_server.py
  ```

  Clone the the [heuristic controller](https://github.com/stanfordroboticsclub/StanfordQuadruped.git):
  ```bash
  git clone https://github.com/stanfordroboticsclub/StanfordQuadruped.git
  cd StanfordQuadruped
  git checkout dji
  ```
  In a separate terminal, navigate to StanfordQuadruped and run 
  ```bash
  python3 run_djipupper_sim.py
  ```

  Keyboard controls:
  * wasd --> moves robot forward/back and left/right
  * arrow keys --> turns robot left/right
  * q --> activates/deactivates robot
  * e --> starts/stops trotting gait
  * ijkl --> tilts and raises robot
</details>
