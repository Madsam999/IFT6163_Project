# Submission Notes

The main code changes for the MJX workflow are inside `puppersimMJX/`.

This folder contains the MJX/Brax environment, training scripts, task profiles,
XML generation utilities, assets, and pretrained policy bundles used for the
experiments.

## Pipeline

The pipeline we followed was:

1. Start from the Pupper v2 URDF.
2. Fix the URDF for MuJoCo/MJX compatibility.
3. Generate a stable MJX XML model.
4. Add the front camera to the MJX XML.
5. Generate AprilTag room assets when training navigation tasks.
6. Train policies using task profiles in `puppersimMJX/tasks/.../training_profiles`.
7. Export policy bundles and test them with `puppersimMJX/play_policy_sim.py`.

The XML and asset generation commands are documented in:

```text
puppersimMJX/xml_generation/README.md
```

## Profiles And Levels

The MJX training profiles are JSON files that configure the environment,
reward, observations, low-level policy usage, and PPO defaults without changing
the training code.

The low-level locomotion profile trains the base command-following controller.
Higher-level AprilTag policies reuse this locomotion policy and output velocity
commands instead of direct motor actions.

AprilTag levels:

* Level 1: test setup for non-privileged camera-like observations.
* Level 2: single-target navigation task.
* Level 3: multi-target navigation task.
* Level 4: more complex multi-room map. The code/profile exists, but we did not
  have time to fully test or tune this level.

There is also pixel-based MJX training code in
`puppersimMJX/pupper_train_ppo_pixels_mjx.py`. This path uses rendered camera
pixels instead of compact camera-like features, but we did not have enough time
to experiment with it properly.

## Main Entry Points

Training:

```text
puppersimMJX/pupper_train_ppo_brax.py
```

Simulation policy playback:

```text
puppersimMJX/play_policy_sim.py
```

Robot policy playback:

```text
puppersimMJX/play_policy_robot.py
```

The practical commands for training and testing pretrained policies are listed
in:

```text
IFT6163_Project/README.md
```
