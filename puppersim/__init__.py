import os

# Gymnasium is optional for MJX/Brax-only workflows.
try:
  import gymnasium as gym
  from gymnasium import register as register
except Exception:
  gym = None
  register = None


if register is not None:
  register(
    id='PupperGymEnv-v0',
    entry_point='puppersim.pupper_gym_env:PupperGymEnv',
    max_episode_steps=150,
    reward_threshold=5.0,
  )

  register(
    id='PupperCommandLocomotionEnv-v0',
    entry_point='puppersim.pupper_gym_env:PupperCommandLocomotionGymEnv',
    max_episode_steps=1000,
    reward_threshold=5.0,
  )

  register(
    id='ReacherEnv-v0',
    entry_point='puppersim.reacher.reacher_env:ReacherEnv',
    max_episode_steps=150,
    reward_threshold=5.0,
  )


def getPupperSimPath():
    resdir = os.path.join(os.path.dirname(__file__))
    return resdir


def getList():
    if gym is None:
      return []
    envs = [spec.id for spec in gym.envs.registry.all() if spec.id.find('Pupper') >= 0]
    return envs
