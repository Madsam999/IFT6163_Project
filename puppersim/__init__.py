import os
from gymnasium import register as register

register(
  id='PupperGymEnv-v0',
  entry_point='puppersim.pupper_gym_env:PupperGymEnv',
  max_episode_steps=150,
  reward_threshold=5.0,
)

register(
  id='ReacherEnv-v0',
  entry_point='puppersim.reacher.reacher_env:ReacherEnv',
  max_episode_steps=150,
  reward_threshold=5.0,
)

register(
    id='PupperGymEnvLong-v0',
    entry_point='puppersim.pupper_gym_env:PupperGymEnv',
    max_episode_steps=1000,
    reward_threshold=5.0,
)

def getPupperSimPath():
    resdir = os.path.join(os.path.dirname(__file__))
    return resdir


def getList():
    envs = [spec.id for spec in gym.envs.registry.all() if spec.id.find('Pupper') >= 0]
    return envs
