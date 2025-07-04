import gymnasium as gym
import time
import torch
from dataclasses import dataclass
import tyro
import puppersim
from pupper_train_ppo_cont_action import Agent
import numpy as np

@dataclass
class Args:
  model_path = ""
  
def make_env():
  def thunk():
    env = gym.make("PupperStandRobotGymEnv-v0")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env
  return thunk

if __name__ == "__main__":
  args = tyro.cli(Args)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  env = gym.vector.SyncVectorEnv([make_env()])
  agent = Agent(env).to(device)
  agent.load_state_dict(torch.load(args.model_path, map_location=device))
  
  returns = []

  obs, _ = env.reset()
  done = False
  totalr = 0.
  steps = 0
  start_time_wall = time.time()
  env_start_time_wall = time.time()
  last_spammy_log = 0.0
  while not done:
    start_time_wall = time.time()
    before_policy = time.time()
    action, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
    after_policy = time.time()
    
    next_obs, r, done, _, infos = env.step(action.cpu().numpy())
    
    totalr += r
    steps += 1
    obs = next_obs
    
  returns.append(totalr)

  print("returns: ", returns)
  print("mean return: ", np.mean(returns))
  print("std of return: ", np.std(returns))
    