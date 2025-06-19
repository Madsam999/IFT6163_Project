import math
import gymnasium
from gymnasium import Env, spaces
from gymnasium.utils import seeding
import numpy as np
import puppersim
import os
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd
from typing import Optional

def create_pupper_env():
  CONFIG_DIR = puppersim.getPupperSimPath()
  _CONFIG_FILE = os.path.join(CONFIG_DIR, "config", "pupper_pmtg_standing.gin")
  #  _NUM_STEPS = 10000
  #  _ENV_RANDOM_SEED = 2

  gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath()+"/")
  gin.parse_config_file(_CONFIG_FILE)
  env = env_loader.load()
  return env


class PupperGymEnv(Env):
  metadata = {
    "render_modes": ["human", "ansi", "rgb_array"],
    "render_fps": 50,
  }

  def __init__(self, render_mode: Optional[str] = None, render=False):
    self.env = create_pupper_env()
    self.observation_space = self.env.observation_space
    self.action_space = self.env.action_space
    self._is_render = render
    self.render_mode = render_mode

  #def _configure(self, display=None):
  #  self.display = display

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    s = int(seed) & 0xffffffff
    self.env.seed(s)

    return [seed]

  def step(self, action):
    retval = self.env.step(action)
    retval = retval + ({}, )
    return retval

  def reset(self, seed=None, options=None):
    retval = self.env.reset()
    return retval, {}

  def update_weights(self, weights):
    self.env.update_weights(weights)

  def render(self, mode='rgb_array', close=False,  **kwargs):
    return self.env.render(mode)

  def configure(self, args):
    self.env.configure(args)

  def close(self):
    self.env.close()

    