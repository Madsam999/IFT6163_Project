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

def create_pupper_env(config_file: Optional[str] = None, enable_rendering: bool = False):
  CONFIG_DIR = puppersim.getPupperSimPath()
  if config_file is None:
    config_file = os.path.join("config", "pupper_pmtg.gin")
  if not os.path.isabs(config_file):
    config_file = os.path.join(CONFIG_DIR, config_file)
  #  _NUM_STEPS = 10000
  #  _ENV_RANDOM_SEED = 2

  gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath()+"/")
  gin.parse_config_file(config_file)
  # Override rendering from caller after parsing config so it can supersede
  # the file default.
  gin.bind_parameter("SimulationParameters.enable_rendering", bool(enable_rendering))
  env = env_loader.load()
  return env


class PupperGymEnv(Env):
  metadata = {
    "render_modes": ["human", "ansi", "rgb_array"],
    "render_fps": 50,
  }

  def __init__(self, render_mode: Optional[str] = None, render=False, config_file: Optional[str] = None):
    # Keep GUI rendering disabled for rgb_array mode (used by training video
    # capture) unless explicitly requested via render=True.
    enable_rendering = bool(render) or render_mode == "human"
    self.env = create_pupper_env(config_file=config_file, enable_rendering=enable_rendering)
    self.observation_space = self.env.observation_space
    self.action_space = self.env.action_space
    self._is_render = render
    self.render_mode = render_mode
    self._cached_task = None

  #def _configure(self, display=None):
  #  self.display = display

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    s = int(seed) & 0xffffffff
    self.env.seed(s)

    return [seed]

  def step(self, action):
    retval = self.env.step(action)
    if isinstance(retval, tuple) and len(retval) == 5:
      observation, reward, terminated, truncated, info = retval
    elif isinstance(retval, tuple) and len(retval) == 4:
      observation, reward, done, info = retval
      terminated = bool(done)
      truncated = False
    else:
      raise ValueError("Unexpected step return format from wrapped env.")

    if info is None:
      info = {}
    else:
      info = dict(info)

    task = self._find_task()
    if task is not None and hasattr(task, "last_reward_terms"):
      try:
        reward_terms = task.last_reward_terms
        for key, value in reward_terms.items():
          info[f"reward/{key}"] = float(value)
      except Exception:
        pass

    return observation, reward, terminated, truncated, info

  def reset(self, seed=None, options=None):
    retval = self.env.reset()
    if isinstance(retval, tuple) and len(retval) == 2:
      observation, info = retval
      if info is None:
        info = {}
      return observation, dict(info)
    return retval, {}

  def update_weights(self, weights):
    self.env.update_weights(weights)

  def render(self, mode='rgb_array', close=False,  **kwargs):
    return self.env.render(mode)

  def configure(self, args):
    self.env.configure(args)

  def close(self):
    self.env.close()

  def _find_task(self):
    if self._cached_task is not None:
      return self._cached_task

    current = self.env
    for _ in range(20):
      if current is None:
        break
      for attr_name in ("task", "_task"):
        task = getattr(current, attr_name, None)
        if task is not None:
          self._cached_task = task
          return task
      next_env = getattr(current, "env", None)
      if next_env is current:
        break
      current = next_env
    return None


class PupperCommandLocomotionGymEnv(PupperGymEnv):
  """Gymnasium wrapper that loads the command locomotion gin config."""

  def __init__(self, render_mode: Optional[str] = None, render=False):
    super().__init__(
      render_mode=render_mode,
      render=render,
      config_file=os.path.join("pupper_tasks", "locomotion_task", "config", "pupper_pmtg_command_locomotion.gin"),
    )

    
