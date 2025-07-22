"""A simple standing task and termination condition."""

from __future__ import absolute_import,division,print_function

import numpy as np
import pybullet as p
import math

import gin

from pybullet_envs.minitaur.envs_v2.tasks import task_interface
from pybullet_envs.minitaur.envs_v2.tasks import task_utils
from pybullet_envs.minitaur.envs_v2.tasks import terminal_conditions
from pybullet_envs.minitaur.envs_v2.utilities import env_utils_v2 as env_utils
from puppersim import pupper_v2


@gin.configurable
class PupperFourLegStandTask(task_interface.Task):
  """
    A basic "standing" task.  We want the robot to stand up and stay still.
    This is a second version heavily based on mintaur_four_leg_stand_env.py
  """

  def __init__(self,
               weight=1.0,
               terminal_condition=terminal_conditions.default_terminal_condition_for_minitaur,
               use_angular_velocity_in_observation=False,
               user_motor_angle_in_observation=False,
               energy_penalty_coef=0.0,
               torque_penalty_coef=0.0,
               min_com_height=0.16, #Initial position = 0.17
               upright_threshold=0.85
               ):
    """Initializes the task.

    Args:
      weight: Float. The scaling factor for the reward.
      terminal_condition: Callable object or function. Determines if the task is
        done.
      energy_penalty_coef: Penalty for energy usage.
      torque_penalty_coef: Penalty for torque usage.
      min_com_height: Minimum height for the center of mass of the robot that
        will be used to terminate the task. This is used to obtain task specific
        gaits and set by the config or gin files based on the task and robot.
      upright_threshold: Minimum z-component of up vector for 'upright'.

    Raises:
      ValueError: The energy coefficient is smaller than zero.
    """
    self._weight = weight
    self._terminal_condition = terminal_condition
    self.use_angular_velocity_in_observation=use_angular_velocity_in_observation
    self.use_motor_angle_in_observation=user_motor_angle_in_observation
    self._min_com_height = min_com_height
    self._upright_threshold = upright_threshold
    self._energy_penalty_coef = energy_penalty_coef
    self._torque_penalty_coef = torque_penalty_coef
    self._env = None
    self._step_count = 0
    self._initial_base_pos = None   # Drift penalty

    if energy_penalty_coef < 0:
      raise ValueError("Energy Penalty Coefficient should be >= 0")

  def __call__(self, env):
    return self.reward(env)

  def reset(self, env):
    self._env = env
    robot = self._env.robot
    # Base position for drift calculations
    self._initial_base_pos = np.array(env_utils.get_robot_base_position(robot))
    
  def update(self, env):
    pass

  def reward(self, env):
    """Reward for standing and stable."""
    del env
    self._step_count += 1
    
    
    robot = self._env.robot
    roll, pitch, yaw = robot.base_roll_pitch_yaw
    return 1.0/ (0.001 + math.fabs(roll) + math.fabs(pitch) + math.fabs(yaw))

  def done(self, env):
    """Ends if robot falls below min height or off balance (uprightness)"""
    del env
    
    robot = self._env.robot
    base_pos = env_utils.get_robot_base_position(robot)
    if base_pos[2] < self._min_com_height:
        return True
    return self._terminal_condition(self._env)

  @property
  def step_count(self):
    return self._step_count