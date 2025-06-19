"""A simple standing task and termination condition."""

from __future__ import absolute_import,division,print_function

import numpy as np
import pybullet as p

import gin

from pybullet_envs.minitaur.envs_v2.tasks import task_interface
from pybullet_envs.minitaur.envs_v2.tasks import task_utils
from pybullet_envs.minitaur.envs_v2.tasks import terminal_conditions
from pybullet_envs.minitaur.envs_v2.utilities import env_utils_v2 as env_utils
from puppersim import pupper_v2


@gin.configurable
class StandingTask(task_interface.Task):
  """A basic "standing" task.  We want the robot to stand up and stay still."""

  def __init__(self,
               weight=1.0,
               terminal_condition=terminal_conditions.default_terminal_condition_for_minitaur,
               energy_penalty_coef=0.0,
               torque_penalty_coef=0.0,
               min_com_height=0.16, #Initial position = 0.17
               upright_threshold=0.85):
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
    base_pos = env_utils.get_robot_base_position(robot)
    base_orientation = env_utils.get_robot_base_orientation(robot)
    rot_matrix = p.getMatrixFromQuaternion(base_orientation)
    up_z = rot_matrix[8]
    
    # Upright and high enough
    upright = float(up_z > self._upright_threshold and base_pos[2] > self._min_com_height)
    reward = upright
    
    # Better if more parallel to the floor
    reward += up_z * 0.1
    
    if self._initial_base_pos is not None:
      # Calculate drift of x,y coordinates based on initial position
      drift = np.linalg.norm(base_pos[:2] - self._initial_base_pos[:2])
      reward -= 0.5 * drift   # Arbitrary coefficient, may need modification
      
    if hasattr(robot, "get_neutral_motor_angles") and hasattr(robot, "motor_angles"):
      neutral_angles = np.array(pupper_v2.Pupper.get_neutral_motor_angles())
      current_angles = np.array(robot.motor_angles)
      joint_deviation = np.linalg.norm(current_angles - neutral_angles)
      reward -= 0.1 * joint_deviation  # Arbitrary coefficient, may need modification

    # Energy
    if self._energy_penalty_coef > 0:
      energy = np.sum(np.abs(robot.motor_torques) * np.abs(robot.motor_velocities))
      reward -= self._energy_penalty_coef * energy

    if self._torque_penalty_coef > 0:
      torque_penalty = np.sum(np.square(robot.motor_torques))
      reward -= self._torque_penalty_coef * torque_penalty

    return reward * self._weight

  def done(self, env):
    """Ends if robot falls below min height or off balance (uprightness)"""
    del env
    
    robot = self._env.robot
    base_pos = env_utils.get_robot_base_position(robot)
    base_orientation = env_utils.get_robot_base_orientation(robot)
    rot_matrix = p.getMatrixFromQuaternion(base_orientation)
    up_z = rot_matrix[8]
    if base_pos[2] < self._min_com_height or up_z < self._upright_threshold:
        return True
    return self._terminal_condition(self._env)

  @property
  def step_count(self):
    return self._step_count