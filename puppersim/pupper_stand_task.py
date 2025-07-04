"""A task to teach pupper to stand still"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import gin
from pybullet_envs.minitaur.envs_v2.tasks import task_interface
from pybullet_envs.minitaur.envs_v2.tasks import task_utils
from pybullet_envs.minitaur.envs_v2.tasks import terminal_conditions
from pybullet_envs.minitaur.envs_v2.utilities import env_utils_v2 as env_utils
from puppersim import pupper_v2

@gin.configurable
class SimpleStandTask(task_interface.Task):
    def __init__(self, 
                 weight=1.0,
                 min_com_height=0.0,
                 terminal_condition=terminal_conditions.default_terminal_condition_for_minitaur,
                 movement_penalty_coef = 1.0,
                 tilt_penalty_coef = 1.0,
                 four_feet_bonus = 3,
                 displacement_penalty_coef = 10,
                 orientation_penalty_coef = 10
                 ):
        
        self.weight = weight
        self._min_com_height = min_com_height
        self._terminal_condition = terminal_condition
        self._env = None
        self._step_count = 0
        self._last_motor_angles = None
        self._last_base_position = None
        self._last_base_orientation = None
        self._movement_penalty_coef = movement_penalty_coef
        self._tilt_penalty_coef = tilt_penalty_coef
        self._four_feet_bonus = four_feet_bonus
        self._displacement_penalty_coef = displacement_penalty_coef
        self._orientation_penalty_coef = orientation_penalty_coef
        
        self.TARGET_MOTOR_ANGLES = np.array(pupper_v2.Pupper.get_neutral_motor_angles())
        self.TARGET_ROLL_PITCH_YAW = np.zeros(3)

    def __call__(self, env):
        return self.reward(env)

    def reset(self, env):
        self._env = env
        self._last_motor_angles = self.TARGET_MOTOR_ANGLES
        self._last_base_position = np.array(env_utils.get_robot_base_position(self._env.robot))
        self._last_base_orientation = np.array(env_utils.get_robot_base_orientation(self._env.robot))

    def update(self, env):
        del env
        self._last_motor_angles = self._env.robot.motor_angles
        self._last_base_position = np.array(env_utils.get_robot_base_position(self._env.robot))
        self._last_base_orientation = np.array(env_utils.get_robot_base_orientation(self._env.robot))

    def reward(self, env):
        del env

        self._step_count += 1
        env = self._env

        current_motor_angles = self._env.robot.motor_angles
        reward = -(np.linalg.norm(self.TARGET_MOTOR_ANGLES - current_motor_angles))
        
        if reward > -0.3:
            reward += (0.2 - np.linalg.norm(self._last_motor_angles - current_motor_angles)) * self._movement_penalty_coef
            
        contact_points = self.count_contact_points()
        if contact_points == 4:
            reward += self._four_feet_bonus
            
        current_tilt = self._env.robot.base_roll_pitch_yaw
        reward += -(np.linalg.norm(self.TARGET_ROLL_PITCH_YAW - current_tilt)) * self._tilt_penalty_coef\
            
        displacement = np.linalg.norm(np.array(env_utils.get_robot_base_position(self._env.robot)) - self._last_base_position)
        reward -= displacement * self._displacement_penalty_coef
        
        orientation_diff = np.linalg.norm(np.array(env_utils.get_robot_base_orientation(self._env.robot)) - self._last_base_orientation)
        reward -= orientation_diff * self._orientation_penalty_coef
        
        
        return reward * self.weight

    def done(self, env):
        del env
        position = env_utils.get_robot_base_position(self._env.robot)
        if self._min_com_height and position[2] < self._min_com_height:
            return True
        return self._terminal_condition(self._env)


    @property
    def step_count(self):
        return self._step_count
    
    def count_contact_points(self):
        contact_forces = self._env.robot.feet_contact_forces()
        return sum(
                   not np.all(row == 0) for row in contact_forces
                  )