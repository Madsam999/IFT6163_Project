"""Reward term functions for command locomotion."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from puppersim.pupper_tasks.locomotion_task.rewards.state import RewardState


_EPS = 1e-6


def _command_norm(state: RewardState) -> float:
  return float(np.linalg.norm(state.command[:3]))


def track_lin_vel_xy_exp(state: RewardState, tracking_sigma: float = 0.25) -> float:
  """Exponential tracking reward for commanded x/y velocity in yaw frame."""
  error = np.sum(np.square(state.command[:2] - state.base_lin_vel_yaw_frame[:2]))
  return float(np.exp(-error / (tracking_sigma + _EPS)))


def track_ang_vel_z_exp(state: RewardState, tracking_sigma: float = 0.25) -> float:
  """Exponential tracking reward for commanded yaw rate."""
  error = np.square(state.command[2] - state.base_ang_vel_world[2])
  return float(np.exp(-error / (tracking_sigma + _EPS)))


def lin_vel_z_l2(state: RewardState) -> float:
  return float(np.square(state.base_lin_vel_world[2]))


def ang_vel_xy_l2(state: RewardState) -> float:
  return float(np.sum(np.square(state.base_ang_vel_world[:2])))


def orientation_flatness_l2(state: RewardState) -> float:
  """Penalizes roll/pitch away from flat base."""
  roll, pitch = state.base_orientation_rpy[:2]
  return float(roll * roll + pitch * pitch)


def torques_l2(state: RewardState) -> float:
  return float(np.sum(np.square(state.motor_torques)))


def mechanical_work_l1(state: RewardState) -> float:
  return float(np.sum(np.abs(state.motor_torques * state.motor_velocities)))


def joint_acceleration_l2(state: RewardState) -> float:
  accel = (state.motor_velocities - state.last_motor_velocities) / max(state.dt, _EPS)
  return float(np.sum(np.square(accel)))


def action_rate_l2(state: RewardState) -> float:
  return float(np.sum(np.square(state.action - state.last_action)))


def feet_air_time(
    state: RewardState,
    minimum_airtime: float = 0.10,
    command_threshold: float = 0.05,
) -> float:
  if _command_norm(state) <= command_threshold:
    return 0.0
  return float(np.sum((state.foot_air_time - minimum_airtime) * state.foot_first_contacts.astype(np.float64)))


def foot_slip_l2(state: RewardState) -> float:
  horizontal_sq_speed = np.sum(np.square(state.foot_velocities_world[:, :2]), axis=1)
  return float(np.sum(horizontal_sq_speed * state.foot_contacts.astype(np.float64)))


def abduction_angle_l2(
    state: RewardState,
    desired_abduction_angles: Optional[Sequence[float]] = None,
) -> float:
  abduction = state.motor_angles[0::3]
  if desired_abduction_angles is None:
    desired = np.zeros_like(abduction)
  else:
    desired = np.asarray(desired_abduction_angles, dtype=np.float64)
  return float(np.sum(np.square(abduction - desired)))


def stand_still_pose_l1(
    state: RewardState,
    default_pose: Sequence[float],
    command_threshold: float = 0.05,
) -> float:
  if _command_norm(state) >= command_threshold:
    return 0.0
  default_pose = np.asarray(default_pose, dtype=np.float64)
  return float(np.sum(np.abs(state.motor_angles - default_pose)))


def undesired_collision_count(state: RewardState) -> float:
  return float(state.undesired_contact_count)


def termination(state: RewardState) -> float:
  return float(state.terminated)

