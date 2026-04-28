"""State extraction utilities for locomotion rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


_EPS = 1e-6


@dataclass
class RewardState:
  """Container passed to reward terms."""

  command: np.ndarray
  dt: float
  base_position: np.ndarray
  base_orientation_rpy: np.ndarray
  base_orientation_quat: np.ndarray
  base_lin_vel_world: np.ndarray
  base_ang_vel_world: np.ndarray
  base_lin_vel_yaw_frame: np.ndarray
  motor_angles: np.ndarray
  motor_velocities: np.ndarray
  last_motor_velocities: np.ndarray
  motor_torques: np.ndarray
  action: np.ndarray
  last_action: np.ndarray
  last_last_action: np.ndarray
  foot_positions_world: np.ndarray
  foot_velocities_world: np.ndarray
  foot_contacts: np.ndarray
  foot_first_contacts: np.ndarray
  foot_air_time: np.ndarray
  contact_forces: np.ndarray
  undesired_contact_count: float
  terminated: bool


def _try_getattr(obj, name: str):
  try:
    return getattr(obj, name)
  except Exception:
    return None


def get_base_kinematics(env) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Returns base position/quaternion/rpy and linear/angular velocities in world frame."""
  robot = env.robot
  client = env.pybullet_client
  robot_id = getattr(robot, "robot_id", None)

  base_position = _try_getattr(robot, "base_position")
  base_quat = _try_getattr(robot, "base_orientation_quaternion")
  base_rpy = _try_getattr(robot, "base_roll_pitch_yaw")
  base_lin_vel = _try_getattr(robot, "base_velocity")
  base_ang_vel = _try_getattr(robot, "base_roll_pitch_yaw_rate")

  if robot_id is not None:
    if base_position is None or base_quat is None:
      pos, quat = client.getBasePositionAndOrientation(robot_id)
      if base_position is None:
        base_position = pos
      if base_quat is None:
        base_quat = quat
    if base_lin_vel is None or base_ang_vel is None:
      lin_vel, ang_vel = client.getBaseVelocity(robot_id)
      if base_lin_vel is None:
        base_lin_vel = lin_vel
      if base_ang_vel is None:
        base_ang_vel = ang_vel

  if base_rpy is None:
    if base_quat is not None:
      base_rpy = client.getEulerFromQuaternion(base_quat)
    else:
      base_rpy = (0.0, 0.0, 0.0)

  if base_position is None:
    base_position = (0.0, 0.0, 0.0)
  if base_quat is None:
    base_quat = (0.0, 0.0, 0.0, 1.0)
  if base_lin_vel is None:
    base_lin_vel = (0.0, 0.0, 0.0)
  if base_ang_vel is None:
    base_ang_vel = (0.0, 0.0, 0.0)

  return (
      np.asarray(base_position, dtype=np.float64),
      np.asarray(base_quat, dtype=np.float64),
      np.asarray(base_rpy, dtype=np.float64),
      np.asarray(base_lin_vel, dtype=np.float64),
      np.asarray(base_ang_vel, dtype=np.float64),
  )


def world_to_yaw_frame(vector_world: np.ndarray, yaw: float) -> np.ndarray:
  """Rotates world-frame vector into yaw-aligned body frame."""
  c = np.cos(yaw)
  s = np.sin(yaw)
  rot_inv = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
  return rot_inv @ np.asarray(vector_world, dtype=np.float64)


def get_foot_link_ids(env, fallback_num_feet: int = 4) -> np.ndarray:
  """Extracts end-effector link ids from the robot URDF loader when available."""
  robot = env.robot
  constants = None
  if hasattr(robot, "get_constants"):
    try:
      constants = robot.get_constants()
    except Exception:
      constants = None
  elif hasattr(robot, "constants"):
    try:
      constants = robot.constants()
    except Exception:
      constants = None

  preferred_names = []
  if constants is not None and hasattr(constants, "END_EFFECTOR_NAMES"):
    preferred_names = list(constants.END_EFFECTOR_NAMES)

  if hasattr(robot, "_urdf_loader"):
    loader = robot._urdf_loader
    if hasattr(loader, "get_end_effector_id_dict"):
      try:
        id_dict = loader.get_end_effector_id_dict()
        if preferred_names:
          ids = [id_dict[name] for name in preferred_names if name in id_dict]
          if ids:
            return np.asarray(ids, dtype=np.int32)
        if id_dict:
          return np.asarray(list(id_dict.values()), dtype=np.int32)
      except Exception:
        pass
    for attr_name in ("_end_effector_ids", "end_effector_ids"):
      ids = _try_getattr(loader, attr_name)
      if ids is not None and len(ids) > 0:
        return np.asarray(ids, dtype=np.int32)

  return np.arange(fallback_num_feet, dtype=np.int32)


def get_foot_states(
    env,
    foot_link_ids: Sequence[int],
    previous_positions: Optional[np.ndarray],
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Returns positions, velocities, contacts and normal forces for each foot."""
  robot_id = env.robot.robot_id
  client = env.pybullet_client

  num_feet = len(foot_link_ids)
  positions = np.zeros((num_feet, 3), dtype=np.float64)
  velocities = np.zeros((num_feet, 3), dtype=np.float64)
  contacts = np.zeros((num_feet,), dtype=bool)
  normal_forces = np.zeros((num_feet,), dtype=np.float64)

  for i, link_id in enumerate(foot_link_ids):
    link_state = client.getLinkState(robot_id, int(link_id), computeLinkVelocity=1)
    positions[i] = np.asarray(link_state[0], dtype=np.float64)
    if len(link_state) >= 8:
      velocities[i] = np.asarray(link_state[6], dtype=np.float64)
    elif previous_positions is not None:
      velocities[i] = (positions[i] - previous_positions[i]) / max(dt, _EPS)

    cps = client.getContactPoints(bodyA=robot_id, linkIndexA=int(link_id))
    in_contact = False
    max_normal_force = 0.0
    for cp in cps:
      body_b = cp[2] if len(cp) > 2 else -1
      normal_force = float(cp[9]) if len(cp) > 9 else 0.0
      if body_b != robot_id and normal_force > _EPS:
        in_contact = True
        if normal_force > max_normal_force:
          max_normal_force = normal_force
    contacts[i] = in_contact
    normal_forces[i] = max_normal_force

  return positions, velocities, contacts, normal_forces


def count_undesired_contacts(env, allowed_link_ids: Sequence[int]) -> float:
  """Counts non-foot contacts (used for collision penalty)."""
  robot_id = env.robot.robot_id
  allowed = set(int(i) for i in allowed_link_ids)
  cps = env.pybullet_client.getContactPoints(bodyA=robot_id)
  count = 0
  for cp in cps:
    body_b = cp[2] if len(cp) > 2 else -1
    link_a = cp[3] if len(cp) > 3 else -2
    normal_force = float(cp[9]) if len(cp) > 9 else 0.0
    if body_b == robot_id:
      continue
    if normal_force <= _EPS:
      continue
    if int(link_a) not in allowed:
      count += 1
  return float(count)
