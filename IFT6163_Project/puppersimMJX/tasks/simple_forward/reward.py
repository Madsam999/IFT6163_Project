"""Simple forward locomotion reward for Pupper Brax/MJX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import jax.numpy as jp


@dataclass(frozen=True)
class SimpleForwardRewardConfig:
    """Config analogous to the PyBullet simple-forward task."""

    weight: float = 1.0
    divide_with_dt: bool = False
    clip_velocity: Optional[float] = None
    weight_action_accel: float = 0.0
    energy_penalty_coef: float = 0.0
    torque_penalty_coef: float = 0.0
    forward_axis: str = "y_neg"  # x | y | y_neg
    allow_action_torque_fallback: bool = False


def _axis_velocity_delta(
    prev_base_position: jp.ndarray,
    current_base_position: jp.ndarray,
    axis: str,
) -> jp.ndarray:
    axis_key = str(axis).lower()
    if axis_key in ("x", "0"):
        return current_base_position[0] - prev_base_position[0]
    if axis_key in ("y", "1"):
        return current_base_position[1] - prev_base_position[1]
    # Match existing Pupper convention: forward is -Y.
    return -(current_base_position[1] - prev_base_position[1])


def compute_simple_forward_reward(
    *,
    prev_base_position: jp.ndarray,
    current_base_position: jp.ndarray,
    dt: float,
    action: jp.ndarray,
    last_action: jp.ndarray,
    last_last_action: jp.ndarray,
    config: SimpleForwardRewardConfig,
    motor_torques: Optional[jp.ndarray] = None,
    motor_velocities: Optional[jp.ndarray] = None,
) -> Tuple[jp.ndarray, Dict[str, jp.ndarray]]:
    """Computes modular simple-forward reward and component terms."""

    velocity = _axis_velocity_delta(
        prev_base_position=prev_base_position,
        current_base_position=current_base_position,
        axis=config.forward_axis,
    )
    if bool(config.divide_with_dt):
        velocity = velocity / jp.maximum(jp.asarray(dt, dtype=velocity.dtype), 1e-8)
    if config.clip_velocity is not None:
        clip = float(config.clip_velocity)
        velocity = jp.clip(velocity, -clip, clip)

    action_acceleration_penalty = jp.asarray(0.0, dtype=velocity.dtype)
    if float(config.weight_action_accel) > 0.0:
        acc = action - 2.0 * last_action + last_last_action
        action_acceleration_penalty = float(config.weight_action_accel) * jp.mean(jp.abs(acc))

    energy_reward = jp.asarray(0.0, dtype=velocity.dtype)
    if float(config.energy_penalty_coef) > 0.0 and motor_torques is not None and motor_velocities is not None:
        energy_reward = -jp.sum(jp.abs(motor_torques * motor_velocities)) * jp.asarray(dt, dtype=velocity.dtype)

    torque_reward = jp.asarray(0.0, dtype=velocity.dtype)
    if float(config.torque_penalty_coef) > 0.0:
        if motor_torques is not None:
            torque_reward = -float(config.torque_penalty_coef) * jp.dot(motor_torques, motor_torques)
        elif bool(config.allow_action_torque_fallback):
            # Optional non-PyBullet fallback retained for compatibility/debugging.
            torque_reward = -float(config.torque_penalty_coef) * jp.dot(action, action)

    reward = velocity
    reward = reward - action_acceleration_penalty
    reward = reward + float(config.energy_penalty_coef) * energy_reward
    reward = reward + torque_reward
    reward = reward * float(config.weight)

    terms = {
        "velocity": velocity,
        "action_accel_penalty": action_acceleration_penalty,
        "energy_reward": energy_reward,
        "torque_reward": torque_reward,
        "task_reward": reward,
    }
    return reward, terms


def build_reward(config: Optional[Dict[str, Any]] = None) -> Callable[[Dict[str, Any]], Tuple[jp.ndarray, Dict[str, jp.ndarray]]]:
    """Builds a reward callable from a JSON-like config dict."""

    config = dict(config or {})
    cfg = SimpleForwardRewardConfig(**config)

    def reward_fn(ctx: Dict[str, Any]) -> Tuple[jp.ndarray, Dict[str, jp.ndarray]]:
        return compute_simple_forward_reward(
            prev_base_position=ctx["prev_base_position"],
            current_base_position=ctx["current_base_position"],
            dt=ctx["dt"],
            action=ctx["action"],
            last_action=ctx["last_action"],
            last_last_action=ctx["last_last_action"],
            config=cfg,
            motor_torques=ctx.get("motor_torques"),
            motor_velocities=ctx.get("motor_velocities"),
        )

    return reward_fn
