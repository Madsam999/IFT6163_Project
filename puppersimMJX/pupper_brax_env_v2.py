"""Local Brax/MJX environment for Pupper v2 (no external repos required)."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

import puppersim.data as pd
from puppersim import pupper_constants


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, _REPO_ROOT)


def _default_urdf_path() -> Path:
    return Path(pd.getDataPath()) / "pupper_v2a.urdf"


def _default_mjcf_path() -> Path:
    return Path(pd.getDataPath()) / "pupper_v2a_mjx.xml"


def _ensure_mjcf(
    model_path: str | os.PathLike[str] | None,
    urdf_path: str | os.PathLike[str] | None,
    auto_generate_mjcf: bool,
    regenerate_mjcf_if_exists: bool,
    kp: float,
    kd: float,
    timestep: float,
    mjx_compatible: bool,
    mjx_iterations: int,
    mjx_ls_iterations: int,
    mjx_impratio: float,
    floating_base: bool,
    spawn_z: float,
    add_tracking_camera: bool,
    collision_mode: str,
) -> Path:
    resolved_model_path = Path(model_path).expanduser().resolve() if model_path else _default_mjcf_path()
    _ = Path(urdf_path).expanduser().resolve() if urdf_path else _default_urdf_path()
    _ = auto_generate_mjcf
    _ = regenerate_mjcf_if_exists
    _ = kp
    _ = kd
    _ = timestep
    _ = mjx_compatible
    _ = mjx_iterations
    _ = mjx_ls_iterations
    _ = mjx_impratio
    _ = floating_base
    _ = spawn_z
    _ = add_tracking_camera
    _ = collision_mode

    if resolved_model_path.exists():
        return resolved_model_path

    raise FileNotFoundError(
        f"MJCF XML not generated: {resolved_model_path}. "
        "Please see puppersimMJX/README.md for generation steps."
    )


def _infer_act_size(sys: Any) -> int:
    nu = getattr(sys, "nu", None)
    if isinstance(nu, int):
        return int(nu)
    if callable(getattr(sys, "act_size", None)):
        return int(sys.act_size())
    if nu is not None:
        return int(nu)
    return 0


def _import_attr(module_name: str, attr_name: str) -> Any:
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(f"Module '{module_name}' has no attribute '{attr_name}'.") from exc


def _motor_init_angles_urdf_convention() -> np.ndarray:
    names = list(pupper_constants.JOINT_NAMES)
    sdk_angles = np.array([pupper_constants.INIT_JOINT_ANGLES[name] for name in names], dtype=np.float64)
    offsets = np.array([pupper_constants.JOINT_OFFSETS[name] for name in names], dtype=np.float64)
    directions = np.array([pupper_constants.JOINT_DIRECTIONS[name] for name in names], dtype=np.float64)
    return (sdk_angles + offsets) * directions


def _init_orientation_wxyz() -> np.ndarray:
    # PyBullet constants store quaternion as [x, y, z, w].
    ori_xyzw = np.array(pupper_constants.INIT_ORIENTATION, dtype=np.float64)
    if ori_xyzw.shape[0] != 4:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.array([ori_xyzw[3], ori_xyzw[0], ori_xyzw[1], ori_xyzw[2]], dtype=np.float64)


class PupperV2BraxEnv:
    pass


def _build_env_class():
    import jax
    from jax import numpy as jp

    from brax import math
    from brax.envs.base import PipelineEnv, State
    from brax.io import mjcf

    class _PupperV2BraxEnvImpl(PipelineEnv):
        """Pupper v2 Brax env with modular reward modes."""

        def __init__(
            self,
            model_path: str | None = None,
            urdf_path: str | None = None,
            auto_generate_mjcf: bool = True,
            regenerate_mjcf_if_exists: bool = True,
            conversion_kp: float = 30.0,
            conversion_kd: float = 1.0,
            conversion_timestep: float = 0.002,
            conversion_mjx_compatible: bool = True,
            conversion_mjx_iterations: int = 1,
            conversion_mjx_ls_iterations: int = 5,
            conversion_mjx_impratio: float = 10.0,
            conversion_floating_base: bool = True,
            conversion_spawn_z: float = 0.17,
            conversion_add_tracking_camera: bool = True,
            conversion_collision_mode: str = "feet_only",
            action_scale: float = 1.0,
            reset_noise_scale: float = 0.01,
            reset_base_noise_scale: float = 0.0,
            reset_qd_noise_scale: float = 0.0,
            terminate_when_unhealthy: bool = True,
            healthy_z_range: Tuple[float, float] = (0.05, 0.6),
            terminal_roll_pitch_threshold: float = 0.4,
            use_pybullet_terminal_condition: bool = True,
            termination_grace_steps: int = 50,
            reward_mode: str = "simple_forward",
            reward_weight: float = 1.0,
            reward_divide_with_dt: bool = False,
            reward_clip_velocity: float | None = None,
            reward_weight_action_accel: float = 0.0,
            reward_energy_penalty_coef: float = 0.0,
            reward_torque_penalty_coef: float = 0.0,
            reward_module: str = "",
            reward_config: Dict[str, Any] | None = None,
            reward_requires_command: bool | None = None,
            # Command-conditioned settings.
            include_command_in_obs: bool = True,
            observation_history: int = 1,
            use_imu: bool = True,
            resample_velocity_step: int = 250,
            lin_vel_x_range: Tuple[float, float] = (-0.75, 0.75),
            lin_vel_y_range: Tuple[float, float] = (-0.5, 0.5),
            ang_vel_yaw_range: Tuple[float, float] = (-2.0, 2.0),
            zero_command_probability: float = 0.02,
            stand_still_command_threshold: float = 0.05,
            tracking_sigma: float = 0.25,
            feet_air_time_minimum: float = 0.10,
            desired_world_z_in_body_frame: Tuple[float, float, float] = (0.0, 0.0, 1.0),
            default_pose: Tuple[float, ...] = (
                0.0,
                0.6,
                -1.2,
                0.0,
                0.6,
                -1.2,
                0.0,
                0.6,
                -1.2,
                0.0,
                0.6,
                -1.2,
            ),
            desired_abduction_angles: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
            command_scale_tracking_lin_vel: float = 1.5,
            command_scale_tracking_ang_vel: float = 0.8,
            command_scale_lin_vel_z: float = -2.0,
            command_scale_ang_vel_xy: float = -0.05,
            command_scale_orientation: float = -5.0,
            command_scale_tracking_orientation: float = 1.0,
            command_scale_torques: float = -0.0002,
            command_scale_joint_acceleration: float = -1e-6,
            command_scale_mechanical_work: float = 0.0,
            command_scale_action_rate: float = -0.01,
            command_scale_feet_air_time: float = 0.2,
            command_scale_stand_still: float = -0.5,
            command_scale_stand_still_joint_velocity: float = -0.1,
            command_scale_abduction_angle: float = -0.1,
            command_scale_termination: float = -100.0,
            command_scale_foot_slip: float = -0.1,
            command_scale_knee_collision: float = -1.0,
            command_scale_body_collision: float = -1.0,
            # Disturbances / observation noise.
            kick_probability: float = 0.0,
            kick_vel: float = 0.10,
            angular_velocity_noise: float = 0.0,
            gravity_noise: float = 0.0,
            motor_angle_noise: float = 0.0,
            last_action_noise: float = 0.0,
            forward_reward_weight: float = 1.0,
            survival_reward: float = 0.0,
            ctrl_cost_weight: float = 0.0,
            forward_axis: str = "y",
            exclude_xy_from_obs: bool = True,
            backend: str = "mjx",
            **kwargs,
        ):
            mjcf_path = _ensure_mjcf(
                model_path=model_path,
                urdf_path=urdf_path,
                auto_generate_mjcf=auto_generate_mjcf,
                regenerate_mjcf_if_exists=regenerate_mjcf_if_exists,
                kp=conversion_kp,
                kd=conversion_kd,
                timestep=conversion_timestep,
                mjx_compatible=conversion_mjx_compatible,
                mjx_iterations=conversion_mjx_iterations,
                mjx_ls_iterations=conversion_mjx_ls_iterations,
                mjx_impratio=conversion_mjx_impratio,
                floating_base=conversion_floating_base,
                spawn_z=conversion_spawn_z,
                add_tracking_camera=conversion_add_tracking_camera,
                collision_mode=conversion_collision_mode,
            )

            sys = mjcf.load(str(mjcf_path))
            if "early_termination_step_threshold" in kwargs:
                kwargs.pop("early_termination_step_threshold", None)
                print(
                    "[PupperV2BraxEnv] ignoring deprecated option: early_termination_step_threshold"
                )
            if "n_frames" not in kwargs:
                kwargs["n_frames"] = 2
            super().__init__(sys=sys, backend=backend, **kwargs)

            self._action_scale = float(action_scale)
            self._reset_noise_scale = float(reset_noise_scale)
            self._reset_base_noise_scale = float(reset_base_noise_scale)
            self._reset_qd_noise_scale = float(reset_qd_noise_scale)
            self._terminate_when_unhealthy = bool(terminate_when_unhealthy)
            self._healthy_z_range = (float(healthy_z_range[0]), float(healthy_z_range[1]))
            self._terminal_roll_pitch_threshold = float(terminal_roll_pitch_threshold)
            self._use_pybullet_terminal_condition = bool(use_pybullet_terminal_condition)
            self._termination_grace_steps = int(max(0, termination_grace_steps))
            self._reward_mode = str(reward_mode).lower()
            self._reward_module = str(reward_module).strip()
            reward_module_explicit = bool(self._reward_module)
            self._include_command_in_obs = bool(include_command_in_obs)
            self._observation_history = int(max(1, observation_history))
            self._use_imu = bool(use_imu)
            self._forward_reward_weight = float(forward_reward_weight)
            self._survival_reward = float(survival_reward)
            self._ctrl_cost_weight = float(ctrl_cost_weight)
            self._exclude_xy_from_obs = bool(exclude_xy_from_obs)
            self._forward_axis = 0 if str(forward_axis).lower() in ("x", "0") else 1

            self._resample_velocity_step = int(max(1, resample_velocity_step))
            self._lin_vel_x_range = (float(lin_vel_x_range[0]), float(lin_vel_x_range[1]))
            self._lin_vel_y_range = (float(lin_vel_y_range[0]), float(lin_vel_y_range[1]))
            self._ang_vel_yaw_range = (float(ang_vel_yaw_range[0]), float(ang_vel_yaw_range[1]))
            self._zero_command_probability = float(zero_command_probability)

            simple_forward_cfg: Dict[str, Any] = {
                "weight": float(reward_weight),
                "divide_with_dt": bool(reward_divide_with_dt),
                "clip_velocity": (None if reward_clip_velocity is None else float(reward_clip_velocity)),
                "weight_action_accel": float(reward_weight_action_accel),
                "energy_penalty_coef": float(reward_energy_penalty_coef),
                "torque_penalty_coef": float(reward_torque_penalty_coef),
                "forward_axis": str(forward_axis),
                "allow_action_torque_fallback": False,
            }
            command_reward_cfg: Dict[str, Any] = {
                "tracking_sigma": float(tracking_sigma),
                "feet_air_time_minimum": float(feet_air_time_minimum),
                "stand_still_command_threshold": float(stand_still_command_threshold),
                "desired_world_z_in_body_frame": tuple(float(x) for x in desired_world_z_in_body_frame),
                "default_pose": tuple(float(x) for x in default_pose),
                "desired_abduction_angles": tuple(float(x) for x in desired_abduction_angles),
                "scales": {
                    "tracking_lin_vel": float(command_scale_tracking_lin_vel),
                    "tracking_ang_vel": float(command_scale_tracking_ang_vel),
                    "lin_vel_z": float(command_scale_lin_vel_z),
                    "ang_vel_xy": float(command_scale_ang_vel_xy),
                    "orientation": float(command_scale_orientation),
                    "tracking_orientation": float(command_scale_tracking_orientation),
                    "torques": float(command_scale_torques),
                    "joint_acceleration": float(command_scale_joint_acceleration),
                    "mechanical_work": float(command_scale_mechanical_work),
                    "action_rate": float(command_scale_action_rate),
                    "feet_air_time": float(command_scale_feet_air_time),
                    "stand_still": float(command_scale_stand_still),
                    "stand_still_joint_velocity": float(command_scale_stand_still_joint_velocity),
                    "abduction_angle": float(command_scale_abduction_angle),
                    "termination": float(command_scale_termination),
                    "foot_slip": float(command_scale_foot_slip),
                    "knee_collision": float(command_scale_knee_collision),
                    "body_collision": float(command_scale_body_collision),
                },
            }

            if not self._reward_module:
                if self._reward_mode == "command_locomotion":
                    self._reward_module = "puppersimMJX.tasks.cc_locomotion.reward"
                elif self._reward_mode == "simple_forward":
                    self._reward_module = "puppersimMJX.tasks.simple_forward.reward"
                else:
                    raise ValueError(
                        "Unknown reward_mode. Set reward_mode to simple_forward/command_locomotion "
                        "or provide reward_module."
                    )

            if reward_config is None:
                if reward_module_explicit:
                    reward_config = {}
                elif self._reward_mode == "command_locomotion":
                    reward_config = command_reward_cfg
                elif self._reward_mode == "simple_forward":
                    reward_config = simple_forward_cfg
                else:
                    reward_config = {}
            if not isinstance(reward_config, dict):
                raise ValueError("reward_config must be a dict when provided.")

            build_reward_fn = _import_attr(self._reward_module, "build_reward")
            self._reward_fn = build_reward_fn(dict(reward_config))
            if reward_requires_command is None:
                self._reward_requires_command = bool(self._reward_mode == "command_locomotion")
            else:
                self._reward_requires_command = bool(reward_requires_command)
            effective_reward_mode = "command_locomotion" if self._reward_requires_command else "simple_forward"
            print(
                "[PupperV2BraxEnv] resolved reward settings: "
                f"reward_mode={self._reward_mode}, "
                f"effective_reward_mode={effective_reward_mode}, "
                f"reward_module={self._reward_module}, "
                f"reward_requires_command={self._reward_requires_command}, "
                f"reward_config_keys={sorted(list(reward_config.keys())) if isinstance(reward_config, dict) else '<none>'}"
            )
            self._kick_probability = float(kick_probability)
            self._kick_vel = float(kick_vel)
            self._angular_velocity_noise = float(angular_velocity_noise)
            self._gravity_noise = float(gravity_noise)
            self._motor_angle_noise = float(motor_angle_noise)
            self._last_action_noise = float(last_action_noise)

            self._act_size = _infer_act_size(self.sys)
            if self._act_size <= 0:
                raise ValueError(
                    "Pupper v2 MJCF has no actuators (nu=0). "
                    "Please see puppersimMJX/README.md for XML generation steps."
                )

            self._torso_idx = 0
            try:
                if "base_link" in self.sys.link_names:
                    self._torso_idx = int(self.sys.link_names.index("base_link"))
            except Exception:
                self._torso_idx = 0

            self._foot_body_indices = []
            for name in ("leftFrontLowerLeg", "leftRearLowerLeg", "rightFrontLowerLeg", "rightRearLowerLeg"):
                try:
                    self._foot_body_indices.append(int(self.sys.link_names.index(name)))
                except Exception:
                    pass
            self._foot_body_indices = tuple(self._foot_body_indices[:4])

            self._knee_body_indices = []
            for name in ("leftFrontUpperLeg", "leftRearUpperLeg", "rightFrontUpperLeg", "rightRearUpperLeg"):
                try:
                    self._knee_body_indices.append(int(self.sys.link_names.index(name)))
                except Exception:
                    pass
            self._knee_body_indices = tuple(self._knee_body_indices[:4])

            low = np.asarray(pupper_constants.MOTOR_ACTION_LOWER_LIMIT, dtype=np.float32)
            high = np.asarray(pupper_constants.MOTOR_ACTION_UPPER_LIMIT, dtype=np.float32)
            if low.shape[0] >= self._act_size:
                low = low[: self._act_size]
                high = high[: self._act_size]
            else:
                low = np.full((self._act_size,), -1.0, dtype=np.float32)
                high = np.full((self._act_size,), 1.0, dtype=np.float32)
            self._action_low = jp.array(low)
            self._action_high = jp.array(high)
            self._action_mid = 0.5 * (self._action_high + self._action_low)
            self._action_half_range = 0.5 * (self._action_high - self._action_low)
            print(
                "[PupperV2BraxEnv] action space: "
                f"act_size={self._act_size}, "
                f"action_scale={self._action_scale}, "
                f"low={low.tolist()}, high={high.tolist()}"
            )

            default_q = np.asarray(self.sys.init_q, dtype=np.float32)
            if default_q.shape[0] >= 19:
                default_q = np.copy(default_q)
                default_q[2] = float(pupper_constants.INIT_POSITION[2])
                default_q[3:7] = _init_orientation_wxyz().astype(np.float32)
                default_q[7:19] = _motor_init_angles_urdf_convention().astype(np.float32)
            self._default_q = jp.array(default_q)

        def _decode_action(self, action):
            normalized = jp.clip(action, -1.0, 1.0) * self._action_scale
            normalized = jp.clip(normalized, -1.0, 1.0)
            return self._action_mid + normalized * self._action_half_range

        def _sample_command(self, rng: jp.ndarray, dtype):
            rng, rx, ry, rz, rzprob = jax.random.split(rng, 5)
            x_cmd = jax.random.uniform(rx, (), minval=self._lin_vel_x_range[0], maxval=self._lin_vel_x_range[1])
            y_cmd = jax.random.uniform(ry, (), minval=self._lin_vel_y_range[0], maxval=self._lin_vel_y_range[1])
            yaw_cmd = jax.random.uniform(rz, (), minval=self._ang_vel_yaw_range[0], maxval=self._ang_vel_yaw_range[1])
            command = jp.array([x_cmd, y_cmd, yaw_cmd], dtype=dtype)
            zero = jax.random.uniform(rzprob, ()) < self._zero_command_probability
            command = jp.where(zero, jp.zeros_like(command), command)
            return rng, command

        def _get_obs(self, pipeline_state, command=None) -> jp.ndarray:
            q = pipeline_state.q
            qd = pipeline_state.qd
            if self._exclude_xy_from_obs and q.shape[0] > 2:
                q = q[2:]
            obs = jp.concatenate([q, qd], axis=0)
            if self._reward_requires_command and self._include_command_in_obs:
                if command is None:
                    command = jp.zeros((3,), dtype=obs.dtype)
                obs = jp.concatenate([obs, command.astype(obs.dtype)], axis=0)
            return obs

        def _flatten_obs_history(self, obs_hist: jp.ndarray) -> jp.ndarray:
            # obs_hist shape: (history, obs_dim) -> (history * obs_dim,)
            return jp.reshape(obs_hist, (-1,))

        def _apply_obs_noise(self, obs: jp.ndarray, rng: jp.ndarray) -> jp.ndarray:
            """Applies sensor-like noise to obs vector."""
            q_size = int(self.sys.q_size()) - (2 if self._exclude_xy_from_obs else 0)
            qd_size = int(self.sys.qd_size())
            out = obs
            rng, k1, k2, k3 = jax.random.split(rng, 4)

            # Motor angle noise on joint-angle slice at end of q-part.
            if self._motor_angle_noise > 0.0 and q_size >= self._act_size:
                idx0 = q_size - self._act_size
                n = self._motor_angle_noise * jax.random.normal(k1, (self._act_size,), dtype=obs.dtype)
                out = out.at[idx0:q_size].add(n)

            # Base angular velocity noise on qd slice (assumes free-joint qd layout: lin xyz + ang xyz + joints).
            if self._angular_velocity_noise > 0.0 and qd_size >= 6:
                idx0 = q_size + 3
                n = self._angular_velocity_noise * jax.random.normal(k2, (3,), dtype=obs.dtype)
                out = out.at[idx0 : idx0 + 3].add(n)

            # Optional gravity/IMU-like bias proxy: add tiny noise to base orientation quaternion entries in q-part.
            if self._gravity_noise > 0.0 and q_size >= 7:
                quat_start = 1 if self._exclude_xy_from_obs else 3
                n = self._gravity_noise * jax.random.normal(k3, (4,), dtype=obs.dtype)
                out = out.at[quat_start : quat_start + 4].add(n)
            return out

        def _body_contact_count(self, contact, body_indices: Tuple[int, ...]):
            if not body_indices or not hasattr(contact, "link_idx"):
                return jp.asarray(0.0)
            out = jp.asarray(0.0)
            for idx in body_indices:
                m = ((contact.link_idx[0] == idx) | (contact.link_idx[1] == idx)) & (contact.dist < 0.0)
                out = out + jp.sum(m.astype(jp.float32))
            return out

        def reset(self, rng: jp.ndarray) -> State:
            rng, rng1, rng2 = jax.random.split(rng, 3)
            q = self._default_q
            if self._reset_noise_scale > 0.0 or self._reset_base_noise_scale > 0.0:
                q_noise = jp.zeros((self.sys.q_size(),), dtype=q.dtype)
                if self._reset_base_noise_scale > 0.0:
                    base_noise = self._reset_base_noise_scale * jax.random.uniform(
                        rng1, shape=(7,), minval=-1.0, maxval=1.0
                    )
                    q_noise = q_noise.at[:7].set(base_noise)
                if self._reset_noise_scale > 0.0 and self.sys.q_size() >= 7 + self._act_size:
                    joint_noise = self._reset_noise_scale * jax.random.uniform(
                        rng1, shape=(self._act_size,), minval=-1.0, maxval=1.0
                    )
                    q_noise = q_noise.at[7 : 7 + self._act_size].set(joint_noise)
                q = q + q_noise
            qd = self._reset_qd_noise_scale * jax.random.normal(rng2, shape=(self.sys.qd_size(),))
            pipeline_state = self.pipeline_init(q, qd)

            command = jp.zeros((3,), dtype=q.dtype)
            if self._reward_requires_command:
                rng, command = self._sample_command(rng, q.dtype)
            obs = self._get_obs(pipeline_state, command=command)
            rng, noise_key = jax.random.split(rng)
            obs = self._apply_obs_noise(obs, noise_key)
            obs_history = jp.tile(obs[None, :], (self._observation_history, 1))
            obs_stacked = self._flatten_obs_history(obs_history)

            zero = jp.zeros(())
            metrics = {
                "reward_forward": zero,
                "reward_task": zero,
                "reward_survival": zero,
                "reward_ctrl_cost": zero,
                "reward_action_accel_penalty": zero,
                "reward_energy": zero,
                "reward_torque": zero,
                "reward_tracking_lin_vel": zero,
                "reward_tracking_ang_vel": zero,
                "reward_orientation": zero,
                "reward_feet_air_time": zero,
                "x_position": zero,
                "y_position": zero,
                "z_position": zero,
                "x_velocity": zero,
                "y_velocity": zero,
                "roll": zero,
                "pitch": zero,
                "command_x": zero,
                "command_y": zero,
                "command_yaw": zero,
                "command_abs_x": zero,
                "command_abs_y": zero,
                "command_abs_yaw": zero,
                "command_norm": zero,
                "command_resampled": zero,
                "tracking_lin_error": zero,
                "tracking_yaw_error": zero,
            }
            info = {
                "last_action": jp.zeros((self._act_size,), dtype=obs.dtype),
                "last_last_action": jp.zeros((self._act_size,), dtype=obs.dtype),
                "episode_step": jp.asarray(0, dtype=jp.int32),
                "command_rng": rng,
                "noise_rng": noise_key,
                "command": command.astype(obs.dtype),
                "obs_history": obs_history,
                "last_joint_vel": jp.zeros((self._act_size,), dtype=obs.dtype),
                "foot_air_time": jp.zeros((len(self._foot_body_indices),), dtype=obs.dtype),
                "prev_foot_contact": jp.zeros((len(self._foot_body_indices),), dtype=jp.bool_),
            }
            return State(pipeline_state, obs_stacked, zero, zero, metrics, info=info)

        def step(self, state: State, action: jp.ndarray) -> State:
            action = self._decode_action(action)
            pipeline_state_0 = state.pipeline_state
            pipeline_state = self.pipeline_step(pipeline_state_0, action)
            noise_rng = state.info["noise_rng"]
            noise_rng, kick_key, kick_dir_key, last_action_noise_key, obs_noise_key = jax.random.split(noise_rng, 5)

            # Random kick perturbation on torso linear velocity.
            if self._kick_probability > 0.0 and self._kick_vel > 0.0:
                do_kick = jax.random.uniform(kick_key, ()) < self._kick_probability
                kick_vec_xy = self._kick_vel * jax.random.uniform(
                    kick_dir_key, (2,), minval=-1.0, maxval=1.0, dtype=action.dtype
                )
                kick_vec = jp.array([kick_vec_xy[0], kick_vec_xy[1], 0.0], dtype=action.dtype)
                xd_new = pipeline_state.xd.replace(
                    vel=pipeline_state.xd.vel.at[self._torso_idx].set(
                        pipeline_state.xd.vel[self._torso_idx] + jp.where(do_kick, kick_vec, jp.zeros_like(kick_vec))
                    )
                )
                pipeline_state = pipeline_state.replace(xd=xd_new)

            pos0 = pipeline_state_0.x.pos[self._torso_idx]
            pos1 = pipeline_state.x.pos[self._torso_idx]
            velocity = (pos1 - pos0) / self.dt
            forward_velocity = velocity[self._forward_axis]

            torso_quat = pipeline_state.x.rot[self._torso_idx]
            qw, qx, qy, qz = torso_quat
            roll = jp.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
            pitch = jp.arcsin(jp.clip(2.0 * (qw * qy - qz * qx), -1.0, 1.0))

            min_z, max_z = self._healthy_z_range
            z = pos1[2]
            is_healthy = jp.where(z < min_z, 0.0, 1.0)
            is_healthy = jp.where(z > max_z, 0.0, is_healthy)
            if self._use_pybullet_terminal_condition:
                rp_limit = jp.asarray(self._terminal_roll_pitch_threshold, dtype=z.dtype)
                is_healthy = jp.where(jp.abs(roll) > rp_limit, 0.0, is_healthy)
                is_healthy = jp.where(jp.abs(pitch) > rp_limit, 0.0, is_healthy)

            raw_done = (1.0 - is_healthy) if self._terminate_when_unhealthy else jp.zeros(())
            episode_step = state.info.get("episode_step", jp.asarray(0, dtype=jp.int32)) + 1
            if self._termination_grace_steps > 0:
                done = jp.where(episode_step <= self._termination_grace_steps, jp.zeros_like(raw_done), raw_done)
            else:
                done = raw_done

            last_action = state.info.get("last_action", jp.zeros_like(action))
            last_last_action = state.info.get("last_last_action", jp.zeros_like(action))
            if self._last_action_noise > 0.0:
                n = self._last_action_noise * jax.random.normal(
                    last_action_noise_key, last_action.shape, dtype=last_action.dtype
                )
                last_action = last_action + n
            command = state.info.get("command", jp.zeros((3,), dtype=action.dtype))

            motor_velocities = None
            if hasattr(pipeline_state, "qd"):
                qd = pipeline_state.qd
                if qd.shape[0] >= self._act_size:
                    motor_velocities = qd[-self._act_size :]
            motor_torques = None
            qfrc_actuator = getattr(pipeline_state, "qfrc_actuator", None)
            if qfrc_actuator is not None and qfrc_actuator.shape[0] >= self._act_size:
                motor_torques = qfrc_actuator[-self._act_size :]

            joint_vel = motor_velocities if motor_velocities is not None else jp.zeros_like(action)
            joint_angles = (
                pipeline_state.q[7 : 7 + self._act_size]
                if pipeline_state.q.shape[0] >= 7 + self._act_size
                else jp.zeros_like(action)
            )
            torques = motor_torques if motor_torques is not None else jp.zeros_like(action)
            last_joint_vel = state.info.get("last_joint_vel", jp.zeros_like(joint_vel))

            foot_contact_list = []
            if hasattr(pipeline_state, "contact") and hasattr(pipeline_state.contact, "link_idx"):
                for idx in self._foot_body_indices:
                    m = ((pipeline_state.contact.link_idx[0] == idx) | (pipeline_state.contact.link_idx[1] == idx)) & (
                        pipeline_state.contact.dist < 0.0
                    )
                    foot_contact_list.append(jp.any(m))
            foot_contact = (
                jp.stack(foot_contact_list).astype(action.dtype)
                if foot_contact_list
                else jp.zeros((len(self._foot_body_indices),), dtype=action.dtype)
            )
            prev_foot_contact = state.info.get(
                "prev_foot_contact", jp.zeros((len(self._foot_body_indices),), dtype=jp.bool_)
            )
            first_contact = foot_contact * (1.0 - prev_foot_contact.astype(action.dtype))
            air_time = state.info.get(
                "foot_air_time", jp.zeros((len(self._foot_body_indices),), dtype=action.dtype)
            )
            air_time = jp.where(foot_contact > 0.5, jp.zeros_like(air_time), air_time + self.dt)

            foot_slip = jp.asarray(0.0, dtype=action.dtype)
            for i, idx in enumerate(self._foot_body_indices):
                vel_xy_sq = jp.sum(jp.square(pipeline_state.xd.vel[idx][:2]))
                foot_slip = foot_slip + vel_xy_sq * foot_contact[i]

            knee_collision = self._body_contact_count(pipeline_state.contact, self._knee_body_indices)
            body_collision = self._body_contact_count(pipeline_state.contact, (self._torso_idx,))

            reward_ctx = {
                "prev_base_position": pos0,
                "current_base_position": pos1,
                "dt": self.dt,
                "action": action,
                "last_action": last_action,
                "last_last_action": last_last_action,
                "motor_torques": motor_torques,
                "motor_velocities": motor_velocities,
                "command": command,
                "torso_quat": torso_quat,
                "base_vel_world": pipeline_state.xd.vel[self._torso_idx],
                "base_ang_world": pipeline_state.xd.ang[self._torso_idx],
                "joint_angles": joint_angles,
                "joint_vel": joint_vel,
                "last_joint_vel": last_joint_vel,
                "torques": torques,
                "air_time": air_time,
                "first_contact": first_contact,
                "foot_slip": foot_slip,
                "knee_collision": knee_collision,
                "body_collision": body_collision,
                "raw_done": raw_done,
                "episode_step": episode_step,
            }
            reward, reward_terms = self._reward_fn(reward_ctx)
            task_reward = reward_terms.get("task_reward", reward)

            default_zero = jp.asarray(0.0, dtype=task_reward.dtype)
            forward_reward = reward_terms.get("velocity", reward_terms.get("tracking_lin_vel", forward_velocity))
            action_accel_penalty = reward_terms.get("action_accel_penalty", reward_terms.get("action_rate", default_zero))
            energy_reward = reward_terms.get("energy_reward", reward_terms.get("mechanical_work", default_zero))
            torque_reward = reward_terms.get("torque_reward", reward_terms.get("torques", default_zero))
            survival_reward = jp.asarray(0.0, dtype=task_reward.dtype)
            ctrl_cost = jp.asarray(0.0, dtype=task_reward.dtype)

            command_rng = state.info["command_rng"]
            command_resampled = jp.asarray(0.0, dtype=reward.dtype)
            if self._reward_requires_command:
                command_rng, sample_key = jax.random.split(command_rng)
                _, new_command = self._sample_command(sample_key, reward.dtype)
                should_resample = (episode_step % self._resample_velocity_step) == 0
                command = jp.where(should_resample, new_command, command)
                command_resampled = should_resample.astype(reward.dtype)

            base_vel_world = pipeline_state.xd.vel[self._torso_idx]
            base_ang_world = pipeline_state.xd.ang[self._torso_idx]
            local_vel = math.rotate(base_vel_world, math.quat_inv(torso_quat))
            local_ang = math.rotate(base_ang_world, math.quat_inv(torso_quat))
            tracking_lin_error = jp.linalg.norm(command[:2] - local_vel[:2])
            tracking_yaw_error = jp.abs(command[2] - local_ang[2])

            obs = self._get_obs(pipeline_state, command=command)
            obs = self._apply_obs_noise(obs, obs_noise_key)
            prev_hist = state.info.get("obs_history", jp.tile(obs[None, :], (self._observation_history, 1)))
            new_hist = jp.concatenate([prev_hist[1:], obs[None, :]], axis=0)
            obs_stacked = self._flatten_obs_history(new_hist)

            metrics: Dict[str, Any] = dict(state.metrics)
            metrics.update(
                reward_forward=forward_reward,
                reward_task=task_reward,
                reward_survival=survival_reward,
                reward_ctrl_cost=-ctrl_cost,
                reward_action_accel_penalty=-action_accel_penalty,
                reward_energy=energy_reward,
                reward_torque=torque_reward,
                x_position=pos1[0],
                y_position=pos1[1],
                z_position=z,
                x_velocity=velocity[0],
                y_velocity=velocity[1],
                roll=roll,
                pitch=pitch,
                command_x=command[0],
                command_y=command[1],
                command_yaw=command[2],
                command_abs_x=jp.abs(command[0]),
                command_abs_y=jp.abs(command[1]),
                command_abs_yaw=jp.abs(command[2]),
                command_norm=jp.linalg.norm(command),
                command_resampled=command_resampled,
                tracking_lin_error=tracking_lin_error,
                tracking_yaw_error=tracking_yaw_error,
            )
            if "tracking_lin_vel" in reward_terms:
                metrics["reward_tracking_lin_vel"] = reward_terms["tracking_lin_vel"]
            if "tracking_ang_vel" in reward_terms:
                metrics["reward_tracking_ang_vel"] = reward_terms["tracking_ang_vel"]
            if "orientation" in reward_terms:
                metrics["reward_orientation"] = reward_terms["orientation"]
            if "feet_air_time" in reward_terms:
                metrics["reward_feet_air_time"] = reward_terms["feet_air_time"]

            info = dict(state.info)
            info["last_last_action"] = last_action
            info["last_action"] = action
            info["episode_step"] = episode_step
            info["command_rng"] = command_rng
            info["noise_rng"] = noise_rng
            info["command"] = command
            info["obs_history"] = new_hist
            if self._reward_requires_command:
                info["last_joint_vel"] = (
                    motor_velocities if motor_velocities is not None else jp.zeros((self._act_size,), dtype=obs.dtype)
                )
                info["foot_air_time"] = air_time
                info["prev_foot_contact"] = foot_contact > 0.5

            return state.replace(
                pipeline_state=pipeline_state,
                obs=obs_stacked,
                reward=reward,
                done=done,
                metrics=metrics,
                info=info,
            )

    return _PupperV2BraxEnvImpl


PupperV2BraxEnv = _build_env_class()
