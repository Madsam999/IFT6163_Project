"""Fast interactive MuJoCo viewer loop for exported Brax policy bundles.

Usage:
  python puppersimMJX/pupper_mjx_live_control.py \
    --bundle-dir runs/<run>/policy_bundle \
    --model-path puppersim/data/pupper_v2a_mjx.xml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import mujoco
import mujoco.viewer
import numpy as np


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from puppersim import pupper_constants
from puppersimMJX.pupper_brax_policy_bundle import BraxPolicyBundle


def _parse_env_kwargs(raw: str) -> Dict:
    if not raw.strip():
        return {}
    p = Path(raw).expanduser()
    if p.exists():
        return dict(json.loads(p.read_text()))
    return dict(json.loads(raw))


def _clip_command(cmd: np.ndarray, xr: Tuple[float, float], yr: Tuple[float, float], yrw: Tuple[float, float]) -> np.ndarray:
    out = cmd.copy()
    out[0] = np.clip(out[0], xr[0], xr[1])
    out[1] = np.clip(out[1], yr[0], yr[1])
    out[2] = np.clip(out[2], yrw[0], yrw[1])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument(
        "--env-kwargs",
        type=str,
        default="puppersimMJX/tasks/cc_locomotion/config/pupper_brax_env_kwargs.command_locomotion.json",
    )
    parser.add_argument("--action-scale", type=float, default=0.75)
    parser.add_argument("--policy-dt", type=float, default=0.04, help="seconds between policy actions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--x-speed", type=float, default=0.4)
    parser.add_argument("--y-speed", type=float, default=0.6)
    parser.add_argument("--yaw-speed", type=float, default=1.0)
    parser.add_argument("--inc-x", type=float, default=0.03, help="increment step for x command in increment mode")
    parser.add_argument("--inc-y", type=float, default=0.03, help="increment step for y command in increment mode")
    parser.add_argument("--inc-yaw", type=float, default=0.08, help="increment step for yaw command in increment mode")
    parser.add_argument("--lin-x-min", type=float, default=-0.75)
    parser.add_argument("--lin-x-max", type=float, default=0.75)
    parser.add_argument("--lin-y-min", type=float, default=-0.5)
    parser.add_argument("--lin-y-max", type=float, default=0.5)
    parser.add_argument("--yaw-min", type=float, default=-2.0)
    parser.add_argument("--yaw-max", type=float, default=2.0)
    parser.add_argument("--command-decay-sec", type=float, default=0.12, help="auto-zero timeout after last key repeat")
    parser.add_argument("--control-mode", type=str, default="increment", choices=["increment", "hold"])
    parser.add_argument("--camera-name", type=str, default="tracking_cam")
    parser.add_argument("--lock-camera", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    _ = np.random.default_rng(args.seed)
    bundle = BraxPolicyBundle(args.bundle_dir)
    env_cfg = _parse_env_kwargs(args.env_kwargs)

    include_command = bool(env_cfg.get("include_command_in_obs", True))
    exclude_xy = bool(env_cfg.get("exclude_xy_from_obs", True))
    obs_hist_len = int(max(1, env_cfg.get("observation_history", 20)))

    model_path = args.model_path
    if model_path is None:
        cfg_model_path = env_cfg.get("model_path", "")
        if cfg_model_path:
            model_path = Path(cfg_model_path)
        else:
            model_path = Path("pupper_v2_final_stable.xml")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if model.nkey > 0:
        try:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        except Exception:
            pass
    mujoco.mj_forward(model, data)

    q_size = int(model.nq)
    qd_size = int(model.nv)
    base_obs_dim = (q_size - 2 if exclude_xy and q_size > 2 else q_size) + qd_size + (3 if include_command else 0)
    expected_dim = obs_hist_len * base_obs_dim
    if expected_dim != bundle.obs_dim:
        raise ValueError(
            f"Bundle obs_dim={bundle.obs_dim} but inferred obs_dim={expected_dim} "
            f"(history={obs_hist_len}, base={base_obs_dim}). Check env kwargs / bundle pair."
        )

    cmd = np.zeros(3, dtype=np.float32)
    cmd_rng_x = (float(args.lin_x_min), float(args.lin_x_max))
    cmd_rng_y = (float(args.lin_y_min), float(args.lin_y_max))
    cmd_rng_yaw = (float(args.yaw_min), float(args.yaw_max))

    low = np.asarray(pupper_constants.MOTOR_ACTION_LOWER_LIMIT, dtype=np.float32)
    high = np.asarray(pupper_constants.MOTOR_ACTION_UPPER_LIMIT, dtype=np.float32)
    act_dim = min(int(model.nu), int(bundle.action_dim), int(low.shape[0]), int(high.shape[0]))
    if act_dim <= 0:
        raise ValueError("No compatible actuators found.")
    low = low[:act_dim]
    high = high[:act_dim]
    action_mid = 0.5 * (low + high)
    action_half = 0.5 * (high - low)

    def _build_base_obs(command: np.ndarray) -> np.ndarray:
        q = np.asarray(data.qpos, dtype=np.float32)
        qd = np.asarray(data.qvel, dtype=np.float32)
        if exclude_xy and q.shape[0] > 2:
            q = q[2:]
        obs = np.concatenate([q, qd], axis=0)
        if include_command:
            obs = np.concatenate([obs, command.astype(np.float32)], axis=0)
        return obs

    first_obs = _build_base_obs(cmd)
    obs_hist = np.tile(first_obs[None, :], (obs_hist_len, 1))

    key_state = {
        "vx+": 0.0,  # A
        "vx-": 0.0,  # D
        "vy+": 0.0,  # W
        "vy-": 0.0,  # S
        "yaw+": 0.0,  # Q
        "yaw-": 0.0,  # E
    }

    now = time.monotonic

    def _key_cb(keycode: int) -> None:
        t = now()
        nonlocal cmd
        if keycode == ord("R"):
            mujoco.mj_resetData(model, data)
            if model.nkey > 0:
                try:
                    mujoco.mj_resetDataKeyframe(model, data, 0)
                except Exception:
                    pass
            mujoco.mj_forward(model, data)
            cmd[:] = 0.0
            return
        # Letter keys from GLFW are uppercase ASCII.
        if args.control_mode == "increment":
            if keycode in (ord("W"), 265):  # up
                cmd[1] += float(args.inc_y)
            elif keycode in (ord("S"), 264):  # down
                cmd[1] -= float(args.inc_y)
            elif keycode in (ord("A"), 263):  # left
                cmd[0] += float(args.inc_x)
            elif keycode in (ord("D"), 262):  # right
                cmd[0] -= float(args.inc_x)
            elif keycode == ord("Q"):
                cmd[2] += float(args.inc_yaw)
            elif keycode == ord("E"):
                cmd[2] -= float(args.inc_yaw)
            elif keycode == ord(" "):
                cmd[:] = 0.0
            cmd = _clip_command(cmd, cmd_rng_x, cmd_rng_y, cmd_rng_yaw)
        else:
            if keycode in (ord("W"), 265):  # up
                key_state["vy+"] = t
            elif keycode in (ord("S"), 264):  # down
                key_state["vy-"] = t
            elif keycode in (ord("A"), 263):  # left
                key_state["vx+"] = t
            elif keycode in (ord("D"), 262):  # right
                key_state["vx-"] = t
            elif keycode == ord("Q"):
                key_state["yaw+"] = t
            elif keycode == ord("E"):
                key_state["yaw-"] = t

    physics_dt = float(model.opt.timestep)
    n_substeps = max(1, int(round(float(args.policy_dt) / max(1e-6, physics_dt))))
    print(f"model_path={model_path}")
    print(f"nu={model.nu}, act_dim={act_dim}, policy_dt={args.policy_dt:.4f}, physics_dt={physics_dt:.4f}, substeps={n_substeps}")
    print("Controls: W/S forward-back, A/D strafe, Q/E yaw, SPACE zero cmd, R reset")
    if args.control_mode == "increment":
        print("control_mode=increment (tap keys to adjust command)")
    else:
        print("control_mode=hold (auto-zero after key repeat timeout)")

    with mujoco.viewer.launch_passive(model, data, key_callback=_key_cb) as viewer:
        cam_id = -1
        if args.camera_name:
            try:
                cam_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_name))
            except Exception:
                cam_id = -1
        if args.lock_camera and cam_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = cam_id
        else:
            viewer.cam.distance = 1.2
            viewer.cam.elevation = -15
        last_log = now()
        while viewer.is_running():
            t = now()
            if args.control_mode == "hold":
                active = lambda ts: (t - ts) <= float(args.command_decay_sec)
                cmd[:] = 0.0
                if active(key_state["vy+"]):
                    cmd[1] += float(args.y_speed)
                if active(key_state["vy-"]):
                    cmd[1] -= float(args.y_speed)
                if active(key_state["vx+"]):
                    cmd[0] += float(args.x_speed)
                if active(key_state["vx-"]):
                    cmd[0] -= float(args.x_speed)
                if active(key_state["yaw+"]):
                    cmd[2] += float(args.yaw_speed)
                if active(key_state["yaw-"]):
                    cmd[2] -= float(args.yaw_speed)
                cmd = _clip_command(cmd, cmd_rng_x, cmd_rng_y, cmd_rng_yaw)

            base_obs = _build_base_obs(cmd)
            obs_hist = np.concatenate([obs_hist[1:], base_obs[None, :]], axis=0)
            stacked_obs = obs_hist.reshape(-1)
            raw_action = bundle.deterministic_action(stacked_obs)[:act_dim]
            motor_target = action_mid + np.clip(raw_action, -1.0, 1.0) * float(args.action_scale) * action_half

            data.ctrl[:act_dim] = motor_target
            for _ in range(n_substeps):
                mujoco.mj_step(model, data)

            if args.lock_camera and cam_id >= 0:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = cam_id
            viewer.sync()

            if now() - last_log > 0.5:
                print(f"cmd=[{cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}] z={data.qpos[2]:.3f}")
                last_log = now()


if __name__ == "__main__":
    main()
