#!/usr/bin/env python3
"""MuJoCo viewer for camera presentation + optional policy testing.

Mode A (camera presentation): run without --bundle-dir.
Mode B (policy test): pass --bundle-dir containing policy_bundle.json/.npz.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import mujoco as mj
import numpy as np
from mujoco.glfw import glfw

try:
    import cv2
except Exception:
    cv2 = None

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from puppersim import pupper_constants
from puppersimMJX.pupper_brax_policy_bundle import BraxPolicyBundle


# Mouse callback globals.
_button_left = False
_button_middle = False
_button_right = False
_lastx = 0.0
_lasty = 0.0


def _parse_env_kwargs(raw: str) -> Dict:
    raw = raw.strip()
    if not raw:
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


def _mouse_button(window, button, act, mods) -> None:
    del button, act, mods
    global _button_left, _button_middle, _button_right
    _button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    _button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    _button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
    glfw.get_cursor_pos(window)


def _mouse_move(window, xpos, ypos, model, scene, cam) -> None:
    global _lastx, _lasty, _button_left, _button_middle, _button_right
    dx = xpos - _lastx
    dy = ypos - _lasty
    _lastx = xpos
    _lasty = ypos
    if (not _button_left) and (not _button_middle) and (not _button_right):
        return

    _, height = glfw.get_window_size(window)
    shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS) or (
        glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    )
    if _button_right:
        action = mj.mjtMouse.mjMOUSE_MOVE_H if shift else mj.mjtMouse.mjMOUSE_MOVE_V
    elif _button_left:
        action = mj.mjtMouse.mjMOUSE_ROTATE_H if shift else mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, dx / max(1, height), dy / max(1, height), scene, cam)


def _scroll(window, xoffset, yoffset, model, scene, cam) -> None:
    del window, xoffset
    mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, cam)


def _key_callback(window, key, scancode, act, mods, model, data, cmd) -> None:
    del window, scancode, mods
    if act != glfw.PRESS:
        return
    if key == glfw.KEY_BACKSPACE or key == glfw.KEY_R:
        mj.mj_resetData(model, data)
        if model.nkey > 0:
            try:
                mj.mj_resetDataKeyframe(model, data, 0)
            except Exception:
                pass
        mj.mj_forward(model, data)
        cmd[:] = 0.0


def _build_obs_fn(model, data, include_command: bool, exclude_xy: bool):
    def _build(command: np.ndarray) -> np.ndarray:
        q = np.asarray(data.qpos, dtype=np.float32)
        qd = np.asarray(data.qvel, dtype=np.float32)
        if exclude_xy and q.shape[0] > 2:
            q = q[2:]
        obs = np.concatenate([q, qd], axis=0)
        if include_command:
            obs = np.concatenate([obs, command.astype(np.float32)], axis=0)
        return obs

    return _build


def _sample_goal_xy(rng: np.random.Generator, base_xy: np.ndarray, r_min: float, r_max: float) -> np.ndarray:
    radius = float(rng.uniform(r_min, r_max))
    angle = float(rng.uniform(-np.pi, np.pi))
    return np.array([base_xy[0] + radius * np.cos(angle), base_xy[1] + radius * np.sin(angle)], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-path", type=Path, default=Path("puppersim/data/pupper_v2_final_stable_cam_goal.xml"))
    parser.add_argument("--camera-name", type=str, default="front_cam")
    parser.add_argument("--simend", type=float, default=0.0, help="<=0 means run until window close")
    parser.add_argument("--print-camera-config", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--inset-scale", type=float, default=0.40)
    parser.add_argument("--show-grayscale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--goal-site-name", type=str, default="goal_marker")
    parser.add_argument("--goal-z", type=float, default=0.02)
    parser.add_argument("--goal-radius-min", type=float, default=0.8)
    parser.add_argument("--goal-radius-max", type=float, default=2.5)
    parser.add_argument("--goal-tolerance", type=float, default=0.25)
    parser.add_argument("--goal-seed", type=int, default=0)

    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument(
        "--env-kwargs",
        type=str,
        default="puppersimMJX/tasks/cc_locomotion/config/pupper_brax_env_kwargs.command_locomotion.json",
    )
    parser.add_argument("--policy-dt", type=float, default=0.04)
    parser.add_argument("--action-scale", type=float, default=0.75)
    parser.add_argument("--lin-x-min", type=float, default=-0.75)
    parser.add_argument("--lin-x-max", type=float, default=0.75)
    parser.add_argument("--lin-y-min", type=float, default=-0.5)
    parser.add_argument("--lin-y-max", type=float, default=0.5)
    parser.add_argument("--yaw-min", type=float, default=-2.0)
    parser.add_argument("--yaw-max", type=float, default=2.0)
    parser.add_argument("--x-speed", type=float, default=0.4)
    parser.add_argument("--y-speed", type=float, default=0.6)
    parser.add_argument("--yaw-speed", type=float, default=1.0)
    args = parser.parse_args()

    model = mj.MjModel.from_xml_path(str(args.xml_path))
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    if model.nkey > 0:
        try:
            mj.mj_resetDataKeyframe(model, data, 0)
        except Exception:
            pass
    mj.mj_forward(model, data)

    # Optional policy setup.
    bundle = None
    cmd = np.zeros(3, dtype=np.float32)
    obs_hist = None
    obs_hist_len = 1
    build_obs = None
    act_dim = int(model.nu)
    low = np.asarray(pupper_constants.MOTOR_ACTION_LOWER_LIMIT, dtype=np.float32)[:act_dim]
    high = np.asarray(pupper_constants.MOTOR_ACTION_UPPER_LIMIT, dtype=np.float32)[:act_dim]
    action_mid = 0.5 * (low + high)
    action_half = 0.5 * (high - low)
    if args.bundle_dir is not None:
        bundle = BraxPolicyBundle(args.bundle_dir)
        env_cfg = _parse_env_kwargs(args.env_kwargs)
        include_command = bool(env_cfg.get("include_command_in_obs", True))
        exclude_xy = bool(env_cfg.get("exclude_xy_from_obs", True))
        obs_hist_len = int(max(1, env_cfg.get("observation_history", 20)))
        build_obs = _build_obs_fn(model, data, include_command=include_command, exclude_xy=exclude_xy)
        base_obs = build_obs(cmd)
        expected_dim = obs_hist_len * int(base_obs.shape[0])
        if expected_dim != bundle.obs_dim:
            raise ValueError(
                f"Bundle obs_dim={bundle.obs_dim} != inferred {expected_dim} "
                f"(history={obs_hist_len}, base={base_obs.shape[0]})."
            )
        obs_hist = np.tile(base_obs[None, :], (obs_hist_len, 1))
        act_dim = min(act_dim, int(bundle.action_dim), int(low.shape[0]), int(high.shape[0]))
        low = low[:act_dim]
        high = high[:act_dim]
        action_mid = action_mid[:act_dim]
        action_half = action_half[:act_dim]

    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    window = glfw.create_window(1200, 900, "Camera + Policy Preview", None, None)
    if window is None:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window.")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)
    scene = mj.MjvScene(model, maxgeom=20000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

    glfw.set_key_callback(window, lambda w, k, s, a, m: _key_callback(w, k, s, a, m, model, data, cmd))
    glfw.set_mouse_button_callback(window, _mouse_button)
    glfw.set_cursor_pos_callback(window, lambda w, x, y: _mouse_move(w, x, y, model, scene, cam))
    glfw.set_scroll_callback(window, lambda w, x, y: _scroll(w, x, y, model, scene, cam))

    camera_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, args.camera_name))
    if camera_id < 0:
        raise ValueError(f"Camera '{args.camera_name}' not found in model.")
    base_body_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base_link"))
    if base_body_id < 0:
        raise ValueError("Body 'base_link' not found in model.")
    goal_site_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, args.goal_site_name))
    goal_enabled = goal_site_id >= 0
    goal_rng = np.random.default_rng(int(args.goal_seed))
    goals_collected = 0
    goal_xy = np.zeros((2,), dtype=np.float64)
    if goal_enabled:
        base_xy0 = np.asarray(data.xpos[base_body_id, :2], dtype=np.float64)
        goal_xy = _sample_goal_xy(goal_rng, base_xy0, float(args.goal_radius_min), float(args.goal_radius_max))
        model.site_pos[goal_site_id, 0] = goal_xy[0]
        model.site_pos[goal_site_id, 1] = goal_xy[1]
        model.site_pos[goal_site_id, 2] = float(args.goal_z)
        print(f"goal_marker={args.goal_site_name} enabled (site_id={goal_site_id})")
    else:
        print(f"goal_marker={args.goal_site_name} not found in XML (visual marker disabled)")

    policy_next_t = 0.0
    wall_prev = time.monotonic()
    cmd_rng_x = (float(args.lin_x_min), float(args.lin_x_max))
    cmd_rng_y = (float(args.lin_y_min), float(args.lin_y_max))
    cmd_rng_yaw = (float(args.yaw_min), float(args.yaw_max))

    while not glfw.window_should_close(window):
        now = time.monotonic()
        wall_dt = max(1e-6, now - wall_prev)
        wall_prev = now

        # Update command from held key state (reliable across GLFW repeat configs).
        vx_in = float(glfw.get_key(window, glfw.KEY_A) == glfw.PRESS) - float(
            glfw.get_key(window, glfw.KEY_D) == glfw.PRESS
        )
        vy_in = float(glfw.get_key(window, glfw.KEY_S) == glfw.PRESS) - float(
            glfw.get_key(window, glfw.KEY_W) == glfw.PRESS
        )
        yaw_in = float(glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS) - float(
            glfw.get_key(window, glfw.KEY_E) == glfw.PRESS
        )
        # RC-style hold behavior: command returns to 0 when keys are released.
        cmd[0] = vx_in * float(args.x_speed)
        cmd[1] = vy_in * float(args.y_speed)
        cmd[2] = yaw_in * float(args.yaw_speed)
        cmd = _clip_command(cmd, cmd_rng_x, cmd_rng_y, cmd_rng_yaw)

        if bundle is not None and build_obs is not None and obs_hist is not None:
            while data.time >= policy_next_t:
                base_obs = build_obs(cmd)
                obs_hist = np.concatenate([obs_hist[1:], base_obs[None, :]], axis=0)
                obs = obs_hist.reshape(-1)
                a_unit = np.asarray(bundle.deterministic_action(obs), dtype=np.float32)[:act_dim]
                target = action_mid + float(args.action_scale) * action_half * a_unit
                data.ctrl[:act_dim] = np.clip(target, low, high)
                policy_next_t += float(args.policy_dt)

        # Simulate at ~60Hz visual frame rate.
        t_prev = data.time
        while data.time - t_prev < 1.0 / 60.0:
            mj.mj_step(model, data)

        if goal_enabled:
            base_xy = np.asarray(data.xpos[base_body_id, :2], dtype=np.float64)
            dist = float(np.linalg.norm(goal_xy - base_xy))
            if dist <= float(args.goal_tolerance):
                goals_collected += 1
                goal_xy = _sample_goal_xy(
                    goal_rng,
                    base_xy,
                    float(args.goal_radius_min),
                    float(args.goal_radius_max),
                )
                print(f"[collect] t={data.time:.2f}s goals_collected={goals_collected}")
            model.site_pos[goal_site_id, 0] = goal_xy[0]
            model.site_pos[goal_site_id, 1] = goal_xy[1]
            model.site_pos[goal_site_id, 2] = float(args.goal_z)

        if args.simend > 0 and data.time >= args.simend:
            break

        viewport_w, viewport_h = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, viewport_w, viewport_h)
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)

        # Inset camera view.
        inset_w = int(max(1, args.inset_scale * 640))
        inset_h = int(max(1, args.inset_scale * 480))
        inset_x = int(viewport_w - inset_w)
        inset_y = int(viewport_h - inset_h)
        inset = mj.MjrRect(inset_x, inset_y, inset_w, inset_h)

        fixed_cam = mj.MjvCamera()
        fixed_cam.type = mj.mjtCamera.mjCAMERA_FIXED
        fixed_cam.fixedcamid = camera_id
        mj.mjv_updateScene(model, data, opt, None, fixed_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(inset, scene, context)

        if args.show_grayscale and cv2 is not None:
            rgb = np.zeros((inset_h, inset_w, 3), dtype=np.uint8)
            depth = np.zeros((inset_h, inset_w), dtype=np.float32)
            mj.mjr_readPixels(rgb, depth, inset, context)
            gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
            bw = np.stack([gray] * 3, axis=-1).astype(np.uint8)
            bw = cv2.flip(bw, 0)
            cv2.imshow("camera_grayscale", cv2.cvtColor(bw, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if args.print_camera_config:
            print(
                f"cam.azimuth={cam.azimuth:.3f}; cam.elevation={cam.elevation:.3f}; "
                f"cam.distance={cam.distance:.3f}; cam.lookat=np.array([{cam.lookat[0]:.4f}, {cam.lookat[1]:.4f}, {cam.lookat[2]:.4f}])"
            )

        glfw.swap_buffers(window)
        glfw.poll_events()

    if cv2 is not None:
        cv2.destroyAllWindows()
    glfw.terminate()


if __name__ == "__main__":
    main()
