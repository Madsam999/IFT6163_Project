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
from typing import Dict, Optional, Tuple

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


def _create_apriltag_detector(family: str):
    if cv2 is None or not hasattr(cv2, "aruco"):
        return None
    fam = str(family).strip().lower()
    mapping = {
        "tag16h5": "DICT_APRILTAG_16h5",
        "tag25h9": "DICT_APRILTAG_25h9",
        "tag36h10": "DICT_APRILTAG_36h10",
        "tag36h11": "DICT_APRILTAG_36h11",
    }
    dict_name = mapping.get(fam, "DICT_APRILTAG_36h11")
    dict_id = getattr(cv2.aruco, dict_name, None)
    if dict_id is None:
        return None
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, params)


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


def _key_callback(window, key, scancode, act, mods, model, data, cmd, reset_flag) -> None:
    del window, scancode, mods
    if act != glfw.PRESS:
        return
    if key == glfw.KEY_BACKSPACE or key == glfw.KEY_R:
        reset_flag[0] = True
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


def _sample_lateral(rng: np.random.Generator, span: float, num_slots: int) -> float:
    if int(num_slots) > 1:
        slot = int(rng.integers(0, int(num_slots)))
        t = float(slot) / float(int(num_slots) - 1)
        return float(-span + (2.0 * span * t))
    return float(rng.uniform(-span, span))


def _sample_apriltag_goal_xy(
    rng: np.random.Generator,
    wall_offset: float,
    wall_span: float,
    front_wall_only: bool,
    wall_num_slots: int,
) -> Tuple[np.ndarray, int, np.ndarray]:
    lateral = _sample_lateral(rng, span=float(wall_span), num_slots=int(wall_num_slots))
    if bool(front_wall_only):
        return np.array([lateral, -wall_offset], dtype=np.float64), 1, np.array([0.0, 1.0, 0.0], dtype=np.float64)
    wall_id = int(rng.integers(0, 4))
    if wall_id == 0:
        return np.array([lateral, wall_offset], dtype=np.float64), 0, np.array([0.0, -1.0, 0.0], dtype=np.float64)
    if wall_id == 1:
        return np.array([lateral, -wall_offset], dtype=np.float64), 1, np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if wall_id == 2:
        return np.array([wall_offset, lateral], dtype=np.float64), 2, np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    return np.array([-wall_offset, lateral], dtype=np.float64), 3, np.array([1.0, 0.0, 0.0], dtype=np.float64)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _quat_inv(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64) / max(1e-12, float(np.dot(q, q)))


def _wall_yaw_from_id(wall_id: int) -> float:
    # User-validated mapping in compiled mesh frame.
    # wall_id: 0 (+Y wall), 1 (-Y wall), 2 (+X wall), 3 (-X wall)
    if int(wall_id) == 0:
        return 0.5 * np.pi
    if int(wall_id) == 1:
        return -0.5 * np.pi
    if int(wall_id) == 2:
        return 0.0
    return np.pi


def _quat_from_wall_id(wall_id: int, display_ref_quat_wxyz: np.ndarray, mesh_quat_wxyz: np.ndarray) -> np.ndarray:
    yaw = _wall_yaw_from_id(int(wall_id))
    q_yaw = np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=np.float64)
    # q_display(wall) = yaw(wall) * q_display_ref, then q_geom = q_display * inv(q_mesh).
    q_display = _quat_mul(q_yaw, display_ref_quat_wxyz.astype(np.float64))
    q = _quat_mul(q_display, _quat_inv(mesh_quat_wxyz.astype(np.float64)))
    q /= max(1e-12, float(np.linalg.norm(q)))
    return q


def _apply_apriltag_pose(
    model: mj.MjModel,
    data: mj.MjData,
    good_geom_id: int,
    bad_geom_id: int,
    good_display_ref_quat: np.ndarray,
    bad_display_ref_quat: np.ndarray,
    good_mesh_quat: np.ndarray,
    bad_mesh_quat: np.ndarray,
    use_bad_tag: bool,
    good_wall_id: int,
    good_xy: np.ndarray,
    tag_height: float,
    wall_offset: float,
    wall_span: float,
    wall_surface_inset: float,
    wall_inner_pos: Optional[Dict[int, float]],
) -> None:
    gx = float(good_xy[0])
    gy = float(good_xy[1])
    if wall_inner_pos is not None:
        if int(good_wall_id) == 0:
            gy = float(wall_inner_pos[0] - wall_surface_inset)
        elif int(good_wall_id) == 1:
            gy = float(wall_inner_pos[1] + wall_surface_inset)
        elif int(good_wall_id) == 2:
            gx = float(wall_inner_pos[2] - wall_surface_inset)
        else:
            gx = float(wall_inner_pos[3] + wall_surface_inset)
    if good_geom_id >= 0:
        model.geom_pos[good_geom_id, 0] = gx
        model.geom_pos[good_geom_id, 1] = gy
        model.geom_pos[good_geom_id, 2] = float(tag_height)
        model.geom_rgba[good_geom_id, 3] = 1.0
        model.geom_quat[good_geom_id, :] = _quat_from_wall_id(
            int(good_wall_id), good_display_ref_quat, good_mesh_quat
        )
    if bad_geom_id >= 0:
        if use_bad_tag:
            bx = float(wall_span)
            by = float(-wall_offset)
            if wall_inner_pos is not None:
                by = float(wall_inner_pos[1] + wall_surface_inset)
            model.geom_pos[bad_geom_id, 0] = bx
            model.geom_pos[bad_geom_id, 1] = by
            model.geom_pos[bad_geom_id, 2] = float(tag_height)
            model.geom_quat[bad_geom_id, :] = _quat_from_wall_id(1, bad_display_ref_quat, bad_mesh_quat)
            model.geom_rgba[bad_geom_id, 3] = 1.0
        else:
            model.geom_rgba[bad_geom_id, 3] = 0.0
    mj.mj_forward(model, data)


def _geom_normal_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    m = np.zeros(9, dtype=np.float64)
    mj.mju_quat2Mat(m, quat_wxyz.astype(np.float64))
    r = m.reshape(3, 3)
    return (r @ np.array([0.0, 0.0, 1.0], dtype=np.float64)).astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-path", type=Path, default=Path("puppersim/data/pupper_v2_final_stable_cam_goal.xml"))
    parser.add_argument("--camera-name", type=str, default="front_cam")
    parser.add_argument("--simend", type=float, default=0.0, help="<=0 means run until window close")
    parser.add_argument("--print-camera-config", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--inset-scale", type=float, default=0.40)
    parser.add_argument("--show-grayscale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--detect-apriltag", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--apriltag-family", type=str, default="tag36h11")
    parser.add_argument(
        "--opencv-input-width",
        type=int,
        default=0,
        help="If >0 with --opencv-input-height, resize camera frame to this width before OpenCV AprilTag detection.",
    )
    parser.add_argument(
        "--opencv-input-height",
        type=int,
        default=0,
        help="If >0 with --opencv-input-width, resize camera frame to this height before OpenCV AprilTag detection.",
    )
    parser.add_argument("--good-tag-id", type=int, default=101)
    parser.add_argument("--bad-tag-id", type=int, default=287)
    parser.add_argument("--goal-site-name", type=str, default="goal_marker")
    parser.add_argument("--goal-z", type=float, default=0.02)
    parser.add_argument("--goal-radius-min", type=float, default=0.8)
    parser.add_argument("--goal-radius-max", type=float, default=2.5)
    parser.add_argument("--goal-tolerance", type=float, default=0.25)
    parser.add_argument("--goal-seed", type=int, default=0)
    parser.add_argument("--tag-randomize-on-reset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tag-randomize-front-wall-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tag-wall-offset", type=float, default=1.95)
    parser.add_argument("--tag-wall-span", type=float, default=0.8)
    parser.add_argument("--tag-height", type=float, default=0.45)
    parser.add_argument("--tag-wall-num-slots", type=int, default=0)
    parser.add_argument("--tag-seed", type=int, default=0)
    parser.add_argument("--tag-collect-radius", type=float, default=0.35)
    parser.add_argument("--tag-deactivate-on-collect", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tag-end-episode-on-collect", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--tag-inactive-x",
        type=float,
        default=1000.0,
        help="X position used when teleporting collected tag out of scene.",
    )
    parser.add_argument("--tag-inactive-y", type=float, default=1000.0)
    parser.add_argument("--tag-inactive-z", type=float, default=-1000.0)

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

    env_cfg = _parse_env_kwargs(args.env_kwargs)
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
    tag_detector = _create_apriltag_detector(args.apriltag_family) if bool(args.detect_apriltag) else None
    if bool(args.detect_apriltag) and tag_detector is None:
        print("warning: AprilTag detection unavailable (opencv-contrib missing or family unsupported).")
    cv_input_w = int(args.opencv_input_width)
    cv_input_h = int(args.opencv_input_height)
    use_cv_resize = cv_input_w > 0 and cv_input_h > 0
    if use_cv_resize:
        print(f"info: OpenCV detector input resized to {cv_input_w}x{cv_input_h}")

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

    reset_requested = [False]
    glfw.set_key_callback(window, lambda w, k, s, a, m: _key_callback(w, k, s, a, m, model, data, cmd, reset_requested))
    glfw.set_mouse_button_callback(window, _mouse_button)
    glfw.set_cursor_pos_callback(window, lambda w, x, y: _mouse_move(w, x, y, model, scene, cam))
    glfw.set_scroll_callback(window, lambda w, x, y: _scroll(w, x, y, model, scene, cam))

    camera_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, args.camera_name))
    if camera_id < 0:
        raise ValueError(f"Camera '{args.camera_name}' not found in model.")
    base_body_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "base_link"))
    if base_body_id < 0:
        raise ValueError("Body 'base_link' not found in model.")
    foot_body_ids = []
    for name in ("leftFrontLowerLeg", "leftRearLowerLeg", "rightFrontLowerLeg", "rightRearLowerLeg"):
        bid = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name))
        if bid >= 0:
            foot_body_ids.append(bid)
    good_geom_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "apriltag_good_panel"))
    bad_geom_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "apriltag_bad_panel"))
    good_base_quat = (
        np.asarray(model.geom_quat[good_geom_id], dtype=np.float64).copy()
        if good_geom_id >= 0
        else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    bad_base_quat = (
        np.asarray(model.geom_quat[bad_geom_id], dtype=np.float64).copy()
        if bad_geom_id >= 0
        else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    good_mesh_quat = (
        np.asarray(model.mesh_quat[int(model.geom_dataid[good_geom_id])], dtype=np.float64).copy()
        if good_geom_id >= 0 and int(model.geom_dataid[good_geom_id]) >= 0
        else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    bad_mesh_quat = (
        np.asarray(model.mesh_quat[int(model.geom_dataid[bad_geom_id])], dtype=np.float64).copy()
        if bad_geom_id >= 0 and int(model.geom_dataid[bad_geom_id]) >= 0
        else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    # Build display-frame reference where wall_id=2 has yaw 0 by definition.
    q_yaw_wall1_inv = np.array([np.cos(0.25 * np.pi), 0.0, 0.0, np.sin(0.25 * np.pi)], dtype=np.float64)
    good_display_init = _quat_mul(good_base_quat, good_mesh_quat)
    bad_display_init = _quat_mul(bad_base_quat, bad_mesh_quat)
    good_display_ref_quat = _quat_mul(q_yaw_wall1_inv, good_display_init)
    bad_display_ref_quat = _quat_mul(q_yaw_wall1_inv, bad_display_init)
    tag_rng = np.random.default_rng(int(args.tag_seed))
    argv = set(sys.argv[1:])
    tag_wall_offset = (
        float(args.tag_wall_offset)
        if "--tag-wall-offset" in argv
        else float(env_cfg.get("apriltag_wall_offset", args.tag_wall_offset))
    )
    tag_wall_span = (
        float(args.tag_wall_span)
        if "--tag-wall-span" in argv
        else float(env_cfg.get("apriltag_wall_span", args.tag_wall_span))
    )
    tag_height = (
        float(args.tag_height)
        if "--tag-height" in argv
        else float(env_cfg.get("apriltag_height", args.tag_height))
    )
    tag_num_slots = (
        int(args.tag_wall_num_slots)
        if "--tag-wall-num-slots" in argv
        else int(env_cfg.get("apriltag_wall_num_slots", args.tag_wall_num_slots))
    )
    if "--tag-randomize-front-wall-only" in argv or "--no-tag-randomize-front-wall-only" in argv:
        tag_front_only = bool(args.tag_randomize_front_wall_only)
    else:
        tag_front_only = bool(env_cfg.get("apriltag_randomize_front_wall_only", args.tag_randomize_front_wall_only))
    if "--tag-randomize-on-reset" in argv or "--no-tag-randomize-on-reset" in argv:
        tag_randomize_on_reset = bool(args.tag_randomize_on_reset)
    else:
        tag_randomize_on_reset = bool(env_cfg.get("apriltag_randomize_good_goal_on_reset", args.tag_randomize_on_reset))
    tag_use_bad = bool(env_cfg.get("apriltag_use_bad_tag", True))
    tag_collect_radius = float(args.tag_collect_radius)
    tag_deactivate_on_collect = bool(args.tag_deactivate_on_collect)
    tag_end_episode_on_collect = bool(args.tag_end_episode_on_collect)
    tag_inactive_xyz = np.array([float(args.tag_inactive_x), float(args.tag_inactive_y), float(args.tag_inactive_z)], dtype=np.float64)
    tag_wall_surface_inset = 0.004
    tag_active = good_geom_id >= 0
    tag_collect_count = 0

    wall_inner_pos = None
    try:
        front_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "room_wall_front"))
        back_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "room_wall_back"))
        right_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "room_wall_right"))
        left_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "room_wall_left"))
        if min(front_id, back_id, right_id, left_id) >= 0:
            front_cy = float(model.geom_pos[front_id, 1])
            back_cy = float(model.geom_pos[back_id, 1])
            right_cx = float(model.geom_pos[right_id, 0])
            left_cx = float(model.geom_pos[left_id, 0])
            front_t = float(model.geom_size[front_id, 1])
            back_t = float(model.geom_size[back_id, 1])
            right_t = float(model.geom_size[right_id, 0])
            left_t = float(model.geom_size[left_id, 0])
            wall_inner_pos = {
                0: back_cy - back_t,    # back wall inner y
                1: front_cy + front_t,  # front wall inner y
                2: right_cx - right_t,  # right wall inner x
                3: left_cx + left_t,    # left wall inner x
            }
    except Exception:
        wall_inner_pos = None

    def _resample_and_apply_tag_pose() -> Optional[Tuple[np.ndarray, int, np.ndarray]]:
        if not tag_randomize_on_reset or good_geom_id < 0:
            return None
        goal_xy, wall_id, inward = _sample_apriltag_goal_xy(
            tag_rng,
            wall_offset=tag_wall_offset,
            wall_span=tag_wall_span,
            front_wall_only=tag_front_only,
            wall_num_slots=tag_num_slots,
        )
        _apply_apriltag_pose(
            model=model,
            data=data,
            good_geom_id=good_geom_id,
            bad_geom_id=bad_geom_id,
            good_display_ref_quat=good_display_ref_quat,
            bad_display_ref_quat=bad_display_ref_quat,
            good_mesh_quat=good_mesh_quat,
            bad_mesh_quat=bad_mesh_quat,
            use_bad_tag=tag_use_bad,
            good_wall_id=wall_id,
            good_xy=goal_xy,
            tag_height=tag_height,
            wall_offset=tag_wall_offset,
            wall_span=tag_wall_span,
            wall_surface_inset=tag_wall_surface_inset,
            wall_inner_pos=wall_inner_pos,
        )
        return goal_xy, wall_id, inward

    tag_init = _resample_and_apply_tag_pose()
    if tag_init is not None:
        goal_xy, wall_id, _ = tag_init
        gpos = np.asarray(model.geom_pos[good_geom_id], dtype=np.float64)
        gquat = np.asarray(model.geom_quat[good_geom_id], dtype=np.float64)
        gnorm = _geom_normal_from_quat_wxyz(gquat)
        print(
            f"[tag] init wall_id={wall_id} sampled=({goal_xy[0]:+.3f},{goal_xy[1]:+.3f},{tag_height:+.3f}) "
            f"applied=({gpos[0]:+.3f},{gpos[1]:+.3f},{gpos[2]:+.3f}) "
            f"normal=({gnorm[0]:+.3f},{gnorm[1]:+.3f},{gnorm[2]:+.3f})"
        )
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
        if reset_requested[0]:
            mj.mj_resetData(model, data)
            if model.nkey > 0:
                try:
                    mj.mj_resetDataKeyframe(model, data, 0)
                except Exception:
                    pass
            mj.mj_forward(model, data)
            cmd[:] = 0.0
            tag_new = _resample_and_apply_tag_pose()
            tag_active = good_geom_id >= 0
            if tag_new is not None:
                goal_xy_r, wall_id_r, _ = tag_new
                gpos = np.asarray(model.geom_pos[good_geom_id], dtype=np.float64)
                gquat = np.asarray(model.geom_quat[good_geom_id], dtype=np.float64)
                gnorm = _geom_normal_from_quat_wxyz(gquat)
                print(
                    f"[tag] reset t={data.time:.2f}s wall_id={wall_id_r} "
                    f"sampled=({goal_xy_r[0]:+.3f},{goal_xy_r[1]:+.3f},{tag_height:+.3f}) "
                    f"applied=({gpos[0]:+.3f},{gpos[1]:+.3f},{gpos[2]:+.3f}) "
                    f"normal=({gnorm[0]:+.3f},{gnorm[1]:+.3f},{gnorm[2]:+.3f})"
                )
            reset_requested[0] = False

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

        if tag_active and good_geom_id >= 0:
            base_xy = np.asarray(data.xpos[base_body_id, :2], dtype=np.float64)
            points_xy = [base_xy]
            for bid in foot_body_ids:
                points_xy.append(np.asarray(data.xpos[bid, :2], dtype=np.float64))
            pts = np.stack(points_xy, axis=0)
            tag_xy = np.asarray(model.geom_pos[good_geom_id, :2], dtype=np.float64)
            dists = np.linalg.norm(pts - tag_xy[None, :], axis=1)
            tag_dist = float(np.min(dists))
            if tag_dist <= tag_collect_radius:
                tag_collect_count += 1
                tag_active = False
                print(
                    f"[tag-collect] t={data.time:.2f}s count={tag_collect_count} "
                    f"dist={tag_dist:.3f} radius={tag_collect_radius:.3f}"
                )
                if tag_deactivate_on_collect:
                    model.geom_pos[good_geom_id, :] = tag_inactive_xyz
                    model.geom_rgba[good_geom_id, 3] = 0.0
                    mj.mj_forward(model, data)
                if tag_end_episode_on_collect:
                    reset_requested[0] = True

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

        tag_status = "AprilTag: detector disabled"
        read_pixels = bool(args.show_grayscale) or (tag_detector is not None)
        if read_pixels and cv2 is not None:
            rgb = np.zeros((inset_h, inset_w, 3), dtype=np.uint8)
            depth = np.zeros((inset_h, inset_w), dtype=np.float32)
            mj.mjr_readPixels(rgb, depth, inset, context)
            rgb = cv2.flip(rgb, 0)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if use_cv_resize:
                gray_cv = cv2.resize(gray, (cv_input_w, cv_input_h), interpolation=cv2.INTER_AREA)
            else:
                gray_cv = gray
            if tag_detector is not None:
                corners, ids, _ = tag_detector.detectMarkers(gray_cv)
                if ids is None or len(ids) == 0:
                    tag_status = "AprilTag: NONE"
                else:
                    tag_ids = [int(x) for x in ids.reshape(-1).tolist()]
                    if int(args.good_tag_id) in tag_ids:
                        tag_status = f"AprilTag: GOOD id={int(args.good_tag_id)}"
                    elif int(args.bad_tag_id) in tag_ids:
                        tag_status = f"AprilTag: BAD id={int(args.bad_tag_id)}"
                    else:
                        tag_status = f"AprilTag: OTHER ids={tag_ids}"
            if args.show_grayscale:
                gray_bgr = cv2.cvtColor(gray_cv, cv2.COLOR_GRAY2BGR)
                if tag_detector is not None and ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedMarkers(gray_bgr, corners, ids)
                cv2.imshow("camera_grayscale", gray_bgr)
                cv2.waitKey(1)
        elif tag_detector is not None:
            tag_status = "AprilTag: readback unavailable"

        mj.mjr_overlay(
            mj.mjtFontScale.mjFONTSCALE_150,
            mj.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            "Tag Status",
            tag_status,
            context,
        )

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
