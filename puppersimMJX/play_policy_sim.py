#!/usr/bin/env python3
"""MuJoCo viewer for camera presentation + optional policy testing.

Mode A (camera presentation): run without --bundle-dir.
Mode B (policy test): pass --bundle-dir containing policy_bundle.json/.npz.
"""

from __future__ import annotations

import argparse
import collections
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

_APRILTAG_LEVEL4_GRID = (
    "11111111111",
    "10001000001",
    "10001011101",
    "10000010001",
    "10001010001",
    "10000000001",
    "11111111111",
)

_POLICY_PRESETS = {
    "cc_locomotion": {
        "bundle_dir": "puppersimMJX/pretrained_policies/cc_locomotion/policy_bundle",
        "env_kwargs": "puppersimMJX/tasks/cc_locomotion/config/pupper_brax_env_kwargs.command_locomotion.json",
    },
    "single_target_baseline_seed1": {
        "bundle_dir": "puppersimMJX/pretrained_policies/single_target_baseline_seed1/policy_bundle",
        "env_kwargs": "puppersimMJX/tasks/apriltag_walls/config/pupper_brax_env_kwargs.apriltag_walls_camera_nopriv_hl_level2.json",
    },
    "single_target_icm_seed1": {
        "bundle_dir": "puppersimMJX/pretrained_policies/single_target_icm_seed1/policy_bundle",
        "env_kwargs": "puppersimMJX/tasks/apriltag_walls/config/pupper_brax_env_kwargs.apriltag_walls_camera_nopriv_hl_level2.json",
    },
    "multi_target_baseline_seed3": {
        "bundle_dir": "puppersimMJX/pretrained_policies/multi_target_baseline_seed3/policy_bundle",
        "env_kwargs": "puppersimMJX/tasks/apriltag_walls/config/pupper_brax_env_kwargs.apriltag_walls_camera_nopriv_hl_level3.json",
    },
    "multi_target_icm_seed3": {
        "bundle_dir": "puppersimMJX/pretrained_policies/multi_target_icm_seed3/policy_bundle",
        "env_kwargs": "puppersimMJX/tasks/apriltag_walls/config/pupper_brax_env_kwargs.apriltag_walls_camera_nopriv_hl_level3.json",
    },
}


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


def _resolve_xml_path(
    xml_path_arg: Path,
    env_cfg: Dict,
    argv_tokens: set[str],
) -> Path:
    # If user explicitly passed --xml-path, honor it.
    if "--xml-path" in argv_tokens:
        p = Path(xml_path_arg).expanduser()
        return p if p.is_absolute() else (Path(_REPO_ROOT) / p).resolve()

    # Otherwise prefer model_path from env kwargs.
    env_model_path = str(env_cfg.get("model_path", "")).strip()
    if env_model_path:
        p = Path(env_model_path).expanduser()
        p = p if p.is_absolute() else (Path(_REPO_ROOT) / p).resolve()
        if p.exists():
            return p

    # Fallback to provided parser default, then common MJX XMLs.
    p = Path(xml_path_arg).expanduser()
    p = p if p.is_absolute() else (Path(_REPO_ROOT) / p).resolve()
    if p.exists():
        return p
    for candidate in (
        Path(_REPO_ROOT) / "puppersim/data/pupper_v2_apriltag_room.xml",
        Path(_REPO_ROOT) / "puppersim/data/pupper_v2_final_stable_cam.xml",
        Path(_REPO_ROOT) / "puppersim/data/pupper_v2_final_stable.xml",
    ):
        if candidate.exists():
            return candidate.resolve()
    return p


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


def _decode_command_action(action: np.ndarray, xr: Tuple[float, float], yr: Tuple[float, float], yrw: Tuple[float, float]) -> np.ndarray:
    a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    return np.asarray(
        [
            0.5 * (xr[0] + xr[1]) + 0.5 * (xr[1] - xr[0]) * a[0],
            0.5 * (yr[0] + yr[1]) + 0.5 * (yr[1] - yr[0]) * a[1],
            0.5 * (yrw[0] + yrw[1]) + 0.5 * (yrw[1] - yrw[0]) * a[2],
        ],
        dtype=np.float32,
    )


def _sample_goal_xy(rng: np.random.Generator, base_xy: np.ndarray, r_min: float, r_max: float) -> np.ndarray:
    radius = float(rng.uniform(r_min, r_max))
    angle = float(rng.uniform(-np.pi, np.pi))
    return np.array([base_xy[0] + radius * np.cos(angle), base_xy[1] + radius * np.sin(angle)], dtype=np.float64)


def _apply_reset_qd_noise(model: mj.MjModel, data: mj.MjData, scale: float, rng: np.random.Generator) -> None:
    if float(scale) <= 0.0:
        return
    data.qvel[:] = np.asarray(data.qvel, dtype=np.float64) + float(scale) * rng.standard_normal(model.nv)


def _apply_spawn_xy_if_configured(
    data: mj.MjData,
    spawn_positions: Optional[np.ndarray],
    spawn_jitter_m: float,
    rng: np.random.Generator,
) -> None:
    if spawn_positions is None or spawn_positions.size == 0:
        return
    idx = int(rng.integers(0, int(spawn_positions.shape[0])))
    xy = np.asarray(spawn_positions[idx], dtype=np.float64).copy()
    if float(spawn_jitter_m) > 0.0:
        xy += rng.uniform(-float(spawn_jitter_m), float(spawn_jitter_m), size=(2,))
    if data.qpos.shape[0] >= 2:
        data.qpos[0] = float(xy[0])
        data.qpos[1] = float(xy[1])


def _sample_lateral(rng: np.random.Generator, span: float, num_slots: int) -> float:
    if int(num_slots) > 1:
        slot = int(rng.integers(0, int(num_slots)))
        t = float(slot) / float(int(num_slots) - 1)
        return float(-span + (2.0 * span * t))
    return float(rng.uniform(-span, span))


def _wall_id_from_normal(inward: np.ndarray) -> int:
    nx = float(inward[0])
    ny = float(inward[1])
    if abs(ny) >= abs(nx):
        return 0 if ny < 0.0 else 1
    return 2 if nx < 0.0 else 3


def _build_apriltag_face_candidates_from_grid(
    *,
    grid_rows: tuple[str, ...],
    cell_size: float,
    corner_margin: float,
) -> list[dict]:
    rows = len(grid_rows)
    cols = len(grid_rows[0]) if rows > 0 else 0
    if rows <= 0 or cols <= 0:
        return []

    occ = np.zeros((rows, cols), dtype=np.int32)
    for r, row in enumerate(grid_rows):
        if len(row) != cols:
            raise ValueError("apriltag wall grid rows must all have equal length")
        for c, ch in enumerate(row):
            if ch not in ("0", "1"):
                raise ValueError("apriltag wall grid must contain only '0' and '1'")
            occ[r, c] = 1 if ch == "1" else 0

    free_cells = [(r, c) for r in range(rows) for c in range(cols) if occ[r, c] == 0]
    if not free_cells:
        raise ValueError("apriltag wall grid has no free cells")
    start = (rows // 2, cols // 2) if occ[rows // 2, cols // 2] == 0 else free_cells[0]
    q = collections.deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                continue
            if occ[rr, cc] != 0:
                continue
            nxt = (rr, cc)
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append(nxt)
    if len(seen) != len(free_cells):
        raise ValueError("apriltag wall grid free space is not fully connected")

    x0 = 0.5 * (cols - 1)
    y0 = 0.5 * (rows - 1)
    half = 0.5 * float(cell_size)
    half_len = half - max(0.0, float(corner_margin))
    if half_len <= 1e-5:
        raise ValueError("apriltag_wall_corner_margin too large for grid cell size")

    candidates: list[dict] = []
    face_dirs = (
        (-1, 0, np.array([0.0, 1.0, 0.0], dtype=np.float64), np.array([1.0, 0.0], dtype=np.float64)),
        (1, 0, np.array([0.0, -1.0, 0.0], dtype=np.float64), np.array([1.0, 0.0], dtype=np.float64)),
        (0, -1, np.array([-1.0, 0.0, 0.0], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64)),
        (0, 1, np.array([1.0, 0.0, 0.0], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64)),
    )
    face_id = 0
    for r in range(rows):
        for c in range(cols):
            if occ[r, c] != 1:
                continue
            cx = (float(c) - x0) * float(cell_size)
            cy = (y0 - float(r)) * float(cell_size)
            for dr, dc, inward, tangent in face_dirs:
                rr = r + dr
                cc = c + dc
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    continue
                if occ[rr, cc] != 0:
                    continue
                center = np.array([cx + inward[0] * half, cy + inward[1] * half], dtype=np.float64)
                candidates.append(
                    {
                        "face_id": int(face_id),
                        "center": center,
                        "inward": inward.copy(),
                        "tangent": tangent.copy(),
                        "half_length": float(half_len),
                        "orient_wall_id": int(_wall_id_from_normal(inward)),
                    }
                )
                face_id += 1
    return candidates


def _sample_apriltag_goal_xy_from_faces(
    rng: np.random.Generator,
    face_candidates: list[dict],
    wall_num_slots: int,
    wall_surface_inset: float,
) -> Tuple[np.ndarray, int, int, np.ndarray]:
    if not face_candidates:
        raise ValueError("face_candidates is empty")
    candidate = face_candidates[int(rng.integers(0, len(face_candidates)))]
    half_len = float(candidate["half_length"])
    if int(wall_num_slots) > 1:
        slot = int(rng.integers(0, int(wall_num_slots)))
        t = float(slot) / float(int(wall_num_slots) - 1)
        lateral = (-1.0 + 2.0 * t) * half_len
    else:
        lateral = float(rng.uniform(-half_len, half_len))
    center = np.asarray(candidate["center"], dtype=np.float64)
    tangent = np.asarray(candidate["tangent"], dtype=np.float64)
    inward = np.asarray(candidate["inward"], dtype=np.float64)
    goal_xy = center + lateral * tangent + float(max(0.0, wall_surface_inset)) * inward[:2]
    return goal_xy, int(candidate["face_id"]), int(candidate["orient_wall_id"]), inward


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


def _quat_rotate_wxyz(quat_wxyz: np.ndarray, vec_xyz: np.ndarray) -> np.ndarray:
    qv = np.array([0.0, vec_xyz[0], vec_xyz[1], vec_xyz[2]], dtype=np.float64)
    out = _quat_mul(_quat_mul(quat_wxyz.astype(np.float64), qv), _quat_inv(quat_wxyz.astype(np.float64)))
    return out[1:].astype(np.float64)


def _camera_like_features(
    base_pos: np.ndarray,
    base_quat_wxyz: np.ndarray,
    tag_pos: np.ndarray,
    wall_rects: Optional[list[tuple[float, float, float, float]]],
    front_camera_offset_in_body: np.ndarray,
    front_camera_forward_in_body: np.ndarray,
    front_camera_up_in_body: np.ndarray,
    camera_tan_half_fov: float,
    camera_obs_uv_bins: int,
    camera_obs_tag_half_size_m: float,
) -> Dict[str, float]:
    cam_pos = base_pos + _quat_rotate_wxyz(base_quat_wxyz, front_camera_offset_in_body)
    cam_fwd = _quat_rotate_wxyz(base_quat_wxyz, front_camera_forward_in_body)
    cam_up = _quat_rotate_wxyz(base_quat_wxyz, front_camera_up_in_body)
    cam_fwd = cam_fwd / max(1e-6, float(np.linalg.norm(cam_fwd)))
    cam_up = cam_up / max(1e-6, float(np.linalg.norm(cam_up)))
    cam_right = np.cross(cam_fwd, cam_up)
    cam_right = cam_right / max(1e-6, float(np.linalg.norm(cam_right)))

    rel = tag_pos - cam_pos
    dist = max(1e-6, float(np.linalg.norm(rel)))
    rel_dir = rel / dist

    z_cam = float(np.dot(rel_dir, cam_fwd))
    rel_z = float(np.dot(rel, cam_fwd))
    x_cam = float(np.dot(rel_dir, cam_right))
    y_cam = float(np.dot(rel_dir, cam_up))
    tan_half = max(1e-6, float(camera_tan_half_fov))
    inv_z = 1.0 / max(z_cam, 1e-4)
    u_raw = (x_cam * inv_z) / tan_half
    v_raw = (y_cam * inv_z) / tan_half
    within = (abs(u_raw) <= 1.0) and (abs(v_raw) <= 1.0)
    u = np.clip(u_raw, -1.0, 1.0)
    v = np.clip(v_raw, -1.0, 1.0)
    if int(camera_obs_uv_bins) > 1:
        bins = float(camera_obs_uv_bins)
        step = 2.0 / max(bins - 1.0, 1.0)
        u = np.clip(np.round((u + 1.0) / step) * step - 1.0, -1.0, 1.0)
        v = np.clip(np.round((v + 1.0) / step) * step - 1.0, -1.0, 1.0)
    occluded = _is_occluded_by_room_walls_np(
        cam_xy=cam_pos[:2],
        tag_xy=tag_pos[:2],
        wall_rects=wall_rects,
    )
    visible = 1.0 if (z_cam > 0.0 and within and (not occluded)) else 0.0
    u = float(u * visible)
    v = float(v * visible)
    centering = max(0.0, 1.0 - np.sqrt(min(1.0, float(u * u + v * v))))
    centering = float(centering * visible)
    dist_norm = 1.0 / (1.0 + dist / 2.0)
    depth = max(rel_z, 1e-4)
    apparent_scale_raw = (2.0 * float(camera_obs_tag_half_size_m)) / (depth * tan_half)
    apparent_scale_raw = float(np.clip(apparent_scale_raw, 0.0, 1.0))
    apparent_scale = apparent_scale_raw * visible
    return {
        "visible": visible,
        "forward_cos": float(z_cam),
        "u": float(u),
        "v": float(v),
        "centering": float(centering),
        "distance_norm": float(dist_norm),
        "apparent_scale": float(apparent_scale),
        "apparent_scale_raw": float(apparent_scale_raw),
    }


def _is_occluded_by_room_walls_np(
    cam_xy: np.ndarray,
    tag_xy: np.ndarray,
    wall_rects: Optional[list[tuple[float, float, float, float]]],
) -> bool:
    if not wall_rects:
        return False

    x0 = float(cam_xy[0])
    y0 = float(cam_xy[1])
    x1 = float(tag_xy[0])
    y1 = float(tag_xy[1])
    dx = x1 - x0
    dy = y1 - y0
    eps = 1e-8
    eps_t = 1e-3
    big = 1e6

    for xmin, ymin, xmax, ymax in wall_rects:
        if xmin <= x0 <= xmax and ymin <= y0 <= ymax:
            return True

        tmin = -big
        tmax = big
        if abs(dx) < eps:
            if x0 < xmin or x0 > xmax:
                continue
        else:
            tx1 = (xmin - x0) / dx
            tx2 = (xmax - x0) / dx
            tmin = max(tmin, min(tx1, tx2))
            tmax = min(tmax, max(tx1, tx2))

        if abs(dy) < eps:
            if y0 < ymin or y0 > ymax:
                continue
        else:
            ty1 = (ymin - y0) / dy
            ty2 = (ymax - y0) / dy
            tmin = max(tmin, min(ty1, ty2))
            tmax = min(tmax, max(ty1, ty2))

        if tmax < 0.0:
            continue
        if tmin > tmax:
            continue

        hit_t = tmin if tmin >= 0.0 else tmax
        if hit_t < 0.0:
            continue
        if eps_t <= hit_t <= (1.0 - eps_t):
            return True
    return False


def _forward_clearance_from_room(
    cam_pos: np.ndarray,
    cam_fwd: np.ndarray,
    wall_inner_pos: Optional[Dict[int, float]],
    max_clearance_m: float,
) -> float:
    if wall_inner_pos is None:
        return 1.0
    x = float(cam_pos[0])
    y = float(cam_pos[1])
    fx = float(cam_fwd[0])
    fy = float(cam_fwd[1])
    eps = 1e-6
    big = 1e6
    x_min = float(wall_inner_pos[3])
    x_max = float(wall_inner_pos[2])
    y_min = float(wall_inner_pos[1])
    y_max = float(wall_inner_pos[0])
    inside = (x_min <= x <= x_max) and (y_min <= y <= y_max)
    if not inside:
        return 0.0

    tx = []
    if abs(fx) > eps:
        t = (x_min - x) / fx
        y_hit = y + t * fy
        tx.append(t if (t > 0.0 and y_min <= y_hit <= y_max) else big)
        t = (x_max - x) / fx
        y_hit = y + t * fy
        tx.append(t if (t > 0.0 and y_min <= y_hit <= y_max) else big)
    else:
        tx.extend([big, big])
    if abs(fy) > eps:
        t = (y_min - y) / fy
        x_hit = x + t * fx
        tx.append(t if (t > 0.0 and x_min <= x_hit <= x_max) else big)
        t = (y_max - y) / fy
        x_hit = x + t * fx
        tx.append(t if (t > 0.0 and x_min <= x_hit <= x_max) else big)
    else:
        tx.extend([big, big])

    dist = min(tx)
    max_d = max(1e-3, float(max_clearance_m))
    dist = min(dist, max_d)
    return float(np.clip(dist / max_d, 0.0, 1.0))


def _build_room_wall_rects(model: mj.MjModel) -> list[tuple[float, float, float, float]]:
    rects: list[tuple[float, float, float, float]] = []
    for gid in range(int(model.ngeom)):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid)
        if not name or not str(name).startswith("room_wall"):
            continue
        # Box geoms encode XY half extents in geom_size[:, :2].
        sx = float(abs(model.geom_size[gid, 0]))
        sy = float(abs(model.geom_size[gid, 1]))
        if sx <= 1e-6 or sy <= 1e-6:
            continue
        cx = float(model.geom_pos[gid, 0])
        cy = float(model.geom_pos[gid, 1])
        rects.append((cx - sx, cy - sy, cx + sx, cy + sy))
    return rects


def _forward_clearance_from_rects(
    cam_pos: np.ndarray,
    cam_fwd: np.ndarray,
    wall_rects: list[tuple[float, float, float, float]],
    max_clearance_m: float,
) -> float:
    if not wall_rects:
        return 1.0

    x = float(cam_pos[0])
    y = float(cam_pos[1])
    fx = float(cam_fwd[0])
    fy = float(cam_fwd[1])
    ray_norm = float(np.hypot(fx, fy))
    if ray_norm <= 1e-8:
        return 1.0
    fx /= ray_norm
    fy /= ray_norm

    eps = 1e-8
    max_d = max(1e-3, float(max_clearance_m))
    best_t = max_d

    for xmin, ymin, xmax, ymax in wall_rects:
        # If camera starts inside any wall volume, treat as fully blocked.
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return 0.0

        tmin = -1e12
        tmax = 1e12

        if abs(fx) < eps:
            if x < xmin or x > xmax:
                continue
        else:
            tx1 = (xmin - x) / fx
            tx2 = (xmax - x) / fx
            tmin = max(tmin, min(tx1, tx2))
            tmax = min(tmax, max(tx1, tx2))

        if abs(fy) < eps:
            if y < ymin or y > ymax:
                continue
        else:
            ty1 = (ymin - y) / fy
            ty2 = (ymax - y) / fy
            tmin = max(tmin, min(ty1, ty2))
            tmax = min(tmax, max(ty1, ty2))

        if tmax < 0.0:
            continue
        if tmin > tmax:
            continue

        hit_t = tmin if tmin >= 0.0 else tmax
        if hit_t < 0.0:
            continue
        if hit_t < best_t:
            best_t = hit_t

    return float(np.clip(best_t / max_d, 0.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-path", type=Path, default=Path("puppersim/data/pupper_v2_final_stable_cam_goal.xml"))
    parser.add_argument("--camera-name", type=str, default="front_cam")
    parser.add_argument("--show-camera-features", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--tag-resample-on-collect", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tag-goals-per-episode", type=int, default=1)
    parser.add_argument("--tag-end-episode-on-all-collected", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--tag-inactive-x",
        type=float,
        default=1000.0,
        help="X position used when teleporting collected tag out of scene.",
    )
    parser.add_argument("--tag-inactive-y", type=float, default=1000.0)
    parser.add_argument("--tag-inactive-z", type=float, default=-1000.0)

    parser.add_argument(
        "--policy",
        choices=sorted(_POLICY_PRESETS.keys()),
        default="",
        help="Shortcut for one of the packaged project policies.",
    )
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
    parser.add_argument("--reset-qd-noise-scale", type=float, default=0.0)
    parser.add_argument("--reset-noise-seed", type=int, default=0)
    args = parser.parse_args()

    argv = set(sys.argv[1:])
    if args.policy:
        preset = _POLICY_PRESETS[args.policy]
        if "--bundle-dir" not in argv:
            args.bundle_dir = Path(preset["bundle_dir"])
        if "--env-kwargs" not in argv:
            args.env_kwargs = preset["env_kwargs"]

    env_cfg = _parse_env_kwargs(args.env_kwargs)
    reset_qd_noise_scale = (
        float(args.reset_qd_noise_scale)
        if "--reset-qd-noise-scale" in argv
        else float(env_cfg.get("reset_qd_noise_scale", args.reset_qd_noise_scale))
    )
    reset_noise_rng = np.random.default_rng(int(args.reset_noise_seed))

    xml_path = _resolve_xml_path(args.xml_path, env_cfg, argv)
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    if model.nkey > 0:
        try:
            mj.mj_resetDataKeyframe(model, data, 0)
        except Exception:
            pass
    _apply_reset_qd_noise(model, data, reset_qd_noise_scale, reset_noise_rng)
    mj.mj_forward(model, data)

    # Optional policy setup.
    bundle = None
    low_level_bundle = None
    policy_mode = "none"
    cmd = np.zeros(3, dtype=np.float32)
    obs_hist = None
    obs_hist_len = 1
    ll_obs_hist = None
    ll_obs_hist_len = 1
    hl_hold_steps = 1
    hl_hold_counter = 0
    high_level_prev_action = np.zeros(3, dtype=np.float32)
    build_obs = None
    build_ll_obs = None
    act_dim = int(model.nu)
    low = np.asarray(pupper_constants.MOTOR_ACTION_LOWER_LIMIT, dtype=np.float32)[:act_dim]
    high = np.asarray(pupper_constants.MOTOR_ACTION_UPPER_LIMIT, dtype=np.float32)[:act_dim]
    action_mid = 0.5 * (low + high)
    action_half = 0.5 * (high - low)
    if args.bundle_dir is not None:
        bundle = BraxPolicyBundle(args.bundle_dir)
        if int(bundle.action_dim) == 3 and bool(env_cfg.get("use_low_level_policy", False)):
            policy_mode = "hierarchical"
            ll_bundle_path = env_cfg.get("low_level_policy_bundle_path", "")
            if not ll_bundle_path:
                raise ValueError("High-level policy requires low_level_policy_bundle_path in env kwargs.")
            low_level_bundle = BraxPolicyBundle(ll_bundle_path)
            ll_obs_hist_len = int(max(1, env_cfg.get("low_level_obs_history", 20)))
            ll_include_command = bool(env_cfg.get("low_level_include_command_in_obs", True))
            exclude_xy = bool(env_cfg.get("exclude_xy_from_obs", True))
            build_ll_obs = _build_obs_fn(model, data, include_command=ll_include_command, exclude_xy=exclude_xy)
            ll_base_obs = build_ll_obs(cmd)
            ll_expected_dim = ll_obs_hist_len * int(ll_base_obs.shape[0])
            if ll_expected_dim != low_level_bundle.obs_dim:
                raise ValueError(
                    f"Low-level bundle obs_dim={low_level_bundle.obs_dim} != inferred {ll_expected_dim} "
                    f"(history={ll_obs_hist_len}, base={ll_base_obs.shape[0]})."
                )
            ll_obs_hist = np.tile(ll_base_obs[None, :], (ll_obs_hist_len, 1))
            hl_hz = float(env_cfg.get("high_level_policy_hz", 0.0))
            hl_hold_steps = int(max(1, round((1.0 / hl_hz) / float(args.policy_dt)))) if hl_hz > 0.0 else 1
            act_dim = min(act_dim, int(low_level_bundle.action_dim), int(low.shape[0]), int(high.shape[0]))
            low = low[:act_dim]
            high = high[:act_dim]
            action_mid = action_mid[:act_dim]
            action_half = action_half[:act_dim]
        else:
            policy_mode = "direct"
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
        print(f"warning: camera '{args.camera_name}' not found in model; camera inset/detection disabled.")
        tag_detector = None
        args.show_grayscale = False
        args.show_camera_features = False
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
    if "--tag-collect-radius" in argv:
        tag_collect_radius = float(args.tag_collect_radius)
    else:
        reward_cfg = env_cfg.get("reward_config", {})
        tag_collect_radius = float(reward_cfg.get("collect_radius_m", args.tag_collect_radius))
    if "--tag-deactivate-on-collect" in argv or "--no-tag-deactivate-on-collect" in argv:
        tag_deactivate_on_collect = bool(args.tag_deactivate_on_collect)
    else:
        tag_deactivate_on_collect = bool(env_cfg.get("apriltag_deactivate_on_collect", args.tag_deactivate_on_collect))
    if "--tag-end-episode-on-collect" in argv or "--no-tag-end-episode-on-collect" in argv:
        tag_end_episode_on_collect = bool(args.tag_end_episode_on_collect)
    else:
        tag_end_episode_on_collect = bool(env_cfg.get("apriltag_end_episode_on_collect", args.tag_end_episode_on_collect))
    if "--tag-resample-on-collect" in argv or "--no-tag-resample-on-collect" in argv:
        tag_resample_on_collect = bool(args.tag_resample_on_collect)
    else:
        tag_resample_on_collect = bool(env_cfg.get("apriltag_resample_on_collect", args.tag_resample_on_collect))
    if "--tag-goals-per-episode" in argv:
        tag_goals_per_episode = int(max(1, args.tag_goals_per_episode))
    else:
        tag_goals_per_episode = int(max(1, env_cfg.get("apriltag_goals_per_episode", args.tag_goals_per_episode)))
    if "--tag-end-episode-on-all-collected" in argv or "--no-tag-end-episode-on-all-collected" in argv:
        tag_end_episode_on_all_collected = bool(args.tag_end_episode_on_all_collected)
    else:
        tag_end_episode_on_all_collected = bool(
            env_cfg.get("apriltag_end_episode_on_all_collected", args.tag_end_episode_on_all_collected)
        )
    tag_collect_requires_visible = bool(env_cfg.get("apriltag_collect_requires_visible", True))
    tag_collect_close_scale_threshold = float(env_cfg.get("apriltag_collect_close_scale_threshold", 0.95))
    tag_collect_close_forward_cos_threshold = float(
        env_cfg.get("apriltag_collect_close_forward_cos_threshold", 0.20)
    )
    tag_inactive_xyz = np.array([float(args.tag_inactive_x), float(args.tag_inactive_y), float(args.tag_inactive_z)], dtype=np.float64)
    tag_wall_layout = str(env_cfg.get("apriltag_wall_layout", "single_room")).strip().lower()
    tag_wall_cell_size = float(env_cfg.get("apriltag_wall_cell_size", 0.6))
    tag_wall_corner_margin = float(env_cfg.get("apriltag_wall_corner_margin", 0.12))
    tag_wall_surface_inset = float(max(0.0, env_cfg.get("apriltag_wall_surface_inset", 0.004)))
    tag_face_candidates: list[dict] = []
    if tag_wall_layout in {"level4_mult_room", "level4_multi_room", "mult_room_level4"}:
        tag_face_candidates = _build_apriltag_face_candidates_from_grid(
            grid_rows=_APRILTAG_LEVEL4_GRID,
            cell_size=tag_wall_cell_size,
            corner_margin=tag_wall_corner_margin,
        )
        print(
            f"info: apriltag wall layout={tag_wall_layout} face_candidates={len(tag_face_candidates)} "
            f"cell_size={tag_wall_cell_size:.3f} corner_margin={tag_wall_corner_margin:.3f}"
        )
    tag_active = good_geom_id >= 0
    tag_collect_count = 0
    tag_episode_collected = 0
    tag_wall_ids_seen: set[int] = set()
    room_wall_rects = _build_room_wall_rects(model)

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

    def _resample_and_apply_tag_pose() -> Optional[Tuple[np.ndarray, int, int, np.ndarray]]:
        if not tag_randomize_on_reset or good_geom_id < 0:
            return None
        if tag_face_candidates:
            goal_xy, wall_id, orient_wall_id, inward = _sample_apriltag_goal_xy_from_faces(
                tag_rng,
                face_candidates=tag_face_candidates,
                wall_num_slots=tag_num_slots,
                wall_surface_inset=tag_wall_surface_inset,
            )
        else:
            goal_xy, orient_wall_id, inward = _sample_apriltag_goal_xy(
                tag_rng,
                wall_offset=tag_wall_offset,
                wall_span=tag_wall_span,
                front_wall_only=tag_front_only,
                wall_num_slots=tag_num_slots,
            )
            wall_id = int(orient_wall_id)
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
            good_wall_id=orient_wall_id,
            good_xy=goal_xy,
            tag_height=tag_height,
            wall_offset=tag_wall_offset,
            wall_span=tag_wall_span,
            wall_surface_inset=tag_wall_surface_inset,
            wall_inner_pos=(None if tag_face_candidates else wall_inner_pos),
        )
        return goal_xy, wall_id, orient_wall_id, inward

    tag_init = _resample_and_apply_tag_pose()
    if tag_init is not None:
        goal_xy, wall_id, orient_wall_id, _ = tag_init
        tag_wall_ids_seen.add(int(wall_id))
        gpos = np.asarray(model.geom_pos[good_geom_id], dtype=np.float64)
        gquat = np.asarray(model.geom_quat[good_geom_id], dtype=np.float64)
        gnorm = _geom_normal_from_quat_wxyz(gquat)
        print(
            f"[tag] init wall_id={wall_id} orient_wall_id={orient_wall_id} "
            f"sampled=({goal_xy[0]:+.3f},{goal_xy[1]:+.3f},{tag_height:+.3f}) "
            f"applied=({gpos[0]:+.3f},{gpos[1]:+.3f},{gpos[2]:+.3f}) "
            f"normal=({gnorm[0]:+.3f},{gnorm[1]:+.3f},{gnorm[2]:+.3f})"
        )
    goal_site_id = int(mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, args.goal_site_name))
    goal_enabled = goal_site_id >= 0
    goal_rng = np.random.default_rng(int(args.goal_seed))
    goals_collected = 0
    has_last_good_seen = 0.0
    last_good_u = 0.0
    last_good_v = 0.0
    steps_since_good_seen = 0.0
    goal_xy = np.zeros((2,), dtype=np.float64)
    spawn_positions = None
    spawn_jitter_m = float(max(0.0, env_cfg.get("apriltag_reset_spawn_jitter_m", 0.0)))
    try:
        raw_spawn_positions = env_cfg.get("apriltag_reset_spawn_positions", None)
        if raw_spawn_positions:
            arr = np.asarray(raw_spawn_positions, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] == 2:
                spawn_positions = arr
    except Exception:
        spawn_positions = None
    spawn_rng = np.random.default_rng(int(args.tag_seed) + 9001)
    _apply_spawn_xy_if_configured(
        data=data,
        spawn_positions=spawn_positions,
        spawn_jitter_m=spawn_jitter_m,
        rng=spawn_rng,
    )
    mj.mj_forward(model, data)
    if spawn_positions is not None and spawn_positions.size > 0:
        print(f"[spawn] init base_xy=({float(data.qpos[0]):+.3f},{float(data.qpos[1]):+.3f})")
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
    front_camera_offset_in_body = np.asarray(
        env_cfg.get("front_camera_offset_in_body", [0.0001, -0.147, -0.0064]), dtype=np.float64
    )
    front_camera_forward_in_body = np.asarray(
        env_cfg.get("front_camera_forward_in_body", [0.0, -1.0, 0.0]), dtype=np.float64
    )
    front_camera_up_in_body = np.asarray(env_cfg.get("front_camera_up_in_body", [0.0, 0.0, 1.0]), dtype=np.float64)
    camera_obs_fov_deg = float(env_cfg.get("camera_obs_fov_deg", 70.0))
    camera_tan_half_fov = float(np.tan(0.5 * np.deg2rad(camera_obs_fov_deg)))
    camera_obs_uv_bins = int(max(0, env_cfg.get("camera_obs_uv_bins", 0)))
    camera_obs_tag_half_size_m = float(max(1e-6, env_cfg.get("camera_obs_tag_half_size_m", 0.20)))
    camera_obs_include_forward_clearance = bool(env_cfg.get("camera_obs_include_forward_clearance", False))
    camera_obs_max_clearance_m = float(max(1e-3, env_cfg.get("camera_obs_max_clearance_m", 3.0)))

    def _current_camera_features(tag_geom_id: int) -> Dict[str, float]:
        if tag_geom_id < 0 or float(model.geom_rgba[tag_geom_id, 3]) <= 0.0:
            return {
                "visible": 0.0,
                "forward_cos": 0.0,
                "u": 0.0,
                "v": 0.0,
                "centering": 0.0,
                "distance_norm": 0.0,
                "apparent_scale": 0.0,
                "apparent_scale_raw": 0.0,
            }
        base_pos = np.asarray(data.xpos[base_body_id], dtype=np.float64)
        base_quat = np.asarray(data.xquat[base_body_id], dtype=np.float64)
        tag_pos = np.asarray(model.geom_pos[tag_geom_id], dtype=np.float64)
        return _camera_like_features(
            base_pos=base_pos,
            base_quat_wxyz=base_quat,
            tag_pos=tag_pos,
            wall_rects=room_wall_rects,
            front_camera_offset_in_body=front_camera_offset_in_body,
            front_camera_forward_in_body=front_camera_forward_in_body,
            front_camera_up_in_body=front_camera_up_in_body,
            camera_tan_half_fov=camera_tan_half_fov,
            camera_obs_uv_bins=camera_obs_uv_bins,
            camera_obs_tag_half_size_m=camera_obs_tag_half_size_m,
        )

    def _current_forward_clearance() -> float:
        base_pos = np.asarray(data.xpos[base_body_id], dtype=np.float64)
        base_quat = np.asarray(data.xquat[base_body_id], dtype=np.float64)
        cam_pos = base_pos + _quat_rotate_wxyz(base_quat, front_camera_offset_in_body)
        cam_fwd = _quat_rotate_wxyz(base_quat, front_camera_forward_in_body)
        cam_fwd = cam_fwd / max(1e-6, float(np.linalg.norm(cam_fwd)))
        if room_wall_rects:
            return _forward_clearance_from_rects(
                cam_pos=cam_pos,
                cam_fwd=cam_fwd,
                wall_rects=room_wall_rects,
                max_clearance_m=camera_obs_max_clearance_m,
            )
        return _forward_clearance_from_room(
            cam_pos=cam_pos,
            cam_fwd=cam_fwd,
            wall_inner_pos=wall_inner_pos,
            max_clearance_m=camera_obs_max_clearance_m,
        )

    def _pack_camera_features(feat: Dict[str, float], include_privileged: bool) -> list[float]:
        if include_privileged:
            return [
                float(feat["visible"]),
                float(feat["u"]),
                float(feat["v"]),
                float(feat["forward_cos"]),
                float(feat["centering"]),
                float(feat["distance_norm"]),
            ]
        return [
            float(feat["visible"]),
            float(feat["u"]),
            float(feat["v"]),
            float(feat["centering"]),
            float(feat["apparent_scale"]),
        ]

    def _build_high_level_obs(prev_action_command: np.ndarray) -> np.ndarray:
        include_privileged = str(env_cfg.get("high_level_obs_mode", "camera_nopriv")).strip().lower() == "camera"
        obs_vals = _pack_camera_features(_current_camera_features(good_geom_id), include_privileged)
        if bool(env_cfg.get("apriltag_use_bad_tag", True)):
            obs_vals.extend(_pack_camera_features(_current_camera_features(bad_geom_id), include_privileged))
        if camera_obs_include_forward_clearance:
            obs_vals.append(float(_current_forward_clearance()))
        if bool(env_cfg.get("reward_requires_command", False)) and bool(env_cfg.get("include_command_in_obs", True)):
            obs_vals.extend([float(x) for x in cmd])
        if bool(env_cfg.get("apriltag_include_goals_collected_in_obs", False)):
            obs_vals.append(float(tag_episode_collected))
        if bool(env_cfg.get("apriltag_include_prev_action_in_obs", False)):
            obs_vals.extend([float(x) for x in prev_action_command])
        if bool(env_cfg.get("apriltag_include_last_good_seen_in_obs", False)):
            obs_vals.extend(
                [
                    float(has_last_good_seen),
                    float(last_good_u),
                    float(last_good_v),
                    float(steps_since_good_seen),
                ]
            )
        return np.asarray(obs_vals, dtype=np.float32)

    if policy_mode == "hierarchical":
        base_obs = _build_high_level_obs(high_level_prev_action)
        if int(base_obs.shape[0]) != int(bundle.obs_dim):
            raise ValueError(f"High-level bundle obs_dim={bundle.obs_dim} != inferred {base_obs.shape[0]}.")
        obs_hist = base_obs[None, :]
        obs_hist_len = 1
        print(
            f"loaded high-level policy preset={args.policy or '<custom>'} "
            f"bundle={args.bundle_dir} low_level={env_cfg.get('low_level_policy_bundle_path')}"
        )

    while not glfw.window_should_close(window):
        if reset_requested[0]:
            mj.mj_resetData(model, data)
            if model.nkey > 0:
                try:
                    mj.mj_resetDataKeyframe(model, data, 0)
                except Exception:
                    pass
            _apply_reset_qd_noise(model, data, reset_qd_noise_scale, reset_noise_rng)
            _apply_spawn_xy_if_configured(
                data=data,
                spawn_positions=spawn_positions,
                spawn_jitter_m=spawn_jitter_m,
                rng=spawn_rng,
            )
            mj.mj_forward(model, data)
            if spawn_positions is not None and spawn_positions.size > 0:
                print(f"[spawn] reset base_xy=({float(data.qpos[0]):+.3f},{float(data.qpos[1]):+.3f})")
            cmd[:] = 0.0
            high_level_prev_action[:] = 0.0
            hl_hold_counter = 0
            has_last_good_seen = 0.0
            last_good_u = 0.0
            last_good_v = 0.0
            steps_since_good_seen = 0.0
            # Re-arm policy timing after simulation time jumps back to 0.
            policy_next_t = 0.0
            if policy_mode == "hierarchical" and build_ll_obs is not None and ll_obs_hist is not None:
                base_obs = _build_high_level_obs(high_level_prev_action)
                obs_hist = base_obs[None, :]
                ll_base_obs = build_ll_obs(cmd)
                ll_obs_hist = np.tile(ll_base_obs[None, :], (ll_obs_hist_len, 1))
            elif bundle is not None and build_obs is not None and obs_hist is not None:
                base_obs = build_obs(cmd)
                obs_hist = np.tile(base_obs[None, :], (obs_hist_len, 1))
            tag_episode_collected = 0
            tag_new = _resample_and_apply_tag_pose()
            tag_active = good_geom_id >= 0
            if tag_new is not None:
                goal_xy_r, wall_id_r, orient_wall_id_r, _ = tag_new
                tag_wall_ids_seen.add(int(wall_id_r))
                gpos = np.asarray(model.geom_pos[good_geom_id], dtype=np.float64)
                gquat = np.asarray(model.geom_quat[good_geom_id], dtype=np.float64)
                gnorm = _geom_normal_from_quat_wxyz(gquat)
                print(
                    f"[tag] reset t={data.time:.2f}s wall_id={wall_id_r} orient_wall_id={orient_wall_id_r} "
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
        if policy_mode != "hierarchical":
            # RC-style hold behavior: command returns to 0 when keys are released.
            cmd[0] = vx_in * float(args.x_speed)
            cmd[1] = vy_in * float(args.y_speed)
            cmd[2] = yaw_in * float(args.yaw_speed)
            cmd = _clip_command(cmd, cmd_rng_x, cmd_rng_y, cmd_rng_yaw)

        if policy_mode == "hierarchical" and bundle is not None and low_level_bundle is not None and build_ll_obs is not None:
            while data.time >= policy_next_t:
                if hl_hold_counter <= 0:
                    obs = _build_high_level_obs(high_level_prev_action)
                    a_unit_hl = np.asarray(bundle.deterministic_action(obs), dtype=np.float32)[:3]
                    high_level_prev_action = a_unit_hl.copy()
                    cmd = _decode_command_action(a_unit_hl, cmd_rng_x, cmd_rng_y, cmd_rng_yaw)
                    hl_hold_counter = int(hl_hold_steps)
                    good_feat_now = _current_camera_features(good_geom_id)
                    if float(good_feat_now["visible"]) > 0.5:
                        has_last_good_seen = 1.0
                        last_good_u = float(good_feat_now["u"])
                        last_good_v = float(good_feat_now["v"])
                        steps_since_good_seen = 0.0
                    else:
                        steps_since_good_seen += 1.0
                hl_hold_counter -= 1

                ll_base_obs = build_ll_obs(cmd)
                ll_obs_hist = np.concatenate([ll_obs_hist[1:], ll_base_obs[None, :]], axis=0)
                ll_obs = ll_obs_hist.reshape(-1)
                a_unit = np.asarray(low_level_bundle.deterministic_action(ll_obs), dtype=np.float32)[:act_dim]
                target = action_mid + float(args.action_scale) * action_half * a_unit
                data.ctrl[:act_dim] = np.clip(target, low, high)
                policy_next_t += float(args.policy_dt)
        elif bundle is not None and build_obs is not None and obs_hist is not None:
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
            collect_ok = tag_dist <= tag_collect_radius
            if collect_ok and tag_collect_requires_visible:
                base_pos = np.asarray(data.xpos[base_body_id], dtype=np.float64)
                base_quat = np.asarray(data.xquat[base_body_id], dtype=np.float64)
                good_pos = np.asarray(model.geom_pos[good_geom_id], dtype=np.float64)
                good_feat = _camera_like_features(
                    base_pos=base_pos,
                    base_quat_wxyz=base_quat,
                    tag_pos=good_pos,
                    wall_rects=room_wall_rects,
                    front_camera_offset_in_body=front_camera_offset_in_body,
                    front_camera_forward_in_body=front_camera_forward_in_body,
                    front_camera_up_in_body=front_camera_up_in_body,
                    camera_tan_half_fov=camera_tan_half_fov,
                    camera_obs_uv_bins=camera_obs_uv_bins,
                    camera_obs_tag_half_size_m=camera_obs_tag_half_size_m,
                )
                visible_ok = float(good_feat["visible"]) > 0.5
                close_override = (
                    float(good_feat["apparent_scale_raw"]) >= float(tag_collect_close_scale_threshold)
                    and float(good_feat["forward_cos"]) >= float(tag_collect_close_forward_cos_threshold)
                )
                collect_ok = bool(visible_ok or close_override)
            if collect_ok:
                tag_collect_count += 1
                tag_episode_collected += 1
                has_last_good_seen = 0.0
                last_good_u = 0.0
                last_good_v = 0.0
                steps_since_good_seen = 0.0
                print(
                    f"[tag-collect] t={data.time:.2f}s count={tag_collect_count} "
                    f"dist={tag_dist:.3f} radius={tag_collect_radius:.3f}"
                )
                episode_target_reached = tag_episode_collected >= int(tag_goals_per_episode)
                should_respawn_now = bool(tag_resample_on_collect) and (not episode_target_reached)
                if should_respawn_now:
                    tag_new = _resample_and_apply_tag_pose()
                    tag_active = good_geom_id >= 0
                    if tag_new is not None:
                        goal_xy_c, wall_id_c, orient_wall_id_c, _ = tag_new
                        tag_wall_ids_seen.add(int(wall_id_c))
                        gpos = np.asarray(model.geom_pos[good_geom_id], dtype=np.float64)
                        gquat = np.asarray(model.geom_quat[good_geom_id], dtype=np.float64)
                        gnorm = _geom_normal_from_quat_wxyz(gquat)
                        print(
                            f"[tag] respawn t={data.time:.2f}s wall_id={wall_id_c} orient_wall_id={orient_wall_id_c} "
                            f"sampled=({goal_xy_c[0]:+.3f},{goal_xy_c[1]:+.3f},{tag_height:+.3f}) "
                            f"applied=({gpos[0]:+.3f},{gpos[1]:+.3f},{gpos[2]:+.3f}) "
                            f"normal=({gnorm[0]:+.3f},{gnorm[1]:+.3f},{gnorm[2]:+.3f}) "
                            f"episode_collected={tag_episode_collected}/{tag_goals_per_episode}"
                        )
                else:
                    tag_active = False
                    if tag_deactivate_on_collect:
                        model.geom_pos[good_geom_id, :] = tag_inactive_xyz
                        model.geom_rgba[good_geom_id, 3] = 0.0
                        mj.mj_forward(model, data)
                if tag_end_episode_on_collect:
                    reset_requested[0] = True
                if tag_end_episode_on_all_collected and episode_target_reached:
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

        tag_status = "AprilTag: detector disabled"
        if camera_id >= 0:
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
        else:
            tag_status = "Camera: unavailable"

        camera_feat_lines = []
        if bool(args.show_camera_features):
            base_pos = np.asarray(data.xpos[base_body_id], dtype=np.float64)
            base_quat = np.asarray(data.xquat[base_body_id], dtype=np.float64)
            if good_geom_id >= 0:
                good_pos = np.asarray(model.geom_pos[good_geom_id], dtype=np.float64)
                good_feat = _camera_like_features(
                    base_pos=base_pos,
                    base_quat_wxyz=base_quat,
                    tag_pos=good_pos,
                    wall_rects=room_wall_rects,
                    front_camera_offset_in_body=front_camera_offset_in_body,
                    front_camera_forward_in_body=front_camera_forward_in_body,
                    front_camera_up_in_body=front_camera_up_in_body,
                    camera_tan_half_fov=camera_tan_half_fov,
                    camera_obs_uv_bins=camera_obs_uv_bins,
                    camera_obs_tag_half_size_m=camera_obs_tag_half_size_m,
                )
                camera_feat_lines.append(
                    "GOOD vis={:.0f} u={:+.2f} v={:+.2f} ctr={:.2f} scale={:.2f}".format(
                        good_feat["visible"],
                        good_feat["u"],
                        good_feat["v"],
                        good_feat["centering"],
                        good_feat["apparent_scale"],
                    )
                )
            else:
                camera_feat_lines.append("GOOD n/a")

            if bad_geom_id >= 0 and bool(tag_use_bad) and float(model.geom_rgba[bad_geom_id, 3]) > 0.0:
                bad_pos = np.asarray(model.geom_pos[bad_geom_id], dtype=np.float64)
                bad_feat = _camera_like_features(
                    base_pos=base_pos,
                    base_quat_wxyz=base_quat,
                    tag_pos=bad_pos,
                    wall_rects=room_wall_rects,
                    front_camera_offset_in_body=front_camera_offset_in_body,
                    front_camera_forward_in_body=front_camera_forward_in_body,
                    front_camera_up_in_body=front_camera_up_in_body,
                    camera_tan_half_fov=camera_tan_half_fov,
                    camera_obs_uv_bins=camera_obs_uv_bins,
                    camera_obs_tag_half_size_m=camera_obs_tag_half_size_m,
                )
                camera_feat_lines.append(
                    "BAD  vis={:.0f} u={:+.2f} v={:+.2f} ctr={:.2f} scale={:.2f}".format(
                        bad_feat["visible"],
                        bad_feat["u"],
                        bad_feat["v"],
                        bad_feat["centering"],
                        bad_feat["apparent_scale"],
                    )
                )
            elif bad_geom_id >= 0 and bool(tag_use_bad):
                camera_feat_lines.append("BAD  inactive")

            if camera_obs_include_forward_clearance:
                cam_pos = base_pos + _quat_rotate_wxyz(base_quat, front_camera_offset_in_body)
                cam_fwd = _quat_rotate_wxyz(base_quat, front_camera_forward_in_body)
                cam_fwd = cam_fwd / max(1e-6, float(np.linalg.norm(cam_fwd)))
                if room_wall_rects:
                    fclr = _forward_clearance_from_rects(
                        cam_pos=cam_pos,
                        cam_fwd=cam_fwd,
                        wall_rects=room_wall_rects,
                        max_clearance_m=camera_obs_max_clearance_m,
                    )
                else:
                    fclr = _forward_clearance_from_room(
                        cam_pos=cam_pos,
                        cam_fwd=cam_fwd,
                        wall_inner_pos=wall_inner_pos,
                        max_clearance_m=camera_obs_max_clearance_m,
                    )
                camera_feat_lines.append(f"FCLR {fclr:.2f}")

        overlay_lines = [tag_status]
        overlay_lines.extend(camera_feat_lines)
        mj.mjr_overlay(
            mj.mjtFontScale.mjFONTSCALE_150,
            mj.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            "Tag Status / Camera Features",
            "\n".join(overlay_lines),
            context,
        )

        if args.print_camera_config:
            print(
                f"cam.azimuth={cam.azimuth:.3f}; cam.elevation={cam.elevation:.3f}; "
                f"cam.distance={cam.distance:.3f}; cam.lookat=np.array([{cam.lookat[0]:.4f}, {cam.lookat[1]:.4f}, {cam.lookat[2]:.4f}])"
            )

        glfw.swap_buffers(window)
        glfw.poll_events()

    if tag_wall_ids_seen:
        print(f"[tag] wall_ids_seen={sorted(tag_wall_ids_seen)}")
    if cv2 is not None:
        cv2.destroyAllWindows()
    glfw.terminate()


if __name__ == "__main__":
    main()
