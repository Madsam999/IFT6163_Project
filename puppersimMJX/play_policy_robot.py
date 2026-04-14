#!/usr/bin/env python3
"""Run a policy bundle on real Pupper with optional PiCam window over SSH."""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import msgpack
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_STANFORD_ROOT = os.path.join(_REPO_ROOT, "StanfordQuadruped")
if _STANFORD_ROOT not in sys.path:
    sys.path.insert(0, _STANFORD_ROOT)

from djipupper import HardwareInterface
from djipupper.IndividualConfig import SERIAL_PORT
from puppersim import pupper_constants
from puppersimMJX.pupper_brax_policy_bundle import BraxPolicyBundle


def _parse_env_kwargs(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}
    p = Path(raw).expanduser()
    if p.exists():
        return dict(json.loads(p.read_text()))
    return dict(json.loads(raw))


def _rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(0.5 * roll)
    sr = np.sin(0.5 * roll)
    cp = np.cos(0.5 * pitch)
    sp = np.sin(0.5 * pitch)
    cy = np.cos(0.5 * yaw)
    sy = np.sin(0.5 * yaw)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float32)


def _clip_command(cmd: np.ndarray, xr: Tuple[float, float], yr: Tuple[float, float], yrw: Tuple[float, float]) -> np.ndarray:
    out = cmd.copy()
    out[0] = np.clip(out[0], xr[0], xr[1])
    out[1] = np.clip(out[1], yr[0], yr[1])
    out[2] = np.clip(out[2], yrw[0], yrw[1])
    return out


class _TerminalKeyReader:
    def __init__(self):
        self._fd = None
        self._old_attrs = None
        self.enabled = False

    def __enter__(self):
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old_attrs = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self.enabled = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and self._fd is not None and self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)

    def read_keys(self) -> list[str]:
        if not self.enabled:
            return []
        out: list[str] = []
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                break
            ch = sys.stdin.read(1)
            if not ch:
                break
            out.append(ch)
        return out


def _latest_packet(hw: HardwareInterface.HardwareInterface) -> Optional[Dict[str, Any]]:
    latest = None
    while True:
        payload = hw.reader.chew()
        if not payload:
            return latest
        try:
            decoded = msgpack.unpackb(payload)
            if isinstance(decoded, dict):
                latest = decoded
        except Exception:
            continue


def _pick_vec(d: Dict[str, Any], candidates: list[str], n: int) -> Optional[np.ndarray]:
    lower = {str(k).lower(): v for k, v in d.items()}
    for k in candidates:
        if k.lower() not in lower:
            continue
        v = lower[k.lower()]
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= n:
            return np.asarray(v[:n], dtype=np.float32)
    return None


def _pick_scalar(d: Dict[str, Any], candidates: list[str], default: float = 0.0) -> float:
    lower = {str(k).lower(): v for k, v in d.items()}
    for k in candidates:
        if k.lower() in lower:
            try:
                return float(lower[k.lower()])
            except Exception:
                pass
    return float(default)


def _as_joint_matrix(cmd12: np.ndarray) -> np.ndarray:
    return np.asarray(cmd12, dtype=np.float32).reshape(4, 3).T


def _camera_setup(source: str, width: int, height: int, fps: int, show: bool):
    if not show:
        return None
    if cv2 is None:
        print("warning: OpenCV not available, disabling camera window.")
        return None
    if not os.environ.get("DISPLAY"):
        print("warning: DISPLAY not set (SSH headless). Camera window disabled.")
        return None
    src: Any = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"warning: cannot open camera source '{source}', disabling window.")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, int(fps))
    return cap


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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument(
        "--env-kwargs",
        type=str,
        default="puppersimMJX/tasks/cc_locomotion/config/pupper_brax_env_kwargs.command_locomotion.json",
    )
    p.add_argument("--serial-port", type=str, default=SERIAL_PORT)
    p.add_argument("--seconds", type=float, default=0.0, help="<=0 means run until Ctrl+C")
    p.add_argument("--policy-dt", type=float, default=0.02)
    p.add_argument("--action-scale", type=float, default=0.75)
    p.add_argument("--base-height", type=float, default=0.17)
    p.add_argument("--kp", type=float, default=50.0)
    p.add_argument("--kd", type=float, default=5.0)
    p.add_argument("--max-current", type=float, default=7.0)
    p.add_argument("--x-speed", type=float, default=0.4)
    p.add_argument("--y-speed", type=float, default=0.6)
    p.add_argument("--yaw-speed", type=float, default=1.0)
    p.add_argument("--lin-x-min", type=float, default=-0.75)
    p.add_argument("--lin-x-max", type=float, default=0.75)
    p.add_argument("--lin-y-min", type=float, default=-0.5)
    p.add_argument("--lin-y-max", type=float, default=0.5)
    p.add_argument("--yaw-min", type=float, default=-2.0)
    p.add_argument("--yaw-max", type=float, default=2.0)
    p.add_argument("--keyboard-decay-sec", type=float, default=0.12)
    p.add_argument("--show-camera", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--camera-source", type=str, default="0")
    p.add_argument("--camera-width", type=int, default=640)
    p.add_argument("--camera-height", type=int, default=480)
    p.add_argument("--camera-fps", type=int, default=30)
    p.add_argument("--camera-grayscale", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--detect-apriltag", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--apriltag-family", type=str, default="tag36h11")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    bundle = BraxPolicyBundle(args.bundle_dir)
    env_cfg = _parse_env_kwargs(args.env_kwargs)
    include_command = bool(env_cfg.get("include_command_in_obs", True))
    exclude_xy = bool(env_cfg.get("exclude_xy_from_obs", True))
    obs_hist_len = int(max(1, env_cfg.get("observation_history", 20)))

    hw = HardwareInterface.HardwareInterface(port=args.serial_port)
    time.sleep(0.1)
    hw.set_joint_space_parameters(float(args.kp), float(args.kd), float(args.max_current))
    hw.activate()
    print(f"activated on {args.serial_port}")
    print("controls: A/D=x, W/S=y, Q/E=yaw, R=zero cmd, X=exit")

    cap = _camera_setup(
        source=str(args.camera_source),
        width=int(args.camera_width),
        height=int(args.camera_height),
        fps=int(args.camera_fps),
        show=bool(args.show_camera),
    )
    tag_detector = None
    if bool(args.detect_apriltag):
        tag_detector = _create_apriltag_detector(args.apriltag_family)
        if tag_detector is None:
            print("warning: AprilTag detection unavailable (opencv-contrib missing or family unsupported).")

    low = np.asarray(pupper_constants.MOTOR_ACTION_LOWER_LIMIT, dtype=np.float32)
    high = np.asarray(pupper_constants.MOTOR_ACTION_UPPER_LIMIT, dtype=np.float32)
    act_dim = min(12, int(bundle.action_dim), int(low.shape[0]), int(high.shape[0]))
    low = low[:act_dim]
    high = high[:act_dim]
    action_mid = 0.5 * (low + high)
    action_half = 0.5 * (high - low)

    cmd = np.zeros(3, dtype=np.float32)
    cmd_rng_x = (float(args.lin_x_min), float(args.lin_x_max))
    cmd_rng_y = (float(args.lin_y_min), float(args.lin_y_max))
    cmd_rng_yaw = (float(args.yaw_min), float(args.yaw_max))
    key_state = {"x+": 0.0, "x-": 0.0, "y+": 0.0, "y-": 0.0, "yaw+": 0.0, "yaw-": 0.0}

    # Wait for first packet with motor position/velocity.
    state_pkt = None
    t0 = time.monotonic()
    while state_pkt is None and (time.monotonic() - t0) < 3.0:
        state_pkt = _latest_packet(hw)
        time.sleep(0.002)
    if state_pkt is None:
        raise RuntimeError("No telemetry packet received from robot within 3s.")

    def build_obs(pkt: Dict[str, Any], command: np.ndarray) -> np.ndarray:
        pos = _pick_vec(pkt, ["position", "pos", "q", "joint_position"], 12)
        vel = _pick_vec(pkt, ["velocity", "vel", "qd", "joint_velocity"], 12)
        if pos is None:
            pos = np.zeros(12, dtype=np.float32)
        if vel is None:
            vel = np.zeros(12, dtype=np.float32)
        roll = _pick_scalar(pkt, ["roll"], 0.0)
        pitch = _pick_scalar(pkt, ["pitch"], 0.0)
        yaw = _pick_scalar(pkt, ["yaw"], 0.0)
        roll_rate = _pick_scalar(pkt, ["roll_rate", "droll"], 0.0)
        pitch_rate = _pick_scalar(pkt, ["pitch_rate", "dpitch"], 0.0)
        yaw_rate = _pick_scalar(pkt, ["yaw_rate", "dyaw"], 0.0)

        q = np.concatenate(
            [np.array([0.0, 0.0, float(args.base_height)], dtype=np.float32), _rpy_to_quat_wxyz(roll, pitch, yaw), pos],
            axis=0,
        )  # 19
        qd = np.concatenate(
            [np.zeros(3, dtype=np.float32), np.array([roll_rate, pitch_rate, yaw_rate], dtype=np.float32), vel],
            axis=0,
        )  # 18
        if exclude_xy and q.shape[0] > 2:
            q = q[2:]
        obs = np.concatenate([q, qd], axis=0)
        if include_command:
            obs = np.concatenate([obs, command.astype(np.float32)], axis=0)
        return obs

    first_obs = build_obs(state_pkt, cmd)
    expected_dim = obs_hist_len * int(first_obs.shape[0])
    if expected_dim != bundle.obs_dim:
        raise ValueError(
            f"Bundle obs_dim={bundle.obs_dim} != inferred {expected_dim} "
            f"(history={obs_hist_len}, base={first_obs.shape[0]})."
        )
    obs_hist = np.tile(first_obs[None, :], (obs_hist_len, 1))

    t_start = time.monotonic()
    next_tick = t_start
    last_print = t_start
    tag_visible = False
    tag_ids: list[int] = []

    try:
        with _TerminalKeyReader() as key_reader:
            while True:
                now = time.monotonic()
                if args.seconds > 0 and (now - t_start) >= float(args.seconds):
                    break

                # SSH terminal keys: hold via repeated chars + short decay.
                for ch in key_reader.read_keys():
                    t = time.monotonic()
                    c = ch.lower()
                    if c == "x":
                        return
                    if c == "r":
                        cmd[:] = 0.0
                        continue
                    if c == "a":
                        key_state["x+"] = t
                    elif c == "d":
                        key_state["x-"] = t
                    elif c == "s":  # same orientation as play_policy_sim
                        key_state["y+"] = t
                    elif c == "w":
                        key_state["y-"] = t
                    elif c == "q":
                        key_state["yaw+"] = t
                    elif c == "e":
                        key_state["yaw-"] = t

                t = time.monotonic()
                decay = float(args.keyboard_decay_sec)
                vx_in = float((t - key_state["x+"]) < decay) - float((t - key_state["x-"]) < decay)
                vy_in = float((t - key_state["y+"]) < decay) - float((t - key_state["y-"]) < decay)
                yaw_in = float((t - key_state["yaw+"]) < decay) - float((t - key_state["yaw-"]) < decay)
                cmd[0] = vx_in * float(args.x_speed)
                cmd[1] = vy_in * float(args.y_speed)
                cmd[2] = yaw_in * float(args.yaw_speed)
                cmd = _clip_command(cmd, cmd_rng_x, cmd_rng_y, cmd_rng_yaw)

                pkt = _latest_packet(hw)
                if pkt is not None:
                    state_pkt = pkt
                base_obs = build_obs(state_pkt, cmd)
                obs_hist = np.concatenate([obs_hist[1:], base_obs[None, :]], axis=0)
                obs = obs_hist.reshape(-1)
                a_unit = np.asarray(bundle.deterministic_action(obs), dtype=np.float32)[:act_dim]
                target = action_mid + float(args.action_scale) * action_half * a_unit

                cmd12 = np.zeros(12, dtype=np.float32)
                cmd12[:act_dim] = np.clip(target, low, high)
                hw.set_actuator_postions(_as_joint_matrix(cmd12))

                if cap is not None:
                    ok, frame = cap.read()
                    if ok:
                        tag_visible = False
                        tag_ids = []
                        if tag_detector is not None:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            corners, ids, _ = tag_detector.detectMarkers(gray)
                            if ids is not None and len(ids) > 0:
                                tag_visible = True
                                tag_ids = [int(x) for x in ids.reshape(-1).tolist()]
                                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                            status = f"AprilTag: {'YES' if tag_visible else 'NO'}"
                            if tag_ids:
                                status += f" ids={tag_ids}"
                            cv2.putText(
                                frame,
                                status,
                                (10, 28),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0) if tag_visible else (0, 0, 255),
                                2,
                                cv2.LINE_AA,
                            )
                        if bool(args.camera_grayscale):
                            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            frame = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                        cv2.imshow("pupper_picam", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("x"), ord("X")):
                            break

                if now - last_print > 0.5:
                    last_print = now
                    tag_msg = " tag=YES" if tag_visible else " tag=NO"
                    if tag_ids:
                        tag_msg += f" ids={tag_ids}"
                    print(
                        f"cmd x={cmd[0]:+.2f} y={cmd[1]:+.2f} yaw={cmd[2]:+.2f}{tag_msg}",
                        end="\r",
                        flush=True,
                    )

                next_tick += float(args.policy_dt)
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.monotonic()
    finally:
        try:
            hw.deactivate()
        except Exception:
            pass
        if cap is not None:
            cap.release()
        if cv2 is not None:
            cv2.destroyAllWindows()
        print("\nrobot deactivated")


if __name__ == "__main__":
    main()
