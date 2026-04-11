"""Run a Brax PPO locomotion policy on the physical DJI Pupper robot.

This runner is intentionally minimal:
1) Loads a Brax policy bundle.
2) Reads robot state from `pupper_hardware_interface`.
3) Builds a Brax-style observation with command + history.
4) Sends position targets at a fixed realtime control rate.
"""

from __future__ import annotations

import argparse
import json
import time
import os
import sys
import select
import termios
import tty
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from puppersim import pupper_constants
from puppersimMJX.pupper_brax_policy_bundle import BraxPolicyBundle


def _parse_env_kwargs(raw: str) -> Dict:
    raw = raw.strip()
    if not raw:
        return {}
    path = Path(raw).expanduser()
    if path.exists():
        return dict(json.loads(path.read_text()))
    return dict(json.loads(raw))


def _clip_command(cmd: np.ndarray, x_rng: Tuple[float, float], y_rng: Tuple[float, float], yaw_rng: Tuple[float, float]) -> np.ndarray:
    out = cmd.copy()
    out[0] = np.clip(out[0], x_rng[0], x_rng[1])
    out[1] = np.clip(out[1], y_rng[0], y_rng[1])
    out[2] = np.clip(out[2], yaw_rng[0], yaw_rng[1])
    return out


def _find_serial_port() -> str:
    from serial.tools import list_ports

    matches = list(list_ports.grep(".*ttyACM0.*"))
    if not matches:
        raise RuntimeError("Could not find ttyACM0 serial port for Pupper.")
    return str(matches[0].device)


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


def _infer_layout(
    bundle: BraxPolicyBundle,
    obs_history: int,
    include_command: bool,
    act_dim: int,
) -> Tuple[int, int, int]:
    if bundle.obs_dim % obs_history != 0:
        raise ValueError(
            f"Bundle obs_dim={bundle.obs_dim} is not divisible by observation_history={obs_history}."
        )
    base_obs_dim = bundle.obs_dim // obs_history
    command_dim = 3 if include_command else 0
    q_plus_qd_dim = base_obs_dim - command_dim
    qd_dim = 6 + act_dim
    q_dim = q_plus_qd_dim - qd_dim
    if q_dim <= 0:
        raise ValueError(
            f"Invalid inferred layout: base_obs_dim={base_obs_dim}, command_dim={command_dim}, qd_dim={qd_dim}."
        )
    return base_obs_dim, q_dim, qd_dim


def _build_base_obs(
    robot_state,
    command: np.ndarray,
    q_dim: int,
    qd_dim: int,
    include_command: bool,
    base_height: float,
) -> np.ndarray:
    motor_pos = np.asarray(robot_state.position, dtype=np.float32)
    motor_vel = np.asarray(robot_state.velocity, dtype=np.float32)
    roll = float(robot_state.roll)
    pitch = float(robot_state.pitch)
    yaw = float(getattr(robot_state, "yaw", 0.0))
    roll_rate = float(robot_state.roll_rate)
    pitch_rate = float(robot_state.pitch_rate)
    yaw_rate = float(getattr(robot_state, "yaw_rate", 0.0))

    quat_wxyz = _rpy_to_quat_wxyz(roll, pitch, yaw)
    full_q = np.concatenate(
        [
            np.array([0.0, 0.0, base_height], dtype=np.float32),  # x, y, z
            quat_wxyz,  # w, x, y, z
            motor_pos,
        ],
        axis=0,
    )  # nominal length 19

    full_qd = np.concatenate(
        [
            np.zeros(3, dtype=np.float32),  # base linear velocity (unknown on hardware)
            np.array([roll_rate, pitch_rate, yaw_rate], dtype=np.float32),  # base angular velocity
            motor_vel,
        ],
        axis=0,
    )  # nominal length 18

    if q_dim == 17:
        q = full_q[2:]  # exclude x, y
    elif q_dim == full_q.shape[0]:
        q = full_q
    elif q_dim < full_q.shape[0]:
        q = full_q[-q_dim:]
    else:
        q = np.concatenate([full_q, np.zeros(q_dim - full_q.shape[0], dtype=np.float32)], axis=0)

    if qd_dim == full_qd.shape[0]:
        qd = full_qd
    elif qd_dim < full_qd.shape[0]:
        qd = full_qd[-qd_dim:]
    else:
        qd = np.concatenate([full_qd, np.zeros(qd_dim - full_qd.shape[0], dtype=np.float32)], axis=0)

    obs = np.concatenate([q.astype(np.float32), qd.astype(np.float32)], axis=0)
    if include_command:
        obs = np.concatenate([obs, command.astype(np.float32)], axis=0)
    return obs


def _set_pose_smooth(hw: Any, start: np.ndarray, end: np.ndarray, duration_sec: float, dt: float) -> None:
    duration = max(0.0, float(duration_sec))
    if duration <= 0.0:
        hw.set_actuator_postions(np.asarray(end, dtype=np.float32))
        return
    n = max(1, int(round(duration / max(1e-4, dt))))
    for i in range(1, n + 1):
        a = float(i) / float(n)
        target = (1.0 - a) * start + a * end
        hw.set_actuator_postions(np.asarray(target, dtype=np.float32))
        time.sleep(dt)


def _auto_export_bundle_from_params(params_path: Path, output_dir: Path) -> None:
    from flax.serialization import from_bytes

    def pick_normalizer(tree):
        for value in tree.values():
            if isinstance(value, dict) and "mean" in value and "std" in value and "std_eps" in value:
                return value
        raise ValueError("Could not find normalizer subtree in params.")

    def pick_policy(tree):
        best = None
        best_out = -1
        for value in tree.values():
            if not isinstance(value, dict) or "params" not in value:
                continue
            params = value["params"]
            if not isinstance(params, dict):
                continue
            layer_names = sorted(
                [k for k in params.keys() if k.startswith("hidden_")],
                key=lambda x: int(x.split("_")[1]),
            )
            if not layer_names:
                continue
            out_bias = params[layer_names[-1]].get("bias")
            if out_bias is None:
                continue
            out_dim = int(np.asarray(out_bias).shape[0])
            if out_dim > best_out:
                best_out = out_dim
                best = params
        if best is None:
            raise ValueError("Could not find policy subtree in params.")
        return best

    params = from_bytes(None, params_path.read_bytes())
    if not isinstance(params, dict):
        raise ValueError(f"Unsupported params root type: {type(params)}")
    norm = pick_normalizer(params)
    policy = pick_policy(params)
    layer_names = sorted([k for k in policy if k.startswith("hidden_")], key=lambda x: int(x.split("_")[1]))
    if not layer_names:
        raise ValueError("No hidden_* layers found in policy subtree.")

    arrays = {
        "obs_mean": np.asarray(norm["mean"], dtype=np.float32),
        "obs_std": np.asarray(norm["std"], dtype=np.float32),
        "obs_std_eps": np.asarray(norm["std_eps"], dtype=np.float32),
    }
    for name in layer_names:
        arrays[f"{name}.kernel"] = np.asarray(policy[name]["kernel"], dtype=np.float32)
        arrays[f"{name}.bias"] = np.asarray(policy[name]["bias"], dtype=np.float32)

    output_dim = int(arrays[f"{layer_names[-1]}.bias"].shape[0])
    action_dim = int(output_dim // 2) if (output_dim % 2 == 0) else int(output_dim)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "policy_bundle.npz", **arrays)
    (output_dir / "policy_bundle.json").write_text(
        json.dumps(
            {
                "source_params_path": str(params_path),
                "activation": "elu",
                "obs_dim": int(arrays["obs_mean"].shape[0]),
                "action_dim": action_dim,
                "policy_output_dim": output_dim,
                "hidden_sizes": [int(arrays[f"{name}.kernel"].shape[1]) for name in layer_names[:-1]],
                "layer_names": layer_names,
                "action_head": "normal_tanh",
                "notes": "Auto-exported by pupper_brax_run_policy_robot.py",
            },
            indent=2,
        )
    )


def _resolve_bundle_dir(checkpoint_dir: Path) -> Path:
    bundle_npz = checkpoint_dir / "policy_bundle.npz"
    bundle_json = checkpoint_dir / "policy_bundle.json"
    if bundle_npz.exists() and bundle_json.exists():
        return checkpoint_dir

    raise FileNotFoundError(
        f"Policy bundle not generated in {checkpoint_dir}. "
        "Please see README for how to generate it "
        "(puppersimMJX/pupper_brax_export_policy_bundle.py)."
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("puppersim/data/pretrained_cc_locomotion"),
        help="Directory containing generated `policy_bundle.npz` and `policy_bundle.json`.",
    )
    parser.add_argument(
        "--env-kwargs",
        type=str,
        default="puppersimMJX/tasks/cc_locomotion/config/pupper_brax_env_kwargs.command_locomotion.json",
        help="Env kwargs json (path or raw json) used during training.",
    )
    parser.add_argument("--policy-dt", type=float, default=0.02, help="Control period in seconds.")
    parser.add_argument("--seconds", type=float, default=60.0, help="Runtime duration.")
    parser.add_argument("--kp", type=float, default=50.0)
    parser.add_argument("--kd", type=float, default=5.0)
    parser.add_argument("--max-current", type=float, default=7.0)
    parser.add_argument("--action-scale", type=float, default=-1.0, help="Override action_scale. Use <0 to read from env kwargs.")
    parser.add_argument("--command-x", type=float, default=0.35)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument(
        "--command-source",
        type=str,
        default="fixed",
        choices=["fixed", "keyboard", "joystick"],
        help="How to update command in realtime.",
    )
    parser.add_argument(
        "--keyboard-control-mode",
        type=str,
        default="increment",
        choices=["increment", "hold"],
        help="increment: tap to adjust command; hold: uses key-repeat with decay timeout.",
    )
    parser.add_argument("--x-speed", type=float, default=0.35, help="|vx| for keyboard hold mode.")
    parser.add_argument("--y-speed", type=float, default=0.25, help="|vy| for keyboard hold mode.")
    parser.add_argument("--yaw-speed", type=float, default=0.8, help="|yaw| for keyboard hold mode.")
    parser.add_argument("--inc-x", type=float, default=0.03, help="Keyboard increment for vx.")
    parser.add_argument("--inc-y", type=float, default=0.03, help="Keyboard increment for vy.")
    parser.add_argument("--inc-yaw", type=float, default=0.08, help="Keyboard increment for yaw.")
    parser.add_argument(
        "--keyboard-command-decay-sec",
        type=float,
        default=0.12,
        help="Hold mode key-repeat timeout before auto-zero.",
    )
    parser.add_argument("--lin-x-min", type=float, default=-0.75)
    parser.add_argument("--lin-x-max", type=float, default=0.75)
    parser.add_argument("--lin-y-min", type=float, default=-0.5)
    parser.add_argument("--lin-y-max", type=float, default=0.5)
    parser.add_argument("--yaw-min", type=float, default=-2.0)
    parser.add_argument("--yaw-max", type=float, default=2.0)
    parser.add_argument("--base-height", type=float, default=0.17, help="Approximate base z used in reconstructed state.")
    parser.add_argument("--startup-pose-seconds", type=float, default=1.5)
    parser.add_argument("--safe-pose-seconds", type=float, default=1.5)
    parser.add_argument(
        "--safe-pose",
        type=str,
        default="-0.2,0.8,-1.2,0.2,0.8,-1.2,-0.2,0.8,-1.2,0.2,0.8,-1.2",
        help="12 joint targets sent at startup and shutdown.",
    )
    return parser.parse_args()


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

    def read_keys(self):
        if not self.enabled:
            return []
        keys = []
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                break
            ch = sys.stdin.read(1)
            if not ch:
                break
            keys.append(ch)
        return keys


def _apply_keyboard_keys(
    command: np.ndarray,
    keys,
    mode: str,
    now_t: float,
    key_state: Dict[str, float],
    x_speed: float,
    y_speed: float,
    yaw_speed: float,
    inc_x: float,
    inc_y: float,
    inc_yaw: float,
    decay_sec: float,
) -> Tuple[np.ndarray, bool]:
    out = command.copy()
    should_stop = False

    for ch in keys:
        c = ch.lower()
        if c == "\x03" or c == "x":
            should_stop = True
            continue
        if mode == "increment":
            if c == "w":
                out[0] += float(inc_x)
            elif c == "s":
                out[0] -= float(inc_x)
            elif c == "a":
                out[1] += float(inc_y)
            elif c == "d":
                out[1] -= float(inc_y)
            elif c == "q":
                out[2] += float(inc_yaw)
            elif c == "e":
                out[2] -= float(inc_yaw)
            elif c == " ":
                out[:] = 0.0
        else:
            if c == "w":
                key_state["vx+"] = now_t
            elif c == "s":
                key_state["vx-"] = now_t
            elif c == "a":
                key_state["vy+"] = now_t
            elif c == "d":
                key_state["vy-"] = now_t
            elif c == "q":
                key_state["yaw+"] = now_t
            elif c == "e":
                key_state["yaw-"] = now_t
            elif c == " ":
                out[:] = 0.0

    if mode == "hold":
        out[:] = 0.0
        active = lambda t: (now_t - t) <= float(decay_sec)
        if active(key_state["vx+"]):
            out[0] += float(x_speed)
        if active(key_state["vx-"]):
            out[0] -= float(x_speed)
        if active(key_state["vy+"]):
            out[1] += float(y_speed)
        if active(key_state["vy-"]):
            out[1] -= float(y_speed)
        if active(key_state["yaw+"]):
            out[2] += float(yaw_speed)
        if active(key_state["yaw-"]):
            out[2] -= float(yaw_speed)

    return out, should_stop


def main() -> None:
    args = _parse_args()
    try:
        from pupper_hardware_interface import interface
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency `pupper_hardware_interface`. "
            "Install it in the active environment on the robot."
        ) from exc

    env_cfg = _parse_env_kwargs(args.env_kwargs)

    bundle_dir = _resolve_bundle_dir(checkpoint_dir=args.checkpoint_dir)
    bundle = BraxPolicyBundle(bundle_dir)

    action_scale = float(args.action_scale)
    if action_scale < 0.0:
        action_scale = float(env_cfg.get("action_scale", 1.0))
    include_command = bool(env_cfg.get("include_command_in_obs", True))
    obs_history = int(max(1, env_cfg.get("observation_history", 1)))

    low = np.asarray(pupper_constants.MOTOR_ACTION_LOWER_LIMIT, dtype=np.float32)
    high = np.asarray(pupper_constants.MOTOR_ACTION_UPPER_LIMIT, dtype=np.float32)
    act_dim = min(int(bundle.action_dim), int(low.shape[0]), int(high.shape[0]))
    if act_dim <= 0:
        raise ValueError("No compatible actuators found.")
    low = low[:act_dim]
    high = high[:act_dim]
    action_mid = 0.5 * (low + high)
    action_half = 0.5 * (high - low)

    base_obs_dim, q_dim, qd_dim = _infer_layout(
        bundle=bundle,
        obs_history=obs_history,
        include_command=include_command,
        act_dim=act_dim,
    )

    x_rng = (float(args.lin_x_min), float(args.lin_x_max))
    y_rng = (float(args.lin_y_min), float(args.lin_y_max))
    yaw_rng = (float(args.yaw_min), float(args.yaw_max))
    command = _clip_command(
        np.array([args.command_x, args.command_y, args.command_yaw], dtype=np.float32),
        x_rng=x_rng,
        y_rng=y_rng,
        yaw_rng=yaw_rng,
    )
    key_state = {
        "vx+": 0.0,
        "vx-": 0.0,
        "vy+": 0.0,
        "vy-": 0.0,
        "yaw+": 0.0,
        "yaw-": 0.0,
    }

    serial_port = _find_serial_port()
    hw = interface.Interface(serial_port)
    time.sleep(0.25)
    hw.set_joint_space_parameters(kp=float(args.kp), kd=float(args.kd), max_current=float(args.max_current))
    hw.read_incoming_data()

    safe_pose = np.fromstring(args.safe_pose, dtype=np.float32, sep=",")
    if safe_pose.shape[0] != act_dim:
        raise ValueError(f"--safe-pose must provide {act_dim} comma-separated values.")

    current_pos = np.asarray(hw.robot_state.position, dtype=np.float32)[:act_dim]
    _set_pose_smooth(
        hw=hw,
        start=current_pos,
        end=safe_pose,
        duration_sec=float(args.startup_pose_seconds),
        dt=float(args.policy_dt),
    )

    first_obs = _build_base_obs(
        robot_state=hw.robot_state,
        command=command,
        q_dim=q_dim,
        qd_dim=qd_dim,
        include_command=include_command,
        base_height=float(args.base_height),
    )
    if first_obs.shape[0] != base_obs_dim:
        raise ValueError(
            f"Constructed base obs dim {first_obs.shape[0]} != inferred base obs dim {base_obs_dim}."
        )
    obs_hist = np.tile(first_obs[None, :], (obs_history, 1))

    print(f"serial_port={serial_port}")
    print(f"bundle_dir={bundle_dir}")
    print(
        f"obs_dim={bundle.obs_dim} base_obs_dim={base_obs_dim} history={obs_history} "
        f"q_dim={q_dim} qd_dim={qd_dim} act_dim={act_dim} action_scale={action_scale:.3f}"
    )
    print(f"initial_command=[{command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f}]")
    print(f"command_source={args.command_source}")
    if args.command_source == "keyboard":
        print("keyboard controls: W/S -> vx, A/D -> vy, Q/E -> yaw, SPACE -> zero, X/Ctrl-C -> stop")
        if args.keyboard_control_mode == "increment":
            print("keyboard mode: increment")
        else:
            print("keyboard mode: hold (uses key-repeat with decay)")
    elif args.command_source == "joystick":
        print("joystick controls: reading puppersim.JoystickInterface (FrSky/BetaFPV HID)")
    print(f"running for {args.seconds:.1f}s at dt={args.policy_dt:.3f}s")

    wall_start = time.time()
    step = 0
    last_log = wall_start
    joystick_control = None
    if args.command_source == "joystick":
        from puppersim.JoystickInterface import JoystickInterface

        joystick_control = JoystickInterface(config=None)

    with _TerminalKeyReader() as key_reader:
        if args.command_source == "keyboard" and not key_reader.enabled:
            raise RuntimeError("Keyboard mode requires an interactive TTY. Run through `ssh -t` / deploy_to_robot.sh.")
        try:
            while True:
                now = time.time()
                if (now - wall_start) >= float(args.seconds):
                    break

                should_stop = False
                if args.command_source == "keyboard":
                    keys = key_reader.read_keys()
                    command, should_stop = _apply_keyboard_keys(
                        command=command,
                        keys=keys,
                        mode=str(args.keyboard_control_mode),
                        now_t=now,
                        key_state=key_state,
                        x_speed=float(args.x_speed),
                        y_speed=float(args.y_speed),
                        yaw_speed=float(args.yaw_speed),
                        inc_x=float(args.inc_x),
                        inc_y=float(args.inc_y),
                        inc_yaw=float(args.inc_yaw),
                        decay_sec=float(args.keyboard_command_decay_sec),
                    )
                    command = _clip_command(command, x_rng=x_rng, y_rng=y_rng, yaw_rng=yaw_rng)
                elif args.command_source == "joystick":
                    joystick_cmd = joystick_control.get_command(None)
                    hv = np.asarray(joystick_cmd.get("horizontal_velocity", [0.0, 0.0]), dtype=np.float32)
                    yaw_rate = float(joystick_cmd.get("yaw_rate", 0.0))
                    # Match existing mapping used in isaac_gym_policy.py.
                    command = np.array([-hv[1], -hv[0], -yaw_rate], dtype=np.float32)
                    command = _clip_command(command, x_rng=x_rng, y_rng=y_rng, yaw_rng=yaw_rng)

                if should_stop:
                    break

                hw.read_incoming_data()
                base_obs = _build_base_obs(
                    robot_state=hw.robot_state,
                    command=command,
                    q_dim=q_dim,
                    qd_dim=qd_dim,
                    include_command=include_command,
                    base_height=float(args.base_height),
                )
                obs_hist = np.concatenate([obs_hist[1:], base_obs[None, :]], axis=0)
                stacked_obs = obs_hist.reshape(-1)

                raw_action = bundle.deterministic_action(stacked_obs)[:act_dim]
                motor_target = action_mid + np.clip(raw_action, -1.0, 1.0) * float(action_scale) * action_half
                hw.set_actuator_postions(np.asarray(motor_target, dtype=np.float32))

                step += 1
                if now - last_log > 0.5:
                    print(
                        f"t={now - wall_start:6.2f}s step={step:05d} "
                        f"rpy=[{hw.robot_state.roll:+.2f},{hw.robot_state.pitch:+.2f},{getattr(hw.robot_state,'yaw',0.0):+.2f}] "
                        f"cmd=[{command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f}]"
                    )
                    last_log = now

                target_elapsed = step * float(args.policy_dt)
                sleep_time = target_elapsed - (time.time() - wall_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            hw.read_incoming_data()
            end_pos = np.asarray(hw.robot_state.position, dtype=np.float32)[:act_dim]
            _set_pose_smooth(
                hw=hw,
                start=end_pos,
                end=safe_pose,
                duration_sec=float(args.safe_pose_seconds),
                dt=float(args.policy_dt),
            )
            print("exiting: robot moved to safe pose")


if __name__ == "__main__":
    main()
