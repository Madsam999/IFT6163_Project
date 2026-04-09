"""Runs an exported Brax policy bundle in the Brax env and optionally saves video."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from puppersim.pupper_brax_policy_bundle import BraxPolicyBundle


def _import_attr(module_name: str, attr_name: str) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _parse_env_kwargs(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}
    candidate = Path(raw).expanduser()
    if candidate.exists():
        data = json.loads(candidate.read_text())
    else:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("env_kwargs must decode into a JSON object/dict.")
    return dict(data)


def _resolve_output_path(base_dir: Path, env_name: str, backend: str) -> Path:
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    return base_dir / f"{env_name}_{backend}_bundle_policy_{run_name}.mp4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--env-name", type=str, default="pupper_v2")
    parser.add_argument("--backend", type=str, default="mjx")
    parser.add_argument(
        "--env-kwargs",
        type=str,
        default="puppersim/config/pupper_brax_env_kwargs.command_locomotion.json",
    )
    parser.add_argument("--custom-env-module", type=str, default="puppersim.pupper_brax_env_v2")
    parser.add_argument("--custom-env-class", type=str, default="PupperV2BraxEnv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--camera", type=str, default="tracking_cam")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-dir", type=Path, default=Path("videos"))
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keyboard-command", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--fixed-command", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--command-x", type=float, default=0.75)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--x-speed", type=float, default=0.5)
    parser.add_argument("--y-speed", type=float, default=0.3)
    parser.add_argument("--yaw-speed", type=float, default=1.0)
    parser.add_argument("--lin-x-min", type=float, default=-0.75)
    parser.add_argument("--lin-x-max", type=float, default=0.75)
    parser.add_argument("--lin-y-min", type=float, default=-0.5)
    parser.add_argument("--lin-y-max", type=float, default=0.5)
    parser.add_argument("--yaw-min", type=float, default=-2.0)
    parser.add_argument("--yaw-max", type=float, default=2.0)
    args = parser.parse_args()

    import imageio.v2 as imageio
    import jax
    from brax import envs

    if bool(args.custom_env_module) != bool(args.custom_env_class):
        raise ValueError("Both --custom-env-module and --custom-env-class must be set together.")
    if args.custom_env_module:
        env_cls = _import_attr(args.custom_env_module, args.custom_env_class)
        envs.register_environment(args.env_name, env_cls)

    env_kwargs = _parse_env_kwargs(args.env_kwargs)
    env_kwargs.setdefault("backend", args.backend)
    env = envs.get_environment(args.env_name, **env_kwargs)

    bundle = BraxPolicyBundle(args.bundle_dir)

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    rng = jax.random.PRNGKey(int(args.seed))
    rng, reset_rng = jax.random.split(rng)
    state = reset_fn(reset_rng)
    rollout = [state.pipeline_state]

    print(f"loaded bundle from {args.bundle_dir}")
    print(f"obs_dim={bundle.obs_dim}, action_dim={bundle.action_dim}, activation={bundle.activation_name}")
    print(f"env_dt={float(env.dt):.6f}")

    cv2 = None
    if args.realtime:
        try:
            import cv2 as _cv2

            cv2 = _cv2
        except Exception:
            print("warning: OpenCV not installed; realtime window disabled. Install `opencv-python`.")
            args.realtime = False

    x_range = (float(args.lin_x_min), float(args.lin_x_max))
    y_range = (float(args.lin_y_min), float(args.lin_y_max))
    yaw_range = (float(args.yaw_min), float(args.yaw_max))

    def _clip_command(cmd: np.ndarray) -> np.ndarray:
        cmd[0] = np.clip(cmd[0], x_range[0], x_range[1])
        cmd[1] = np.clip(cmd[1], y_range[0], y_range[1])
        cmd[2] = np.clip(cmd[2], yaw_range[0], yaw_range[1])
        return cmd

    def _command_from_key(key_code: int) -> np.ndarray:
        # Zero every frame by default: if key is not being repeated/held,
        # command returns to zero as requested.
        cmd = np.zeros(3, dtype=np.float32)
        # User-facing mapping:
        #   W/S -> forward/backward command
        #   A/D -> left/right strafe command
        # For this env, forward corresponds to command_y.
        if key_code in (ord("w"), 82):  # w / up-arrow
            cmd[1] += float(args.x_speed)
        if key_code in (ord("s"), 84):  # s / down-arrow
            cmd[1] -= float(args.x_speed)
        if key_code in (ord("a"), 81):  # a / left-arrow
            cmd[0] += float(args.y_speed)
        if key_code in (ord("d"), 83):  # d / right-arrow
            cmd[0] -= float(args.y_speed)
        if key_code == ord("q"):
            cmd[2] += float(args.yaw_speed)
        if key_code == ord("e"):
            cmd[2] -= float(args.yaw_speed)
        return _clip_command(cmd)

    if args.keyboard_command and not args.realtime:
        print("warning: --keyboard-command requires --realtime window; disabling keyboard control.")
        args.keyboard_command = False
    if args.keyboard_command and args.fixed_command:
        print("warning: both --keyboard-command and --fixed-command set; using keyboard command.")
        args.fixed_command = False

    if args.fixed_command and "command" in state.info:
        cmd0 = np.asarray(
            _clip_command(np.array([args.command_x, args.command_y, args.command_yaw], dtype=np.float32)),
            dtype=np.float32,
        )
        info = dict(state.info)
        info["command"] = jax.numpy.asarray(cmd0, dtype=state.obs.dtype)
        state = state.replace(info=info)

    wall_start = time.time()
    for i in range(max(1, int(args.steps))):
        key = -1
        if args.realtime and cv2 is not None:
            # Poll current key first, render after step below.
            key = cv2.waitKey(1) & 0xFF

        if args.keyboard_command and "command" in state.info:
            cmd = _command_from_key(key)
            info = dict(state.info)
            info["command"] = jax.numpy.asarray(cmd, dtype=state.obs.dtype)
            state = state.replace(info=info)

        obs = np.asarray(state.obs, dtype=np.float32)
        action_np = bundle.deterministic_action(obs)
        action = jax.numpy.asarray(action_np, dtype=state.obs.dtype)
        state = step_fn(state, action)
        rollout.append(state.pipeline_state)

        if i % 200 == 0:
            reward = float(np.asarray(state.reward))
            done = float(np.asarray(state.done))
            if (args.keyboard_command or args.fixed_command) and "command" in state.info:
                cmd_now = np.asarray(state.info["command"])
                print(
                    f"step={i:05d} reward={reward:+.4f} done={done:.1f} "
                    f"cmd=[{cmd_now[0]:+.2f},{cmd_now[1]:+.2f},{cmd_now[2]:+.2f}]"
                )
            else:
                print(f"step={i:05d} reward={reward:+.4f} done={done:.1f}")

        if args.fixed_command and "command" in state.info:
            cmd = np.asarray(
                _clip_command(np.array([args.command_x, args.command_y, args.command_yaw], dtype=np.float32)),
                dtype=np.float32,
            )
            info = dict(state.info)
            info["command"] = jax.numpy.asarray(cmd, dtype=state.obs.dtype)
            state = state.replace(info=info)

        if args.realtime and cv2 is not None and (i % max(1, int(args.render_every)) == 0):
            frame = env.render(
                [state.pipeline_state],
                height=int(args.height),
                width=int(args.width),
                camera=args.camera if args.camera else None,
            )[0]
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("pupper_brax_bundle", bgr)
            if key == 27:  # ESC
                break
            target_elapsed = (i + 1) * float(env.dt)
            sleep_time = target_elapsed - (time.time() - wall_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if float(np.asarray(state.done)) >= 0.5:
            rng, reset_rng = jax.random.split(rng)
            state = reset_fn(reset_rng)
            rollout.append(state.pipeline_state)

    if not args.save_video:
        if cv2 is not None:
            cv2.destroyAllWindows()
        return

    render_kwargs: Dict[str, Any] = {"height": int(args.height), "width": int(args.width)}
    if args.camera:
        render_kwargs["camera"] = args.camera
    frames = env.render(rollout, **render_kwargs)
    output_path = _resolve_output_path(args.video_dir, args.env_name, args.backend)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inferred_fps = int(round(1.0 / float(env.dt))) if float(env.dt) > 0 else 30
    fps = int(args.fps) if int(args.fps) > 0 else inferred_fps
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"saved_video={output_path}")
    if cv2 is not None:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
