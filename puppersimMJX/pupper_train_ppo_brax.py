"""Train PPO with Brax/MJX using a CLI similar to the PyTorch trainer."""

import functools
import importlib
import inspect
import json
import os
import random
import re
import sys
import time
from collections import deque
from collections.abc import Mapping as CbMapping
from collections.abc import Sequence as CbSequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np
import tyro
from torch.utils.tensorboard import SummaryWriter


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_DEFAULT_ENV_KWARGS = "puppersimMJX/tasks/simple_forward/config/pupper_brax_env_kwargs.v3like_stable.json"


@dataclass
class Args:
    exp_name: str = "puppermjx_train_ppo"
    """the name of this experiment"""
    task_name: str = ""
    """optional task name used in run/save naming (defaults to profile name or env_name)"""
    seed: int = 1
    """seed for numpy/python/jax"""
    track: bool = True
    """if toggled, this experiment is tracked with Weights and Biases"""
    wandb_project_name: str = "pupper_mjx"
    """the wandb project name"""
    wandb_entity: Optional[str] = None
    """wandb entity (team/user)"""
    save_model: bool = True
    """save final parameters under runs/{run_name}"""
    save_checkpoints: bool = True
    """save periodic checkpoints during training when callback is supported"""
    params_format: str = "npz"
    """parameter serialization format: npz, flax-bytes, or pickle"""
    checkpoint_interval: int = 10_000_000
    """save a checkpoint every N environment steps"""
    checkpoint_dirname: str = "checkpoints"
    """subdirectory under runs/{run_name} for periodic checkpoints"""

    env_name: str = "pupper_v2"
    """brax environment name (defaults to local Pupper v2 env)"""
    backend: str = "mjx"
    """brax backend, typically mjx/positional/spring/generalized"""
    env_kwargs: str = _DEFAULT_ENV_KWARGS
    """JSON string or JSON file path with kwargs passed to envs.get_environment"""
    profile: str = ""
    """optional JSON string/file path for a training profile (env kwargs + randomization defaults)"""
    custom_env_module: str = "puppersimMJX.pupper_brax_env_v2"
    """module path for custom env registration (defaults to local Pupper v2 env)"""
    custom_env_class: str = "PupperV2BraxEnv"
    """class name for custom env registration"""
    randomization_module: str = ""
    """optional module path that contains domain randomization function"""
    randomization_fn: str = ""
    """optional randomization function name (e.g. domain_randomize)"""
    randomization_kwargs: str = ""
    """JSON string or JSON file path with kwargs for randomization_fn"""

    num_timesteps: int = 50_000_000
    """total timesteps for PPO training"""
    episode_length: int = 500
    """episode length"""
    eval_interval: int = 5_000_000
    """target interval in env steps between eval/log updates"""
    reward_scaling: float = 1.0
    """reward scaling used by Brax PPO"""
    normalize_observations: bool = True
    """toggle observation normalization"""
    action_repeat: int = 1
    """action repeat"""
    unroll_length: int = 20
    """trajectory length before update"""
    num_minibatches: int = 32
    """number of minibatches"""
    num_updates_per_batch: int = 4
    """gradient updates per batch"""
    discounting: float = 0.97
    """discount factor gamma"""
    learning_rate: float = 3e-4
    """optimizer learning rate"""
    learning_rate_schedule: str = "constant"
    """learning rate schedule: constant or linear"""
    entropy_cost: float = 1e-2
    """entropy regularization coefficient"""
    num_envs: int = 8192
    """number of parallel environments"""
    batch_size: int = 256
    """training batch size"""

    network_hidden_sizes: str = "256,128,128,128"
    """comma-separated hidden layer sizes for the policy/value MLP"""
    activation: str = "elu"
    """activation name: elu, tanh, relu, swish, gelu, sigmoid, leaky_relu"""
    save_policy_inference_fn: bool = False
    """store a pickled inference function (less portable, usually unnecessary)"""
    capture_video: bool = False
    """whether to render and save a final policy rollout video"""
    capture_video_during_training: bool = False
    """whether to save periodic rollout videos during training"""
    video_eval_interval: int = 10_000_000
    """save/log a training video every N env steps (when callback runs)"""
    video_steps: int = 1000
    """number of env steps to render in the saved video"""
    video_width: int = 640
    """video width in pixels"""
    video_height: int = 480
    """video height in pixels"""
    video_camera: str = "tracking_cam"
    """camera name from MJCF; set empty string to use MuJoCo default camera"""
    video_front_camera_inset: bool = True
    """overlay front camera inset at top-right when available"""
    video_front_camera_name: str = "front_cam"
    """front camera name used for inset overlay"""
    video_front_inset_scale: float = 0.33
    """inset size as fraction of output width/height"""
    video_randomize_goal_marker: bool = True
    """randomize static goal marker site position per saved video when available"""
    video_goal_marker_site_name: str = "goal_marker"
    """site name for the static goal marker in MJCF"""
    video_goal_marker_radius_min: float = 0.8
    """minimum radial distance for randomized goal marker position"""
    video_goal_marker_radius_max: float = 2.5
    """maximum radial distance for randomized goal marker position"""
    video_goal_marker_z: float = 0.02
    """z-height for randomized goal marker site"""
    video_sync_goal_marker_to_collectible: bool = True
    """sync rendered goal marker to true collectible goal position at each frame when available"""
    video_fps: int = 0
    """video fps (<=0 means inferred from env.dt)"""
    video_dirname: str = "videos"
    """base directory for saved videos"""


def _slugify_name(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return text or "task"


def _resolve_task_name(args: Args, profile: Mapping[str, Any]) -> str:
    if str(args.task_name).strip():
        return _slugify_name(args.task_name)
    profile_name = profile.get("name", "")
    if str(profile_name).strip():
        return _slugify_name(str(profile_name))
    profile_raw = str(args.profile or "").strip()
    if profile_raw and Path(profile_raw).expanduser().exists():
        return _slugify_name(Path(profile_raw).stem)
    return _slugify_name(args.env_name)


def _artifact_prefix(exp_name: str) -> str:
    base = _slugify_name(exp_name)
    if base.startswith("puppermjx_"):
        return base
    return f"puppermjx_{base}"


def _build_run_name(task_name: str) -> str:
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{_slugify_name(task_name)}_{date_str}_mjx"


def _import_attr(module_name: str, attr_name: str) -> Any:
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(f"Module '{module_name}' has no attribute '{attr_name}'.") from exc


def _parse_hidden_sizes(spec: str) -> Tuple[int, ...]:
    parts = [piece.strip() for piece in spec.split(",") if piece.strip()]
    if not parts:
        raise ValueError("network_hidden_sizes cannot be empty.")
    hidden_sizes = tuple(int(part) for part in parts)
    if any(size <= 0 for size in hidden_sizes):
        raise ValueError("network_hidden_sizes must be positive integers.")
    return hidden_sizes


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


def _parse_profile(raw: str) -> Dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {}
    candidate = Path(raw).expanduser()
    if candidate.exists():
        data = json.loads(candidate.read_text())
    else:
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("profile must decode into a JSON object/dict.")
    return dict(data)


def _resolve_profile_dict(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        return _parse_env_kwargs(value)
    raise ValueError(f"profile field '{field_name}' must be a dict or a JSON/path string.")


def _resolve_profile_kwargs_raw(value: Any, field_name: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value)
    raise ValueError(f"profile field '{field_name}' must be a dict or a JSON/path string.")


def _filter_kwargs(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        # Some callables/modules in older Brax packaging do not expose a
        # signature reliably; in that case pass kwargs through.
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _simplify_metric_key(key: str) -> str:
    raw = str(key).strip().strip("/")
    if raw.startswith("training/"):
        return f"train/{raw[len('training/'):]}"
    return raw


def _activation_fn(name: str):
    import jax.nn as jnn

    activation_map = {
        "relu": jnn.relu,
        "tanh": jnn.tanh,
        "elu": jnn.elu,
        "swish": jnn.swish,
        "silu": jnn.silu,
        "gelu": jnn.gelu,
        "sigmoid": jnn.sigmoid,
        "leaky_relu": jnn.leaky_relu,
    }
    key = name.strip().lower()
    if key not in activation_map:
        choices = ", ".join(sorted(activation_map))
        raise ValueError(f"Unsupported activation '{name}'. Choose one of: {choices}")
    return activation_map[key]


def _resolve_learning_rate(
    schedule_name: str,
    base_learning_rate: float,
    total_timesteps: int,
):
    schedule = str(schedule_name or "constant").strip().lower()
    base_lr = float(base_learning_rate)
    if base_lr <= 0:
        raise ValueError("learning_rate must be > 0.")
    if schedule == "constant":
        return base_lr
    if schedule == "linear":
        # Brax PPO versions that cast `learning_rate` with `jnp.array(...)`
        # only accept scalar values, not callables/schedules.
        print(
            "warning: learning_rate_schedule=linear requested, but this Brax PPO "
            "version expects scalar learning_rate. Falling back to constant."
        )
        return base_lr
    raise ValueError("learning_rate_schedule must be one of: constant, linear")


def _save_params(path: Path, params: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".npz":
        flat_arrays: Dict[str, np.ndarray] = {}
        if isinstance(params, Mapping):
            stack: list[Tuple[Tuple[str, ...], Any]] = [(tuple(), params)]
            while stack:
                prefix, node = stack.pop()
                if isinstance(node, Mapping):
                    for key, value in node.items():
                        stack.append((prefix + (str(key),), value))
                    continue
                key = "/".join(prefix) if prefix else "params"
                flat_arrays[key] = np.asarray(node)
        else:
            flat_arrays["params"] = np.asarray(params)
        np.savez(path, **flat_arrays)
        return "npz"
    if path.suffix == ".pkl":
        import pickle

        path.write_bytes(pickle.dumps(params))
        return "pickle"
    try:
        from flax.serialization import to_bytes

        path.write_bytes(to_bytes(params))
        return "flax-bytes"
    except Exception:
        import pickle

        path.write_bytes(pickle.dumps(params))
        return "pickle"


def _normalize_params_format(raw: str) -> str:
    fmt = str(raw or "").strip().lower()
    if fmt in ("npz",):
        return "npz"
    if fmt in ("flax", "flax-bytes", "flax_bytes"):
        return "flax-bytes"
    if fmt in ("pickle", "pkl"):
        return "pickle"
    raise ValueError("params_format must be one of: npz, flax-bytes, pickle")


def _params_extension(params_format: str) -> str:
    if params_format == "npz":
        return ".npz"
    if params_format == "pickle":
        return ".pkl"
    return ".params"


def _maybe_register_custom_env(args: Args, envs_module: Any):
    if bool(args.custom_env_module) != bool(args.custom_env_class):
        raise ValueError("Both custom_env_module and custom_env_class must be set together.")
    if not args.custom_env_module:
        return
    env_cls = _import_attr(args.custom_env_module, args.custom_env_class)
    envs_module.register_environment(args.env_name, env_cls)


def _maybe_get_randomization_fn(
    randomization_module: str,
    randomization_fn_name: str,
    randomization_kwargs_raw: str,
) -> Optional[Callable[..., Any]]:
    if bool(randomization_module) != bool(randomization_fn_name):
        raise ValueError("Both randomization_module and randomization_fn must be set together.")
    if not randomization_module:
        return None
    fn = _import_attr(randomization_module, randomization_fn_name)
    randomization_kwargs = _parse_env_kwargs(randomization_kwargs_raw)
    if not randomization_kwargs:
        return fn
    return functools.partial(fn, **randomization_kwargs)


def _resolve_ppo_train_callable(ppo_train_obj: Any) -> Callable[..., Any]:
    """Supports Brax versions where `ppo.train` is a function or a module."""
    if callable(ppo_train_obj):
        return ppo_train_obj
    candidate = getattr(ppo_train_obj, "train", None)
    if callable(candidate):
        return candidate
    raise TypeError(
        "Could not resolve a callable PPO trainer from brax.training.agents.ppo.train. "
        f"Got object type: {type(ppo_train_obj)}"
    )


def _save_rollout_video_from_policy_builder(
    env: Any,
    make_policy_builder: Callable[..., Any],
    params: Any,
    seed: int,
    output_path: Path,
    num_steps: int,
    width: int,
    height: int,
    camera: Optional[str],
    front_camera_inset: bool,
    front_camera_name: Optional[str],
    front_inset_scale: float,
    randomize_goal_marker: bool,
    goal_marker_site_name: str,
    goal_marker_radius_min: float,
    goal_marker_radius_max: float,
    goal_marker_z: float,
    sync_goal_marker_to_collectible: bool,
    fps: int,
) -> None:
    import jax
    import numpy as np
    from jax import numpy as jp

    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError(
            "imageio is required to save videos. Install with `pip install imageio imageio-ffmpeg`."
        ) from exc

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    policy = make_policy_builder(params, deterministic=True)
    policy = jax.jit(policy)

    goal_positions_xy: list[np.ndarray] = []

    if randomize_goal_marker:
        try:
            sys_obj = getattr(env, "sys", None)
            site_names = list(getattr(sys_obj, "site_names", [])) if sys_obj is not None else []
            if goal_marker_site_name in site_names:
                site_idx = int(site_names.index(goal_marker_site_name))
                rng_vis = np.random.default_rng(int(seed) + 1337)
                r_lo = float(min(goal_marker_radius_min, goal_marker_radius_max))
                r_hi = float(max(goal_marker_radius_min, goal_marker_radius_max))
                radius = float(rng_vis.uniform(r_lo, r_hi))
                angle = float(rng_vis.uniform(-np.pi, np.pi))
                gx = radius * np.cos(angle)
                gy = radius * np.sin(angle)
                site_pos = np.asarray(getattr(sys_obj, "site_pos"))
                if 0 <= site_idx < int(site_pos.shape[0]):
                    site_pos = np.array(site_pos, copy=True)
                    site_pos[site_idx, 0] = gx
                    site_pos[site_idx, 1] = gy
                    site_pos[site_idx, 2] = float(goal_marker_z)
                    if hasattr(sys_obj, "replace"):
                        env.sys = sys_obj.replace(site_pos=jp.asarray(site_pos))
        except Exception as exc:
            print(f"warning: goal marker randomization skipped ({exc})")

    rng = jax.random.PRNGKey(int(seed))
    rng, reset_rng = jax.random.split(rng)
    state = reset_fn(reset_rng)
    rollout = [state.pipeline_state]
    goal_positions_xy.append(np.asarray(state.info.get("goal_position", jp.zeros((2,), dtype=state.obs.dtype))))

    for _ in range(max(1, int(num_steps))):
        rng, policy_rng = jax.random.split(rng)
        action, _ = policy(state.obs, policy_rng)
        state = step_fn(state, action)
        rollout.append(state.pipeline_state)
        goal_positions_xy.append(np.asarray(state.info.get("goal_position", jp.zeros((2,), dtype=state.obs.dtype))))
        done_flag = float(np.asarray(state.done))
        if done_flag >= 0.5:
            rng, reset_rng = jax.random.split(rng)
            state = reset_fn(reset_rng)
            rollout.append(state.pipeline_state)
            goal_positions_xy.append(np.asarray(state.info.get("goal_position", jp.zeros((2,), dtype=state.obs.dtype))))

    render_kwargs = {
        "height": int(height),
        "width": int(width),
    }
    if camera:
        render_kwargs["camera"] = camera

    def _render_with_synced_goal(
        camera_name: Optional[str],
        frame_w: int,
        frame_h: int,
    ):
        sys_obj = getattr(env, "sys", None)
        if sys_obj is None or not hasattr(sys_obj, "site_names") or not hasattr(sys_obj, "site_pos"):
            raise RuntimeError("env.sys has no site metadata")
        site_names = list(getattr(sys_obj, "site_names"))
        if goal_marker_site_name not in site_names:
            raise RuntimeError(f"site '{goal_marker_site_name}' not found")
        site_idx = int(site_names.index(goal_marker_site_name))
        if len(goal_positions_xy) != len(rollout):
            raise RuntimeError("goal/rollout length mismatch")

        original_sys = env.sys
        out_frames = []
        try:
            for pstate, gxy in zip(rollout, goal_positions_xy):
                gx = float(np.asarray(gxy)[0])
                gy = float(np.asarray(gxy)[1])
                site_pos = np.array(np.asarray(env.sys.site_pos), copy=True)
                site_pos[site_idx, 0] = gx
                site_pos[site_idx, 1] = gy
                site_pos[site_idx, 2] = float(goal_marker_z)
                env.sys = env.sys.replace(site_pos=jp.asarray(site_pos))

                kwargs = {"height": int(frame_h), "width": int(frame_w)}
                if camera_name:
                    kwargs["camera"] = camera_name
                frame = env.render([pstate], **kwargs)[0]
                out_frames.append(np.array(frame, copy=True))
        finally:
            env.sys = original_sys
        return out_frames

    def _render_with_mujoco_goal_sync(
        camera_name: Optional[str],
        frame_w: int,
        frame_h: int,
    ):
        try:
            import mujoco as mj
        except Exception as exc:
            raise RuntimeError(f"mujoco import failed: {exc}") from exc

        model_path = getattr(env, "_model_path", "")
        if not model_path:
            raise RuntimeError("env has no _model_path for MuJoCo fallback rendering")

        m = mj.MjModel.from_xml_path(str(model_path))
        d = mj.MjData(m)
        renderer = mj.Renderer(m, height=int(frame_h), width=int(frame_w))

        site_id = int(mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, goal_marker_site_name))
        if site_id < 0:
            raise RuntimeError(f"site '{goal_marker_site_name}' not found in MuJoCo model")

        cam_id = -1
        if camera_name:
            cam_id = int(mj.mj_name2id(m, mj.mjtObj.mjOBJ_CAMERA, str(camera_name)))
            if cam_id < 0:
                raise RuntimeError(f"camera '{camera_name}' not found in MuJoCo model")

        if len(goal_positions_xy) != len(rollout):
            raise RuntimeError("goal/rollout length mismatch")

        out_frames = []
        try:
            for pstate, gxy in zip(rollout, goal_positions_xy):
                q = np.asarray(pstate.q, dtype=np.float64)
                qd = np.asarray(pstate.qd, dtype=np.float64)
                if q.shape[0] != m.nq or qd.shape[0] != m.nv:
                    raise RuntimeError(
                        f"state shape mismatch (q={q.shape[0]}/{m.nq}, qd={qd.shape[0]}/{m.nv})"
                    )

                d.qpos[:] = q
                d.qvel[:] = qd
                m.site_pos[site_id, 0] = float(np.asarray(gxy)[0])
                m.site_pos[site_id, 1] = float(np.asarray(gxy)[1])
                m.site_pos[site_id, 2] = float(goal_marker_z)
                mj.mj_forward(m, d)

                if cam_id >= 0:
                    renderer.update_scene(d, camera=str(camera_name))
                else:
                    renderer.update_scene(d)
                out_frames.append(np.array(renderer.render(), copy=True))
        finally:
            try:
                renderer.close()
            except Exception:
                pass
        return out_frames

    goal_sync_active = False
    goal_sync_backend = "none"  # one of: none, brax, mujoco
    if sync_goal_marker_to_collectible:
        try:
            frames = _render_with_synced_goal(camera_name=camera, frame_w=int(width), frame_h=int(height))
            goal_sync_active = True
            goal_sync_backend = "brax"
        except Exception as exc:
            print(f"warning: goal marker sync disabled ({exc})")
            try:
                frames = _render_with_mujoco_goal_sync(camera_name=camera, frame_w=int(width), frame_h=int(height))
                goal_sync_active = True
                goal_sync_backend = "mujoco"
                print("info: using MuJoCo fallback renderer for goal-marker sync")
            except Exception as exc2:
                print(f"warning: MuJoCo goal sync fallback disabled ({exc2})")
                frames = env.render(rollout, **render_kwargs)
    else:
        frames = env.render(rollout, **render_kwargs)

    if front_camera_inset and front_camera_name:
        try:
            inset_scale = float(np.clip(float(front_inset_scale), 0.1, 0.95))
            inset_w = max(1, int(round(int(width) * inset_scale)))
            inset_h = max(1, int(round(int(height) * inset_scale)))
            if goal_sync_active and goal_sync_backend == "brax":
                inset_frames = _render_with_synced_goal(
                    camera_name=str(front_camera_name), frame_w=inset_w, frame_h=inset_h
                )
            elif goal_sync_active and goal_sync_backend == "mujoco":
                inset_frames = _render_with_mujoco_goal_sync(
                    camera_name=str(front_camera_name), frame_w=inset_w, frame_h=inset_h
                )
            else:
                inset_frames = env.render(rollout, height=inset_h, width=inset_w, camera=str(front_camera_name))

            composed = []
            for base, inset in zip(frames, inset_frames):
                base_np = np.array(base, copy=True)
                inset_np = np.array(inset, copy=False)
                h0, w0 = base_np.shape[0], base_np.shape[1]
                h1, w1 = inset_np.shape[0], inset_np.shape[1]
                y0 = 0
                x0 = max(0, w0 - w1)
                y1 = min(h0, h1)
                x1 = min(w0, x0 + w1)
                base_np[y0:y1, x0:x1] = inset_np[: (y1 - y0), : (x1 - x0)]
                composed.append(base_np)
            frames = composed
        except Exception as exc:
            print(f"warning: front camera inset disabled ({exc})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    inferred_fps = int(round(1.0 / float(env.dt))) if float(env.dt) > 0 else 30
    use_fps = int(fps) if int(fps) > 0 else inferred_fps
    imageio.mimsave(output_path, frames, fps=use_fps)


def _save_rollout_video(
    env: Any,
    make_inference_fn: Callable[..., Any],
    params: Any,
    seed: int,
    output_path: Path,
    num_steps: int,
    width: int,
    height: int,
    camera: Optional[str],
    front_camera_inset: bool,
    front_camera_name: Optional[str],
    front_inset_scale: float,
    randomize_goal_marker: bool,
    goal_marker_site_name: str,
    goal_marker_radius_min: float,
    goal_marker_radius_max: float,
    goal_marker_z: float,
    sync_goal_marker_to_collectible: bool,
    fps: int,
) -> None:
    return _save_rollout_video_from_policy_builder(
        env=env,
        make_policy_builder=make_inference_fn,
        params=params,
        seed=seed,
        output_path=output_path,
        num_steps=num_steps,
        width=width,
        height=height,
        camera=camera,
        front_camera_inset=front_camera_inset,
        front_camera_name=front_camera_name,
        front_inset_scale=front_inset_scale,
        randomize_goal_marker=randomize_goal_marker,
        goal_marker_site_name=goal_marker_site_name,
        goal_marker_radius_min=goal_marker_radius_min,
        goal_marker_radius_max=goal_marker_radius_max,
        goal_marker_z=goal_marker_z,
        sync_goal_marker_to_collectible=sync_goal_marker_to_collectible,
        fps=fps,
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    profile = _parse_profile(args.profile)
    task_name = _resolve_task_name(args, profile)
    run_name = _build_run_name(task_name)
    wandb_run_name = task_name
    save_prefix = _artifact_prefix(args.exp_name)

    try:
        import jax
        from brax import envs
        from brax.training.agents.ppo import networks as ppo_networks
        from brax.training.agents.ppo import train as ppo_train_obj
    except Exception as exc:
        raise SystemExit(
            "Brax/MJX dependencies are missing. Install with:\n"
            "  pip install \"jax[cpu]\" brax mujoco flax orbax-checkpoint tyro\n"
            "Python 3.10+ is required for this MJX path.\n"
            f"Import error: {exc}"
        ) from exc

    writer = SummaryWriter(os.path.join("runs", run_name))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    wandb = None
    if args.track:
        try:
            import wandb as _wandb
            if not callable(getattr(_wandb, "init", None)):
                raise RuntimeError(
                    f"Imported wandb module has no init() (module={getattr(_wandb, '__file__', 'namespace')})."
                )

            _wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=vars(args),
                name=wandb_run_name,
                save_code=True,
            )
            wandb = _wandb
        except Exception as exc:
            print(f"wandb init failed, disabling tracking: {exc}")

    ppo_train = _resolve_ppo_train_callable(ppo_train_obj)

    _maybe_register_custom_env(args, envs)
    profile_env_kwargs = _resolve_profile_dict(profile.get("env_kwargs"), "env_kwargs")
    use_cli_env_kwargs = True
    if profile and str(args.env_kwargs).strip() == _DEFAULT_ENV_KWARGS:
        # Keep profile defaults when user didn't explicitly override env_kwargs.
        use_cli_env_kwargs = False
    cli_env_kwargs = _parse_env_kwargs(args.env_kwargs) if use_cli_env_kwargs else {}
    env_kwargs = {**profile_env_kwargs, **cli_env_kwargs}
    env_kwargs.setdefault("backend", args.backend)
    if profile:
        profile_name = profile.get("name") or Path(args.profile).name
        print(f"using training profile: {profile_name}")
    print(
        "reward setup: "
        f"module={env_kwargs.get('reward_module', '<default>')}, "
        f"requires_command={env_kwargs.get('reward_requires_command', '<default>')}, "
        f"resample_velocity_step={env_kwargs.get('resample_velocity_step', '<default>')}, "
        f"lin_vel_x_range={env_kwargs.get('lin_vel_x_range', '<default>')}, "
        f"lin_vel_y_range={env_kwargs.get('lin_vel_y_range', '<default>')}, "
        f"ang_vel_yaw_range={env_kwargs.get('ang_vel_yaw_range', '<default>')}, "
        f"zero_command_probability={env_kwargs.get('zero_command_probability', '<default>')}"
    )

    env = envs.get_environment(args.env_name, **env_kwargs)
    eval_env_kwargs = dict(env_kwargs)
    if "regenerate_mjcf_if_exists" in eval_env_kwargs:
        eval_env_kwargs["regenerate_mjcf_if_exists"] = False
    eval_env = envs.get_environment(args.env_name, **eval_env_kwargs)

    activation = _activation_fn(args.activation)
    hidden_sizes = _parse_hidden_sizes(args.network_hidden_sizes)

    network_factory_kwargs = _filter_kwargs(
        ppo_networks.make_ppo_networks,
        {
            "policy_hidden_layer_sizes": hidden_sizes,
            "value_hidden_layer_sizes": hidden_sizes,
            "activation": activation,
        },
    )
    make_networks_factory = functools.partial(ppo_networks.make_ppo_networks, **network_factory_kwargs)

    checkpoint_dir = Path("runs") / run_name / args.checkpoint_dirname
    params_format = _normalize_params_format(args.params_format)
    params_ext = _params_extension(params_format)
    if args.save_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_state = {"next_step": max(1, int(args.checkpoint_interval))}
    eval_log_state = {"next_step": 0}
    video_state = {
        "next_step": 0,
        "initial_saved": False,
    }
    log_state = {"max_step": 0}
    train_reward_window = deque(maxlen=10)
    reward_window = deque(maxlen=10)
    last_video_state: Dict[str, Optional[Path]] = {"path": None}

    training_start = time.time()

    def progress_fn(num_steps: int, metrics: Mapping[str, Any]):
        step = int(num_steps)
        log_state["max_step"] = max(log_state["max_step"], step)
        if step < eval_log_state["next_step"]:
            return
        while eval_log_state["next_step"] <= step:
            eval_log_state["next_step"] += max(1, int(args.eval_interval))

        elapsed = max(1e-6, time.time() - training_start)
        sps = int(step / elapsed)
        writer.add_scalar("sps", sps, step)
        wandb_payload = {"sps": sps, "num_steps": step}
        for key, value in metrics.items():
            scalar_value = _as_float(value)
            if scalar_value is None:
                continue
            metric_key = _simplify_metric_key(key)
            writer.add_scalar(metric_key, scalar_value, step)
            wandb_payload[metric_key] = scalar_value
            if key == "training/episode_reward" or metric_key == "train/episode_reward":
                train_reward_window.append(scalar_value)
                reward_window.append(scalar_value)
            elif metric_key == "eval/episode_reward":
                reward_window.append(scalar_value)
        if not train_reward_window:
            fallback_train_reward = _as_float(metrics.get("training/episode_reward"))
            if fallback_train_reward is not None:
                train_reward_window.append(fallback_train_reward)
                reward_window.append(fallback_train_reward)
        if train_reward_window:
            avg10 = float(np.mean(train_reward_window))
            writer.add_scalar("train/avg_episode_reward", avg10, step)
            wandb_payload["train/avg_episode_reward"] = avg10
        elif reward_window:
            # Fallback when Brax does not report training/episode_reward at this step.
            avg10 = float(np.mean(reward_window))
            writer.add_scalar("train/avg_episode_reward", avg10, step)
            wandb_payload["train/avg_episode_reward"] = avg10
        if wandb is not None:
            try:
                wandb.log(wandb_payload, step=step)
            except Exception as exc:
                print(f"wandb.log failed at step={step}: {exc}")
        reward_keys = ("eval/episode_reward", "eval/episode_reward_std", "training/episode_reward")
        reward_summary = ", ".join(
            [f"{key}={_as_float(metrics.get(key)):.3f}" for key in reward_keys if _as_float(metrics.get(key)) is not None]
        )
        suffix = f", {reward_summary}" if reward_summary else ""
        print(f"num_steps={num_steps}, SPS={sps}{suffix}")

    def policy_params_fn(current_step: int, _make_policy: Callable[..., Any], params: Any):
        step = int(current_step)
        log_state["max_step"] = max(log_state["max_step"], step)

        if args.capture_video and not bool(video_state["initial_saved"]):
            initial_video_path = (
                Path(args.video_dirname)
                / run_name
                / f"{save_prefix}_initial.mp4"
            )
            _save_rollout_video_from_policy_builder(
                env=eval_env,
                make_policy_builder=_make_policy,
                params=params,
                seed=args.seed + 7777,
                output_path=initial_video_path,
                num_steps=args.video_steps,
                width=args.video_width,
                height=args.video_height,
                camera=args.video_camera or None,
                front_camera_inset=bool(args.video_front_camera_inset),
                front_camera_name=(args.video_front_camera_name or None),
                front_inset_scale=float(args.video_front_inset_scale),
                randomize_goal_marker=bool(args.video_randomize_goal_marker),
                goal_marker_site_name=str(args.video_goal_marker_site_name),
                goal_marker_radius_min=float(args.video_goal_marker_radius_min),
                goal_marker_radius_max=float(args.video_goal_marker_radius_max),
                goal_marker_z=float(args.video_goal_marker_z),
                sync_goal_marker_to_collectible=bool(args.video_sync_goal_marker_to_collectible),
                fps=args.video_fps,
            )
            video_state["initial_saved"] = True
            last_video_state["path"] = initial_video_path
            print(f"initial video saved to {initial_video_path}")
            if wandb is not None:
                try:
                    wandb.log(
                        {
                            "episode_video": wandb.Video(
                                str(initial_video_path),
                                format="mp4",
                            ),
                            "num_steps": step,
                        },
                        step=step,
                    )
                except Exception as exc:
                    print(f"wandb initial video log failed at step={step}: {exc}")

        if args.save_checkpoints and step >= checkpoint_state["next_step"]:
            checkpoint_path = checkpoint_dir / f"{save_prefix}_step_{step:012d}{params_ext}"
            fmt = _save_params(checkpoint_path, params)
            print(f"checkpoint saved to {checkpoint_path} ({fmt})")
            while checkpoint_state["next_step"] <= step:
                checkpoint_state["next_step"] += max(1, int(args.checkpoint_interval))

        if (
            args.capture_video_during_training
            and step >= video_state["next_step"]
        ):
            video_output_path = (
                Path(args.video_dirname)
                / run_name
                / f"{save_prefix}_step_{step:012d}.mp4"
            )
            _save_rollout_video_from_policy_builder(
                env=eval_env,
                make_policy_builder=_make_policy,
                params=params,
                seed=args.seed + 9000 + (step // max(1, int(args.video_eval_interval))),
                output_path=video_output_path,
                num_steps=args.video_steps,
                width=args.video_width,
                height=args.video_height,
                camera=args.video_camera or None,
                front_camera_inset=bool(args.video_front_camera_inset),
                front_camera_name=(args.video_front_camera_name or None),
                front_inset_scale=float(args.video_front_inset_scale),
                randomize_goal_marker=bool(args.video_randomize_goal_marker),
                goal_marker_site_name=str(args.video_goal_marker_site_name),
                goal_marker_radius_min=float(args.video_goal_marker_radius_min),
                goal_marker_radius_max=float(args.video_goal_marker_radius_max),
                goal_marker_z=float(args.video_goal_marker_z),
                sync_goal_marker_to_collectible=bool(args.video_sync_goal_marker_to_collectible),
                fps=args.video_fps,
            )
            last_video_state["path"] = video_output_path
            print(f"training video saved to {video_output_path}")
            if wandb is not None:
                try:
                    wandb.log(
                        {
                            "episode_video": wandb.Video(
                                str(video_output_path),
                                format="mp4",
                            ),
                            "num_steps": step,
                        },
                        step=step,
                    )
                except Exception as exc:
                    print(f"wandb video log failed at step={step}: {exc}")
            while video_state["next_step"] <= step:
                video_state["next_step"] += max(1, int(args.video_eval_interval))

    eval_interval = max(1, int(args.eval_interval))
    computed_num_evals = max(1, int(args.num_timesteps) // eval_interval + 1)
    learning_rate_value = _resolve_learning_rate(
        schedule_name=args.learning_rate_schedule,
        base_learning_rate=args.learning_rate,
        total_timesteps=args.num_timesteps,
    )
    train_kwargs = {
        "num_timesteps": args.num_timesteps,
        "episode_length": args.episode_length,
        "num_evals": computed_num_evals,
        "reward_scaling": args.reward_scaling,
        "normalize_observations": args.normalize_observations,
        "action_repeat": args.action_repeat,
        "unroll_length": args.unroll_length,
        "num_minibatches": args.num_minibatches,
        "num_updates_per_batch": args.num_updates_per_batch,
        "discounting": args.discounting,
        "learning_rate": learning_rate_value,
        "entropy_cost": args.entropy_cost,
        "num_envs": args.num_envs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "network_factory": make_networks_factory,
        "eval_env": eval_env,
        "progress_fn": progress_fn,
        "policy_params_fn": policy_params_fn,
    }

    randomization_module = str(args.randomization_module or profile.get("randomization_module", "")).strip()
    randomization_fn_name = str(args.randomization_fn or profile.get("randomization_fn", "")).strip()
    randomization_kwargs_raw = args.randomization_kwargs
    if not randomization_kwargs_raw.strip():
        randomization_kwargs_raw = _resolve_profile_kwargs_raw(
            profile.get("randomization_kwargs"),
            "randomization_kwargs",
        )
    randomization_fn = _maybe_get_randomization_fn(
        randomization_module=randomization_module,
        randomization_fn_name=randomization_fn_name,
        randomization_kwargs_raw=randomization_kwargs_raw,
    )
    if randomization_fn is not None:
        train_kwargs["randomization_fn"] = randomization_fn

    filtered_train_kwargs = _filter_kwargs(ppo_train, train_kwargs)
    if args.save_checkpoints and "policy_params_fn" not in filtered_train_kwargs:
        print("warning: this Brax PPO version has no policy_params_fn callback; periodic checkpoints disabled.")

    train_output = ppo_train(environment=env, **filtered_train_kwargs)
    if isinstance(train_output, CbSequence) and len(train_output) >= 2:
        make_inference_fn = train_output[0]
        params = train_output[1]
        metrics = train_output[2] if len(train_output) >= 3 else {}
    else:
        raise RuntimeError("Unexpected return value from brax PPO train().")

    train_seconds = time.time() - training_start
    print(f"training complete in {train_seconds:.1f}s")
    writer.add_scalar("training_seconds", train_seconds, args.num_timesteps)

    if isinstance(metrics, CbMapping):
        for key, value in metrics.items():
            scalar_value = _as_float(value)
            if scalar_value is None:
                continue
            writer.add_scalar(f"final_metrics/{key}", scalar_value, args.num_timesteps)

    if args.save_model:
        final_model_path = Path("runs") / run_name / f"{save_prefix}{params_ext}"
        serialization = _save_params(final_model_path, params)
        print(f"model saved to {final_model_path} ({serialization})")

        if args.save_policy_inference_fn:
            import pickle

            fn_path = Path("runs") / run_name / f"{save_prefix}.inference_fn.pkl"
            fn_path.write_bytes(pickle.dumps(make_inference_fn))
            print(f"inference function saved to {fn_path} (pickle)")

    if args.capture_video:
        video_output_path = Path(args.video_dirname) / run_name / f"{save_prefix}_final.mp4"
        _save_rollout_video(
            env=eval_env,
            make_inference_fn=make_inference_fn,
            params=params,
            seed=args.seed + 12345,
            output_path=video_output_path,
            num_steps=args.video_steps,
            width=args.video_width,
            height=args.video_height,
            camera=args.video_camera or None,
            front_camera_inset=bool(args.video_front_camera_inset),
            front_camera_name=(args.video_front_camera_name or None),
            front_inset_scale=float(args.video_front_inset_scale),
            randomize_goal_marker=bool(args.video_randomize_goal_marker),
            goal_marker_site_name=str(args.video_goal_marker_site_name),
            goal_marker_radius_min=float(args.video_goal_marker_radius_min),
            goal_marker_radius_max=float(args.video_goal_marker_radius_max),
            goal_marker_z=float(args.video_goal_marker_z),
            sync_goal_marker_to_collectible=bool(args.video_sync_goal_marker_to_collectible),
            fps=args.video_fps,
        )
        last_video_state["path"] = video_output_path
        print(f"video saved to {video_output_path}")
        if wandb is not None:
            try:
                final_log_step = max(log_state["max_step"] + 1, int(args.num_timesteps))
                wandb.log(
                    {
                        "episode_video": wandb.Video(
                            str(video_output_path),
                            format="mp4",
                        ),
                        "num_steps": int(args.num_timesteps),
                    },
                    step=final_log_step,
                )
                log_state["max_step"] = max(log_state["max_step"], final_log_step)
            except Exception as exc:
                print(f"wandb final video log failed: {exc}")

    if wandb is not None and last_video_state["path"] is not None:
        try:
            last_video_step = max(log_state["max_step"] + 1, int(args.num_timesteps))
            wandb.log(
                {
                    "episode_video": wandb.Video(
                        str(last_video_state["path"]),
                        format="mp4",
                    ),
                    "num_steps": int(args.num_timesteps),
                },
                step=last_video_step,
            )
            log_state["max_step"] = max(log_state["max_step"], last_video_step)
        except Exception as exc:
            print(f"wandb last video log failed: {exc}")

    writer.close()
    if wandb is not None:
        wandb.finish()

    print(f"jax platform: {jax.default_backend()}")
