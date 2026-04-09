"""Train PPO with Brax/MJX using a CLI similar to the PyTorch trainer."""

import functools
import importlib
import inspect
import json
import os
import random
import sys
import time
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


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed for numpy/python/jax"""
    track: bool = False
    """if toggled, this experiment is tracked with Weights and Biases"""
    wandb_project_name: str = "pupper-v2-brax"
    """the wandb project name"""
    wandb_entity: Optional[str] = None
    """wandb entity (team/user)"""
    save_model: bool = True
    """save final parameters under runs/{run_name}"""
    save_checkpoints: bool = True
    """save periodic checkpoints during training when callback is supported"""
    checkpoint_interval: int = 10_000_000
    """save a checkpoint every N environment steps"""
    checkpoint_dirname: str = "checkpoints"
    """subdirectory under runs/{run_name} for periodic checkpoints"""

    env_name: str = "pupper_v2"
    """brax environment name (defaults to local Pupper v2 env)"""
    backend: str = "mjx"
    """brax backend, typically mjx/positional/spring/generalized"""
    env_kwargs: str = "puppersim/config/pupper_brax_env_kwargs.v3like_stable.json"
    """JSON string or JSON file path with kwargs passed to envs.get_environment"""
    custom_env_module: str = "puppersim.pupper_brax_env_v2"
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
    num_evals: int = 11
    """number of eval phases during training"""
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
    video_interval_steps: int = 10_000_000
    """save a training video every N env steps (when callback runs)"""
    video_max_count: int = 3
    """maximum number of periodic videos to save during training"""
    video_steps: int = 1000
    """number of env steps to render in the saved video"""
    video_width: int = 640
    """video width in pixels"""
    video_height: int = 480
    """video height in pixels"""
    video_camera: str = "tracking_cam"
    """camera name from MJCF; set empty string to use MuJoCo default camera"""
    video_fps: int = 0
    """video fps (<=0 means inferred from env.dt)"""
    video_dirname: str = "videos"
    """base directory for saved videos"""


def _build_run_name(args: Args) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{args.env_name}__{args.exp_name}__{timestamp}"


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


def _save_params(path: Path, params: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from flax.serialization import to_bytes

        path.write_bytes(to_bytes(params))
        return "flax-bytes"
    except Exception:
        import pickle

        path.write_bytes(pickle.dumps(params))
        return "pickle"


def _maybe_register_custom_env(args: Args, envs_module: Any):
    if bool(args.custom_env_module) != bool(args.custom_env_class):
        raise ValueError("Both custom_env_module and custom_env_class must be set together.")
    if not args.custom_env_module:
        return
    env_cls = _import_attr(args.custom_env_module, args.custom_env_class)
    envs_module.register_environment(args.env_name, env_cls)


def _maybe_get_randomization_fn(args: Args) -> Optional[Callable[..., Any]]:
    if bool(args.randomization_module) != bool(args.randomization_fn):
        raise ValueError("Both randomization_module and randomization_fn must be set together.")
    if not args.randomization_module:
        return None
    fn = _import_attr(args.randomization_module, args.randomization_fn)
    randomization_kwargs = _parse_env_kwargs(args.randomization_kwargs)
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
    fps: int,
) -> None:
    import jax
    import numpy as np

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

    rng = jax.random.PRNGKey(int(seed))
    rng, reset_rng = jax.random.split(rng)
    state = reset_fn(reset_rng)
    rollout = [state.pipeline_state]

    for _ in range(max(1, int(num_steps))):
        rng, policy_rng = jax.random.split(rng)
        action, _ = policy(state.obs, policy_rng)
        state = step_fn(state, action)
        rollout.append(state.pipeline_state)
        done_flag = float(np.asarray(state.done))
        if done_flag >= 0.5:
            rng, reset_rng = jax.random.split(rng)
            state = reset_fn(reset_rng)
            rollout.append(state.pipeline_state)

    render_kwargs = {
        "height": int(height),
        "width": int(width),
    }
    if camera:
        render_kwargs["camera"] = camera
    frames = env.render(rollout, **render_kwargs)

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
        fps=fps,
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    random.seed(args.seed)
    np.random.seed(args.seed)

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

    run_name = _build_run_name(args)
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
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                save_code=True,
            )
            wandb = _wandb
        except Exception as exc:
            print(f"wandb init failed, disabling tracking: {exc}")

    ppo_train = _resolve_ppo_train_callable(ppo_train_obj)

    _maybe_register_custom_env(args, envs)
    env_kwargs = _parse_env_kwargs(args.env_kwargs)
    env_kwargs.setdefault("backend", args.backend)

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
    if args.save_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_state = {"next_step": max(1, int(args.checkpoint_interval))}
    video_state = {
        "next_step": max(1, int(args.video_interval_steps)),
        "count": 0,
    }

    training_start = time.time()

    def progress_fn(num_steps: int, metrics: Mapping[str, Any]):
        elapsed = max(1e-6, time.time() - training_start)
        sps = int(num_steps / elapsed)
        writer.add_scalar("charts/SPS", sps, int(num_steps))
        for key, value in metrics.items():
            scalar_value = _as_float(value)
            if scalar_value is None:
                continue
            writer.add_scalar(f"metrics/{key}", scalar_value, int(num_steps))
        reward_keys = ("eval/episode_reward", "eval/episode_reward_std", "training/episode_reward")
        reward_summary = ", ".join(
            [f"{key}={_as_float(metrics.get(key)):.3f}" for key in reward_keys if _as_float(metrics.get(key)) is not None]
        )
        suffix = f", {reward_summary}" if reward_summary else ""
        print(f"num_steps={num_steps}, SPS={sps}{suffix}")

    def policy_params_fn(current_step: int, _make_policy: Callable[..., Any], params: Any):
        step = int(current_step)
        if args.save_checkpoints and step >= checkpoint_state["next_step"]:
            checkpoint_path = checkpoint_dir / f"{args.exp_name}_step_{step:012d}.params"
            fmt = _save_params(checkpoint_path, params)
            print(f"checkpoint saved to {checkpoint_path} ({fmt})")
            while checkpoint_state["next_step"] <= step:
                checkpoint_state["next_step"] += max(1, int(args.checkpoint_interval))

        if (
            args.capture_video_during_training
            and args.video_max_count > 0
            and video_state["count"] < int(args.video_max_count)
            and step >= video_state["next_step"]
        ):
            video_output_path = (
                Path(args.video_dirname)
                / run_name
                / f"{args.exp_name}_step_{step:012d}.mp4"
            )
            _save_rollout_video_from_policy_builder(
                env=eval_env,
                make_policy_builder=_make_policy,
                params=params,
                seed=args.seed + 9000 + video_state["count"],
                output_path=video_output_path,
                num_steps=args.video_steps,
                width=args.video_width,
                height=args.video_height,
                camera=args.video_camera or None,
                fps=args.video_fps,
            )
            video_state["count"] += 1
            print(f"training video saved to {video_output_path}")
            while video_state["next_step"] <= step:
                video_state["next_step"] += max(1, int(args.video_interval_steps))

    train_kwargs = {
        "num_timesteps": args.num_timesteps,
        "episode_length": args.episode_length,
        "num_evals": args.num_evals,
        "reward_scaling": args.reward_scaling,
        "normalize_observations": args.normalize_observations,
        "action_repeat": args.action_repeat,
        "unroll_length": args.unroll_length,
        "num_minibatches": args.num_minibatches,
        "num_updates_per_batch": args.num_updates_per_batch,
        "discounting": args.discounting,
        "learning_rate": args.learning_rate,
        "entropy_cost": args.entropy_cost,
        "num_envs": args.num_envs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "network_factory": make_networks_factory,
        "eval_env": eval_env,
        "progress_fn": progress_fn,
        "policy_params_fn": policy_params_fn,
    }

    randomization_fn = _maybe_get_randomization_fn(args)
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
    writer.add_scalar("charts/training_seconds", train_seconds, args.num_timesteps)

    if isinstance(metrics, CbMapping):
        for key, value in metrics.items():
            scalar_value = _as_float(value)
            if scalar_value is None:
                continue
            writer.add_scalar(f"final_metrics/{key}", scalar_value, args.num_timesteps)

    if args.save_model:
        final_model_path = Path("runs") / run_name / f"{args.exp_name}.params"
        serialization = _save_params(final_model_path, params)
        print(f"model saved to {final_model_path} ({serialization})")

        if args.save_policy_inference_fn:
            import pickle

            fn_path = Path("runs") / run_name / f"{args.exp_name}.inference_fn.pkl"
            fn_path.write_bytes(pickle.dumps(make_inference_fn))
            print(f"inference function saved to {fn_path} (pickle)")

    if args.capture_video:
        video_output_path = Path(args.video_dirname) / run_name / f"{args.exp_name}_final.mp4"
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
            fps=args.video_fps,
        )
        print(f"video saved to {video_output_path}")

    writer.close()
    if wandb is not None:
        wandb.finish()

    print(f"jax platform: {jax.default_backend()}")
