"""Train PPO with Brax vision networks on MJX-rendered pixel observations.

This script keeps the Pupper MJX environment and swaps the custom PPO loop for
Brax PPO's trainer/loss stack, using vision observations keyed as `pixels/...`.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import os
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import jax
import jax.numpy as jp
import numpy as np
import tyro
from brax.envs.base import Wrapper
from torch.utils.tensorboard import SummaryWriter

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_DEFAULT_ENV_KWARGS = "puppersimMJX/tasks/simple_forward/config/pupper_brax_env_kwargs.v3like_stable.json"
_RENDER_CONTEXT_KEEPALIVE: list[Any] = []
_VIDEO_RENDERER_KEEPALIVE: list[Any] = []


@dataclass
class Args:
    exp_name: str = "puppermjx_train_ppo_pixels"
    task_name: str = ""
    seed: int = 1
    track: bool = True
    wandb_project_name: str = "pupper_mjx"
    wandb_entity: Optional[str] = None

    env_name: str = "pupper_v2"
    backend: str = "mjx"
    profile: str = "puppersimMJX/tasks/apriltag_walls/config/training_profiles/apriltag_walls_camera_hl.json"
    env_kwargs: str = _DEFAULT_ENV_KWARGS
    custom_env_module: str = "puppersimMJX.pupper_brax_env_v2"
    custom_env_class: str = "PupperV2BraxEnv"

    total_env_steps: int = 100_000_000
    episode_length: int = 1500
    eval_interval: int = 500_000

    nworld: int = 64
    rollout_length: int = 256
    batch_size: int = 512
    update_epochs: int = 4
    num_minibatches: int = 32

    learning_rate: float = 3e-4
    learning_rate_schedule: str = "constant"
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_coef_value: float = 0.2
    ent_coef: float = 1e-2
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.01
    init_action_std: float = 0.4

    normalize_observations: bool = False
    reward_scaling: float = 1.0

    image_width: int = 80
    image_height: int = 80
    camera_name: str = "front_cam"
    include_state_in_obs: bool = False
    state_obs_key: str = "state"
    pixels_obs_key: str = "pixels/front"
    obs_grayscale: bool = False
    obs_grayscale_keep_rgb_channels: bool = False

    policy_hidden_sizes: str = "256,256"
    value_hidden_sizes: str = "256,256"
    cnn_channels: str = "32,64,64"
    cnn_kernels: str = "8,4,3"
    cnn_strides: str = "4,2,1"
    cnn_padding: str = "zeros"
    cnn_global_pool: str = "avg"
    normalise_channels: bool = False
    augment_pixels: bool = False

    capture_video: bool = False
    capture_video_during_training: bool = False
    video_interval: int = 1_000_000
    video_steps: int = 1500
    video_fps: int = 60
    video_dirname: str = "videos"
    video_renderer: str = "mujoco"
    video_width: int = 480
    video_height: int = 480
    video_main_camera_name: str = "tracking_cam"
    video_inset_camera_name: str = "front_cam"
    video_inset_scale: float = 0.28
    video_inset_margin: int = 4
    video_brightness: float = 1.0
    video_gamma: float = 1.0

    debug_eval_obs_sampling: bool = False
    debug_eval_episodes: int = 2
    debug_eval_max_steps_per_episode: int = 8
    debug_eval_values_to_print: int = 12
    debug_eval_save_frames: bool = False
    debug_eval_frame_obs_key: str = "pixels/front"
    debug_eval_frames_dirname: str = "debug_eval_frames"
    debug_eval_frame_max_saved: int = 64

    save_model: bool = True
    params_format: str = "npz"


def _import_attr(module_name: str, attr_name: str) -> Any:
    module = importlib.import_module(module_name)
    if not hasattr(module, attr_name):
        raise ValueError(f"Module '{module_name}' has no attribute '{attr_name}'.")
    return getattr(module, attr_name)


def _parse_json_or_file(raw: str) -> Dict[str, Any]:
    raw = str(raw or "").strip()
    if not raw:
        return {}
    p = Path(raw).expanduser()
    data = json.loads(p.read_text()) if p.exists() else json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")
    return dict(data)


def _parse_profile(raw: str) -> Dict[str, Any]:
    return _parse_json_or_file(raw)


def _parse_int_tuple(spec: str, name: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    if not parts:
        raise ValueError(f"{name} cannot be empty")
    vals = tuple(int(x) for x in parts)
    if any(v <= 0 for v in vals):
        raise ValueError(f"{name} must contain positive integers")
    return vals


def _ensure_custom_env_registered(args: Args, envs_module: Any) -> None:
    env_cls = _import_attr(args.custom_env_module, args.custom_env_class)
    envs_module.register_environment(args.env_name, env_cls)


def _save_video_mp4(frames_u8: np.ndarray, out_path: Path, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("imageio is required for video export (`pip install imageio imageio-ffmpeg`).") from exc

    arr = np.asarray(frames_u8, dtype=np.uint8)
    if arr.ndim != 4:
        raise ValueError(f"Expected [T,H,W,C], got shape {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), arr, fps=max(1, int(fps)))


def _resize_nearest_u8(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    src = np.asarray(image, dtype=np.uint8)
    if src.ndim != 3 or src.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 uint8 image, got shape {src.shape}")
    h, w = src.shape[:2]
    th = max(1, int(target_h))
    tw = max(1, int(target_w))
    if h == th and w == tw:
        return src
    ys = np.clip(np.linspace(0, h - 1, th).astype(np.int32), 0, h - 1)
    xs = np.clip(np.linspace(0, w - 1, tw).astype(np.int32), 0, w - 1)
    return src[ys[:, None], xs[None, :], :]


def _overlay_inset(base: np.ndarray, inset: np.ndarray, inset_scale: float, margin: int) -> np.ndarray:
    frame = np.asarray(base, dtype=np.uint8).copy()
    h, w = frame.shape[:2]
    tgt_w = max(1, int(round(w * float(inset_scale))))
    tgt_h = max(1, int(round(h * float(inset_scale))))
    pip = _resize_nearest_u8(inset, tgt_h, tgt_w)

    m = max(0, int(margin))
    y0 = min(m, max(0, h - tgt_h))
    # Place inset in top-right corner.
    x0 = max(0, w - tgt_w - m)
    y1 = min(h, y0 + tgt_h)
    x1 = min(w, x0 + tgt_w)
    frame[y0:y1, x0:x1, :] = pip[: (y1 - y0), : (x1 - x0), :]

    # Thin border to keep inset readable over bright backgrounds.
    by0 = max(0, y0 - 1)
    bx0 = max(0, x0 - 1)
    by1 = min(h, y1 + 1)
    bx1 = min(w, x1 + 1)
    frame[by0:by1, bx0, :] = 255
    frame[by0:by1, bx1 - 1, :] = 255
    frame[by0, bx0:bx1, :] = 255
    frame[by1 - 1, bx0:bx1, :] = 255
    return frame


def _enhance_video_frame(image: np.ndarray, brightness: float, gamma: float) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    b = max(0.1, float(brightness))
    g = max(0.1, float(gamma))
    x = np.clip(arr.astype(np.float32) / 255.0, 0.0, 1.0)
    x = np.clip(x * b, 0.0, 1.0)
    x = np.power(x, g)
    return np.asarray(np.clip(x * 255.0, 0.0, 255.0), dtype=np.uint8)


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


def _assert_warp_render_ready() -> None:
    from mujoco import mjx
    import mujoco.mjx.warp as mjxw

    if not hasattr(mjx, "create_render_context"):
        raise RuntimeError("mujoco.mjx.create_render_context is unavailable.")
    if not bool(getattr(mjxw, "WARP_INSTALLED", False)):
        raise RuntimeError("MuJoCo Warp backend is not available. Install `warp-lang`.")
    try:
        cuda_devices = jax.devices("cuda")
    except Exception as exc:
        raise RuntimeError(f"CUDA backend not available to JAX: {exc}") from exc
    if len(cuda_devices) == 0:
        raise RuntimeError("No CUDA devices found for MJX Warp rendering.")


def _extract_packed_rgb(render_out: Any, min_pixels: int) -> jp.ndarray:
    leaves = jax.tree_util.tree_leaves(render_out)
    candidates = []
    for leaf in leaves:
        if not hasattr(leaf, "dtype"):
            continue
        if np.dtype(leaf.dtype) != np.dtype(np.uint32):
            continue
        arr = jp.asarray(leaf)
        if arr.ndim == 0:
            continue
        last_dim = int(arr.shape[-1]) if arr.shape else 0
        if last_dim >= int(min_pixels):
            candidates.append(arr)
    if not candidates:
        raise RuntimeError("Could not find packed uint32 RGB buffer in mjx.render output.")
    # Prefer the candidate with the largest contiguous pixel span in the last dimension.
    return max(candidates, key=lambda a: int(a.shape[-1]))


def _decode_packed_rgb(packed: jp.ndarray, image_w: int, image_h: int) -> jp.ndarray:
    pixels = int(image_w) * int(image_h)
    p = packed
    if p.ndim == 1:
        p = p[None, :]
    elif p.ndim > 2:
        p = p.reshape((p.shape[0], -1))
    p = p[:, :pixels]
    b = (p & jp.asarray(0xFF, dtype=jp.uint32)).astype(jp.float32) / 255.0
    g = ((p >> jp.asarray(8, dtype=jp.uint32)) & jp.asarray(0xFF, dtype=jp.uint32)).astype(jp.float32) / 255.0
    r = ((p >> jp.asarray(16, dtype=jp.uint32)) & jp.asarray(0xFF, dtype=jp.uint32)).astype(jp.float32) / 255.0
    r = r.reshape((r.shape[0], r.shape[1], 1))
    g = g.reshape((g.shape[0], g.shape[1], 1))
    b = b.reshape((b.shape[0], b.shape[1], 1))
    rgb = jp.concatenate([r, g, b], axis=-1)
    return rgb.reshape((rgb.shape[0], int(image_h), int(image_w), 3))


def _build_single_world_renderer(
    env: Any,
    args: Args,
    camera_name: Optional[str] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> Callable[[Any], jp.ndarray]:
    import mujoco
    from mujoco import mjx
    from mujoco.mjx.third_party.warp._src.jax_experimental import ffi as warp_ffi

    mj_model = env.sys.mj_model
    cam_name = str(args.camera_name if camera_name is None else camera_name).strip()
    if cam_name:
        camera_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name))
        if camera_id < 0:
            raise ValueError(f"Camera '{cam_name}' not found in model.")
    else:
        camera_id = 0

    cam_active = [False] * int(mj_model.ncam)
    cam_active[camera_id] = True

    render_w = int(args.image_width if image_width is None else image_width)
    render_h = int(args.image_height if image_height is None else image_height)

    rc_local = mjx.create_render_context(
        mjm=mj_model,
        nworld=1,
        cam_res=(render_w, render_h),
        render_rgb=True,
        render_depth=True,
        use_shadows=False,
        use_textures=True,
        enabled_geom_groups=[0, 1, 2],
        cam_active=cam_active,
    )
    _RENDER_CONTEXT_KEEPALIVE.append(rc_local)
    rc_tree = rc_local.pytree()

    mx_warp = mjx.put_model(mj_model, impl="warp", graph_mode=warp_ffi.GraphMode.NONE)
    data_template = mjx.make_data(mj_model, impl="warp")

    def _render_one(pipeline_state: Any) -> jp.ndarray:
        # Avoid an extra Warp forward() pass; camera/geom transforms already
        # exist in pipeline_state and are enough for rendering + BVH refit.
        d = data_template.replace(
            cam_xmat=pipeline_state.cam_xmat,
            cam_xpos=pipeline_state.cam_xpos,
            geom_xmat=pipeline_state.geom_xmat,
            geom_xpos=pipeline_state.geom_xpos,
        )
        d = mjx.refit_bvh(mx_warp, d, rc_tree)
        packed = _extract_packed_rgb(mjx.render(mx_warp, d, rc_tree), min_pixels=render_w * render_h)
        rgb = _decode_packed_rgb(packed, render_w, render_h)
        return jp.clip(rgb[0], 0.0, 1.0).astype(jp.float32)

    return jax.jit(_render_one)


def _unwrap_env(env: Any) -> Any:
    cur = env
    while hasattr(cur, "env"):
        cur = cur.env
    return cur


def _build_single_world_mujoco_renderer(
    env: Any,
    args: Args,
    camera_name: Optional[str] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> Callable[[Any, Optional[Mapping[str, Any]]], np.ndarray]:
    import mujoco as mj

    base_env = _unwrap_env(env)
    model_path = str(getattr(base_env, "_model_path", "")).strip()
    if not model_path:
        raise RuntimeError("MuJoCo video renderer requires env._model_path")

    cam_name = str(args.camera_name if camera_name is None else camera_name).strip()
    render_w = int(args.image_width if image_width is None else image_width)
    render_h = int(args.image_height if image_height is None else image_height)

    m = mj.MjModel.from_xml_path(model_path)
    d = mj.MjData(m)
    renderer = mj.Renderer(m, height=render_h, width=render_w)
    if cam_name:
        cam_id = int(mj.mj_name2id(m, mj.mjtObj.mjOBJ_CAMERA, cam_name))
        if cam_id < 0:
            raise ValueError(f"Camera '{cam_name}' not found in model.")
    good_geom_id = int(mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "apriltag_good_panel"))
    last_tag_sig: Optional[tuple] = None

    _VIDEO_RENDERER_KEEPALIVE.append((m, d, renderer, cam_name))

    def _render_one(pipeline_state: Any, state_info: Optional[Mapping[str, Any]] = None) -> np.ndarray:
        nonlocal last_tag_sig
        q = np.asarray(pipeline_state.q, dtype=np.float64)
        qd = np.asarray(pipeline_state.qd, dtype=np.float64)
        if q.shape[0] != m.nq or qd.shape[0] != m.nv:
            raise RuntimeError(f"Pipeline state shape mismatch (q={q.shape[0]}/{m.nq}, qd={qd.shape[0]}/{m.nv})")
        d.qpos[:] = q
        d.qvel[:] = qd
        if good_geom_id >= 0 and hasattr(pipeline_state, "geom_xpos") and hasattr(pipeline_state, "geom_xmat"):
            gpos = np.asarray(pipeline_state.geom_xpos[good_geom_id], dtype=np.float64).reshape(-1)
            gmat = np.asarray(pipeline_state.geom_xmat[good_geom_id], dtype=np.float64).reshape(-1)
            q_geom = np.zeros(4, dtype=np.float64)
            mj.mju_mat2Quat(q_geom, gmat)
            sig = (
                float(gpos[0]),
                float(gpos[1]),
                float(gpos[2]),
                float(q_geom[0]),
                float(q_geom[1]),
                float(q_geom[2]),
                float(q_geom[3]),
            )
            if last_tag_sig != sig:
                m.geom_pos[good_geom_id, :] = gpos[:3]
                m.geom_quat[good_geom_id, :] = q_geom
                last_tag_sig = sig
        mj.mj_forward(m, d)
        if cam_name:
            renderer.update_scene(d, camera=cam_name)
        else:
            renderer.update_scene(d)
        return np.asarray(renderer.render(), dtype=np.uint8)

    return _render_one


def _build_batched_renderer(
    env: Any,
    args: Args,
    camera_name: Optional[str] = None,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None,
) -> Callable[[Any], jp.ndarray]:
    import mujoco
    from mujoco import mjx
    from mujoco.mjx.third_party.warp._src.jax_experimental import ffi as warp_ffi

    mj_model = env.sys.mj_model
    cam_name = str(args.camera_name if camera_name is None else camera_name).strip()
    if cam_name:
        camera_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name))
        if camera_id < 0:
            raise ValueError(f"Camera '{cam_name}' not found in model.")
    else:
        camera_id = 0

    nworld = int(args.nworld)
    cam_active = [False] * int(mj_model.ncam)
    cam_active[camera_id] = True

    render_w = int(args.image_width if image_width is None else image_width)
    render_h = int(args.image_height if image_height is None else image_height)

    rc_local = mjx.create_render_context(
        mjm=mj_model,
        nworld=nworld,
        cam_res=(render_w, render_h),
        render_rgb=True,
        render_depth=True,
        use_shadows=False,
        use_textures=True,
        enabled_geom_groups=[0, 1, 2],
        cam_active=cam_active,
    )
    _RENDER_CONTEXT_KEEPALIVE.append(rc_local)
    rc_tree = rc_local.pytree()

    mx_warp = mjx.put_model(mj_model, impl="warp", graph_mode=warp_ffi.GraphMode.NONE)
    data_template = mjx.make_data(mj_model, impl="warp")

    def _render_batch(pipeline_state_batched: Any) -> jp.ndarray:
        d = data_template.replace(
            cam_xmat=pipeline_state_batched.cam_xmat,
            cam_xpos=pipeline_state_batched.cam_xpos,
            geom_xmat=pipeline_state_batched.geom_xmat,
            geom_xpos=pipeline_state_batched.geom_xpos,
        )
        d = mjx.refit_bvh(mx_warp, d, rc_tree)
        packed = _extract_packed_rgb(mjx.render(mx_warp, d, rc_tree), min_pixels=render_w * render_h)
        rgb = _decode_packed_rgb(packed, render_w, render_h)
        return jp.clip(rgb, 0.0, 1.0).astype(jp.float32)

    return jax.jit(_render_batch)


class PixelObsWrapper(Wrapper):
    """Wraps a Pupper env so obs is a dict containing Brax vision pixel keys."""

    def __init__(self, env: Any, args: Args):
        super().__init__(env)
        self._nworld = int(args.nworld)
        self._pixels_key = str(args.pixels_obs_key)
        self._include_state = bool(args.include_state_in_obs)
        self._state_key = str(args.state_obs_key)
        self._obs_grayscale = bool(args.obs_grayscale)
        self._obs_grayscale_keep_rgb_channels = bool(args.obs_grayscale_keep_rgb_channels)
        self._render_one = _build_single_world_renderer(env, args)
        self._render_batch = _build_batched_renderer(env, args)

    def _maybe_grayscale(self, pixels: jp.ndarray) -> jp.ndarray:
        if not self._obs_grayscale:
            return pixels
        # Input is expected as (..., H, W, 3) float32 in [0, 1].
        r = pixels[..., 0:1]
        g = pixels[..., 1:2]
        b = pixels[..., 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        if self._obs_grayscale_keep_rgb_channels:
            return jp.repeat(gray, repeats=3, axis=-1)
        return gray

    def _make_obs(self, state_obs: jp.ndarray, pipeline_state: Any) -> Dict[str, jp.ndarray]:
        pixels = self._maybe_grayscale(self._render_one(pipeline_state))
        obs = {self._pixels_key: pixels}
        if self._include_state:
            obs[self._state_key] = state_obs
        return obs

    def _make_obs_batched(self, state_obs_batched: jp.ndarray, pipeline_state_batched: Any) -> Dict[str, jp.ndarray]:
        batch_n = int(state_obs_batched.shape[0])
        if batch_n == self._nworld:
            pixels = self._render_batch(pipeline_state_batched)
        else:
            pixels = jax.vmap(self._render_one)(pipeline_state_batched)
        pixels = self._maybe_grayscale(pixels)
        obs = {self._pixels_key: pixels}
        if self._include_state:
            obs[self._state_key] = state_obs_batched
        return obs

    def reset(self, rng: jp.ndarray) -> Any:
        if hasattr(rng, "ndim") and int(rng.ndim) == 2:
            state = jax.vmap(self.env.reset)(rng)
            obs = self._make_obs_batched(state.obs, state.pipeline_state)
            return state.replace(obs=obs)
        state = self.env.reset(rng)
        obs = self._make_obs(state.obs, state.pipeline_state)
        return state.replace(obs=obs)

    def step(self, state: Any, action: jp.ndarray) -> Any:
        if hasattr(action, "ndim") and int(action.ndim) == 2:
            nstate = jax.vmap(self.env.step)(state, action)
            obs = self._make_obs_batched(nstate.obs, nstate.pipeline_state)
            return nstate.replace(obs=obs)
        nstate = self.env.step(state, action)
        obs = self._make_obs(nstate.obs, nstate.pipeline_state)
        return nstate.replace(obs=obs)


def _build_video_rollout(
    env: Any,
    make_policy_builder: Callable[..., Any],
    params: Any,
    seed: int,
    num_steps: int,
    render_main_fn: Optional[Callable[[Any], jp.ndarray]] = None,
    render_inset_fn: Optional[Callable[[Any], jp.ndarray]] = None,
    inset_scale: float = 0.28,
    inset_margin: int = 4,
    video_brightness: float = 1.0,
    video_gamma: float = 1.0,
) -> np.ndarray:
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    policy = make_policy_builder(params, deterministic=True)
    policy = jax.jit(policy)

    rng = jax.random.PRNGKey(int(seed))
    rng, reset_rng = jax.random.split(rng)
    state = reset_fn(reset_rng)

    def _obs_to_frame(obs: Mapping[str, jp.ndarray]) -> np.ndarray:
        pix = None
        for k, v in obs.items():
            if str(k).startswith("pixels/"):
                pix = v
                break
        if pix is None:
            raise RuntimeError("No `pixels/...` key found in observation for video export")
        arr = np.asarray(jp.clip(pix, 0.0, 1.0) * 255.0, dtype=np.uint8)
        return arr

    def _render_to_frame(render_fn: Callable[[Any], jp.ndarray], state_obj: Any) -> np.ndarray:
        try:
            rendered = render_fn(state_obj.pipeline_state, getattr(state_obj, "info", None))
        except TypeError:
            rendered = render_fn(state_obj.pipeline_state)
        arr = np.asarray(rendered)
        if arr.dtype == np.uint8:
            return arr
        return np.asarray(jp.clip(rendered, 0.0, 1.0) * 255.0, dtype=np.uint8)

    def _compose_frame(state_obj: Any) -> np.ndarray:
        if render_main_fn is not None:
            frame = _render_to_frame(render_main_fn, state_obj)
        else:
            frame = _obs_to_frame(state_obj.obs)
        if render_inset_fn is not None:
            inset = _render_to_frame(render_inset_fn, state_obj)
            frame = _overlay_inset(frame, inset, inset_scale=float(inset_scale), margin=int(inset_margin))
        if float(video_brightness) != 1.0 or float(video_gamma) != 1.0:
            return _enhance_video_frame(frame, brightness=float(video_brightness), gamma=float(video_gamma))
        return frame

    frames = [_compose_frame(state)]
    for _ in range(max(1, int(num_steps))):
        rng, policy_rng = jax.random.split(rng)
        action, _ = policy(state.obs, policy_rng)
        state = step_fn(state, action)
        frames.append(_compose_frame(state))
        done_flag = float(np.asarray(state.done))
        if done_flag >= 0.5:
            rng, reset_rng = jax.random.split(rng)
            state = reset_fn(reset_rng)
            frames.append(_compose_frame(state))

    return np.stack(frames, axis=0)


def _resolve_learning_rate(schedule_name: str, base_learning_rate: float) -> float:
    schedule = str(schedule_name or "constant").strip().lower()
    base_lr = float(base_learning_rate)
    if base_lr <= 0:
        raise ValueError("learning_rate must be > 0")
    if schedule == "constant":
        return base_lr
    if schedule == "linear":
        print(
            "warning: learning_rate_schedule=linear requested, but this Brax PPO version "
            "expects scalar learning_rate. Falling back to constant."
        )
        return base_lr
    raise ValueError("learning_rate_schedule must be one of: constant, linear")


def _filter_kwargs(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _is_cuda_graph_capture_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "cuda graph capture failed" in text
        or "cuda_error_stream_capture_unsupported" in text
        or "ffi callback error" in text
    )


def _wrap_env_no_vmap(env: Any, episode_length: int, action_repeat: int, randomization_fn: Optional[Callable[..., Any]] = None):
    from brax.envs.wrappers import training as training_wrappers

    if randomization_fn is not None:
        print("warning: randomization_fn provided but ignored by custom wrap_env_fn")
    env = training_wrappers.EpisodeWrapper(env, int(episode_length), int(action_repeat))
    env = training_wrappers.AutoResetWrapper(env)
    return env


def _print_eval_obs_sample(obs: Mapping[str, Any], episode_idx: int, step_idx: int, max_values: int) -> None:
    print(f"[debug-eval-obs] episode={episode_idx} step={step_idx}")
    for key, value in obs.items():
        arr = np.asarray(value)
        arr_f = arr.astype(np.float32, copy=False)
        flat = arr_f.reshape(-1)
        nvals = max(1, int(max_values))
        preview = np.array2string(flat[:nvals], precision=4, separator=", ")
        print(
            f"  key={key!r} shape={arr.shape} dtype={arr.dtype} "
            f"min={float(np.min(arr_f)):.6f} max={float(np.max(arr_f)):.6f} "
            f"mean={float(np.mean(arr_f)):.6f}"
        )
        print(f"    sample[0:{nvals}]={preview}")


def _debug_eval_obs_sampling(
    env: Any,
    make_policy_builder: Callable[..., Any],
    params: Any,
    seed: int,
    num_episodes: int,
    max_steps_per_episode: int,
    values_to_print: int,
    save_frames: bool,
    frame_obs_key: str,
    frames_dir: Optional[Path],
    max_saved_frames: int,
    train_step: int,
) -> None:
    print(
        "[debug-eval-obs] starting eval observation sampling: "
        f"episodes={int(max(1, num_episodes))}, max_steps={int(max(1, max_steps_per_episode))}"
    )
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    policy = make_policy_builder(params, deterministic=True)
    policy = jax.jit(policy)

    rng = jax.random.PRNGKey(int(seed))
    episodes = int(max(1, num_episodes))
    max_steps = int(max(1, max_steps_per_episode))
    frame_obs_key = str(frame_obs_key)
    max_saved_frames = int(max(1, max_saved_frames))
    num_saved_frames = 0

    def _to_u8_image(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x)
        if arr.ndim == 3 and arr.shape[-1] in (1, 3):
            img = arr
        elif arr.ndim == 3 and arr.shape[0] in (1, 3):
            img = np.transpose(arr, (1, 2, 0))
        else:
            raise ValueError(f"Unsupported image tensor shape for PNG export: {arr.shape}")
        img = img.astype(np.float32, copy=False)
        if float(np.max(img)) > 1.5:
            img = np.clip(img, 0.0, 255.0)
        else:
            img = np.clip(img, 0.0, 1.0) * 255.0
        out = img.astype(np.uint8)
        if out.shape[-1] == 1:
            out = np.repeat(out, 3, axis=-1)
        return out

    def _tile_images(images: np.ndarray) -> np.ndarray:
        imgs = np.asarray(images)
        n = int(imgs.shape[0])
        cols = int(np.ceil(np.sqrt(float(n))))
        rows = int(np.ceil(n / float(cols)))
        h, w, c = imgs.shape[1], imgs.shape[2], imgs.shape[3]
        canvas = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
        for i in range(n):
            r = i // cols
            cidx = i % cols
            canvas[r * h : (r + 1) * h, cidx * w : (cidx + 1) * w, :] = imgs[i]
        return canvas

    for ep in range(episodes):
        rng, reset_rng = jax.random.split(rng)
        state = reset_fn(reset_rng)
        done = False
        step_idx = 0
        while (not done) and (step_idx < max_steps):
            _print_eval_obs_sample(
                obs=state.obs,
                episode_idx=ep,
                step_idx=step_idx,
                max_values=int(values_to_print),
            )
            if bool(save_frames) and frames_dir is not None and num_saved_frames < max_saved_frames:
                if frame_obs_key in state.obs:
                    frame_arr = np.asarray(state.obs[frame_obs_key])
                    try:
                        if frame_arr.ndim == 4:
                            imgs = np.stack([_to_u8_image(frame_arr[i]) for i in range(int(frame_arr.shape[0]))], axis=0)
                            out_img = _tile_images(imgs)
                        else:
                            out_img = _to_u8_image(frame_arr)
                        out_path = frames_dir / (
                            f"train_step_{int(train_step):012d}_ep_{ep:03d}_step_{step_idx:04d}.png"
                        )
                        try:
                            import imageio.v2 as imageio

                            imageio.imwrite(str(out_path), out_img)
                            print(f"[debug-eval-obs] saved frame: {out_path}")
                            num_saved_frames += 1
                        except Exception as exc:
                            print(f"[debug-eval-obs] warning: failed to save PNG frame: {exc}")
                    except Exception as exc:
                        print(f"[debug-eval-obs] warning: failed to convert frame_obs_key={frame_obs_key!r}: {exc}")
                else:
                    print(f"[debug-eval-obs] warning: frame_obs_key={frame_obs_key!r} missing in obs keys={list(state.obs.keys())}")
            rng, policy_rng = jax.random.split(rng)
            action, _ = policy(state.obs, policy_rng)
            state = step_fn(state, action)
            done = float(np.asarray(state.done)) >= 0.5
            step_idx += 1
        print(f"[debug-eval-obs] episode={ep} sampled_steps={step_idx} done={done}")


def main(args: Args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Keep Brax simulation off Warp to avoid MJX+Warp graph-capture crashes.
    # We still use Warp explicitly for camera rendering in the pixel wrapper.
    if os.environ.get("MJX_GPU_DEFAULT_WARP", "").strip().lower() != "false":
        print("info: setting MJX_GPU_DEFAULT_WARP=false for simulation stability.")
    os.environ["MJX_GPU_DEFAULT_WARP"] = "false"

    if int(args.image_width) <= 0 or int(args.image_height) <= 0:
        raise ValueError("image_width and image_height must be positive")
    if int(args.video_width) < 0 or int(args.video_height) < 0:
        raise ValueError("video_width/video_height must be >= 0")

    from brax import envs
    from brax.training.agents.ppo import networks_vision as ppo_networks_vision
    from brax.training.agents.ppo import train as ppo_train_module

    _ensure_custom_env_registered(args, envs)

    profile = _parse_profile(args.profile)
    profile_env_kwargs = (
        dict(profile.get("env_kwargs", {}))
        if isinstance(profile.get("env_kwargs", {}), dict)
        else _parse_json_or_file(str(profile.get("env_kwargs", "")))
    )
    cli_env_kwargs = {} if str(args.env_kwargs).strip() == _DEFAULT_ENV_KWARGS and profile else _parse_json_or_file(args.env_kwargs)
    env_kwargs = {**profile_env_kwargs, **cli_env_kwargs}
    env_kwargs.setdefault("backend", args.backend)

    base_env = envs.get_environment(args.env_name, **env_kwargs)
    _assert_warp_render_ready()

    vision_env = PixelObsWrapper(base_env, args)
    video_main_renderer: Optional[Callable[[Any], jp.ndarray]] = None
    video_inset_renderer: Optional[Callable[[Any], jp.ndarray]] = None
    if bool(args.capture_video) or bool(args.capture_video_during_training):
        try:
            video_w = int(args.video_width) if int(args.video_width) > 0 else int(args.image_width)
            video_h = int(args.video_height) if int(args.video_height) > 0 else int(args.image_height)
            video_renderer_mode = str(args.video_renderer or "mujoco").strip().lower()
            if video_renderer_mode not in {"mujoco", "mjx"}:
                raise ValueError("video_renderer must be one of: mujoco, mjx")
            if video_renderer_mode == "mujoco":
                video_main_renderer = _build_single_world_mujoco_renderer(
                    base_env,
                    args,
                    camera_name=args.video_main_camera_name,
                    image_width=video_w,
                    image_height=video_h,
                )
            else:
                video_main_renderer = _build_single_world_renderer(
                    base_env,
                    args,
                    camera_name=args.video_main_camera_name,
                    image_width=video_w,
                    image_height=video_h,
                )
            inset_name = str(args.video_inset_camera_name or "").strip()
            if inset_name:
                if video_renderer_mode == "mujoco":
                    video_inset_renderer = _build_single_world_mujoco_renderer(
                        base_env,
                        args,
                        camera_name=inset_name,
                        image_width=video_w,
                        image_height=video_h,
                    )
                else:
                    video_inset_renderer = _build_single_world_renderer(
                        base_env,
                        args,
                        camera_name=inset_name,
                        image_width=video_w,
                        image_height=video_h,
                    )
            print(
                "info: video cameras configured: "
                f"main={args.video_main_camera_name}, inset={args.video_inset_camera_name or 'disabled'}, "
                f"resolution={video_w}x{video_h}, renderer={video_renderer_mode}"
            )
        except Exception as exc:
            print(f"warning: failed to configure dedicated video cameras: {exc}")
            print("info: using observation pixels for videos to avoid renderer instability in headless mode")
    if bool(args.normalize_observations) and not bool(args.include_state_in_obs):
        print("warning: normalize_observations=true ignored for pixels-only obs; disabling normalization.")
        args.normalize_observations = False

    policy_hidden = _parse_int_tuple(args.policy_hidden_sizes, "policy_hidden_sizes")
    value_hidden = _parse_int_tuple(args.value_hidden_sizes, "value_hidden_sizes")
    cnn_channels = _parse_int_tuple(args.cnn_channels, "cnn_channels")
    cnn_kernels = _parse_int_tuple(args.cnn_kernels, "cnn_kernels")
    cnn_strides = _parse_int_tuple(args.cnn_strides, "cnn_strides")

    if not (len(cnn_channels) == len(cnn_kernels) == len(cnn_strides)):
        raise ValueError("cnn_channels, cnn_kernels, cnn_strides must have equal lengths")

    vision_factory_kwargs = {
        "policy_hidden_layer_sizes": policy_hidden,
        "value_hidden_layer_sizes": value_hidden,
        "normalise_channels": bool(args.normalise_channels),
        "policy_obs_key": (args.state_obs_key if bool(args.include_state_in_obs) else ""),
        "value_obs_key": (args.state_obs_key if bool(args.include_state_in_obs) else ""),
        # Newer Brax versions may support these; older versions ignore them.
        "distribution_type": "tanh_normal",
        "noise_std_type": "scalar",
        "init_noise_std": float(args.init_action_std),
        "cnn_output_channels": cnn_channels,
        "cnn_kernel_size": cnn_kernels,
        "cnn_stride": cnn_strides,
        "cnn_padding": str(args.cnn_padding),
        "cnn_global_pool": str(args.cnn_global_pool),
    }
    filtered_vision_factory_kwargs = _filter_kwargs(ppo_networks_vision.make_ppo_networks_vision, vision_factory_kwargs)
    dropped_vision_keys = sorted(set(vision_factory_kwargs.keys()) - set(filtered_vision_factory_kwargs.keys()))
    if dropped_vision_keys:
        print(
            "warning: installed Brax vision network factory does not support: "
            + ", ".join(dropped_vision_keys)
        )

    network_factory = functools.partial(
        ppo_networks_vision.make_ppo_networks_vision,
        **filtered_vision_factory_kwargs,
    )

    run_name = f"{args.exp_name}_{time.strftime('%Y%m%d-%H%M%S')}_brax_vision"
    writer = SummaryWriter(os.path.join("runs", run_name))
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n" + "\n".join(f"|{k}|{v}|" for k, v in vars(args).items()))

    wandb = None
    if args.track:
        try:
            import wandb as _wandb

            _wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=vars(args),
                name=run_name,
                save_code=True,
            )
            wandb = _wandb
        except Exception as exc:
            print(f"wandb init failed, disabling tracking: {exc}")

    video_state = {"next_step": max(1, int(args.video_interval)), "disabled": False}
    default_video_fps = max(1, int(round(1.0 / float(base_env.dt))))
    video_fps = int(args.video_fps) if int(args.video_fps) > 0 else default_video_fps

    def progress_fn(step: int, metrics: Mapping[str, Any]) -> None:
        metric_floats = {}
        for k, v in metrics.items():
            try:
                metric_floats[str(k)] = float(np.asarray(v))
            except Exception:
                continue

        for k, v in metric_floats.items():
            writer.add_scalar(k, v, step)

        if wandb is not None and metric_floats:
            payload = dict(metric_floats)
            payload["num_steps"] = int(step)
            try:
                wandb.log(payload, step=step)
            except Exception:
                pass

        reward = metric_floats.get("eval/episode_reward", metric_floats.get("training/episode_reward", np.nan))
        kl = metric_floats.get("training/kl_mean", np.nan)
        sps = metric_floats.get("training/sps", np.nan)
        print(f"step={step}, reward={reward:.3f}, kl={kl:.5f}, sps={sps:.0f}")

    def policy_params_fn(step: int, make_policy_builder: Callable[..., Any], params: Any) -> None:
        if bool(video_state["disabled"]):
            return

        if bool(args.debug_eval_obs_sampling):
            frames_dir: Optional[Path] = None
            if bool(args.debug_eval_save_frames):
                frames_dir = Path(str(args.debug_eval_frames_dirname)) / run_name
                frames_dir.mkdir(parents=True, exist_ok=True)
            _debug_eval_obs_sampling(
                env=vision_env,
                make_policy_builder=make_policy_builder,
                params=params,
                seed=args.seed + 123_000 + int(step),
                num_episodes=int(args.debug_eval_episodes),
                max_steps_per_episode=int(args.debug_eval_max_steps_per_episode),
                values_to_print=int(args.debug_eval_values_to_print),
                save_frames=bool(args.debug_eval_save_frames),
                frame_obs_key=str(args.debug_eval_frame_obs_key),
                frames_dir=frames_dir,
                max_saved_frames=int(args.debug_eval_frame_max_saved),
                train_step=int(step),
            )

        do_periodic = bool(args.capture_video_during_training) and int(step) >= int(video_state["next_step"])
        do_initial = bool(args.capture_video or args.capture_video_during_training) and int(step) == 0

        if not (do_periodic or do_initial):
            return

        try:
            frames = _build_video_rollout(
                env=vision_env,
                make_policy_builder=make_policy_builder,
                params=params,
                seed=args.seed + 10_000 + int(step),
                num_steps=int(args.video_steps),
                render_main_fn=video_main_renderer,
                render_inset_fn=video_inset_renderer,
                inset_scale=float(args.video_inset_scale),
                inset_margin=int(args.video_inset_margin),
                video_brightness=float(args.video_brightness),
                video_gamma=float(args.video_gamma),
            )
        except Exception as exc:
            if _is_cuda_graph_capture_error(exc):
                video_state["disabled"] = True
                print(
                    "warning: disabling training-time video capture after CUDA graph capture failure: "
                    f"{exc}"
                )
                return
            raise
        video_path = Path(args.video_dirname) / run_name / f"{args.exp_name}_step_{int(step):012d}.mp4"
        _save_video_mp4(frames, video_path, fps=video_fps)
        print(f"video saved: {video_path}")

        if wandb is not None:
            try:
                wandb.log({"episode_video": wandb.Video(str(video_path), format="mp4"), "num_steps": int(step)}, step=int(step))
            except Exception:
                pass

        while int(video_state["next_step"]) <= int(step):
            video_state["next_step"] += max(1, int(args.video_interval))

    computed_num_evals = max(2, int(args.total_env_steps) // max(1, int(args.eval_interval)) + 1)

    ppo_train = ppo_train_module.train if hasattr(ppo_train_module, "train") else ppo_train_module

    train_kwargs = {
        "num_timesteps": int(args.total_env_steps),
        "episode_length": int(args.episode_length),
        "num_evals": int(computed_num_evals),
        "num_eval_envs": int(args.nworld),
        "reward_scaling": float(args.reward_scaling),
        "normalize_observations": bool(args.normalize_observations),
        "action_repeat": 1,
        "unroll_length": int(args.rollout_length),
        "batch_size": int(args.batch_size),
        "num_minibatches": int(args.num_minibatches),
        "num_updates_per_batch": int(args.update_epochs),
        "discounting": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "learning_rate": _resolve_learning_rate(args.learning_rate_schedule, args.learning_rate),
        "entropy_cost": float(args.ent_coef),
        "clipping_epsilon": float(args.clip_coef),
        "clipping_epsilon_value": float(args.clip_coef_value),
        "max_grad_norm": float(args.max_grad_norm),
        "desired_kl": float(args.target_kl),
        "num_envs": int(args.nworld),
        "seed": int(args.seed),
        "vision": True,
        "augment_pixels": bool(args.augment_pixels),
        "wrap_env_fn": _wrap_env_no_vmap,
        "max_devices_per_host": 1,
        "use_pmap_on_reset": False,
        "network_factory": network_factory,
        "progress_fn": progress_fn,
        "policy_params_fn": policy_params_fn,
        "eval_env": vision_env,
    }

    filtered_train_kwargs = _filter_kwargs(ppo_train, train_kwargs)

    training_start = time.time()
    train_output = ppo_train(environment=vision_env, **filtered_train_kwargs)
    if not isinstance(train_output, Sequence) or len(train_output) < 2:
        raise RuntimeError("Unexpected return value from brax PPO train().")

    make_inference_fn = train_output[0]
    params = train_output[1]
    metrics = train_output[2] if len(train_output) >= 3 else {}

    train_seconds = time.time() - training_start
    print(f"training complete in {train_seconds:.1f}s")
    writer.add_scalar("training_seconds", train_seconds, int(args.total_env_steps))

    if isinstance(metrics, Mapping):
        for key, value in metrics.items():
            try:
                writer.add_scalar(f"final_metrics/{key}", float(np.asarray(value)), int(args.total_env_steps))
            except Exception:
                continue

    if bool(args.save_model):
        params_format = _normalize_params_format(args.params_format)
        params_ext = _params_extension(params_format)
        out_path = Path("runs") / run_name / f"{args.exp_name}{params_ext}"
        serialization = _save_params(out_path, params)
        print(f"model saved: {out_path} ({serialization})")

    if bool(args.capture_video):
        frames = _build_video_rollout(
            env=vision_env,
            make_policy_builder=make_inference_fn,
            params=params,
            seed=args.seed + 99_999,
            num_steps=int(args.video_steps),
            render_main_fn=video_main_renderer,
            render_inset_fn=video_inset_renderer,
            inset_scale=float(args.video_inset_scale),
            inset_margin=int(args.video_inset_margin),
            video_brightness=float(args.video_brightness),
            video_gamma=float(args.video_gamma),
        )
        final_video_path = Path(args.video_dirname) / run_name / f"{args.exp_name}_final.mp4"
        _save_video_mp4(frames, final_video_path, fps=video_fps)
        print(f"final video saved: {final_video_path}")
        if wandb is not None:
            try:
                wandb.log({"episode_video": wandb.Video(str(final_video_path), format="mp4"), "num_steps": int(args.total_env_steps)})
            except Exception:
                pass

    writer.close()
    if wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
