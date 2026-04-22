from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np


def _base_env(env: Any) -> Any:
    cur = env
    while hasattr(cur, "env"):
        cur = cur.env
    return cur


def _safe_xy_from_state(state: Any) -> np.ndarray:
    try:
        q = np.asarray(state.pipeline_state.q)
        if q.size >= 2:
            return np.asarray(q[:2], dtype=np.float64)
    except Exception:
        pass
    try:
        gp = state.info.get("goal_position", None)
        if gp is not None:
            g = np.asarray(gp, dtype=np.float64)
            if g.size >= 2:
                return g[:2]
    except Exception:
        pass
    return np.zeros((2,), dtype=np.float64)


def _extract_world_static_rects(env: Any) -> list[tuple[float, float, float, float]]:
    rects: list[tuple[float, float, float, float]] = []
    base = _base_env(env)
    sys_obj = getattr(base, "sys", None)
    mjm = getattr(sys_obj, "mj_model", None)
    if mjm is None:
        return rects

    try:
        geom_pos = np.asarray(mjm.geom_pos, dtype=np.float64)
        geom_size = np.asarray(mjm.geom_size, dtype=np.float64)
        geom_bodyid = np.asarray(mjm.geom_bodyid, dtype=np.int32)
    except Exception:
        return rects

    n = int(min(geom_pos.shape[0], geom_size.shape[0], geom_bodyid.shape[0]))
    for i in range(n):
        # World-body geoms usually correspond to static map layout.
        if int(geom_bodyid[i]) != 0:
            continue
        cx = float(geom_pos[i, 0])
        cy = float(geom_pos[i, 1])
        sx = float(abs(geom_size[i, 0]))
        sy = float(abs(geom_size[i, 1]))
        if sx <= 1e-5 and sy <= 1e-5:
            continue
        half_w = max(sx, 0.02)
        half_h = max(sy, 0.02)
        rects.append((cx - half_w, cy - half_h, 2.0 * half_w, 2.0 * half_h))
    return rects


def _collect_rollout(env: Any, make_policy_builder: Any, params: Any, seed: int, num_steps: int):
    import jax

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    policy = jax.jit(make_policy_builder(params, deterministic=True))

    rng = jax.random.PRNGKey(int(seed))
    rng, reset_rng = jax.random.split(rng)
    state = reset_fn(reset_rng)

    states = [state]
    rewards = [float(np.asarray(state.reward))]
    goals = [np.asarray(state.info.get("goal_position", np.zeros((2,), dtype=np.float32)), dtype=np.float64).reshape(-1)[:2]]

    for _ in range(max(1, int(num_steps))):
        rng, policy_rng = jax.random.split(rng)
        action, _ = policy(state.obs, policy_rng)
        state = step_fn(state, action)
        states.append(state)
        rewards.append(float(np.asarray(state.reward)))
        goals.append(np.asarray(state.info.get("goal_position", np.zeros((2,), dtype=np.float32)), dtype=np.float64).reshape(-1)[:2])

        done_flag = float(np.asarray(state.done))
        if done_flag >= 0.5:
            rng, reset_rng = jax.random.split(rng)
            state = reset_fn(reset_rng)
            states.append(state)
            rewards.append(float(np.asarray(state.reward)))
            goals.append(np.asarray(state.info.get("goal_position", np.zeros((2,), dtype=np.float32)), dtype=np.float64).reshape(-1)[:2])

    traj = np.stack([_safe_xy_from_state(s) for s in states], axis=0)
    goals_arr = np.stack(goals, axis=0)
    return states, traj, np.asarray(rewards, dtype=np.float64), goals_arr


def _compute_map_bounds(
    rects: list[tuple[float, float, float, float]],
    traj_xy: np.ndarray,
    goals_xy: np.ndarray,
    margin: float,
) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []

    for x, y, w, h in rects:
        xs.extend([float(x), float(x + w)])
        ys.extend([float(y), float(y + h)])

    if traj_xy.size:
        xs.extend([float(np.min(traj_xy[:, 0])), float(np.max(traj_xy[:, 0]))])
        ys.extend([float(np.min(traj_xy[:, 1])), float(np.max(traj_xy[:, 1]))])

    if goals_xy.size:
        xs.extend([float(np.min(goals_xy[:, 0])), float(np.max(goals_xy[:, 0]))])
        ys.extend([float(np.min(goals_xy[:, 1])), float(np.max(goals_xy[:, 1]))])

    if not xs or not ys:
        return (-3.0, 3.0, -3.0, 3.0)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = max(0.1, float(margin))
    return (xmin - pad, xmax + pad, ymin - pad, ymax + pad)


def _try_render_frame(env: Any, pstate: Any, camera: Optional[str], w: int, h: int) -> Optional[np.ndarray]:
    kwargs: dict[str, Any] = {"width": int(w), "height": int(h)}
    if camera:
        kwargs["camera"] = str(camera)
    try:
        frame = env.render([pstate], **kwargs)[0]
        return np.asarray(frame, dtype=np.uint8)
    except Exception:
        return None


def save_rollout_report(
    *,
    env: Any,
    make_policy_builder: Any,
    params: Any,
    seed: int,
    output_path: Path,
    num_steps: int,
    num_stages: int = 4,
    front_camera_name: Optional[str] = "front_cam",
    map_margin: float = 0.25,
    topdown_camera_name: Optional[str] = "tracking_cam",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle

    output_path = Path(output_path)
    states, traj_xy, rewards, goals_xy = _collect_rollout(
        env=env,
        make_policy_builder=make_policy_builder,
        params=params,
        seed=int(seed),
        num_steps=int(num_steps),
    )

    rects = _extract_world_static_rects(env)
    xmin, xmax, ymin, ymax = _compute_map_bounds(rects, traj_xy, goals_xy, margin=float(map_margin))

    n = max(2, int(num_stages))
    stage_idx = np.linspace(0, max(0, len(traj_xy) - 1), n, dtype=np.int32)

    fig = plt.figure(figsize=(16, 10), dpi=120, facecolor="#08111f")
    gs = fig.add_gridspec(2, n, height_ratios=[1.0, 1.65], hspace=0.18, wspace=0.08)

    def _draw_room(ax):
        ax.set_facecolor("#0a1425")
        for x, y, w, h in rects:
            ax.add_patch(Rectangle((x, y), w, h, facecolor="#394554", edgecolor="#7b8794", linewidth=0.6, alpha=0.9))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    cmap = plt.cm.turbo
    total_steps = max(1, len(traj_xy) - 1)

    for panel_i, idx in enumerate(stage_idx.tolist()):
        ax = fig.add_subplot(gs[0, panel_i])
        _draw_room(ax)

        if idx >= 1:
            pts = traj_xy[: idx + 1]
            seg = np.stack([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(seg, cmap=cmap, linewidths=2.2)
            lc.set_array(np.linspace(0.0, 1.0, seg.shape[0]))
            ax.add_collection(lc)

        start = traj_xy[0]
        cur = traj_xy[idx]
        ax.scatter([start[0]], [start[1]], s=42, c="#58d26f", edgecolors="none")
        ax.scatter([cur[0]], [cur[1]], s=68, c="#ff6f3c", edgecolors="#0b0b0b", linewidths=0.8)

        g = goals_xy[min(idx, len(goals_xy) - 1)]
        ax.scatter([g[0]], [g[1]], s=56, c="#f1d14b", marker="s", edgecolors="#111", linewidths=0.8)

        stage_reward = float(np.sum(rewards[: idx + 1]))
        ax.set_title(
            f"STAGE {panel_i}\\nstep={idx}  reward={stage_reward:.2f}",
            color="#d7dce3",
            fontsize=12,
            pad=8,
        )

        # Optional light background image from a top-down camera, if available.
        if topdown_camera_name:
            frame = _try_render_frame(
                env=env,
                pstate=states[idx].pipeline_state,
                camera=topdown_camera_name,
                w=420,
                h=320,
            )
            if frame is not None:
                ax.imshow(frame, extent=(xmin, xmax, ymin, ymax), alpha=0.18, origin="upper")

    map_ax = fig.add_subplot(gs[1, : n - 1])
    _draw_room(map_ax)
    if len(traj_xy) >= 2:
        seg = np.stack([traj_xy[:-1], traj_xy[1:]], axis=1)
        lc = LineCollection(seg, cmap=cmap, linewidths=3.2)
        lc.set_array(np.linspace(0.0, 1.0, seg.shape[0]))
        map_ax.add_collection(lc)
    map_ax.scatter([traj_xy[0, 0]], [traj_xy[0, 1]], s=70, c="#58d26f", edgecolors="none", label="start")
    map_ax.scatter([traj_xy[-1, 0]], [traj_xy[-1, 1]], s=110, c="#ff6f3c", edgecolors="#111", linewidths=1.0, label="current")

    # Show a few distinct goal samples seen in this rollout.
    shown_goals = goals_xy[:: max(1, len(goals_xy) // 8)] if len(goals_xy) else np.zeros((0, 2))
    if shown_goals.shape[0] > 0:
        map_ax.scatter(shown_goals[:, 0], shown_goals[:, 1], s=46, c="#f1d14b", marker="s", edgecolors="#111", linewidths=0.7)

    map_ax.set_title("AERIAL MAP VIEW (dynamic)", color="#d7dce3", fontsize=15, pad=10)

    side_ax = fig.add_subplot(gs[1, n - 1])
    side_ax.set_facecolor("#0a1425")
    side_ax.set_xticks([])
    side_ax.set_yticks([])
    for spine in side_ax.spines.values():
        spine.set_color("#2c394a")

    final_front = None
    if front_camera_name:
        final_front = _try_render_frame(
            env=env,
            pstate=states[-1].pipeline_state,
            camera=front_camera_name,
            w=480,
            h=320,
        )

    if final_front is not None:
        side_ax.imshow(final_front)
        side_ax.set_title("FRONT CAMERA", color="#d7dce3", fontsize=12, pad=8)
    else:
        side_ax.text(
            0.5,
            0.55,
            "FRONT CAMERA\nunavailable",
            color="#d7dce3",
            ha="center",
            va="center",
            fontsize=12,
            transform=side_ax.transAxes,
        )

    info_lines = [
        f"steps: {int(total_steps)}",
        f"reward sum: {float(np.sum(rewards)):.3f}",
        f"x range: [{xmin:.2f}, {xmax:.2f}]",
        f"y range: [{ymin:.2f}, {ymax:.2f}]",
        f"goals seen: {int(goals_xy.shape[0])}",
    ]
    fig.text(
        0.012,
        0.012,
        " | ".join(info_lines),
        color="#95a4b8",
        fontsize=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
