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
            return np.asarray(q, dtype=np.float64).reshape(-1)[:2]
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
    episode_ids = [0]
    episode_id = 0

    for _ in range(max(1, int(num_steps))):
        rng, policy_rng = jax.random.split(rng)
        action, _ = policy(state.obs, policy_rng)
        state = step_fn(state, action)
        states.append(state)
        rewards.append(float(np.asarray(state.reward)))
        goals.append(np.asarray(state.info.get("goal_position", np.zeros((2,), dtype=np.float32)), dtype=np.float64).reshape(-1)[:2])
        episode_ids.append(int(episode_id))

        done_flag = float(np.asarray(state.done))
        if done_flag >= 0.5:
            rng, reset_rng = jax.random.split(rng)
            state = reset_fn(reset_rng)
            episode_id += 1
            states.append(state)
            rewards.append(float(np.asarray(state.reward)))
            goals.append(np.asarray(state.info.get("goal_position", np.zeros((2,), dtype=np.float32)), dtype=np.float64).reshape(-1)[:2])
            episode_ids.append(int(episode_id))

    traj = np.stack([_safe_xy_from_state(s) for s in states], axis=0)
    goals_arr = np.stack(goals, axis=0)
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    episode_ids_arr = np.asarray(episode_ids, dtype=np.int32)

    ep_returns: list[float] = []
    ep_lengths: list[int] = []
    if episode_ids_arr.size > 0:
        for eid in range(int(np.max(episode_ids_arr)) + 1):
            mask = episode_ids_arr == eid
            if not np.any(mask):
                continue
            ep_returns.append(float(np.sum(rewards_arr[mask])))
            ep_lengths.append(int(np.sum(mask)))

    stats = {
        "episode_ids": episode_ids_arr,
        "episodes": int(max(1, len(ep_returns))),
        "ep_returns": np.asarray(ep_returns, dtype=np.float64),
        "ep_lengths": np.asarray(ep_lengths, dtype=np.int32),
    }
    return states, traj, rewards_arr, goals_arr, stats


def _filter_xy_samples(xy: np.ndarray, *, max_abs: float = 100.0) -> np.ndarray:
    if xy.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    arr = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    finite = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    sane = (np.abs(arr[:, 0]) < float(max_abs)) & (np.abs(arr[:, 1]) < float(max_abs))
    return arr[finite & sane]


def _stage_reward_for_index(rewards: np.ndarray, episode_ids: np.ndarray, idx: int) -> float:
    if rewards.size == 0:
        return 0.0
    j = int(np.clip(idx, 0, rewards.shape[0] - 1))
    if episode_ids.size != rewards.size:
        return float(np.sum(rewards[: j + 1]))
    eid = int(episode_ids[j])
    mask = episode_ids[: j + 1] == eid
    return float(np.sum(rewards[: j + 1][mask]))


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


def _filter_goal_samples(
    goals_xy: np.ndarray,
    rects: list[tuple[float, float, float, float]],
    *,
    slack_m: float = 1.0,
) -> np.ndarray:
    if goals_xy.size == 0:
        return goals_xy
    g = np.asarray(goals_xy, dtype=np.float64).reshape(-1, 2)
    finite = np.isfinite(g[:, 0]) & np.isfinite(g[:, 1])
    # Drop clearly invalid/inactive sentinels like [1000, 1000].
    sane = (np.abs(g[:, 0]) < 100.0) & (np.abs(g[:, 1]) < 100.0)
    keep = finite & sane
    if not np.any(keep):
        return np.zeros((0, 2), dtype=np.float64)
    g = g[keep]
    if not rects:
        return g

    xs: list[float] = []
    ys: list[float] = []
    for x, y, w, h in rects:
        xs.extend([float(x), float(x + w)])
        ys.extend([float(y), float(y + h)])
    if not xs or not ys:
        return g
    xmin, xmax = min(xs) - float(slack_m), max(xs) + float(slack_m)
    ymin, ymax = min(ys) - float(slack_m), max(ys) + float(slack_m)
    inside = (g[:, 0] >= xmin) & (g[:, 0] <= xmax) & (g[:, 1] >= ymin) & (g[:, 1] <= ymax)
    return g[inside]


def _build_exploration_maps(
    *,
    rects: list[tuple[float, float, float, float]],
    traj_xy: np.ndarray,
    bounds: tuple[float, float, float, float],
    cell_size: float,
) -> dict[str, Any]:
    xmin, xmax, ymin, ymax = bounds
    cell = max(0.02, float(cell_size))
    nx = max(1, int(np.ceil((xmax - xmin) / cell)))
    ny = max(1, int(np.ceil((ymax - ymin) / cell)))

    xs = xmin + (np.arange(nx, dtype=np.float64) + 0.5) * cell
    ys = ymin + (np.arange(ny, dtype=np.float64) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    blocked = np.zeros((ny, nx), dtype=bool)
    for x, y, w, h in rects:
        x0 = float(x)
        x1 = float(x + w)
        y0 = float(y)
        y1 = float(y + h)
        blocked |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    free = ~blocked

    visit_count = np.zeros((ny, nx), dtype=np.int32)
    first_visit = np.full((ny, nx), -1, dtype=np.int32)
    if traj_xy.size > 0:
        ix = np.floor((traj_xy[:, 0] - xmin) / cell).astype(np.int32)
        iy = np.floor((traj_xy[:, 1] - ymin) / cell).astype(np.int32)
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)
        for t, (cx, cy) in enumerate(zip(ix.tolist(), iy.tolist())):
            if not free[cy, cx]:
                continue
            visit_count[cy, cx] += 1
            if first_visit[cy, cx] < 0:
                first_visit[cy, cx] = int(t)

    visited = (visit_count > 0) & free
    unique_cells = int(np.count_nonzero(visited))
    free_cells = int(np.count_nonzero(free))
    total_visits = int(np.sum(visit_count))
    traj_steps = int(max(1, traj_xy.shape[0] - 1))
    coverage_pct = (100.0 * unique_cells / free_cells) if free_cells > 0 else 0.0
    revisit_ratio = (float(total_visits) / float(max(1, unique_cells))) if unique_cells > 0 else 0.0
    new_cells_per_100 = float(unique_cells) / (float(max(1, traj_steps)) / 100.0)

    return {
        "visit_count": visit_count,
        "first_visit": first_visit,
        "free_mask": free,
        "unique_cells": unique_cells,
        "free_cells": free_cells,
        "coverage_pct": coverage_pct,
        "revisit_ratio": revisit_ratio,
        "new_cells_per_100_steps": new_cells_per_100,
    }


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
    coverage_cell_size: float = 0.15,
    num_rollouts: int = 1,
    rollout_seed_stride: int = 1,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle

    output_path = Path(output_path)
    runs = max(1, int(num_rollouts))
    seed_stride = max(1, int(rollout_seed_stride))
    rollouts: list[dict[str, Any]] = []
    for i in range(runs):
        run_seed = int(seed) + i * seed_stride
        states_i, traj_i, rewards_i, goals_i, stats_i = _collect_rollout(
            env=env,
            make_policy_builder=make_policy_builder,
            params=params,
            seed=run_seed,
            num_steps=int(num_steps),
        )
        rollouts.append(
            {
                "states": states_i,
                "traj": traj_i,
                "rewards": rewards_i,
                "goals": goals_i,
                "stats": stats_i,
            }
        )

    # Use the first rollout for stage snapshots/front-camera panel.
    states = rollouts[0]["states"]
    traj_xy = rollouts[0]["traj"]
    rewards = rollouts[0]["rewards"]
    goals_xy = rollouts[0]["goals"]
    episode_ids = np.asarray(rollouts[0]["stats"].get("episode_ids", np.zeros_like(rewards, dtype=np.int32)), dtype=np.int32)

    all_traj = np.concatenate([_filter_xy_samples(r["traj"]) for r in rollouts], axis=0) if rollouts else _filter_xy_samples(traj_xy)
    all_goals_raw = np.concatenate([r["goals"] for r in rollouts], axis=0) if rollouts else goals_xy

    rects = _extract_world_static_rects(env)
    all_goals = _filter_goal_samples(all_goals_raw, rects, slack_m=1.0)
    xmin, xmax, ymin, ymax = _compute_map_bounds(rects, all_traj, all_goals, margin=float(map_margin))

    explore_runs = [
        _build_exploration_maps(
            rects=rects,
            traj_xy=_filter_xy_samples(r["traj"]),
            bounds=(xmin, xmax, ymin, ymax),
            cell_size=float(coverage_cell_size),
        )
        for r in rollouts
    ]
    visit_count = np.sum([e["visit_count"] for e in explore_runs], axis=0)
    free_mask = explore_runs[0]["free_mask"] if explore_runs else np.zeros((1, 1), dtype=bool)
    first_visit_stack = np.stack(
        [np.where(e["first_visit"] >= 0, e["first_visit"], 10**9) for e in explore_runs],
        axis=0,
    )
    first_visit = np.min(first_visit_stack, axis=0)
    first_visit = np.where(first_visit >= 10**9, -1, first_visit)

    coverage_vals = np.asarray([float(e["coverage_pct"]) for e in explore_runs], dtype=np.float64)
    revisit_vals = np.asarray([float(e["revisit_ratio"]) for e in explore_runs], dtype=np.float64)
    new_cells_vals = np.asarray([float(e["new_cells_per_100_steps"]) for e in explore_runs], dtype=np.float64)
    reward_sums = np.asarray([float(np.sum(r["rewards"])) for r in rollouts], dtype=np.float64)
    unique_vals = np.asarray([int(e["unique_cells"]) for e in explore_runs], dtype=np.int32)
    free_cells = int(explore_runs[0]["free_cells"]) if explore_runs else 0
    episodes_vals = np.asarray([int(r["stats"].get("episodes", 1)) for r in rollouts], dtype=np.int32)
    ep_lens_all = np.concatenate(
        [np.asarray(r["stats"].get("ep_lengths", np.zeros((0,), dtype=np.int32)), dtype=np.int32) for r in rollouts],
        axis=0,
    )
    mean_ep_len = float(np.mean(ep_lens_all)) if ep_lens_all.size > 0 else 0.0

    fig = plt.figure(figsize=(14, 7), dpi=120, facecolor="#08111f")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.08)

    def _draw_room(
        ax,
        *,
        bg_color: str = "#0a1425",
        wall_face: str = "#394554",
        wall_edge: str = "#7b8794",
    ):
        ax.set_facecolor(bg_color)
        for x, y, w, h in rects:
            ax.add_patch(Rectangle((x, y), w, h, facecolor=wall_face, edgecolor=wall_edge, linewidth=0.6, alpha=0.9))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    total_steps = max(1, len(traj_xy) - 1)
    traj_ax = fig.add_subplot(gs[0, 0])
    _draw_room(traj_ax)
    ep_cmap = plt.get_cmap("tab20")
    total_episodes = 0
    for run_idx, r in enumerate(rollouts):
        txy_raw = np.asarray(r["traj"], dtype=np.float64).reshape(-1, 2)
        eids = np.asarray(r["stats"].get("episode_ids", np.zeros((txy_raw.shape[0],), dtype=np.int32)), dtype=np.int32)
        if eids.shape[0] != txy_raw.shape[0]:
            eids = np.zeros((txy_raw.shape[0],), dtype=np.int32)
        unique_eids = np.unique(eids)
        total_episodes += int(unique_eids.size)
        for local_i, eid in enumerate(unique_eids.tolist()):
            idx = np.where(eids == int(eid))[0]
            if idx.size < 2:
                continue
            pts = txy_raw[idx]
            keep = (
                np.isfinite(pts[:, 0])
                & np.isfinite(pts[:, 1])
                & (np.abs(pts[:, 0]) < 100.0)
                & (np.abs(pts[:, 1]) < 100.0)
            )
            pts = pts[keep]
            if pts.shape[0] < 2:
                continue
            color = ep_cmap((run_idx * 7 + local_i) % 20)
            seg = np.stack([pts[:-1], pts[1:]], axis=1)
            lc = LineCollection(seg, colors=[color], linewidths=1.8, alpha=0.82)
            traj_ax.add_collection(lc)
            traj_ax.scatter([pts[0, 0]], [pts[0, 1]], s=14, c=[color], edgecolors="none", alpha=0.95)

    # Show a few distinct goal samples seen in this rollout.
    shown_goals = all_goals[:: max(1, len(all_goals) // 10)] if len(all_goals) else np.zeros((0, 2))
    if shown_goals.shape[0] > 0:
        traj_ax.scatter(shown_goals[:, 0], shown_goals[:, 1], s=46, c="#f1d14b", marker="s", edgecolors="#111", linewidths=0.7)

    traj_ax.set_title("AERIAL TRAJECTORIES (per episode/reset)", color="#d7dce3", fontsize=14, pad=10)

    # Mark start and end from the first rollout for orientation.
    traj_xy_sane = _filter_xy_samples(traj_xy)
    if traj_xy_sane.shape[0] > 0:
        traj_ax.scatter([traj_xy_sane[0, 0]], [traj_xy_sane[0, 1]], s=74, c="#58d26f", edgecolors="none")
        traj_ax.scatter([traj_xy_sane[-1, 0]], [traj_xy_sane[-1, 1]], s=105, c="#ff6f3c", edgecolors="#111", linewidths=1.0)

    heat_ax = fig.add_subplot(gs[0, 1])
    _draw_room(
        heat_ax,
        bg_color="#f6f0e1",
        wall_face="#c9c0ae",
        wall_edge="#9d9380",
    )
    if np.any(visit_count > 0):
        heat = np.ma.array(visit_count.astype(np.float64), mask=~free_mask)
        heat_ax.imshow(
            heat,
            extent=(xmin, xmax, ymin, ymax),
            origin="lower",
            cmap="magma",
            alpha=0.88,
            interpolation="nearest",
        )
    heat_ax.set_title("EXPLORATION HEATMAP", color="#d7dce3", fontsize=14, pad=10)

    info_lines = [
        f"runs: {runs}",
        f"episodes total: {total_episodes}",
        f"episodes/run: {float(np.mean(episodes_vals)):.2f}±{float(np.std(episodes_vals)):.2f}",
        f"avg ep len: {mean_ep_len:.1f}",
        f"steps: {int(total_steps)}",
        f"reward sum: {float(np.mean(reward_sums)):.3f}±{float(np.std(reward_sums)):.3f}",
        f"coverage: {float(np.mean(coverage_vals)):.1f}%±{float(np.std(coverage_vals)):.1f}",
        f"unique/free cells: {int(np.mean(unique_vals))}/{int(free_cells)}",
        f"revisit ratio: {float(np.mean(revisit_vals)):.2f}±{float(np.std(revisit_vals)):.2f}",
        f"new cells/100: {float(np.mean(new_cells_vals)):.2f}±{float(np.std(new_cells_vals)):.2f}",
        f"x range: [{xmin:.2f}, {xmax:.2f}]",
        f"y range: [{ymin:.2f}, {ymax:.2f}]",
        f"goals seen(valid/raw): {int(all_goals.shape[0])}/{int(all_goals_raw.shape[0])}",
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
