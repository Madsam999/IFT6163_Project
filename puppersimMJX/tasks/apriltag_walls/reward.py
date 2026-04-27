"""Reward for watching a randomized AprilTag placed on one surrounding wall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import jax.numpy as jp


@dataclass(frozen=True)
class AprilTagWallsRewardConfig:
    """Config for camera-visibility based AprilTag watching reward."""

    use_sparse_distance_collect_reward: bool = False
    distance_penalty_weight: float = 1.0
    use_distance_collect_reward: bool = False
    progress_weight: float = 1.0
    use_visibility_gated_distance_reward: bool = True
    not_visible_penalty: float = 1.0
    visible_base_reward: float = 1.0
    visible_distance_bonus_weight: float = 1.0
    collect_radius_m: float = 0.35
    collect_consecutive_steps: int = 10
    collect_bonus: float = 10.0
    require_visibility_for_collect: bool = True
    visible_reward: float = 1.0
    alignment_weight: float = 0.6
    centering_weight: float = 0.8
    distance_weight: float = 0.15
    bad_visible_penalty: float = 1.2
    bad_alignment_penalty: float = 0.5
    bad_centering_penalty: float = 0.7
    bad_distance_penalty: float = 0.2
    contrastive_distance_weight: float = 0.2
    action_penalty_weight: float = 0.0


def compute_apriltag_walls_reward(
    *,
    apriltag_visible: jp.ndarray,
    apriltag_forward_cos: jp.ndarray,
    apriltag_centering: jp.ndarray,
    apriltag_distance_norm: jp.ndarray,
    apriltag_bad_visible: jp.ndarray,
    apriltag_bad_forward_cos: jp.ndarray,
    apriltag_bad_centering: jp.ndarray,
    apriltag_bad_distance_norm: jp.ndarray,
    action: jp.ndarray,
    config: AprilTagWallsRewardConfig,
) -> Tuple[jp.ndarray, Dict[str, jp.ndarray]]:
    """Computes a dense reward from good-vs-bad AprilTag camera features."""

    good_visible = jp.clip(apriltag_visible, 0.0, 1.0)
    good_alignment = jp.maximum(apriltag_forward_cos, 0.0)
    good_centering = jp.clip(apriltag_centering, 0.0, 1.0)
    good_proximity = jp.clip(apriltag_distance_norm, 0.0, 1.0)

    bad_visible = jp.clip(apriltag_bad_visible, 0.0, 1.0)
    bad_alignment = jp.maximum(apriltag_bad_forward_cos, 0.0)
    bad_centering = jp.clip(apriltag_bad_centering, 0.0, 1.0)
    bad_proximity = jp.clip(apriltag_bad_distance_norm, 0.0, 1.0)

    action_penalty = jp.mean(jp.square(action))
    good_reward = (
        float(config.visible_reward) * good_visible
        + float(config.alignment_weight) * good_alignment * good_visible
        + float(config.centering_weight) * good_centering * good_visible
        + float(config.distance_weight) * good_proximity * good_visible
    )
    bad_penalty = (
        float(config.bad_visible_penalty) * bad_visible
        + float(config.bad_alignment_penalty) * bad_alignment * bad_visible
        + float(config.bad_centering_penalty) * bad_centering * bad_visible
        + float(config.bad_distance_penalty) * bad_proximity * bad_visible
    )
    contrastive_distance = float(config.contrastive_distance_weight) * (good_proximity - bad_proximity)
    reward = (
        good_reward
        - bad_penalty
        + contrastive_distance
        - float(config.action_penalty_weight) * action_penalty
    )

    terms = {
        "apriltag_good_visible": good_visible,
        "apriltag_good_alignment": good_alignment,
        "apriltag_good_centering": good_centering,
        "apriltag_good_distance_norm": good_proximity,
        "apriltag_bad_visible": bad_visible,
        "apriltag_bad_alignment": bad_alignment,
        "apriltag_bad_centering": bad_centering,
        "apriltag_bad_distance_norm": bad_proximity,
        "apriltag_good_reward": good_reward,
        "apriltag_bad_penalty": bad_penalty,
        "apriltag_contrastive_distance": contrastive_distance,
        "action_penalty": action_penalty,
        "task_reward": reward,
    }
    return reward, terms


def build_reward(
    config: Optional[Dict[str, Any]] = None,
) -> Callable[[Dict[str, Any]], Tuple[jp.ndarray, Dict[str, jp.ndarray]]]:
    """Builds reward function from JSON-like config."""

    cfg = AprilTagWallsRewardConfig(**dict(config or {}))

    def reward_fn(ctx: Dict[str, Any]) -> Tuple[jp.ndarray, Dict[str, jp.ndarray]]:
        if bool(cfg.use_sparse_distance_collect_reward):
            d_curr = jp.asarray(ctx.get("goal_distance_override", ctx.get("apriltag_distance", 0.0)))
            visible = jp.clip(jp.asarray(ctx.get("apriltag_visible", 0.0)), 0.0, 1.0)
            within_radius = d_curr < float(cfg.collect_radius_m)
            if bool(cfg.require_visibility_for_collect):
                within_radius = within_radius & (visible > 0.5)
            collected = within_radius.astype(d_curr.dtype)
            action_penalty = jp.mean(jp.square(ctx["action"]))
            reward = (
                -float(cfg.distance_penalty_weight) * d_curr
                + float(cfg.collect_bonus) * collected
                - float(cfg.action_penalty_weight) * action_penalty
            )
            terms = {
                "task_reward": reward,
                "goal_reached": collected,
                "distance_to_tag": d_curr,
                "within_collect_radius": within_radius.astype(d_curr.dtype),
                "apriltag_visible": visible,
                "action_penalty": action_penalty,
            }
            return reward, terms

        if bool(cfg.use_distance_collect_reward):
            d_curr = jp.asarray(ctx.get("goal_distance_override", ctx.get("apriltag_distance", 0.0)))
            d_prev = jp.asarray(ctx.get("goal_distance_prev", d_curr))
            progress = d_prev - d_curr
            progress_reward = float(cfg.progress_weight) * progress

            visible = jp.clip(ctx["apriltag_visible"], 0.0, 1.0)
            distance_norm = jp.clip(jp.asarray(ctx.get("apriltag_distance_norm", 0.0)), 0.0, 1.0)
            visible_gate_reward = jp.asarray(0.0, dtype=d_curr.dtype)
            if bool(cfg.use_visibility_gated_distance_reward):
                # Requested shaping:
                # - if tag not visible: -1
                # - if visible: base + distance bonus (higher when closer)
                visible_case = float(cfg.visible_base_reward) + float(cfg.visible_distance_bonus_weight) * distance_norm
                not_visible_case = -float(cfg.not_visible_penalty)
                visible_gate_reward = jp.where(visible > 0.5, visible_case, not_visible_case)

            within_radius = d_curr < float(cfg.collect_radius_m)
            if bool(cfg.require_visibility_for_collect):
                within_radius = within_radius & (visible > 0.5)

            streak_prev = jp.asarray(ctx.get("collect_streak_prev", jp.zeros_like(d_curr)))
            streak = jp.where(within_radius, streak_prev + 1.0, jp.zeros_like(streak_prev))
            collected = (streak >= float(max(1, int(cfg.collect_consecutive_steps)))).astype(d_curr.dtype)

            action_penalty = jp.mean(jp.square(ctx["action"]))
            reward = (
                + progress_reward
                + visible_gate_reward
                + float(cfg.collect_bonus) * collected
                - float(cfg.action_penalty_weight) * action_penalty
            )
            terms = {
                "task_reward": reward,
                "goal_reached": collected,
                "collect_streak": streak,
                "distance_progress": progress,
                "distance_progress_reward": progress_reward,
                "distance_to_tag": d_curr,
                "distance_norm_to_tag": distance_norm,
                "within_collect_radius": within_radius.astype(d_curr.dtype),
                "apriltag_visible": visible,
                "visibility_gate_reward": visible_gate_reward,
                "action_penalty": action_penalty,
            }
            return reward, terms

        # Backward compatible defaults for older environments that only provide
        # a single AprilTag feature stream.
        zeros = jp.zeros_like(ctx["apriltag_visible"])
        return compute_apriltag_walls_reward(
            apriltag_visible=ctx["apriltag_visible"],
            apriltag_forward_cos=ctx["apriltag_forward_cos"],
            apriltag_centering=ctx["apriltag_centering"],
            apriltag_distance_norm=ctx["apriltag_distance_norm"],
            apriltag_bad_visible=ctx.get("apriltag_bad_visible", zeros),
            apriltag_bad_forward_cos=ctx.get("apriltag_bad_forward_cos", zeros),
            apriltag_bad_centering=ctx.get("apriltag_bad_centering", zeros),
            apriltag_bad_distance_norm=ctx.get("apriltag_bad_distance_norm", zeros),
            action=ctx["action"],
            config=cfg,
        )

    return reward_fn
