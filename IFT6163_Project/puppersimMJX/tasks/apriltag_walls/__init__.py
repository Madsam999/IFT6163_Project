"""AprilTag-on-walls high-level navigation task package."""

from puppersimMJX.tasks.apriltag_walls.reward import AprilTagWallsRewardConfig
from puppersimMJX.tasks.apriltag_walls.reward import build_reward
from puppersimMJX.tasks.apriltag_walls.reward import compute_apriltag_walls_reward

__all__ = ["AprilTagWallsRewardConfig", "compute_apriltag_walls_reward", "build_reward"]
