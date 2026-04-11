"""Simple forward locomotion task package."""

from puppersimMJX.tasks.simple_forward.reward import SimpleForwardRewardConfig
from puppersimMJX.tasks.simple_forward.reward import build_reward
from puppersimMJX.tasks.simple_forward.reward import compute_simple_forward_reward

__all__ = ["SimpleForwardRewardConfig", "compute_simple_forward_reward", "build_reward"]
