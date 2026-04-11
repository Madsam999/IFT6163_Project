"""Command-conditioned locomotion task package."""

from puppersimMJX.tasks.cc_locomotion.reward import CommandRewardConfig
from puppersimMJX.tasks.cc_locomotion.reward import CommandRewardScales
from puppersimMJX.tasks.cc_locomotion.reward import build_reward
from puppersimMJX.tasks.cc_locomotion.reward import compute_command_reward

__all__ = ["CommandRewardConfig", "CommandRewardScales", "compute_command_reward", "build_reward"]
