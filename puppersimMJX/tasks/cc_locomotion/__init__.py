"""Command-conditioned locomotion task package."""

from puppersimMJX.tasks.cc_locomotion.nav_controller import NavController
from puppersimMJX.tasks.cc_locomotion.nav_controller import NavControllerConfig
from puppersimMJX.tasks.cc_locomotion.reward import CommandRewardConfig
from puppersimMJX.tasks.cc_locomotion.reward import CommandRewardScales
from puppersimMJX.tasks.cc_locomotion.reward import build_reward
from puppersimMJX.tasks.cc_locomotion.reward import compute_command_reward

__all__ = [
    "NavController",
    "NavControllerConfig",
    "CommandRewardConfig",
    "CommandRewardScales",
    "compute_command_reward",
    "build_reward",
]
