"""Wrappers around Gymnasium environments: the STL/reward-machine layer.

``wrap_env_specs`` builds the stack; ``SpecWrapper`` implements it;
``RewardMachine`` is the FSM it drives.
"""

from .reward_machine import RewardMachine
from .spec_wrapper import SpecWrapper, wrap_env_specs

__all__ = ["SpecWrapper", "wrap_env_specs", "RewardMachine"]
