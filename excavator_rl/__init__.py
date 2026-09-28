"""Excavator digging task for Isaac Lab.

Importing this package registers the gym environments:

    Excavator-Digging-v0        training config
    Excavator-Digging-Play-v0   small, visualised config

Registration uses string entry points, so importing the package costs nothing
and ``excavator_rl.soil`` stays usable (and unit-testable) on a machine with
only torch installed.
"""

from __future__ import annotations

import importlib

__all__ = ["resolve_entry_point"]


def resolve_entry_point(spec):
    """Turn a ``"module:attr"`` string (or a class) into the object itself."""
    if isinstance(spec, str):
        module, _, attr = spec.partition(":")
        return getattr(importlib.import_module(module), attr)
    return spec


try:
    import gymnasium as gym

    gym.register(
        id="Excavator-Digging-v0",
        entry_point="excavator_rl.digging_env:DiggingEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "excavator_rl.digging_env_cfg:DiggingEnvCfg",
            "rsl_rl_cfg_entry_point": "excavator_rl.agents.rsl_rl_ppo_cfg:ExcavatorPPORunnerCfg",
        },
    )

    gym.register(
        id="Excavator-Digging-Play-v0",
        entry_point="excavator_rl.digging_env:DiggingEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "excavator_rl.digging_env_cfg:DiggingEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": "excavator_rl.agents.rsl_rl_ppo_cfg:ExcavatorPPORunnerCfg",
        },
    )
except ImportError:  # gymnasium absent: soil model and configs still importable
    pass
