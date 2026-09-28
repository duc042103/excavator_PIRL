"""Excavator digging task for Isaac Lab.

Importing this package registers the gym environments:

    Excavator-Digging-v0        training config
    Excavator-Digging-Play-v0   small, visualised config

Each id carries an agent config for every RL library Isaac Lab ships, so the
task runs with any of them (all PPO):

    rsl_rl_cfg_entry_point    excavator_rl/agents/rsl_rl_ppo_cfg.py
    skrl_cfg_entry_point      excavator_rl/agents/skrl_ppo_cfg.yaml
    rl_games_cfg_entry_point  excavator_rl/agents/rl_games_ppo_cfg.yaml
    sb3_cfg_entry_point       excavator_rl/agents/sb3_ppo_cfg.yaml

Registration uses string entry points, so importing the package costs nothing,
never touches Isaac Lab (safe before the simulator app is launched), and
``excavator_rl.soil`` stays usable (and unit-testable) on a machine with only
torch installed.
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


#: agent configs shared by the training and the play task
_AGENT_CFGS = {
    "rsl_rl_cfg_entry_point": "excavator_rl.agents.rsl_rl_ppo_cfg:ExcavatorPPORunnerCfg",
    "skrl_cfg_entry_point": "excavator_rl.agents:skrl_ppo_cfg.yaml",
    "rl_games_cfg_entry_point": "excavator_rl.agents:rl_games_ppo_cfg.yaml",
    "sb3_cfg_entry_point": "excavator_rl.agents:sb3_ppo_cfg.yaml",
}

try:
    import gymnasium as gym

    for _id, _cfg in (
        ("Excavator-Digging-v0", "excavator_rl.digging_env_cfg:DiggingEnvCfg"),
        ("Excavator-Digging-Play-v0", "excavator_rl.digging_env_cfg:DiggingEnvCfg_PLAY"),
    ):
        if _id not in gym.registry:  # re-importing must not warn about re-registration
            gym.register(
                id=_id,
                entry_point="excavator_rl.digging_env:DiggingEnv",
                disable_env_checker=True,
                kwargs={"env_cfg_entry_point": _cfg, **_AGENT_CFGS},
            )
except ImportError:  # gymnasium absent: soil model and configs still importable
    pass
