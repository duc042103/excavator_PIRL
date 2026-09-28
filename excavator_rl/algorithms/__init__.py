"""The four RL algorithms of vortexRL, re-implemented in PyTorch for many
parallel Isaac Lab environments.

=============  ==============  =============================================
``--algo``     action space    vortexRL source
=============  ==============  =============================================
``reinforce``  discrete (27)   ``algorithms/Reinforce_discrete.py``
``ppo``        discrete (27)*  ``algorithms/PPO_agent.py`` (*or continuous:
                               ``PPO_agentcontinuous.py``)
``trpo``       continuous      ``algorithms/TRPO.py``
``ddpg``       continuous      ``algorithms/DDPG.py``
=============  ==============  =============================================

Pure torch: nothing here imports Isaac Lab, so the algorithms run (and are
tested, see ``tests/test_algorithms.py``) against any vectorised environment
that follows the contract in :mod:`excavator_rl.algorithms.common`.
"""

from __future__ import annotations

import torch

from .common import Agent, Logger, cfg_from_overrides
from .ddpg import DDPG, DDPGCfg
from .ppo import PPO, PPOCfg
from .reinforce import Reinforce, ReinforceCfg
from .trpo import TRPO, TRPOCfg

ALGORITHMS: dict[str, type[Agent]] = {
    "reinforce": Reinforce,
    "ppo": PPO,
    "trpo": TRPO,
    "ddpg": DDPG,
}

__all__ = [
    "ALGORITHMS",
    "Agent",
    "DDPG",
    "DDPGCfg",
    "Logger",
    "PPO",
    "PPOCfg",
    "Reinforce",
    "ReinforceCfg",
    "TRPO",
    "TRPOCfg",
    "cfg_from_overrides",
    "load_agent",
    "make_agent",
]


def make_agent(algo: str, obs_dim: int, act_dim: int, device, overrides: list[str] | None = None) -> Agent:
    """Create an agent by name, applying ``key=value`` config overrides."""
    if algo not in ALGORITHMS:
        raise ValueError(f"unknown algorithm '{algo}', choose from {sorted(ALGORITHMS)}")
    cls = ALGORITHMS[algo]
    return cls(obs_dim, act_dim, device, cfg_from_overrides(cls.Cfg, overrides))


def load_agent(path: str, device, load_optimizers: bool = False) -> tuple[Agent, dict]:
    """Rebuild an agent (network sizes, action space, normaliser) from a checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cls = ALGORITHMS[ckpt["algo"]]
    cfg = cls.Cfg(**{k: tuple(v) if isinstance(v, list) else v for k, v in ckpt["cfg"].items()})
    agent = cls(ckpt["obs_dim"], ckpt["act_dim"], device, cfg)
    agent.load_state(ckpt, load_optimizers=load_optimizers)
    return agent, ckpt
