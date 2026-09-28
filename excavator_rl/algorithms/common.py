"""Building blocks shared by the vortexRL algorithm ports.

Everything here is plain torch -- no Isaac Lab import -- so the algorithms can
be unit-tested on a laptop against a toy environment (``tests/test_algorithms.py``).

Vectorised environment contract (see :class:`excavator_rl.algorithms.isaaclab_env.IsaacLabVecEnv`)::

    env.num_envs, env.obs_dim, env.act_dim, env.device
    obs = env.reset()                                   # (N, obs_dim)
    obs, rew, terminated, truncated, info = env.step(a) # a: (N, act_dim) in [-1, 1]

Environments reset themselves: after a step, ``obs`` of a finished environment
is already the first observation of its next episode.  ``terminated`` means a
real terminal state (task done), ``truncated`` a time-out.
"""

from __future__ import annotations

import itertools
import os
import time
from collections import deque
from dataclasses import asdict, fields
from typing import Any

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# networks
# --------------------------------------------------------------------------- #
_ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}


def mlp(in_dim: int, hidden: tuple[int, ...] | list[int], out_dim: int, activation: str = "elu") -> nn.Sequential:
    """Fully connected network ``in -> hidden... -> out`` (no output activation)."""
    act = _ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


def discrete_action_table(act_dim: int, levels: tuple[float, ...] = (-1.0, 0.0, 1.0)) -> torch.Tensor:
    """Every combination of per-joint levels: vortexRL's ``action_dict``.

    With 3 joints and levels {-1, 0, 1} this is the 27-action space that
    vortexRL's REINFORCE and PPO use (``itertools.product([-1., 1., 0], repeat=3)``).
    """
    return torch.tensor(list(itertools.product(levels, repeat=act_dim)), dtype=torch.float32)


class RunningMeanStd(nn.Module):
    """Running observation normaliser (parallel Welford update), saved with the model."""

    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))
        self.clip = clip
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).float()
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta.pow(2) * self.count * b_n / tot) / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)


class ParamHolder(nn.Module):
    """Wrap a bare parameter (e.g. a log-std) so it saves / loads like a module."""

    def __init__(self, p: nn.Parameter):
        super().__init__()
        self.p = p


# --------------------------------------------------------------------------- #
# returns / advantages
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over a (T, N) rollout.

    ``dones[t]`` marks that the episode ended *after* step t (so no bootstrap
    from ``values[t + 1]``).  Returns ``(advantages, returns)``.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * not_done - values[t]
        last_gae = delta + gamma * lam * not_done * last_gae
        adv[t] = last_gae
    return adv, adv + values


# --------------------------------------------------------------------------- #
# bookkeeping
# --------------------------------------------------------------------------- #
class EpisodeTracker:
    """Per-environment episode return / length, plus a moving window of finished ones."""

    def __init__(self, num_envs: int, device, window: int = 200):
        self.ret = torch.zeros(num_envs, device=device)
        self.len = torch.zeros(num_envs, device=device)
        self.returns: deque[float] = deque(maxlen=window)
        self.lengths: deque[float] = deque(maxlen=window)
        self.successes: deque[float] = deque(maxlen=window)
        self.total_episodes = 0

    @torch.no_grad()
    def update(self, rew: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor) -> None:
        self.ret += rew
        self.len += 1
        done = terminated | truncated
        if done.any():
            idx = done.nonzero(as_tuple=False).squeeze(1)
            self.returns.extend(self.ret[idx].tolist())
            self.lengths.extend(self.len[idx].tolist())
            # the excavator task terminates only on success; time-outs are failures
            self.successes.extend(terminated[idx].float().tolist())
            self.total_episodes += idx.numel()
            self.ret[idx] = 0.0
            self.len[idx] = 0.0

    def summary(self) -> dict[str, float]:
        if not self.returns:
            return {}
        n = len(self.returns)
        return {
            "Episode/return": sum(self.returns) / n,
            "Episode/length": sum(self.lengths) / n,
            "Episode/success_rate": sum(self.successes) / n,
        }


class Logger:
    """TensorBoard + console logging.  ``log_dir=None`` prints only."""

    def __init__(self, log_dir: str | None, print_every: int = 10):
        self.writer = None
        if log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=log_dir, flush_secs=10)
            except ImportError:  # pragma: no cover
                print("[algorithms] tensorboard not installed -- console logging only")
        self.print_every = max(1, print_every)
        self.t0 = time.time()

    def log(self, it: int, total_it: int, env_steps: int, scalars: dict[str, float]) -> None:
        if self.writer is not None:
            for k, v in scalars.items():
                self.writer.add_scalar(k, v, env_steps)
        if it % self.print_every == 0 or it == total_it - 1:
            fps = env_steps / max(time.time() - self.t0, 1e-6)
            parts = [f"it {it + 1:>5}/{total_it}", f"steps {env_steps:>11,d}", f"{fps:>8,.0f} steps/s"]
            for key in ("Episode/return", "Episode/length", "Episode/success_rate"):
                if key in scalars:
                    parts.append(f"{key.split('/')[1]} {scalars[key]:8.2f}")
            print("  ".join(parts), flush=True)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


# --------------------------------------------------------------------------- #
# agent base class
# --------------------------------------------------------------------------- #
class Agent:
    """Common save / load / config handling.  Subclasses set ``name`` and ``Cfg``."""

    name: str = ""
    Cfg: type = object

    def __init__(self, obs_dim: int, act_dim: int, device, cfg=None):
        #: iterations already done before this run (set when resuming), used
        #: to keep checkpoint numbers increasing
        self.iteration_offset = 0
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.device = torch.device(device)
        self.cfg = cfg if cfg is not None else self.Cfg()
        self.obs_norm = RunningMeanStd(obs_dim).to(self.device)

    # -- modules that make up the agent (for save / load / device moves)
    def modules(self) -> dict[str, nn.Module]:
        raise NotImplementedError

    def optimizers(self) -> dict[str, torch.optim.Optimizer]:
        return {}

    # -- interface used by the scripts
    def train(self, env, num_iterations: int, logger: Logger, ckpt_dir: str | None = None, save_interval: int = 100) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action in [-1, 1] for evaluation."""
        raise NotImplementedError

    def save(self, path: str, extra: dict[str, Any] | None = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "algo": self.name,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "cfg": asdict(self.cfg),
                "obs_norm": self.obs_norm.state_dict(),
                "modules": {k: m.state_dict() for k, m in self.modules().items()},
                "optimizers": {k: o.state_dict() for k, o in self.optimizers().items()},
                **(extra or {}),
            },
            path,
        )

    def load_state(self, ckpt: dict[str, Any], load_optimizers: bool = True) -> None:
        self.obs_norm.load_state_dict(ckpt["obs_norm"])
        for k, m in self.modules().items():
            m.load_state_dict(ckpt["modules"][k])
        if load_optimizers:
            for k, o in self.optimizers().items():
                if k in ckpt.get("optimizers", {}):
                    o.load_state_dict(ckpt["optimizers"][k])

    def _maybe_save(self, it: int, num_iterations: int, ckpt_dir: str | None, save_interval: int) -> None:
        if ckpt_dir is None:
            return
        if (it + 1) % save_interval == 0 or it == num_iterations - 1:
            n = self.iteration_offset + it + 1
            self.save(os.path.join(ckpt_dir, f"model_{n}.pt"), extra={"iteration": n})


def cfg_from_overrides(cfg_cls: type, overrides: list[str] | None):
    """Build a config dataclass from ``["key=value", ...]`` command-line overrides."""
    cfg = cfg_cls()
    types = {f.name: f.type for f in fields(cfg_cls)}
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep or key not in types:
            raise ValueError(f"unknown config override '{item}'; valid keys: {sorted(types)}")
        default = getattr(cfg, key)
        if isinstance(default, bool):
            value: Any = raw.lower() in ("1", "true", "yes", "on")
        elif isinstance(default, int):
            value = int(float(raw))
        elif isinstance(default, float):
            value = float(raw)
        elif isinstance(default, (tuple, list)):
            value = tuple(int(v) for v in raw.strip("()[] ").split(",") if v.strip())
        else:
            value = raw
        setattr(cfg, key, value)
    return cfg
