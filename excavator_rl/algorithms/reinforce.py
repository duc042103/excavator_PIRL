"""REINFORCE (Monte-Carlo policy gradient), port of vortexRL ``algorithms/Reinforce_discrete.py``.

Reference: Sutton, McAllester, Singh, Mansour -- *Policy Gradient Methods for
Reinforcement Learning with Function Approximation* (NIPS 1999).

Kept from vortexRL
    * discrete action space: every combination of {-1, 0, +1} per joint
      (3 joints -> 27 actions), softmax policy, no critic;
    * loss = -sum_t G_t * log pi(a_t | s_t), discounted Monte-Carlo return G_t;
    * a baseline on the return (vortexRL added a constant +2.5 to every reward;
      here the returns are normalised, which is the same idea without a
      hand-tuned constant).

Changed for thousands of parallel environments
    * one update per rollout of ``horizon`` steps x N environments instead of
      one update per episode;
    * environments reset themselves, so an episode can still be running when
      the rollout ends.  Its Monte-Carlo return is then cut short.  Only steps
      whose return is (nearly) complete are used: the episode ended inside the
      rollout, or at least ``3 / (1 - gamma)`` steps of it follow (the part
      cut off is then weighted by gamma^k < e^-3 ~ 5 %).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .common import Agent, EpisodeTracker, discrete_action_table, mlp


@dataclass
class ReinforceCfg:
    num_envs: int = 512
    horizon: int = 512
    """Rollout length per update [control steps]; 512 steps = 17 s at 30 Hz."""
    gamma: float = 0.99
    learning_rate: float = 1.0e-3
    hidden_dims: tuple = (128, 64)
    activation: str = "elu"
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    action_levels: tuple = (-1, 0, 1)
    """Command levels per joint (x action_scale of the env).  vortexRL: {-1, 0, 1}."""


class Reinforce(Agent):
    name = "reinforce"
    Cfg = ReinforceCfg

    def __init__(self, obs_dim: int, act_dim: int, device, cfg: ReinforceCfg | None = None):
        super().__init__(obs_dim, act_dim, device, cfg)
        c = self.cfg
        self.table = discrete_action_table(act_dim, tuple(float(v) for v in c.action_levels)).to(self.device)
        self.policy = mlp(obs_dim, c.hidden_dims, self.table.shape[0], c.activation).to(self.device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=c.learning_rate)
        # minimum number of following steps for a return to count as complete
        self.min_tail = int(round(3.0 / max(1.0 - c.gamma, 1e-6)))

    def modules(self) -> dict[str, nn.Module]:
        return {"policy": self.policy}

    def optimizers(self):
        return {"policy": self.opt}

    def _dist(self, obs_n: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.policy(obs_n))

    @torch.no_grad()
    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self.table[self.policy(self.obs_norm(obs)).argmax(dim=-1)]

    def train(self, env, num_iterations, logger, ckpt_dir=None, save_interval=100):
        c = self.cfg
        N, T = env.num_envs, c.horizon
        tracker = EpisodeTracker(N, self.device)
        obs = env.reset()
        env_steps = 0

        for it in range(num_iterations):
            obs_buf = torch.zeros(T, N, self.obs_dim, device=self.device)
            act_buf = torch.zeros(T, N, dtype=torch.long, device=self.device)
            rew_buf = torch.zeros(T, N, device=self.device)
            done_buf = torch.zeros(T, N, device=self.device)

            with torch.no_grad():
                for t in range(T):
                    self.obs_norm.update(obs)
                    obs_buf[t] = obs
                    a = self._dist(self.obs_norm(obs)).sample()
                    obs, rew, term, trunc, _ = env.step(self.table[a])
                    act_buf[t], rew_buf[t] = a, rew
                    done_buf[t] = (term | trunc).float()
                    tracker.update(rew, term, trunc)
            env_steps += T * N

            # discounted Monte-Carlo returns + which of them are complete enough
            with torch.no_grad():
                ret = torch.zeros(T, N, device=self.device)
                valid = torch.zeros(T, N, dtype=torch.bool, device=self.device)
                # the episode that contains step t ended inside the rollout iff
                # some done flag is set at t' >= t
                ended = torch.zeros(N, dtype=torch.bool, device=self.device)
                g = torch.zeros(N, device=self.device)
                for t in reversed(range(T)):
                    d = done_buf[t].bool()
                    g = rew_buf[t] + c.gamma * g * (~d).float()
                    ret[t] = g
                    ended |= d
                    valid[t] = ended | (T - t > self.min_tail)
                n_valid = int(valid.sum())
                g_v = ret[valid]
                adv = (g_v - g_v.mean()) / (g_v.std() + 1e-8) if n_valid > 1 else g_v

            logp_ent = None
            if n_valid > 1:
                dist = self._dist(self.obs_norm(obs_buf[valid]))
                logp = dist.log_prob(act_buf[valid])
                entropy = dist.entropy().mean()
                loss = -(logp * adv).mean() - c.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), c.max_grad_norm)
                self.opt.step()
                logp_ent = (loss.item(), entropy.item())

            scalars = tracker.summary()
            scalars["Train/valid_fraction"] = n_valid / (T * N)
            if logp_ent is not None:
                scalars["Loss/policy"], scalars["Policy/entropy"] = logp_ent
            logger.log(it, num_iterations, env_steps, scalars)
            self._maybe_save(it, num_iterations, ckpt_dir, save_interval)
