"""DDPG, port of vortexRL ``algorithms/DDPG.py``.

Reference: Lillicrap et al. -- *Continuous control with deep reinforcement
learning* (ICLR 2016).

Kept from vortexRL
    * actor 400-300 ReLU with tanh output, final layer initialised
      uniform(+-3e-3); critic: state -> 400, concatenated with the action ->
      300 -> Q (the layout of the paper and of vortexRL);
    * Ornstein-Uhlenbeck exploration noise (theta 0.15, sigma 0.1), soft
      target updates tau = 5e-3, Adam 5e-4 for actor and critic, replay buffer.

Changed for thousands of parallel environments
    * every control step adds N transitions (one per environment) to a GPU
      replay buffer and runs ``updates_per_step`` gradient steps with larger
      batches;
    * one OU process per environment, reset when that environment resets;
    * ``gamma`` 0.99 at 30 Hz (vortexRL used 0.9 at 2 Hz: 0.9^(1/15) = 0.993);
    * transitions that end in a time-out are not stored: their next
      observation is already the reset state, which would corrupt the target.
      Real terminal states (task success) are stored with done = 1.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn

from .common import Agent, EpisodeTracker


@dataclass
class DDPGCfg:
    num_envs: int = 256
    steps_per_iteration: int = 32
    """Control steps between two log lines / checkpoint counts."""
    buffer_size: int = 2_000_000
    batch_size: int = 1024
    updates_per_step: int = 2
    learning_starts: int = 20_000
    """Transitions collected with uniform random actions before learning starts."""
    gamma: float = 0.99
    tau: float = 5.0e-3
    actor_lr: float = 5.0e-4
    critic_lr: float = 5.0e-4
    ou_theta: float = 0.15
    ou_sigma: float = 0.10
    hidden1: int = 400
    hidden2: int = 300
    max_grad_norm: float = 1.0


class DDPGActor(nn.Module):
    def __init__(self, obs_dim, act_dim, h1, h2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, act_dim), nn.Tanh(),
        )
        nn.init.uniform_(self.net[4].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[4].bias, -3e-3, 3e-3)

    def forward(self, obs):
        return self.net(obs)


class DDPGCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, h1, h2):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, h1)
        self.l2 = nn.Linear(h1 + act_dim, h2)
        self.l3 = nn.Linear(h2, 1)
        nn.init.uniform_(self.l3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.l3.bias, -3e-3, 3e-3)

    def forward(self, obs, act):
        x = torch.relu(self.l1(obs))
        x = torch.relu(self.l2(torch.cat([x, act], dim=-1)))
        return self.l3(x).squeeze(-1)


class ReplayBuffer:
    """Ring buffer of raw (un-normalised) transitions, kept on the training device."""

    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity, self.device = int(capacity), device
        self.obs = torch.zeros(self.capacity, obs_dim, device=device)
        self.next_obs = torch.zeros(self.capacity, obs_dim, device=device)
        self.act = torch.zeros(self.capacity, act_dim, device=device)
        self.rew = torch.zeros(self.capacity, device=device)
        self.done = torch.zeros(self.capacity, device=device)
        self.ptr, self.size = 0, 0

    def add(self, obs, act, rew, next_obs, done):
        n = obs.shape[0]
        if n == 0:
            return
        idx = (self.ptr + torch.arange(n, device=self.device)) % self.capacity
        self.obs[idx], self.act[idx], self.rew[idx] = obs, act, rew
        self.next_obs[idx], self.done[idx] = next_obs, done
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch):
        idx = torch.randint(0, self.size, (batch,), device=self.device)
        return self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx]


class DDPG(Agent):
    name = "ddpg"
    Cfg = DDPGCfg

    def __init__(self, obs_dim: int, act_dim: int, device, cfg: DDPGCfg | None = None):
        super().__init__(obs_dim, act_dim, device, cfg)
        c = self.cfg
        self.actor = DDPGActor(obs_dim, act_dim, c.hidden1, c.hidden2).to(self.device)
        self.critic = DDPGCritic(obs_dim, act_dim, c.hidden1, c.hidden2).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=c.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=c.critic_lr)

    def modules(self):
        return {
            "actor": self.actor,
            "critic": self.critic,
            "actor_target": self.actor_target,
            "critic_target": self.critic_target,
        }

    def optimizers(self):
        return {"actor": self.actor_opt, "critic": self.critic_opt}

    @torch.no_grad()
    def act_inference(self, obs):
        return self.actor(self.obs_norm(obs))

    @torch.no_grad()
    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.lerp_(sp, self.cfg.tau)

    def _learn(self, buffer: ReplayBuffer):
        c = self.cfg
        obs, act, rew, next_obs, done = buffer.sample(c.batch_size)
        obs, next_obs = self.obs_norm(obs), self.obs_norm(next_obs)

        with torch.no_grad():
            q_next = self.critic_target(next_obs, self.actor_target(next_obs))
            y = rew + c.gamma * (1.0 - done) * q_next
        loss_q = (self.critic(obs, act) - y).pow(2).mean()
        self.critic_opt.zero_grad()
        loss_q.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)
        self.critic_opt.step()

        loss_pi = -self.critic(obs, self.actor(obs)).mean()
        self.actor_opt.zero_grad()
        loss_pi.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), c.max_grad_norm)
        self.actor_opt.step()

        self._soft_update(self.critic_target, self.critic)
        self._soft_update(self.actor_target, self.actor)
        return loss_q.item(), loss_pi.item()

    def train(self, env, num_iterations, logger, ckpt_dir=None, save_interval=100):
        c = self.cfg
        N = env.num_envs
        tracker = EpisodeTracker(N, self.device)
        buffer = ReplayBuffer(c.buffer_size, self.obs_dim, self.act_dim, self.device)
        noise = torch.zeros(N, self.act_dim, device=self.device)
        obs = env.reset()
        env_steps = 0

        for it in range(num_iterations):
            q_losses, pi_losses = [], []
            for _ in range(c.steps_per_iteration):
                with torch.no_grad():
                    self.obs_norm.update(obs)
                    if buffer.size < c.learning_starts:
                        a = torch.rand(N, self.act_dim, device=self.device) * 2.0 - 1.0
                    else:
                        noise += c.ou_theta * (0.0 - noise) + c.ou_sigma * torch.randn_like(noise)
                        a = (self.actor(self.obs_norm(obs)) + noise).clamp(-1.0, 1.0)
                    next_obs, rew, term, trunc, _ = env.step(a)
                    tracker.update(rew, term, trunc)
                    keep = ~(trunc & ~term)          # drop time-outs (next_obs is a reset state)
                    buffer.add(obs[keep], a[keep], rew[keep], next_obs[keep], term[keep].float())
                    noise[term | trunc] = 0.0        # fresh OU process for the new episode
                    obs = next_obs
                env_steps += N

                if buffer.size >= max(c.learning_starts, c.batch_size):
                    for _ in range(c.updates_per_step):
                        lq, lp = self._learn(buffer)
                        q_losses.append(lq)
                        pi_losses.append(lp)

            scalars = tracker.summary()
            scalars["Train/buffer_size"] = float(buffer.size)
            if q_losses:
                scalars["Loss/critic"] = sum(q_losses) / len(q_losses)
                scalars["Loss/actor"] = sum(pi_losses) / len(pi_losses)
            logger.log(it, num_iterations, env_steps, scalars)
            self._maybe_save(it, num_iterations, ckpt_dir, save_interval)
