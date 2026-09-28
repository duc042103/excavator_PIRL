"""PPO-Clip, port of vortexRL ``algorithms/PPO_agent.py`` (discrete) and
``algorithms/PPO_agentcontinuous.py`` (continuous).

Reference: Schulman et al. -- *Proximal Policy Optimization Algorithms* (2017).

Kept from vortexRL
    * separate actor and critic networks with separate Adam optimisers
      (policy lr 3e-4, value lr 1e-3), clip ratio 0.2, GAE lambda 0.97;
    * early stopping of the policy epochs once the approximate KL exceeds
      1.5 x ``target_kl`` (0.01);
    * both action spaces: ``action_type="discrete"`` is vortexRL's default PPO
      (27 actions, every {-1, 0, +1} combination), ``"continuous"`` its
      Gaussian variant.

Changed for thousands of parallel environments
    * rollouts of ``horizon`` steps x N envs, minibatched epochs instead of 80
      full-batch passes over a 50-step buffer;
    * time-outs are bootstrapped with the critic (``r += gamma * V(s)``) so a
      truncated episode is not mistaken for a terminal state.

For comparison, the tuned PPO of rsl_rl / skrl / rl_games is available
through ``scripts/train.py`` and ``scripts/run_rl.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .common import Agent, EpisodeTracker, ParamHolder, compute_gae, discrete_action_table, mlp


@dataclass
class PPOCfg:
    num_envs: int = 4096
    horizon: int = 32
    action_type: str = "discrete"
    """``discrete`` (vortexRL PPO_agent.py) or ``continuous`` (PPO_agentcontinuous.py)."""
    action_levels: tuple = (-1, 0, 1)
    gamma: float = 0.99
    lam: float = 0.97
    clip_ratio: float = 0.2
    policy_lr: float = 3.0e-4
    value_lr: float = 1.0e-3
    epochs: int = 5
    num_minibatches: int = 4
    target_kl: float = 0.01
    entropy_coef: float = 0.005
    max_grad_norm: float = 1.0
    hidden_dims: tuple = (256, 128, 64)
    activation: str = "elu"
    init_log_std: float = -0.22
    """Continuous only: initial log std of the Gaussian (std 0.8)."""


class PPO(Agent):
    name = "ppo"
    Cfg = PPOCfg

    def __init__(self, obs_dim: int, act_dim: int, device, cfg: PPOCfg | None = None):
        super().__init__(obs_dim, act_dim, device, cfg)
        c = self.cfg
        if c.action_type not in ("discrete", "continuous"):
            raise ValueError(f"action_type must be 'discrete' or 'continuous', got {c.action_type!r}")
        self.discrete = c.action_type == "discrete"
        if self.discrete:
            self.table = discrete_action_table(act_dim, tuple(float(v) for v in c.action_levels)).to(self.device)
            n_out = self.table.shape[0]
        else:
            n_out = act_dim
        self.actor = mlp(obs_dim, c.hidden_dims, n_out, c.activation).to(self.device)
        self.critic = mlp(obs_dim, c.hidden_dims, 1, c.activation).to(self.device)
        self.log_std = nn.Parameter(torch.full((act_dim,), c.init_log_std, device=self.device))
        actor_params = list(self.actor.parameters()) + ([] if self.discrete else [self.log_std])
        self.policy_opt = torch.optim.Adam(actor_params, lr=c.policy_lr)
        self.value_opt = torch.optim.Adam(self.critic.parameters(), lr=c.value_lr)

    def modules(self):
        return {"actor": self.actor, "critic": self.critic, "log_std": ParamHolder(self.log_std)}

    def optimizers(self):
        return {"policy": self.policy_opt, "value": self.value_opt}

    # ------------------------------------------------------------------ #
    def _dist(self, obs_n: torch.Tensor):
        out = self.actor(obs_n)
        if self.discrete:
            return torch.distributions.Categorical(logits=out)
        return torch.distributions.Normal(out, self.log_std.exp().expand_as(out))

    def _log_prob(self, dist, a):
        lp = dist.log_prob(a)
        return lp if self.discrete else lp.sum(-1)

    def _entropy(self, dist):
        e = dist.entropy()
        return e if self.discrete else e.sum(-1)

    def _to_env(self, a: torch.Tensor) -> torch.Tensor:
        return self.table[a] if self.discrete else a.clamp(-1.0, 1.0)

    @torch.no_grad()
    def act_inference(self, obs):
        out = self.actor(self.obs_norm(obs))
        return self.table[out.argmax(-1)] if self.discrete else out.clamp(-1.0, 1.0)

    # ------------------------------------------------------------------ #
    def train(self, env, num_iterations, logger, ckpt_dir=None, save_interval=100):
        c = self.cfg
        N, T = env.num_envs, c.horizon
        tracker = EpisodeTracker(N, self.device)
        obs = env.reset()
        env_steps = 0
        act_shape = () if self.discrete else (self.act_dim,)
        act_dtype = torch.long if self.discrete else torch.float32

        for it in range(num_iterations):
            obs_buf = torch.zeros(T, N, self.obs_dim, device=self.device)
            act_buf = torch.zeros(T, N, *act_shape, dtype=act_dtype, device=self.device)
            logp_buf = torch.zeros(T, N, device=self.device)
            val_buf = torch.zeros(T, N, device=self.device)
            rew_buf = torch.zeros(T, N, device=self.device)
            done_buf = torch.zeros(T, N, device=self.device)

            with torch.no_grad():
                for t in range(T):
                    self.obs_norm.update(obs)
                    obs_n = self.obs_norm(obs)
                    dist = self._dist(obs_n)
                    a = dist.sample()
                    v = self.critic(obs_n).squeeze(-1)
                    obs_buf[t], act_buf[t], logp_buf[t], val_buf[t] = obs, a, self._log_prob(dist, a), v
                    obs, rew, term, trunc, _ = env.step(self._to_env(a))
                    tracker.update(rew, term, trunc)
                    # bootstrap time-outs with the value of the state they left
                    rew_buf[t] = rew + c.gamma * v * (trunc & ~term).float()
                    done_buf[t] = (term | trunc).float()
                last_v = self.critic(self.obs_norm(obs)).squeeze(-1)
                adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_v, c.gamma, c.lam)
            env_steps += T * N

            b_obs = self.obs_norm(obs_buf.reshape(T * N, -1))
            b_act = act_buf.reshape(T * N, *act_shape)
            b_logp, b_ret = logp_buf.reshape(-1), ret.reshape(-1)
            b_adv = adv.reshape(-1)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            mb = max(1, (T * N) // c.num_minibatches)
            kl = torch.zeros((), device=self.device)
            stop_policy = False
            pol_losses, val_losses, epochs_done = [], [], 0
            for _ in range(c.epochs):
                perm = torch.randperm(T * N, device=self.device)
                for start in range(0, T * N, mb):
                    idx = perm[start:start + mb]
                    # -- policy (PPO-Clip), stopped early on a KL blow-up as in vortexRL
                    if not stop_policy:
                        dist = self._dist(b_obs[idx])
                        logp = self._log_prob(dist, b_act[idx])
                        ratio = torch.exp(logp - b_logp[idx])
                        a_ = b_adv[idx]
                        surr = torch.min(ratio * a_, ratio.clamp(1 - c.clip_ratio, 1 + c.clip_ratio) * a_)
                        loss_pi = -surr.mean() - c.entropy_coef * self._entropy(dist).mean()
                        self.policy_opt.zero_grad()
                        loss_pi.backward()
                        nn.utils.clip_grad_norm_(self.policy_opt.param_groups[0]["params"], c.max_grad_norm)
                        self.policy_opt.step()
                        pol_losses.append(loss_pi.item())
                    # -- value function
                    loss_v = (self.critic(b_obs[idx]).squeeze(-1) - b_ret[idx]).pow(2).mean()
                    self.value_opt.zero_grad()
                    loss_v.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)
                    self.value_opt.step()
                    val_losses.append(loss_v.item())
                epochs_done += 1
                if not stop_policy:
                    with torch.no_grad():
                        new_logp = self._log_prob(self._dist(b_obs), b_act)
                        kl = (b_logp - new_logp).mean()
                    stop_policy = bool(kl > 1.5 * c.target_kl)

            scalars = tracker.summary()
            scalars.update({
                "Loss/policy": sum(pol_losses) / max(len(pol_losses), 1),
                "Loss/value": sum(val_losses) / max(len(val_losses), 1),
                "Policy/approx_kl": float(kl),
                "Policy/early_stopped": float(stop_policy),
            })
            if not self.discrete:
                scalars["Policy/mean_std"] = float(self.log_std.detach().exp().mean())
            logger.log(it, num_iterations, env_steps, scalars)
            self._maybe_save(it, num_iterations, ckpt_dir, save_interval)
