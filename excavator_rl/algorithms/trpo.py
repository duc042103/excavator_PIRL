"""TRPO, port of vortexRL ``algorithms/TRPO.py``.

Reference: Schulman, Levine, Moritz, Jordan, Abbeel -- *Trust Region Policy
Optimization* (ICML 2015); GAE from Schulman et al. 2016.

vortexRL's file is Patrick Coady's TensorFlow "TRPO" -- strictly speaking a
KL-*penalty* method (Adam on surrogate + beta * KL + hinge, beta servoed to a
KL target).  This port implements the trust-region update of the paper
itself, which is what vortexRL's write-up describes and compares:

    maximise   L(theta) = E[ pi_theta(a|s) / pi_old(a|s) * A ]
    subject to E[ KL(pi_old || pi_theta) ] <= max_kl

solved with a natural-gradient step (conjugate gradient on Fisher-vector
products) followed by a backtracking line search that enforces the KL bound
and requires the surrogate to improve.

Kept from vortexRL: continuous Gaussian policy with a state-independent log
variance, separate value network fitted by regression on discounted returns,
GAE with lambda 0.98, running observation normalisation (its ``Scaler``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .common import Agent, EpisodeTracker, ParamHolder, compute_gae, mlp


@dataclass
class TRPOCfg:
    num_envs: int = 1024
    horizon: int = 64
    gamma: float = 0.99
    lam: float = 0.98
    max_kl: float = 0.01
    """Trust-region radius (mean KL per update).  vortexRL targeted 0.003."""
    cg_iters: int = 10
    cg_damping: float = 0.1
    line_search_steps: int = 10
    line_search_decay: float = 0.5
    fvp_subsample: float = 0.25
    """Fraction of the batch used for Fisher-vector products (speed)."""
    value_lr: float = 1.0e-3
    value_epochs: int = 5
    value_minibatches: int = 4
    max_grad_norm: float = 1.0
    hidden_dims: tuple = (256, 128, 64)
    activation: str = "tanh"
    init_log_std: float = -0.5
    """vortexRL: init_logvar = -1  ->  log std = -0.5."""


class TRPO(Agent):
    name = "trpo"
    Cfg = TRPOCfg

    def __init__(self, obs_dim: int, act_dim: int, device, cfg: TRPOCfg | None = None):
        super().__init__(obs_dim, act_dim, device, cfg)
        c = self.cfg
        self.actor = mlp(obs_dim, c.hidden_dims, act_dim, c.activation).to(self.device)
        self.log_std = nn.Parameter(torch.full((act_dim,), c.init_log_std, device=self.device))
        self.critic = mlp(obs_dim, c.hidden_dims, 1, c.activation).to(self.device)
        self.value_opt = torch.optim.Adam(self.critic.parameters(), lr=c.value_lr)
        self.policy_params = list(self.actor.parameters()) + [self.log_std]

    def modules(self):
        return {"actor": self.actor, "critic": self.critic, "log_std": ParamHolder(self.log_std)}

    def optimizers(self):
        return {"value": self.value_opt}

    def _dist(self, obs_n):
        mu = self.actor(obs_n)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    @torch.no_grad()
    def act_inference(self, obs):
        return self.actor(self.obs_norm(obs)).clamp(-1.0, 1.0)

    # ------------------------------------------------------------------ #
    # natural-gradient machinery
    # ------------------------------------------------------------------ #
    def _mean_kl(self, obs_n, mu_old, std_old):
        new = self._dist(obs_n)
        old = torch.distributions.Normal(mu_old, std_old)
        return torch.distributions.kl_divergence(old, new).sum(-1).mean()

    def _fvp(self, v, obs_n, mu_old, std_old):
        kl = self._mean_kl(obs_n, mu_old, std_old)
        grads = torch.autograd.grad(kl, self.policy_params, create_graph=True)
        flat = torch.cat([g.reshape(-1) for g in grads])
        hv = torch.autograd.grad((flat * v).sum(), self.policy_params)
        return torch.cat([g.reshape(-1) for g in hv]) + self.cfg.cg_damping * v

    def _conjugate_gradient(self, fvp, b):
        x = torch.zeros_like(b)
        r, p = b.clone(), b.clone()
        rr = r.dot(r)
        for _ in range(self.cfg.cg_iters):
            Ap = fvp(p)
            alpha = rr / (p.dot(Ap) + 1e-10)
            x += alpha * p
            r -= alpha * Ap
            rr_new = r.dot(r)
            if rr_new < 1e-10:
                break
            p = r + (rr_new / rr) * p
            rr = rr_new
        return x

    def _policy_update(self, obs_n, act, adv):
        c = self.cfg
        with torch.no_grad():
            d_old = self._dist(obs_n)
            mu_old, std_old = d_old.loc.clone(), d_old.scale.clone()
            logp_old = d_old.log_prob(act).sum(-1)

        def surrogate():
            logp = self._dist(obs_n).log_prob(act).sum(-1)
            return (torch.exp(logp - logp_old) * adv).mean()

        surr = surrogate()
        g = torch.cat([x.reshape(-1) for x in torch.autograd.grad(surr, self.policy_params)])

        n_fvp = max(1, int(obs_n.shape[0] * c.fvp_subsample))
        sub = torch.randperm(obs_n.shape[0], device=self.device)[:n_fvp]
        fvp = lambda v: self._fvp(v, obs_n[sub], mu_old[sub], std_old[sub])  # noqa: E731

        step_dir = self._conjugate_gradient(fvp, g)
        shs = 0.5 * step_dir.dot(fvp(step_dir))
        if not torch.isfinite(shs) or shs <= 0:
            return {"Policy/kl": 0.0, "Policy/surrogate_gain": 0.0, "Policy/line_search_ok": 0.0}
        full_step = step_dir * torch.sqrt(c.max_kl / shs)
        expected = g.dot(full_step)

        old_params = parameters_to_vector(self.policy_params).detach()
        surr_old = surr.item()
        frac, accepted, kl_v, gain = 1.0, False, 0.0, 0.0
        with torch.no_grad():
            for _ in range(c.line_search_steps):
                vector_to_parameters(old_params + frac * full_step, self.policy_params)
                gain = surrogate().item() - surr_old
                kl_v = self._mean_kl(obs_n, mu_old, std_old).item()
                if kl_v <= 1.5 * c.max_kl and gain > 0.0:
                    accepted = True
                    break
                frac *= c.line_search_decay
            if not accepted:
                vector_to_parameters(old_params, self.policy_params)
                kl_v, gain = 0.0, 0.0
        return {
            "Policy/kl": kl_v,
            "Policy/surrogate_gain": gain,
            "Policy/expected_gain": float(expected),
            "Policy/line_search_ok": float(accepted),
            "Policy/mean_std": float(self.log_std.detach().exp().mean()),
        }

    # ------------------------------------------------------------------ #
    def train(self, env, num_iterations, logger, ckpt_dir=None, save_interval=100):
        c = self.cfg
        N, T = env.num_envs, c.horizon
        tracker = EpisodeTracker(N, self.device)
        obs = env.reset()
        env_steps = 0

        for it in range(num_iterations):
            obs_buf = torch.zeros(T, N, self.obs_dim, device=self.device)
            act_buf = torch.zeros(T, N, self.act_dim, device=self.device)
            val_buf = torch.zeros(T, N, device=self.device)
            rew_buf = torch.zeros(T, N, device=self.device)
            done_buf = torch.zeros(T, N, device=self.device)

            with torch.no_grad():
                for t in range(T):
                    self.obs_norm.update(obs)
                    obs_n = self.obs_norm(obs)
                    a = self._dist(obs_n).sample()
                    v = self.critic(obs_n).squeeze(-1)
                    obs_buf[t], act_buf[t], val_buf[t] = obs, a, v
                    obs, rew, term, trunc, _ = env.step(a.clamp(-1.0, 1.0))
                    tracker.update(rew, term, trunc)
                    rew_buf[t] = rew + c.gamma * v * (trunc & ~term).float()
                    done_buf[t] = (term | trunc).float()
                last_v = self.critic(self.obs_norm(obs)).squeeze(-1)
                adv, ret = compute_gae(rew_buf, val_buf, done_buf, last_v, c.gamma, c.lam)
            env_steps += T * N

            b_obs = self.obs_norm(obs_buf.reshape(T * N, -1))
            b_act = act_buf.reshape(T * N, -1)
            b_ret = ret.reshape(-1)
            b_adv = adv.reshape(-1)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            stats = self._policy_update(b_obs, b_act, b_adv)

            mb = max(1, (T * N) // c.value_minibatches)
            v_losses = []
            for _ in range(c.value_epochs):
                perm = torch.randperm(T * N, device=self.device)
                for start in range(0, T * N, mb):
                    idx = perm[start:start + mb]
                    loss_v = (self.critic(b_obs[idx]).squeeze(-1) - b_ret[idx]).pow(2).mean()
                    self.value_opt.zero_grad()
                    loss_v.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), c.max_grad_norm)
                    self.value_opt.step()
                    v_losses.append(loss_v.item())

            scalars = tracker.summary()
            scalars.update(stats)
            scalars["Loss/value"] = sum(v_losses) / max(len(v_losses), 1)
            logger.log(it, num_iterations, env_steps, scalars)
            self._maybe_save(it, num_iterations, ckpt_dir, save_interval)
