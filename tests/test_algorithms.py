#!/usr/bin/env python3
"""Learning tests for the vortexRL algorithm ports.  Plain torch, CPU, no Isaac Sim.

Every algorithm trains for a short budget on a toy vectorised task with the
same interface, action range and termination semantics as the excavator
environment, and must clearly beat a random policy.  This catches the bugs
that "runs without crashing" does not: wrong signs, broken returns/advantages,
targets bootstrapped across resets, checkpoints that do not restore.

    python tests/test_algorithms.py            # all algorithms (~1-3 min on CPU)
    python tests/test_algorithms.py ppo ddpg   # a subset
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from excavator_rl.algorithms import Logger, load_agent, make_agent  # noqa: E402

OK, FAIL = "  ok  ", " FAIL "
_failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{OK if cond else FAIL}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        _failures.append(name)


class PointReach:
    """Move a point to a random target with 3 velocity commands in [-1, 1].

    Terminates (success) inside a 0.15 radius, times out after 40 steps; auto
    resets like an Isaac Lab DirectRLEnv.
    """

    def __init__(self, num_envs=64, device="cpu", max_steps=40, seed=0):
        self.num_envs, self.device, self.max_steps = num_envs, torch.device(device), max_steps
        self.obs_dim, self.act_dim = 6, 3
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.pos = torch.zeros(num_envs, 3)
        self.goal = torch.zeros(num_envs, 3)
        self.t = torch.zeros(num_envs, dtype=torch.long)

    def _reset_idx(self, idx):
        n = idx.numel()
        self.pos[idx] = torch.rand(n, 3, generator=self.gen) * 2 - 1
        self.goal[idx] = torch.rand(n, 3, generator=self.gen) * 2 - 1
        self.t[idx] = 0

    def _obs(self):
        return torch.cat([self.pos, self.goal - self.pos], dim=1)

    def reset(self):
        self._reset_idx(torch.arange(self.num_envs))
        return self._obs()

    def step(self, a):
        a = a.clamp(-1, 1)
        self.pos = (self.pos + 0.1 * a).clamp(-1.5, 1.5)
        self.t += 1
        dist = (self.goal - self.pos).norm(dim=1)
        terminated = dist < 0.15
        truncated = (self.t >= self.max_steps) & ~terminated
        rew = -dist + 10.0 * terminated.float() - 0.01 * a.pow(2).sum(1)
        done = terminated | truncated
        if done.any():
            self._reset_idx(done.nonzero().squeeze(1))
        return self._obs(), rew, terminated, truncated, {}


def evaluate(policy, episodes_env=None, seed=123):
    """Mean undiscounted return and success rate over one episode per env."""
    env = episodes_env or PointReach(num_envs=256, seed=seed)
    obs = env.reset()
    ret = torch.zeros(env.num_envs)
    finished = torch.zeros(env.num_envs, dtype=torch.bool)
    success = torch.zeros(env.num_envs, dtype=torch.bool)
    for _ in range(env.max_steps):
        with torch.no_grad():
            a = policy(obs)
        obs, r, term, trunc, _ = env.step(a)
        ret += r * (~finished).float()
        success |= term & ~finished
        finished |= term | trunc
    return float(ret.mean()), float(success.float().mean())


# algorithm -> (config overrides, training iterations)
BUDGETS = {
    "ppo": (["num_envs=64", "horizon=32", "hidden_dims=64,64", "policy_lr=1e-3"], 60),
    "ppo_continuous": (["num_envs=64", "horizon=32", "hidden_dims=64,64", "action_type=continuous",
                        "policy_lr=1e-3", "init_log_std=-0.5"], 60),
    "trpo": (["num_envs=64", "horizon=64", "hidden_dims=64,64", "max_kl=0.02"], 60),
    "reinforce": (["num_envs=128", "horizon=160", "hidden_dims=64,64", "learning_rate=3e-3", "gamma=0.95"], 60),
    "ddpg": (["num_envs=64", "steps_per_iteration=32", "hidden1=128", "hidden2=128", "batch_size=256",
              "learning_starts=2000", "buffer_size=200000", "gamma=0.95", "ou_sigma=0.2"], 40),
}


def run(name: str) -> None:
    algo = name.split("_")[0]
    overrides, iters = BUDGETS[name]
    torch.manual_seed(0)
    n_envs = int(next(o for o in overrides if o.startswith("num_envs=")).split("=")[1])
    env = PointReach(num_envs=n_envs, seed=1)
    agent = make_agent(algo, env.obs_dim, env.act_dim, "cpu", overrides)

    rand_ret, rand_succ = evaluate(lambda o: torch.rand(o.shape[0], 3) * 2 - 1)
    before, _ = evaluate(agent.act_inference)

    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        agent.train(env, iters, Logger(None, print_every=10**9), ckpt_dir=d, save_interval=iters)
        after, succ = evaluate(agent.act_inference)
        dt = time.time() - t0

        check(f"{name}: learns (random {rand_ret:.1f} / untrained {before:.1f} -> trained {after:.1f})",
              after > rand_ret + 8.0 and after > before + 5.0, f"{dt:.0f} s")
        check(f"{name}: reaches the goal", succ > max(0.5, rand_succ + 0.3),
              f"success {succ:.0%} (random {rand_succ:.0%})")

        ckpt = os.path.join(d, f"model_{iters}.pt")
        check(f"{name}: checkpoint written", os.path.isfile(ckpt))
        loaded, meta = load_agent(ckpt, "cpu")
        obs = PointReach(num_envs=32, seed=7).reset()
        same = torch.allclose(loaded.act_inference(obs), agent.act_inference(obs), atol=1e-6)
        check(f"{name}: checkpoint restores the same policy", same and meta["algo"] == algo)


if __name__ == "__main__":
    names = sys.argv[1:] or list(BUDGETS)
    names = [n for n in BUDGETS if n in names or n.split("_")[0] in names]
    for n in names:
        print(f"\n{n}")
        run(n)
    print("\n" + "=" * 52)
    if _failures:
        print(f"{len(_failures)} FAILED: {_failures}")
        sys.exit(1)
    print("all algorithm tests passed")
