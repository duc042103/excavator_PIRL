"""Adapter from an Isaac Lab ``DirectRLEnv`` to the plain vectorised-env
contract used by :mod:`excavator_rl.algorithms`.

Duck-typed on purpose (no Isaac Lab import), so it is importable anywhere.
"""

from __future__ import annotations

import torch


class IsaacLabVecEnv:
    def __init__(self, env, obs_group: str = "policy"):
        self.env = env                      # the gym.make(...) env (wrappers included)
        self.unwrapped = env.unwrapped
        self.obs_group = obs_group
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.obs_dim = int(self.unwrapped.cfg.observation_space)
        self.act_dim = int(self.unwrapped.cfg.action_space)

    def reset(self) -> torch.Tensor:
        obs, _ = self.env.reset()
        return obs[self.obs_group]

    def randomize_episode_lengths(self) -> None:
        """Spread episode starts so that resets do not all happen together."""
        u = self.unwrapped
        u.episode_length_buf = torch.randint_like(u.episode_length_buf, high=int(u.max_episode_length))

    def step(self, actions: torch.Tensor):
        obs, rew, terminated, truncated, extras = self.env.step(actions)
        return obs[self.obs_group], rew, terminated.bool(), truncated.bool(), extras

    def close(self) -> None:
        self.env.close()
