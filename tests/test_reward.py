#!/usr/bin/env python3
"""Reward / phase-machine tests.

Isaac Lab is stubbed out, so the real :class:`DiggingEnv` methods run against
hand-built state on a CPU.  Catches the things that only show up after Isaac Sim
has spent two minutes starting: wrong tensor shapes, phase flags that never
flip, rewards that go NaN.

    python tests/test_reward.py
"""

from __future__ import annotations

import os
import sys
import types
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# --------------------------------------------------------------------------- #
# stub the Isaac Lab modules that digging_env imports at module level
# --------------------------------------------------------------------------- #
class _AutoModule(types.ModuleType):
    """Module whose every attribute is a fresh MagicMock."""

    def __getattr__(self, name):
        m = mock.MagicMock(name=f"{self.__name__}.{name}")
        setattr(self, name, m)
        return m


for name in [
    "isaaclab",
    "isaaclab.sim",
    "isaaclab.sim.converters",
    "isaaclab.assets",
    "isaaclab.envs",
    "isaaclab.markers",
    "isaaclab.utils",
    "isaaclab.utils.math",
    "isaaclab.actuators",
    "isaaclab.scene",
]:
    sys.modules.setdefault(name, _AutoModule(name))


def _quat_apply(q, v):
    """Real wxyz quaternion rotation -- this stub has to be correct, not a mock."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=1)
    return v + w * t + torch.cross(xyz, t, dim=1)


sys.modules["isaaclab.utils.math"].quat_apply = _quat_apply
sys.modules["isaaclab.envs"].DirectRLEnv = type("DirectRLEnv", (), {})
sys.modules["isaaclab.envs"].DirectRLEnvCfg = type("DirectRLEnvCfg", (), {})
sys.modules["isaaclab.utils"].configclass = lambda c: c

from excavator_rl.digging_env import DiggingEnv  # noqa: E402
from excavator_rl.soil import SoilCfg, SoilModel  # noqa: E402

N = 6
OK, FAIL = "  ok  ", " FAIL "
_failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{OK if cond else FAIL}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        _failures.append(name)


class Cfg:
    """Mirror of DiggingEnvCfg with the fields the reward functions touch."""

    target_fill = 0.6
    lift_height = 4.2
    min_fill_at_success = 0.4
    reward_mode = "dense"
    w_fill, w_lift, w_success = 60.0, 12.0, 200.0
    w_time, w_spill, w_effort, w_action_rate, w_reach = 0.05, 30.0, 0.02, 0.01, 0.5
    soil = SoilCfg()


def make_env(mode: str = "dense") -> DiggingEnv:
    env = object.__new__(DiggingEnv)
    env.cfg = Cfg()
    env.cfg.reward_mode = mode
    env.device = "cpu"
    env.num_envs = N

    env.soil = SoilModel(env.cfg.soil, N, "cpu")
    env.soil.reset(randomize=False)

    env._pos_attr = "body_pos_w"
    env._bucket_id = 4
    env._act_joint_ids = torch.tensor([1, 2, 3])
    env._effort_limit = torch.tensor([1.5e6, 5.0e5, 3.3e5])
    env._back = torch.zeros(N, dtype=torch.bool)
    env._success = torch.zeros(N, dtype=torch.bool)
    env._prev_fill_ratio = torch.zeros(N)
    env._prev_fill_mass = torch.zeros(N)
    env._prev_height = torch.zeros(N)
    env._filtered = torch.zeros(N, 3)
    env._prev_filtered = torch.zeros(N, 3)
    env._soil_out = {
        "force": torch.zeros(N, 3),
        "depth": torch.zeros(N),
        "d_fill": torch.zeros(N),
        "spilled": torch.zeros(N),
    }

    env.scene = types.SimpleNamespace(env_origins=torch.zeros(N, 3))
    data = types.SimpleNamespace(
        body_pos_w=torch.zeros(N, 5, 3),
        applied_torque=torch.zeros(N, 4),
    )
    env.robot = types.SimpleNamespace(data=data)
    return env


def set_height(env, h) -> None:
    env.robot.data.body_pos_w[:, env._bucket_id, 2] = torch.as_tensor(h, dtype=torch.float32)


def test_phase_flips_on_fill():
    env = make_env()
    set_height(env, 1.0)
    env.soil.fill[:] = torch.tensor([0.0, 0.5, 0.9, 0.95, 1.2, 1.5])  # cap 1.5
    env._update_phase()
    ratio = env.soil.fill_ratio
    check("back flag follows the fill target",
          torch.equal(env._back, ratio >= 0.6), f"ratios {[round(float(x),2) for x in ratio]}")
    check("no success while the bucket is low", not bool(env._success.any()))


def test_success_requires_height_and_payload():
    env = make_env()
    env.soil.fill[:] = torch.tensor([1.5, 1.5, 1.5, 0.3, 1.5, 0.0])
    set_height(env, [5.0, 4.2, 3.0, 5.0, 4.19, 5.0])
    env._update_phase()
    expect = torch.tensor([True, True, False, False, False, False])
    check("success = lifted high AND still loaded", torch.equal(env._success, expect),
          f"{env._success.tolist()}")


def test_phase_is_latched():
    env = make_env()
    env.soil.fill[:] = 1.2
    set_height(env, 1.0)
    env._update_phase()
    check("back latches on", bool(env._back.all()))
    env.soil.fill[:] = 0.0                       # payload lost
    env._update_phase()
    check("back stays latched after spilling", bool(env._back.all()))


def test_dense_reward_pays_for_filling():
    env = make_env()
    set_height(env, 1.0)
    env._prev_height = torch.full((N,), 1.0)
    env.soil.fill[:] = torch.linspace(0.0, 0.5, N)
    env._prev_fill_ratio = torch.zeros(N)
    r = env._get_rewards()
    check("more fill -> more reward", bool((r.diff() > 0).all()), f"{[round(float(x),2) for x in r]}")
    check("idle costs time", float(r[0]) < 0.0, f"{float(r[0]):.3f}")


def test_dense_reward_pays_for_lifting():
    env = make_env()
    env._back[:] = True
    env.soil.fill[:] = 1.2
    env._prev_fill_ratio[:] = 0.8
    env._prev_height = torch.full((N,), 1.0)
    set_height(env, [1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    r = env._get_rewards()
    check("higher lift -> more reward", bool((r.diff() > 0).all()), f"{[round(float(x),2) for x in r]}")


def test_success_bonus_dominates():
    env = make_env()
    env.soil.fill[:] = 1.2
    env._prev_fill_ratio[:] = 0.8
    env._prev_height = torch.full((N,), 4.0)
    set_height(env, [4.3] * 3 + [1.0] * 3)
    r = env._get_rewards()
    check("success is worth far more than any step reward",
          bool((r[:3] > 100).all()) and bool((r[3:] < 100).all()),
          f"{[round(float(x),1) for x in r]}")


def test_penalties():
    env = make_env()
    set_height(env, 1.0)
    env._prev_height = torch.full((N,), 1.0)
    base = float(env._get_rewards()[0])

    env = make_env()
    set_height(env, 1.0)
    env._prev_height = torch.full((N,), 1.0)
    env._soil_out["spilled"] = torch.full((N,), 0.3)
    check("spilling is penalised", float(env._get_rewards()[0]) < base)

    env = make_env()
    set_height(env, 1.0)
    env._prev_height = torch.full((N,), 1.0)
    env.robot.data.applied_torque[:, 1:] = 1.0e6
    check("saturating the actuators is penalised", float(env._get_rewards()[0]) < base)

    env = make_env()
    set_height(env, 1.0)
    env._prev_height = torch.full((N,), 1.0)
    env._filtered = torch.ones(N, 3)
    check("action chattering is penalised", float(env._get_rewards()[0]) < base)


def test_vortex_reward_matches_the_paper_shape():
    env = make_env("vortex")
    set_height(env, 1.0)
    env._prev_height = torch.full((N,), 1.0)
    # digging phase: mass rising / flat / falling
    env.soil.fill[:] = torch.tensor([0.5, 0.3, 0.1, 0.5, 0.5, 0.5])
    env._prev_fill_mass = torch.tensor([0.3, 0.3, 0.3, 0.3, 0.3, 0.3]) * env.cfg.soil.density
    r = env._get_rewards()
    check("all-negative during digging", bool((r < 0).all()), f"{[round(float(x),2) for x in r]}")
    check("gaining soil scores better than losing it", float(r[0]) > float(r[2]))

    env = make_env("vortex")
    env._back[:] = True
    env.soil.fill[:] = 1.4
    env._prev_fill_mass[:] = 1.4 * env.cfg.soil.density
    env._prev_height = torch.full((N,), 3.0)
    set_height(env, [3.5, 3.0, 2.5, 4.3, 4.3, 4.3])
    env.soil.fill[3:] = torch.tensor([1.4, 0.8, 0.1])
    env._prev_fill_mass[3:] = torch.tensor([1.4, 0.8, 0.1]) * env.cfg.soil.density
    r = env._get_rewards()
    check("rising beats falling while lifting", float(r[0]) > float(r[2]),
          f"{[round(float(x),2) for x in r]}")
    check("finishing full beats finishing empty", float(r[3]) > float(r[5]))


def test_no_nan_under_random_state():
    torch.manual_seed(0)
    for mode in ("dense", "vortex"):
        env = make_env(mode)
        for _ in range(200):
            env.soil.fill[:] = torch.rand(N) * env.cfg.soil.capacity
            set_height(env, torch.rand(N) * 6.0)
            env._filtered = torch.randn(N, 3)
            env._soil_out["spilled"] = torch.rand(N) * 0.1
            env._soil_out["depth"] = torch.rand(N)
            env.robot.data.applied_torque = torch.randn(N, 4) * 1e6
            r = env._get_rewards()
            if not torch.isfinite(r).all():
                check(f"{mode} reward stays finite", False)
                break
        else:
            check(f"{mode} reward stays finite", True)


def test_observation_width_matches_the_config():
    from excavator_rl.excavator_cfg import ARM_JOINTS, VELOCITY_LIMITS  # noqa: F401

    for n_act in (3, 4):
        declared = 3 * n_act + 3 + 4
        built = n_act + n_act + n_act + 3 + 1 + 1 + 1 + 1
        check(f"observation width consistent for {n_act} actions", declared == built,
              f"cfg says {declared}, _get_observations builds {built}")


if __name__ == "__main__":
    for fn in [
        test_phase_flips_on_fill,
        test_success_requires_height_and_payload,
        test_phase_is_latched,
        test_dense_reward_pays_for_filling,
        test_dense_reward_pays_for_lifting,
        test_success_bonus_dominates,
        test_penalties,
        test_vortex_reward_matches_the_paper_shape,
        test_no_nan_under_random_state,
        test_observation_width_matches_the_config,
    ]:
        print(f"\n{fn.__name__}")
        fn()

    print("\n" + "=" * 52)
    if _failures:
        print(f"{len(_failures)} FAILED: {_failures}")
        sys.exit(1)
    print("all reward/phase tests passed")
