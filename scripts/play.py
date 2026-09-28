#!/usr/bin/env python3
"""Run a trained excavator policy and print per-episode dig statistics.

    ./isaaclab.sh -p scripts/play.py --checkpoint logs/excavator/model_final.pt
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained excavator policy.")
parser.add_argument("--task", type=str, default="Excavator-Digging-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--usd", type=str, default=None)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--export", action="store_true", help="also export the policy to TorchScript")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
except ImportError:  # pragma: no cover
    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper  # type: ignore

from excavator_rl import resolve_entry_point  # noqa: E402  (also registers the gym ids)
from excavator_rl.agents.rsl_rl_ppo_cfg import ExcavatorPPORunnerCfg  # noqa: E402


def main() -> None:
    env_cfg = resolve_entry_point(gym.spec(args_cli.task).kwargs["env_cfg_entry_point"])()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.usd:
        env_cfg.usd_path = args_cli.usd
        env_cfg.robot.spawn.usd_path = args_cli.usd

    agent_cfg = ExcavatorPPORunnerCfg()

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[play] loaded {args_cli.checkpoint}")

    if args_cli.export:
        try:
            from isaaclab_rl.rsl_rl import export_policy_as_jit

            out = os.path.dirname(os.path.abspath(args_cli.checkpoint))
            export_policy_as_jit(runner.alg.actor_critic, runner.obs_normalizer, out, "policy.pt")
            print(f"[play] exported TorchScript policy to {out}/policy.pt")
        except Exception as exc:
            print(f"[play] export skipped: {exc}")

    core = env.unwrapped
    obs, _ = env.reset()
    best_fill = torch.zeros(core.num_envs, device=core.device)

    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)[:4]

        best_fill = torch.maximum(best_fill, core.soil.fill_ratio)
        if dones.any():
            idx = dones.nonzero(as_tuple=False).squeeze(1)
            for i in idx.tolist():
                print(
                    f"  step {step:5d} env {i}: peak fill {best_fill[i]:.2f} "
                    f"({best_fill[i] * core.cfg.soil.capacity * core.cfg.soil.density:.0f} kg)  "
                    f"soil moved {core.soil.moved_volume()[i]:.2f} m^3  "
                    f"success={bool(core._success[i])}"
                )
                best_fill[i] = 0.0

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
