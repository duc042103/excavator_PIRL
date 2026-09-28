#!/usr/bin/env python3
"""Run a policy trained with scripts/train_algo.py and print per-episode statistics.

    # newest checkpoint of the newest run of that algorithm
    ~/IsaacLab/isaaclab.sh -p scripts/play_algo.py --algo ddpg

    # a specific checkpoint (the algorithm is read from the file)
    ~/IsaacLab/isaaclab.sh -p scripts/play_algo.py --checkpoint logs/trpo/excavator_digging/<run>/model_1500.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from excavator_rl.algorithms import ALGORITHMS  # noqa: E402

parser = argparse.ArgumentParser(description="Play a vortexRL-algorithm policy.")
parser.add_argument("--algo", choices=sorted(ALGORITHMS), default=None,
                    help="pick the newest checkpoint of this algorithm (if --checkpoint is not given)")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--task", type=str, default="Excavator-Digging-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--usd", type=str, default=None)
parser.add_argument("--log_root", type=str, default="logs")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from excavator_rl import resolve_entry_point  # noqa: E402
from excavator_rl.algorithms import load_agent  # noqa: E402
from excavator_rl.algorithms.isaaclab_env import IsaacLabVecEnv  # noqa: E402


def latest_checkpoint(algo: str) -> str:
    root = os.path.join(args_cli.log_root, algo, "excavator_digging")
    for run in sorted(glob.glob(os.path.join(root, "*")), reverse=True):
        ckpts = glob.glob(os.path.join(run, "model_*.pt"))
        if ckpts:
            def key(p):
                m = re.search(r"model_(\d+)\.pt$", p)
                return (1, 0) if m is None else (0, int(m.group(1)))  # model_final.pt last
            return max(ckpts, key=key)
    raise FileNotFoundError(f"no checkpoint under '{root}' -- train first or pass --checkpoint")


def main() -> None:
    if args_cli.checkpoint is None and args_cli.algo is None:
        raise SystemExit("pass --algo <name> or --checkpoint <file.pt>")
    ckpt_path = os.path.abspath(os.path.expanduser(args_cli.checkpoint or latest_checkpoint(args_cli.algo)))

    env_cfg = resolve_entry_point(gym.spec(args_cli.task).kwargs["env_cfg_entry_point"])()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.usd:
        env_cfg.usd_path = os.path.abspath(os.path.expanduser(args_cli.usd))
        env_cfg.robot.spawn.usd_path = env_cfg.usd_path

    env = IsaacLabVecEnv(gym.make(args_cli.task, cfg=env_cfg, render_mode=None))
    agent, _ = load_agent(ckpt_path, env.device)
    print(f"[play_algo] {agent.name} policy from {ckpt_path}")

    core = env.unwrapped
    with torch.inference_mode():
        obs = env.reset()
        ep_ret = torch.zeros(env.num_envs, device=env.device)
        best_fill = torch.zeros(env.num_envs, device=env.device)
        for step in range(args_cli.steps):
            if not simulation_app.is_running():
                break
            obs, rew, term, trunc, _ = env.step(agent.act_inference(obs))
            ep_ret += rew
            best_fill = torch.maximum(best_fill, core.soil.fill_ratio)
            done = term | trunc
            for i in done.nonzero(as_tuple=False).squeeze(1).tolist():
                print(f"  step {step:5d} env {i}: return {ep_ret[i]:8.1f}  peak fill {best_fill[i]:.2f}  "
                      f"success={bool(term[i])}")
                ep_ret[i], best_fill[i] = 0.0, 0.0
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
