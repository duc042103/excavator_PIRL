#!/usr/bin/env python3
"""Train the excavator with the vortexRL algorithms: REINFORCE, PPO, TRPO, DDPG.

    ~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo ppo       --headless
    ~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo trpo      --headless
    ~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo ddpg      --headless
    ~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo reinforce --headless

    # config overrides (field names in excavator_rl/algorithms/<algo>.py)
    ~/IsaacLab/isaaclab.sh -p scripts/train_algo.py --algo ppo --headless \\
        --cfg action_type=continuous policy_lr=1e-4

Runs are stored in ``logs/<algo>/excavator_digging/<date>_<time>/``
(checkpoints ``model_<iteration>.pt`` + TensorBoard).  ``--num_envs`` defaults
to the algorithm's own ``num_envs`` setting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from excavator_rl.algorithms import ALGORITHMS, cfg_from_overrides  # noqa: E402  (pure torch, safe before app launch)

parser = argparse.ArgumentParser(description="Train the excavator with a vortexRL algorithm.")
parser.add_argument("--algo", required=True, choices=sorted(ALGORITHMS))
parser.add_argument("--task", type=str, default="Excavator-Digging-v0")
parser.add_argument("--num_envs", type=int, default=None, help="default: the algorithm's num_envs")
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--cfg", nargs="*", default=[], metavar="KEY=VALUE", help="algorithm config overrides")
parser.add_argument("--usd", type=str, default=None, help="override the excavator USD path")
parser.add_argument("--reward_mode", type=str, default=None, choices=["dense", "vortex"])
parser.add_argument("--resume", type=str, default=None, help="checkpoint .pt to continue from")
parser.add_argument("--log_root", type=str, default="logs")
parser.add_argument("--run_name", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# validate the overrides before spending a minute starting Isaac Sim
_probe_cfg = cfg_from_overrides(ALGORITHMS[args_cli.algo].Cfg, args_cli.cfg)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from excavator_rl import resolve_entry_point  # noqa: E402  (registers the gym ids)
from excavator_rl.algorithms import Logger, load_agent, make_agent  # noqa: E402
from excavator_rl.algorithms.isaaclab_env import IsaacLabVecEnv  # noqa: E402


def main() -> None:
    torch.manual_seed(args_cli.seed)
    num_envs = args_cli.num_envs or _probe_cfg.num_envs

    env_cfg = resolve_entry_point(gym.spec(args_cli.task).kwargs["env_cfg_entry_point"])()
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.usd:
        env_cfg.usd_path = os.path.abspath(os.path.expanduser(args_cli.usd))
        env_cfg.robot.spawn.usd_path = env_cfg.usd_path
    if args_cli.reward_mode:
        env_cfg.reward_mode = args_cli.reward_mode
    if args_cli.headless:
        env_cfg.debug_vis = False
    if not os.path.isfile(env_cfg.usd_path):
        raise FileNotFoundError(
            f"excavator USD not found at '{env_cfg.usd_path}'. Run scripts/setup_assets.sh first, "
            "pass --usd, or set EXCAVATOR_USD."
        )

    env = IsaacLabVecEnv(gym.make(args_cli.task, cfg=env_cfg, render_mode=None))
    device = env.device

    if args_cli.resume:
        agent, ckpt = load_agent(os.path.abspath(os.path.expanduser(args_cli.resume)), device, load_optimizers=True)
        if agent.name != args_cli.algo:
            raise ValueError(f"checkpoint is '{agent.name}', not '{args_cli.algo}'")
        agent.iteration_offset = int(ckpt.get("iteration", 0))
        print(f"[train_algo] resumed from {args_cli.resume} (iteration {agent.iteration_offset})")
    else:
        agent = make_agent(args_cli.algo, env.obs_dim, env.act_dim, device, args_cli.cfg + [f"num_envs={num_envs}"])

    run = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + (f"_{args_cli.run_name}" if args_cli.run_name else "")
    log_dir = os.path.abspath(os.path.join(args_cli.log_root, args_cli.algo, "excavator_digging", run))
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump({"algo": args_cli.algo, "task": args_cli.task, "num_envs": num_envs,
                   "reward_mode": env_cfg.reward_mode, "seed": args_cli.seed,
                   "agent": asdict(agent.cfg)}, f, indent=2, default=str)

    print(f"[train_algo] {args_cli.algo}  envs={num_envs}  obs={env.obs_dim}  act={env.act_dim}  "
          f"reward={env_cfg.reward_mode}  device={device}\n[train_algo] logging to {log_dir}")
    env.randomize_episode_lengths()
    logger = Logger(log_dir)
    try:
        agent.train(env, args_cli.max_iterations, logger, ckpt_dir=log_dir, save_interval=args_cli.save_interval)
    finally:
        logger.close()
        agent.save(os.path.join(log_dir, "model_final.pt"),
                   extra={"iteration": agent.iteration_offset + args_cli.max_iterations})
        print(f"[train_algo] saved {os.path.join(log_dir, 'model_final.pt')}")
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
