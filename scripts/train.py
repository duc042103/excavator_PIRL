#!/usr/bin/env python3
"""Train the excavator digging policy with PPO (rsl_rl).

    ./isaaclab.sh -p scripts/train.py --num_envs 4096 --headless
    ./isaaclab.sh -p scripts/train.py --num_envs 32            # watch it
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the excavator digging policy.")
parser.add_argument("--task", type=str, default="Excavator-Digging-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--usd", type=str, default=None, help="override the excavator USD path")
parser.add_argument("--reward_mode", type=str, default=None, choices=["dense", "vortex"])
parser.add_argument("--control_swing", action="store_true", help="add the slew joint to the actions")
parser.add_argument("--resume", type=str, default=None, help="checkpoint .pt to resume from")
parser.add_argument("--log_dir", type=str, default="logs/excavator")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------- #
# everything below must be imported *after* the app is up
# --------------------------------------------------------------------------- #
import gymnasium as gym  # noqa: E402
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
    env_cfg.seed = args_cli.seed
    if args_cli.usd:
        env_cfg.usd_path = args_cli.usd
        env_cfg.robot.spawn.usd_path = args_cli.usd
    if args_cli.reward_mode:
        env_cfg.reward_mode = args_cli.reward_mode
    if args_cli.control_swing:
        env_cfg.control_swing = True
        env_cfg.__post_init__()
        env_cfg.scene.num_envs = args_cli.num_envs
    # markers are pointless (and slow) when nobody is watching
    if args_cli.headless:
        env_cfg.debug_vis = False

    if not os.path.isfile(env_cfg.usd_path):
        raise FileNotFoundError(
            f"excavator USD not found at '{env_cfg.usd_path}'.\n"
            "Run scripts/convert_urdf.py first, or pass --usd /path/to/excavator.usd, "
            "or set the EXCAVATOR_USD environment variable."
        )

    agent_cfg = ExcavatorPPORunnerCfg()
    agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations:
        agent_cfg.max_iterations = args_cli.max_iterations

    log_dir = os.path.abspath(args_cli.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args_cli.resume:
        runner.load(args_cli.resume)
        print(f"[train] resumed from {args_cli.resume}")

    print(
        f"[train] task={args_cli.task}  envs={env_cfg.scene.num_envs}  "
        f"obs={env_cfg.observation_space}  act={env_cfg.action_space}  "
        f"reward={env_cfg.reward_mode}  device={agent_cfg.device}"
    )
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    ckpt = os.path.join(log_dir, "model_final.pt")
    runner.save(ckpt)
    print(f"[train] saved {ckpt}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
