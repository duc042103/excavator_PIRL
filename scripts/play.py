#!/usr/bin/env python3
"""Run a trained excavator policy (rsl_rl) and print per-episode dig statistics.

    # newest checkpoint of the newest run in logs/rsl_rl/excavator_digging/
    ~/IsaacLab/isaaclab.sh -p scripts/play.py

    # a specific checkpoint
    ~/IsaacLab/isaaclab.sh -p scripts/play.py --checkpoint logs/rsl_rl/excavator_digging/<run>/model_2999.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained excavator policy.")
parser.add_argument("--task", type=str, default="Excavator-Digging-Play-v0")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="policy .pt; default: newest model_*.pt under --log_root")
parser.add_argument("--log_root", type=str, default="logs/rsl_rl/excavator_digging")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--usd", type=str, default=None)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--export", action="store_true", help="also export the policy to TorchScript + ONNX")
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


def latest_checkpoint(log_root: str) -> str:
    """Highest-iteration model_*.pt of the most recent run under ``log_root``."""
    runs = sorted(d for d in glob.glob(os.path.join(log_root, "*")) if os.path.isdir(d))
    for run in reversed(runs):
        ckpts = glob.glob(os.path.join(run, "model_*.pt"))
        if ckpts:
            def key(p):
                m = re.search(r"model_(\d+)\.pt$", p)
                return (1, 0) if m is None else (0, int(m.group(1)))  # model_final.pt last
            return max(ckpts, key=key)
    raise FileNotFoundError(f"no model_*.pt found under '{log_root}' -- train first or pass --checkpoint")


def main() -> None:
    spec_kwargs = gym.spec(args_cli.task).kwargs
    env_cfg = resolve_entry_point(spec_kwargs["env_cfg_entry_point"])()
    agent_cfg = resolve_entry_point(spec_kwargs["rsl_rl_cfg_entry_point"])()

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    if args_cli.usd:
        env_cfg.usd_path = os.path.abspath(os.path.expanduser(args_cli.usd))
        env_cfg.robot.spawn.usd_path = env_cfg.usd_path

    checkpoint = args_cli.checkpoint or latest_checkpoint(args_cli.log_root)
    checkpoint = os.path.abspath(os.path.expanduser(checkpoint))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[play] loaded {checkpoint}")

    if args_cli.export:
        export_policy(runner, os.path.join(os.path.dirname(checkpoint), "exported"))

    core = env.unwrapped
    obs, _ = env.reset()
    best_fill = torch.zeros(core.num_envs, device=core.device)

    for step in range(args_cli.steps):
        if not simulation_app.is_running():
            break
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


def export_policy(runner, out_dir: str) -> None:
    """Export the actor (with its observation normaliser) to TorchScript and ONNX."""
    try:
        from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
    except ImportError as exc:  # pragma: no cover
        print(f"[play] export skipped: {exc}")
        return

    # rsl-rl-lib >= 2.3 renamed alg.actor_critic -> alg.policy, and 3.x moved
    # the normaliser from the runner into the policy
    policy_nn = getattr(runner.alg, "policy", None)
    if policy_nn is None:
        policy_nn = runner.alg.actor_critic
    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    if normalizer is None:
        normalizer = getattr(runner, "obs_normalizer", None)

    os.makedirs(out_dir, exist_ok=True)
    try:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=out_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=out_dir, filename="policy.onnx")
        print(f"[play] exported policy.pt / policy.onnx to {out_dir}")
    except Exception as exc:
        print(f"[play] export failed: {exc}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
