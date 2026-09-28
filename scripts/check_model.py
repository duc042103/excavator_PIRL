#!/usr/bin/env python3
"""Sanity-check the excavator before spending GPU hours on RL.

Prints what PhysX *actually* loaded (joint order, limits, drive gains, link
masses), verifies the bucket-tip offset, then runs a scripted IK dig cycle
(excavator_rl/scripted.py) that must fill the bucket and complete the task.

    ~/IsaacLab/isaaclab.sh -p scripts/check_model.py                  # with a viewport
    ~/IsaacLab/isaaclab.sh -p scripts/check_model.py --headless       # numbers only
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect the excavator articulation and soil model.")
parser.add_argument("--usd", type=str, default=None)
parser.add_argument("--cycle_s", type=float, default=14.0, help="length of the scripted dig cycle [s]")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import excavator_rl  # noqa: F401,E402
from excavator_rl.digging_env_cfg import DiggingEnvCfg  # noqa: E402
from excavator_rl.excavator_params import ARM_JOINTS, BUCKET_TIP_OFFSET, VELOCITY_LIMITS  # noqa: E402
from excavator_rl.scripted import ArmKinematics, ScriptedDigger  # noqa: E402


def banner(txt: str) -> None:
    print("\n" + txt + "\n" + "-" * len(txt))


def main() -> None:
    cfg = DiggingEnvCfg()
    cfg.scene.num_envs = 1
    cfg.debug_vis = not args_cli.headless
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device
    if args_cli.usd:
        cfg.usd_path = os.path.abspath(os.path.expanduser(args_cli.usd))
        cfg.robot.spawn.usd_path = cfg.usd_path
    if not os.path.isfile(cfg.usd_path):
        raise FileNotFoundError(
            f"excavator USD not found at '{cfg.usd_path}' -- run scripts/setup_assets.sh "
            "or pass --usd"
        )

    env = gym.make("Excavator-Digging-v0", cfg=cfg).unwrapped
    env.reset()
    robot = env.robot
    data = robot.data

    def column(*names):
        """First of `names` that this Isaac Lab build actually exposes."""
        for n in names:
            v = getattr(data, n, None)
            if v is not None:
                return v
        return None

    banner("JOINTS (index order is what the action vector maps to)")
    print(f"{'idx':>3} {'name':<22} {'pos limits [rad]':>22} {'effort [N.m]':>14} {'vel [rad/s]':>12}")
    limits = data.soft_joint_pos_limits[0]
    eff_col = column("joint_effort_limits_sim", "joint_effort_limits")
    vel_col = column("joint_velocity_limits_sim", "joint_velocity_limits", "joint_vel_limits")
    for i, name in enumerate(robot.joint_names):
        eff = float(eff_col[0, i]) if eff_col is not None else float("nan")
        vel = float(vel_col[0, i]) if vel_col is not None else float("nan")
        print(f"{i:>3} {name:<22} {limits[i,0]:>10.3f} .. {limits[i,1]:<8.3f} {eff:>14.4g} {vel:>12.3g}")

    print("\nactuated by the policy:", [robot.joint_names[i] for i in env._act_joint_ids.tolist()])

    banner("DRIVE GAINS as loaded (stiffness must be 0 for velocity control)")
    for i, name in enumerate(robot.joint_names):
        print(f"  {name:<22} stiffness={data.joint_stiffness[0,i]:>12.4g}  damping={data.joint_damping[0,i]:>12.4g}")

    banner("LINKS")
    masses = robot.root_physx_view.get_masses()[0]
    for i, name in enumerate(robot.body_names):
        print(f"  {i} {name:<22} mass = {masses[i]:>12.1f} kg")
    print(f"  total (incl. the root-fixed undercarriage): {masses.sum():.0f} kg")

    banner("BUCKET TIP CALIBRATION")
    pos, quat, tip_w, tip_vel, open_dir, _, _ = env._bucket_state()
    local = (tip_w - env.scene.env_origins)[0]
    pivot = (pos - env.scene.env_origins)[0]
    print(f"  tip offset in bucket frame : {BUCKET_TIP_OFFSET}")
    print(f"  bucket pivot (env frame)   : {pivot.tolist()}")
    print(f"  cutting edge (env frame)   : {local.tolist()}")
    print(f"  reach r = {torch.linalg.norm(local[:2]):.2f} m,  z = {local[2]:+.2f} m")
    print(f"  bucket opening points      : {open_dir[0].tolist()}  (z ~ +1 = curled/holding)")
    print(f"  soil surface under the tip : {env.soil.surface_height(torch.linalg.norm(local[:2]).unsqueeze(0))[0]:+.2f} m")
    print("\n  If the edge is not where you expect, adjust BUCKET_TIP_OFFSET in excavator_params.py.")

    banner("SCRIPTED DIG CYCLE (IK expert: bite, drag, curl, lift)")
    env.reset()
    expert = ScriptedDigger(ArmKinematics(), torch.tensor([VELOCITY_LIMITS[jn] for jn in ARM_JOINTS]))
    arm_ids = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
    dt = env.step_dt
    print(f"{'t':>6} {'phase':<10} {'depth':>7} {'F_soil':>10} {'fill':>7} {'fill_kg':>9} {'tooth_z':>8} {'pivot_z':>8}")
    peak_fill, moved, success = 0.0, 0.0, False
    for k in range(int(args_cli.cycle_s / dt)):
        t = k * dt
        moved = float(env.soil.moved_volume()[0])  # read before a success resets the soil
        a = expert.action(t, data.joint_pos[0, arm_ids].cpu())
        _, _, term, _, _ = env.step(a.unsqueeze(0).float().to(env.device))
        if bool(term[0]):
            success = True
            print(f"{t:>6.1f} SUCCESS -- bucket lifted to {cfg.lift_height} m with soil")
            break
        peak_fill = max(peak_fill, float(env.soil.fill_ratio[0]))
        if k % int(0.5 / dt) == 0:
            p, _, tw, _, _, _, _ = env._bucket_state()
            tl = (tw - env.scene.env_origins)[0]
            pv = (p - env.scene.env_origins)[0]
            f = torch.linalg.norm(env._soil_out["force"][0])
            print(
                f"{t:>6.1f} {expert.phase(t):<10} {env._soil_out['depth'][0]:>7.3f} "
                f"{f:>10.0f} {env.soil.fill_ratio[0]:>7.2f} {env.soil.fill_mass[0]:>9.0f} "
                f"{tl[2]:>8.2f} {pv[2]:>8.2f}"
            )

    banner("RESULT")
    print(f"  peak bucket fill : {peak_fill:.2f} of capacity (target {cfg.target_fill})")
    print(f"  soil displaced   : {moved:.2f} m^3")
    print(f"  task completed   : {success}")
    print(
        "\n  The IK expert completes the task in tools/pybullet_twin.py.  If it fails here, the Isaac Sim\n"
        "  articulation behaves differently (drive gains, limits, joint signs): compare the tables above\n"
        "  with `python tools/pybullet_twin.py check` before training."
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
