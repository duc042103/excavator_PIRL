#!/usr/bin/env python3
"""Run the excavator task on PyBullet (CPU) -- a smoke test without Isaac Sim / GPU.

The *real* task code runs here: ``excavator_rl.digging_env.DiggingEnv`` (its
observations, soil model, rewards, phases and resets) and the real
``DiggingEnvCfg``.  Only the Isaac Lab layer underneath is swapped for a small
PyBullet adapter that loads the same URDF with the same actuator limits.

What it is good for: catching geometry / kinematics / reward / soil mistakes
and trying the RL algorithms end-to-end on a machine without an RTX GPU.
What it is not: a replacement for Isaac Sim.  PyBullet's velocity motors are
stiff constraints capped at the effort limit (Isaac Lab uses a damped drive),
contact and solver details differ, and it is ~1000x slower than 4096 GPU envs.

    pip install pybullet torch gymnasium
    python tools/pybullet_twin.py check                  # model, pose, drives, scripted dig
    python tools/pybullet_twin.py train --algo ppo --num_envs 16 --iterations 50 --log_dir logs/twin_ppo
    python tools/pybullet_twin.py play logs/twin_ppo/model_50.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
import types
from unittest import mock

import gymnasium as gym
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
DEFAULT_URDF = os.path.join(REPO, "assets", "MathScavator9000_flat", "urdf", "MathScavator9000_flat.SLDASM.urdf")


# --------------------------------------------------------------------------- #
# 1. stand-ins for the Isaac Lab modules digging_env imports
# --------------------------------------------------------------------------- #
class _AutoModule(types.ModuleType):
    def __getattr__(self, name):
        m = mock.MagicMock(name=f"{self.__name__}.{name}")
        setattr(self, name, m)
        return m


def _quat_apply(q, v):
    """wxyz quaternion rotation (same convention as isaaclab.utils.math)."""
    w, xyz = q[:, :1], q[:, 1:]
    t = 2.0 * torch.cross(xyz, v, dim=1)
    return v + w * t + torch.cross(xyz, t, dim=1)


class _DirectRLEnvCfg:
    pass


class _DirectRLEnv(gym.Env):
    """The part of isaaclab.envs.DirectRLEnv that DiggingEnv relies on."""

    physics_dt = 1.0 / 120.0

    def __init__(self, cfg, render_mode=None, **kwargs):
        self.cfg = cfg
        self.num_envs = cfg.scene.num_envs
        self.device = "cpu"
        spacing = float(cfg.scene.env_spacing)
        self.scene = types.SimpleNamespace(
            env_origins=torch.tensor([[i * spacing, 0.0, 0.0] for i in range(self.num_envs)]),
            articulations={},
            clone_environments=lambda **kw: None,
            filter_collisions=lambda **kw: None,
        )
        self.step_dt = self.physics_dt * cfg.decimation
        self.max_episode_length = math.ceil(cfg.episode_length_s / self.step_dt)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self._setup_scene()

    def _reset_idx(self, env_ids):
        self.robot.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

    def reset(self):
        self._reset_idx(self.robot._ALL_INDICES)
        self.robot.update()
        return self._get_observations(), {}

    def step(self, action):
        self._pre_physics_step(action)
        for _ in range(self.cfg.decimation):
            self._apply_action()
            self.robot.write_data_to_sim()
            self.robot.sim_step()
            self.robot.update()
        self.episode_length_buf += 1
        term, trunc = self._get_dones()
        rew = self._get_rewards()
        ids = (term | trunc).nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            self._reset_idx(ids)
        return self._get_observations(), rew, term, trunc, {}

    def close(self):
        self.robot.disconnect()


def install_isaaclab_stubs() -> None:
    for name in ["isaaclab", "isaaclab.sim", "isaaclab.assets", "isaaclab.envs", "isaaclab.markers",
                 "isaaclab.utils", "isaaclab.utils.math", "isaaclab.actuators", "isaaclab.scene"]:
        sys.modules[name] = _AutoModule(name)
    sys.modules["isaaclab.utils.math"].quat_apply = _quat_apply
    sys.modules["isaaclab.utils"].configclass = lambda c: c
    sys.modules["isaaclab.envs"].DirectRLEnv = _DirectRLEnv
    sys.modules["isaaclab.envs"].DirectRLEnvCfg = _DirectRLEnvCfg
    sys.modules["isaaclab.assets"].Articulation = PyBulletArticulation


# --------------------------------------------------------------------------- #
# 2. PyBullet articulation with the Isaac Lab Articulation interface
# --------------------------------------------------------------------------- #
class _Composer:
    def __init__(self, n):
        self.force_b = torch.zeros(n, 3)
        self.torque_b = torch.zeros(n, 3)

    def set_forces_and_torques(self, forces, torques, body_ids=None, env_ids=None, is_global=False):
        assert not is_global
        ids = slice(None) if env_ids is None else env_ids
        self.force_b[ids] = forces[:, 0]
        self.torque_b[ids] = torques[:, 0]


class PyBulletArticulation:
    urdf_path = DEFAULT_URDF
    origins = None

    def __init__(self, cfg=None):
        import pybullet as p

        from excavator_rl.excavator_params import (
            ALL_JOINTS, BASE_HEIGHT, BUCKET_LINK, DEFAULT_JOINT_POS, EFFORT_LIMITS, JOINT_LIMITS,
            VELOCITY_DRIVE_DAMPING, VELOCITY_LIMITS,
        )

        self.p = p
        self.cid = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)
        p.setTimeStep(_DirectRLEnv.physics_dt, physicsClientId=self.cid)
        urdf = _fixed_urdf(self.urdf_path)
        origins = PyBulletArticulation.origins if PyBulletArticulation.origins is not None else torch.zeros(1, 3)
        self.n = origins.shape[0]
        self.bodies = []
        for i in range(self.n):
            b = p.loadURDF(urdf, basePosition=[float(origins[i, 0]), float(origins[i, 1]), BASE_HEIGHT],
                           useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=self.cid)
            for link in range(-1, p.getNumJoints(b, physicsClientId=self.cid)):
                p.setCollisionFilterGroupMask(b, link, 0, 0, physicsClientId=self.cid)  # soil is analytic
            self.bodies.append(b)
        b0 = self.bodies[0]
        self.joint_names = [p.getJointInfo(b0, j, physicsClientId=self.cid)[1].decode() for j in range(p.getNumJoints(b0, physicsClientId=self.cid))]
        assert self.joint_names == list(ALL_JOINTS), self.joint_names
        self.body_names = ["base_link"] + [p.getJointInfo(b0, j, physicsClientId=self.cid)[12].decode() for j in range(len(self.joint_names))]
        self.masses = [p.getDynamicsInfo(b0, j, physicsClientId=self.cid)[0] for j in range(-1, len(self.joint_names))]
        nj, nb = len(self.joint_names), len(self.body_names)
        self.effort = torch.tensor([EFFORT_LIMITS[j] for j in self.joint_names])
        self.vmax = torch.tensor([VELOCITY_LIMITS[j] for j in self.joint_names])
        self._ALL_INDICES = torch.arange(self.n)
        self.bucket_body = self.body_names.index(BUCKET_LINK)
        self.vel_target = torch.zeros(self.n, nj)
        self.permanent_wrench_composer = _Composer(self.n)
        lim = torch.tensor([JOINT_LIMITS[j] for j in self.joint_names])
        root = torch.zeros(self.n, 13)
        root[:, 2], root[:, 3] = BASE_HEIGHT, 1.0
        self.data = types.SimpleNamespace(
            joint_pos=torch.zeros(self.n, nj), joint_vel=torch.zeros(self.n, nj),
            soft_joint_pos_limits=lim.unsqueeze(0).repeat(self.n, 1, 1),
            default_joint_pos=torch.tensor([DEFAULT_JOINT_POS[j] for j in self.joint_names]).repeat(self.n, 1),
            default_root_state=root,
            applied_torque=torch.zeros(self.n, nj),
            body_link_pos_w=torch.zeros(self.n, nb, 3), body_link_quat_w=torch.zeros(self.n, nb, 4),
            body_link_lin_vel_w=torch.zeros(self.n, nb, 3), body_ang_vel_w=torch.zeros(self.n, nb, 3),
            body_com_pos_w=torch.zeros(self.n, nb, 3),
            joint_stiffness=torch.zeros(self.n, nj),
            joint_damping=torch.tensor([VELOCITY_DRIVE_DAMPING[j] for j in self.joint_names]).repeat(self.n, 1),
            joint_effort_limits=self.effort.repeat(self.n, 1),
            joint_vel_limits=self.vmax.repeat(self.n, 1),
        )
        masses = torch.tensor(self.masses)
        self.root_physx_view = types.SimpleNamespace(get_masses=lambda: masses.repeat(self.n, 1))

    # -- Isaac Lab Articulation API ------------------------------------------
    def find_joints(self, names, preserve_order=False):
        names = [names] if isinstance(names, str) else list(names)
        return [self.joint_names.index(n) for n in names], names

    def find_bodies(self, names, preserve_order=False):
        names = [names] if isinstance(names, str) else list(names)
        return [self.body_names.index(n) for n in names], names

    def set_joint_velocity_target(self, target, joint_ids=None, env_ids=None):
        self.vel_target[:, joint_ids] = target

    def write_joint_state_to_sim(self, pos, vel, joint_ids=None, env_ids=None):
        for k, e in enumerate(env_ids.tolist()):
            for j in range(len(self.joint_names)):
                self.p.resetJointState(self.bodies[e], j, float(pos[k, j]), float(vel[k, j]), physicsClientId=self.cid)
        self.data.joint_pos[env_ids] = pos
        self.data.joint_vel[env_ids] = vel

    def write_root_pose_to_sim(self, pose, env_ids=None):
        pass  # fixed base

    def write_root_velocity_to_sim(self, vel, env_ids=None):
        pass

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.permanent_wrench_composer.force_b[ids] = 0.0
        self.permanent_wrench_composer.torque_b[ids] = 0.0
        self.vel_target[ids] = 0.0

    def write_data_to_sim(self):
        p, c = self.p, self.cid
        cmd = torch.maximum(torch.minimum(self.vel_target, self.vmax), -self.vmax)
        f_b, t_b = self.permanent_wrench_composer.force_b, self.permanent_wrench_composer.torque_b
        q = self.data.body_link_quat_w[:, self.bucket_body]
        f_w, t_w = _quat_apply(q, f_b), _quat_apply(q, t_b)
        com = self.data.body_com_pos_w[:, self.bucket_body]
        link = self.bucket_body - 1
        for e, b in enumerate(self.bodies):
            p.setJointMotorControlArray(b, list(range(len(self.joint_names))), p.VELOCITY_CONTROL,
                                        targetVelocities=cmd[e].tolist(), forces=self.effort.tolist(), physicsClientId=c)
            p.applyExternalForce(b, link, f_w[e].tolist(), com[e].tolist(), p.WORLD_FRAME, physicsClientId=c)
            p.applyExternalTorque(b, link, t_w[e].tolist(), p.WORLD_FRAME, physicsClientId=c)

    def sim_step(self):
        self.p.stepSimulation(physicsClientId=self.cid)

    def update(self):
        p, c, d = self.p, self.cid, self.data
        nl = len(self.joint_names)
        for e, b in enumerate(self.bodies):
            js = p.getJointStates(b, list(range(nl)), physicsClientId=c)
            d.joint_pos[e] = torch.tensor([s[0] for s in js])
            d.joint_vel[e] = torch.tensor([s[1] for s in js])
            d.applied_torque[e] = torch.tensor([s[3] for s in js])
            pos, orn = p.getBasePositionAndOrientation(b, physicsClientId=c)
            d.body_link_pos_w[e, 0] = torch.tensor(pos)
            d.body_com_pos_w[e, 0] = torch.tensor(pos)
            d.body_link_quat_w[e, 0] = torch.tensor([orn[3], orn[0], orn[1], orn[2]])
            for j, s in enumerate(p.getLinkStates(b, list(range(nl)), computeLinkVelocity=1,
                                                  computeForwardKinematics=1, physicsClientId=c)):
                com, lpos, lorn = torch.tensor(s[0]), torch.tensor(s[4]), s[5]
                v_com, w = torch.tensor(s[6]), torch.tensor(s[7])
                d.body_com_pos_w[e, j + 1] = com
                d.body_link_pos_w[e, j + 1] = lpos
                d.body_link_quat_w[e, j + 1] = torch.tensor([lorn[3], lorn[0], lorn[1], lorn[2]])
                d.body_link_lin_vel_w[e, j + 1] = v_com + torch.cross(w, lpos - com, dim=0)
                d.body_ang_vel_w[e, j + 1] = w

    def disconnect(self):
        self.p.disconnect(physicsClientId=self.cid)


def _fixed_urdf(src: str) -> str:
    """URDF with package:// mesh paths made absolute (same repair as scripts/convert_urdf.py)."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(src)
    mesh_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(src))), "meshes")
    for m in tree.getroot().iter("mesh"):
        m.set("filename", os.path.join(mesh_dir, os.path.basename(m.get("filename", ""))))
    out = os.path.join(tempfile.gettempdir(), "excavator_pybullet.urdf")
    tree.write(out)
    return out


# --------------------------------------------------------------------------- #
# 3. building the real env on top
# --------------------------------------------------------------------------- #
def make_env(num_envs: int = 1, reward_mode: str = "dense", play: bool = False):
    install_isaaclab_stubs()
    from excavator_rl.digging_env import DiggingEnv
    from excavator_rl.digging_env_cfg import DiggingEnvCfg, DiggingEnvCfg_PLAY

    cfg = (DiggingEnvCfg_PLAY if play else DiggingEnvCfg)()
    cfg.scene = types.SimpleNamespace(num_envs=num_envs, env_spacing=30.0)
    cfg.__post_init__()
    cfg.scene.num_envs = num_envs
    cfg.debug_vis = False
    cfg.reward_mode = reward_mode
    PyBulletArticulation.origins = torch.tensor([[i * 30.0, 0.0, 0.0] for i in range(num_envs)])
    return DiggingEnv(cfg)


class TwinVecEnv:
    """DiggingEnv on PyBullet with the contract of excavator_rl.algorithms."""

    def __init__(self, env):
        self.env, self.unwrapped = env, env
        self.num_envs, self.device = env.num_envs, torch.device("cpu")
        self.obs_dim, self.act_dim = int(env.cfg.observation_space), int(env.cfg.action_space)

    def reset(self):
        return self.env.reset()[0]["policy"]

    def step(self, a):
        obs, rew, term, trunc, extras = self.env.step(a)
        return obs["policy"], rew, term.bool(), trunc.bool(), extras


# --------------------------------------------------------------------------- #
# 4. commands
# --------------------------------------------------------------------------- #
def banner(t):
    print("\n" + t + "\n" + "-" * len(t))


def cmd_check(args) -> int:
    from excavator_rl.scripted import ArmKinematics, ScriptedDigger

    env = make_env(1, play=True)
    env.reset()
    r, d = env.robot, env.robot.data
    problems = []

    banner("MODEL")
    for name, m in zip(r.body_names, r.masses):
        print(f"  {name:<20} {m:>10.0f} kg")
    print(f"  moving mass {sum(r.masses[1:]) / 1000:.1f} t")
    for j, name in enumerate(r.joint_names):
        lo, hi = d.soft_joint_pos_limits[0, j].tolist()
        print(f"  {name:<20} limits {lo:+.3f} .. {hi:+.3f} rad   effort {r.effort[j]:.3g} N.m   vmax {r.vmax[j]:.2f} rad/s")

    banner("DIG-ENTRY POSE (DEFAULT_JOINT_POS)")
    pos, quat, tip, _, open_dir, _, _ = env._bucket_state()
    tip, pivot = tip[0] - env.scene.env_origins[0], pos[0] - env.scene.env_origins[0]
    reach = float(torch.linalg.norm(tip[:2]))
    surf = float(env.soil.surface_height(torch.tensor([reach])))
    print(f"  bucket pivot    y {pivot[1]:+.2f}  z {pivot[2]:+.2f} m")
    print(f"  cutting edge    y {tip[1]:+.2f}  z {tip[2]:+.2f} m   (soil surface {surf:+.2f} m)")
    print(f"  bucket opening  {[round(v, 2) for v in open_dir[0].tolist()]}")
    if tip[2] < surf - 0.15:
        problems.append(f"cutting edge starts {surf - tip[2]:.2f} m under the soil")
    if tip[2] > surf + 1.0:
        problems.append(f"cutting edge starts {tip[2] - surf:.2f} m above the soil")

    banner("HOLD STILL 3 s (zero command)")
    q0 = d.joint_pos[0].clone()
    for _ in range(90):
        env.step(torch.zeros(1, env.cfg.action_space))
    drift = (d.joint_pos[0] - q0)[1:]
    print(f"  joint drift boom/stick/bucket: {[round(v, 4) for v in drift.tolist()]} rad")
    if drift.abs().max() > 0.02:
        problems.append(f"arm drifts {drift.abs().max():.3f} rad in 3 s under zero command")

    banner("SCRIPTED DIG CYCLE (IK expert: bite, drag, curl, lift)")
    env.reset()
    kin = ArmKinematics(PyBulletArticulation.urdf_path)
    expert = ScriptedDigger(kin, r.vmax[1:])
    print(f"{'t':>5} {'phase':<10} {'depth':>6} {'F_soil':>9} {'fill':>5} {'tooth_y':>7} {'tooth_z':>7} {'pivot_z':>7} {'reward':>7}")
    peak_fill, peak_depth, max_pivot, total_r, success = 0.0, 0.0, -9.0, 0.0, False
    fill_at_end = 0.0
    for k in range(int(args.cycle_s / env.step_dt)):
        t = k * env.step_dt
        a = expert.action(t, d.joint_pos[0, 1:])
        _, rew, term, trunc, _ = env.step(a.unsqueeze(0).float())
        total_r += float(rew[0])
        pos, _, tip, _, _, _, _ = env._bucket_state()
        o = env.scene.env_origins[0]
        if bool(term[0]):
            success = True
            print(f"{t:>5.1f} SUCCESS -- bucket lifted to {env.cfg.lift_height} m with soil, episode terminated")
            break
        peak_fill = max(peak_fill, float(env.soil.fill_ratio[0]))
        peak_depth = max(peak_depth, float(env._soil_out["depth"][0]))
        max_pivot = max(max_pivot, float(pos[0, 2] - o[2]))
        fill_at_end = float(env.soil.fill_ratio[0])
        if k % 15 == 0:
            print(f"{t:>5.1f} {expert.phase(t):<10} {env._soil_out['depth'][0]:>6.2f} {torch.linalg.norm(env._soil_out['force'][0]):>9.0f} "
                  f"{env.soil.fill_ratio[0]:>5.2f} {tip[0, 1] - o[1]:>7.2f} {tip[0, 2] - o[2]:>7.2f} {pos[0, 2] - o[2]:>7.2f} {rew[0]:>7.2f}")
    print(f"\n  peak depth {peak_depth:.2f} m   peak fill {peak_fill:.2f} (target {env.cfg.target_fill})   "
          f"highest pivot {max_pivot:.2f} m (target {env.cfg.lift_height} m)   return {total_r:.1f}   success {success}")
    if peak_depth <= 0.0:
        problems.append("scripted cycle never reaches the soil")
    if peak_fill < env.cfg.target_fill:
        problems.append(f"scripted cycle fills only {peak_fill:.2f} of the bucket (target {env.cfg.target_fill})")
    if not success:
        problems.append(f"scripted cycle does not complete the task (fill left {fill_at_end:.2f}, "
                        f"highest pivot {max_pivot:.2f} m)")

    banner("RESULT")
    for msg in problems:
        print("  PROBLEM:", msg)
    if not problems:
        print("  no problems found")
    env.close()
    return 1 if problems else 0


def cmd_train(args) -> int:
    from excavator_rl.algorithms import Logger, make_agent

    torch.manual_seed(args.seed)
    env = TwinVecEnv(make_env(args.num_envs, reward_mode=args.reward_mode))
    u = env.unwrapped
    u.episode_length_buf = torch.randint_like(u.episode_length_buf, high=int(u.max_episode_length))
    agent = make_agent(args.algo, env.obs_dim, env.act_dim, "cpu", [f"num_envs={args.num_envs}"] + args.cfg)
    t0 = time.time()
    agent.train(env, args.iterations, Logger(args.log_dir, print_every=args.print_every),
                ckpt_dir=args.log_dir, save_interval=max(1, args.iterations))
    print(f"[pybullet_twin] {args.algo}: {args.iterations} iterations in {time.time() - t0:.0f} s")
    return 0


def cmd_play(args) -> int:
    from excavator_rl.algorithms import load_agent

    agent, _ = load_agent(args.checkpoint, "cpu")
    env = TwinVecEnv(make_env(1, play=True))
    u = env.unwrapped
    print(f"[pybullet_twin] {agent.name} policy from {args.checkpoint}")
    print(f"{'t':>5} {'action':>22} {'depth':>6} {'F_soil':>8} {'fill':>5} {'tooth_y':>7} {'tooth_z':>7} {'pivot_z':>7} {'reward':>7}")
    obs, ret, episodes, successes, t = env.reset(), 0.0, 0, 0, 0
    while episodes < args.episodes:
        with torch.no_grad():
            a = agent.act_inference(obs)
        obs, rew, term, trunc, _ = env.step(a)
        ret += float(rew[0])
        if bool(term[0] | trunc[0]):
            episodes += 1
            successes += int(term[0])
            print(f"  episode {episodes}: {'SUCCESS' if term[0] else 'time-out'} after {t + 1} steps "
                  f"({(t + 1) * u.step_dt:.1f} s), return {ret:.1f}")
            ret, t = 0.0, 0
            continue
        if t % args.every == 0:
            pos, _, tip, _, _, _, _ = u._bucket_state()
            o = u.scene.env_origins[0]
            print(f"{t * u.step_dt:>5.1f} {str([round(v, 2) for v in a[0].tolist()]):>22} {u._soil_out['depth'][0]:>6.2f} "
                  f"{torch.linalg.norm(u._soil_out['force'][0]):>8.0f} {u.soil.fill_ratio[0]:>5.2f} {tip[0, 1] - o[1]:>7.2f} "
                  f"{tip[0, 2] - o[2]:>7.2f} {pos[0, 2] - o[2]:>7.2f} {rew[0]:>7.2f}")
        t += 1
    print(f"[pybullet_twin] success {successes}/{episodes}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="model, dig-entry pose, drives, scripted dig cycle")
    c.add_argument("--cycle_s", type=float, default=14.0)
    t = sub.add_parser("train", help="short training run with an excavator_rl.algorithms agent")
    t.add_argument("--algo", default="ppo", choices=["reinforce", "ppo", "trpo", "ddpg"])
    t.add_argument("--num_envs", type=int, default=16)
    t.add_argument("--iterations", type=int, default=50)
    t.add_argument("--reward_mode", default="dense", choices=["dense", "vortex"])
    t.add_argument("--cfg", nargs="*", default=[])
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--print_every", type=int, default=5)
    t.add_argument("--log_dir", default=None)
    pl = sub.add_parser("play", help="run a checkpoint saved by `train` (or scripts/train_algo.py)")
    pl.add_argument("checkpoint")
    pl.add_argument("--episodes", type=int, default=3)
    pl.add_argument("--every", type=int, default=6, help="print every N control steps")
    for s in (c, t, pl):
        s.add_argument("--urdf", default=DEFAULT_URDF)
    args = ap.parse_args()
    PyBulletArticulation.urdf_path = args.urdf
    return {"check": cmd_check, "train": cmd_train, "play": cmd_play}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
