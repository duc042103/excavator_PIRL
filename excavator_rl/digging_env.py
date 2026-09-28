"""Isaac Lab direct-workflow RL environment: excavator digging + lifting.

Port of Task 1 from `hanzunye/vortexRL <https://github.com/hanzunye/vortexRL>`_
(Han & Stein, KIT -- "Applying Reinforcement Learning to Digital Twin of
Excavator to Dig Automatically") onto Isaac Sim / Isaac Lab.

What carries over from the reference
------------------------------------
* 3 continuous actions = boom / stick / bucket velocity commands, low-pass
  filtered to imitate spool dynamics.
* the segmented, phase-based reward (dig -> lift), with a ``back`` phase flag
  in the observation -- available here as ``reward_mode="vortex"``.
* task definition: load the bucket, then raise it above 4.2 m without losing
  the payload.

What necessarily changes
------------------------
* Vortex Studio's particle/mesh soil solver is replaced by the analytic soil
  model in :mod:`excavator_rl.soil` (Isaac Sim has no equivalent solver that
  runs thousands of environments in parallel).
* joint-space instead of cylinder-space commands: the CAD has no hydraulic
  cylinder linkage, only the four revolute joints.
* thousands of parallel environments instead of one, so the default reward is
  a smooth dense variant; the faithful stepwise reward is still selectable.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

try:  # Isaac Lab >= 2.0
    from isaaclab.utils.math import quat_apply
except ImportError:  # pragma: no cover - older naming
    from isaaclab.utils.math import quat_rotate as quat_apply

from .digging_env_cfg import DiggingEnvCfg
from .excavator_params import ARM_JOINTS, BUCKET_LINK, EFFORT_LIMITS, SWING_JOINT
from .soil import SoilModel


class DiggingEnv(DirectRLEnv):
    cfg: DiggingEnvCfg

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def __init__(self, cfg: DiggingEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        joint_names = [SWING_JOINT] + ARM_JOINTS if cfg.control_swing else list(ARM_JOINTS)
        self._act_joint_ids, _ = self.robot.find_joints(joint_names, preserve_order=True)
        self._act_joint_ids = torch.tensor(self._act_joint_ids, device=self.device, dtype=torch.long)
        self._bucket_id, _ = self.robot.find_bodies(BUCKET_LINK)
        self._bucket_id = int(self._bucket_id[0])

        n_act = len(joint_names)
        self._action_scale = torch.tensor(cfg.action_scale, device=self.device)
        #: per-joint effort ceiling, taken from the actuator table rather than
        #: from a version-dependent ArticulationData attribute
        self._effort_limit = torch.tensor(
            [EFFORT_LIMITS[n] for n in joint_names], device=self.device
        ).clamp(min=1.0)
        self._raw_actions = torch.zeros(self.num_envs, n_act, device=self.device)
        self._filtered = torch.zeros_like(self._raw_actions)
        self._prev_filtered = torch.zeros_like(self._raw_actions)

        self._tip_offset = torch.tensor(cfg.tip_offset, device=self.device).repeat(self.num_envs, 1)
        self._com_offset = torch.tensor(cfg.bucket_com_offset, device=self.device).repeat(self.num_envs, 1)
        self._open_dir_b = torch.tensor(cfg.bucket_open_dir, device=self.device).repeat(self.num_envs, 1)

        self.soil = SoilModel(cfg.soil, self.num_envs, self.device)

        # phase flag: the "back" signal of vortexRL -- True once the bucket holds
        # enough soil and the agent should be lifting rather than digging.
        self._back = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._prev_fill_ratio = torch.zeros(self.num_envs, device=self.device)
        self._prev_height = torch.zeros(self.num_envs, device=self.device)
        self._prev_fill_mass = torch.zeros(self.num_envs, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._soil_out = {
            "force": torch.zeros(self.num_envs, 3, device=self.device),
            "depth": torch.zeros(self.num_envs, device=self.device),
            "d_fill": torch.zeros(self.num_envs, device=self.device),
            "spilled": torch.zeros(self.num_envs, device=self.device),
        }

        # Isaac Lab >= 2.3 routes external wrenches through a WrenchComposer and
        # logs a deprecation warning on every set_external_force_and_torque call
        # (i.e. 30 times a second here); use the composer directly when present.
        self._wrench_composer = getattr(self.robot, "permanent_wrench_composer", None)
        self._bucket_ids_t = torch.tensor([self._bucket_id], device=self.device, dtype=torch.long)

        # resolve the (version dependent) body state accessors once
        data = self.robot.data
        self._pos_attr = "body_link_pos_w" if hasattr(data, "body_link_pos_w") else "body_pos_w"
        self._quat_attr = "body_link_quat_w" if hasattr(data, "body_link_quat_w") else "body_quat_w"
        if hasattr(data, "body_link_lin_vel_w"):
            self._vel_attr, self._vel_is_com = "body_link_lin_vel_w", False
        elif hasattr(data, "body_com_lin_vel_w"):
            self._vel_attr, self._vel_is_com = "body_com_lin_vel_w", True
        else:
            self._vel_attr, self._vel_is_com = "body_lin_vel_w", False
        self._angvel_attr = (
            "body_ang_vel_w" if hasattr(data, "body_ang_vel_w") else "body_com_ang_vel_w"
        )

        self._soil_markers: VisualizationMarkers | None = None
        if self.cfg.debug_vis:
            self._init_markers()

    # ------------------------------------------------------------------ #
    # scene
    # ------------------------------------------------------------------ #
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)

        # Safety floor well below the deepest trench: the excavator root is
        # fixed, and soil contact is analytic, so this plane is only a visual
        # reference and a catch-all collider.
        ground = sim_utils.GroundPlaneCfg(size=(200.0, 200.0))
        ground.func("/World/ground", ground, translation=(0.0, 0.0, self.cfg.ground_z))

        self.scene.clone_environments(copy_from_source=False)
        # CPU PhysX needs the cloned environments' collisions filtered explicitly
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot

        light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.92))
        light.func("/World/Light", light)

    # ------------------------------------------------------------------ #
    # control
    # ------------------------------------------------------------------ #
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._raw_actions = actions.clone().clamp(-1.0, 1.0)
        self._prev_filtered = self._filtered.clone()
        a = self.cfg.action_lowpass
        self._filtered = a * self._raw_actions + (1.0 - a) * self._filtered

        # soil is integrated once per control step, and the resulting wrench
        # persists through the decimated physics steps
        self._update_soil()

    def _apply_action(self) -> None:
        vel_cmd = self._filtered * self._action_scale
        self.robot.set_joint_velocity_target(vel_cmd, joint_ids=self._act_joint_ids)

    # ------------------------------------------------------------------ #
    # kinematics helpers
    # ------------------------------------------------------------------ #
    def _bucket_state(self):
        data = self.robot.data
        pos = getattr(data, self._pos_attr)[:, self._bucket_id]
        quat = getattr(data, self._quat_attr)[:, self._bucket_id]
        lin = getattr(data, self._vel_attr)[:, self._bucket_id]
        ang = getattr(data, self._angvel_attr)[:, self._bucket_id]

        r_tip = quat_apply(quat, self._tip_offset)
        r_com = quat_apply(quat, self._com_offset)
        tip_w = pos + r_tip
        # velocity of the cutting edge: rigid-body transfer from whichever
        # reference point this Isaac Lab version reports
        lever = r_tip - r_com if self._vel_is_com else r_tip
        tip_vel = lin + torch.cross(ang, lever, dim=1)
        open_dir_w = quat_apply(quat, self._open_dir_b)
        return pos, quat, tip_w, tip_vel, open_dir_w, r_tip, r_com

    def _update_soil(self) -> None:
        pos, quat, tip_w, tip_vel, open_dir_w, r_tip, r_com = self._bucket_state()
        tip_local = tip_w - self.scene.env_origins

        out = self.soil.step(
            tip_pos=tip_local,
            tip_vel=tip_vel,
            open_dir_w=open_dir_w,
            dt=self.step_dt,
        )
        self._soil_out = out

        # resistance at the cutting edge + payload weight at the bucket COM
        payload = torch.zeros_like(out["force"])
        payload[:, 2] = -self.soil.fill_mass * 9.81
        force_w = out["force"] + payload
        # external wrenches act at the COM, so move the tip force there
        torque_w = torch.cross(r_tip - r_com, out["force"], dim=1)

        # Rotate into the bucket link frame ourselves and always pass a *local*
        # wrench.  The world-frame option is not safe to rely on: Isaac Lab 2.3's
        # WrenchComposer converts global wrenches with link poses it caches until
        # the next reset, i.e. with a stale bucket orientation.
        quat_inv = quat * torch.tensor([1.0, -1.0, -1.0, -1.0], device=self.device)
        f = quat_apply(quat_inv, force_w)
        t = quat_apply(quat_inv, torque_w)
        self._set_bucket_wrench(f, t)

    def _set_bucket_wrench(self, force_b: torch.Tensor, torque_b: torch.Tensor, env_ids=None) -> None:
        """Hold a body-frame wrench (applied at the COM) on the bucket until changed."""
        f = force_b.unsqueeze(1).contiguous()
        t = torque_b.unsqueeze(1).contiguous()
        if self._wrench_composer is not None:
            self._wrench_composer.set_forces_and_torques(
                forces=f, torques=t, body_ids=self._bucket_ids_t, env_ids=env_ids, is_global=False
            )
        else:
            self.robot.set_external_force_and_torque(f, t, body_ids=[self._bucket_id], env_ids=env_ids)

    # ------------------------------------------------------------------ #
    # observations
    # ------------------------------------------------------------------ #
    def _get_observations(self) -> dict:
        pos, _, tip_w, _, _, _, _ = self._bucket_state()
        tip_local = tip_w - self.scene.env_origins
        pivot_local = pos - self.scene.env_origins

        q = self.robot.data.joint_pos[:, self._act_joint_ids]
        qd = self.robot.data.joint_vel[:, self._act_joint_ids]
        lo = self.robot.data.soft_joint_pos_limits[:, self._act_joint_ids, 0]
        hi = self.robot.data.soft_joint_pos_limits[:, self._act_joint_ids, 1]
        q_n = 2.0 * (q - lo) / (hi - lo).clamp(min=1e-6) - 1.0
        qd_n = qd / self._action_scale

        fill_ratio = self.soil.fill_ratio.unsqueeze(1)
        depth = self._soil_out["depth"].unsqueeze(1)
        phase = self._back.float().unsqueeze(1)
        dh = ((self.cfg.lift_height - pivot_local[:, 2]) / self.cfg.lift_height).unsqueeze(1)

        obs = torch.cat(
            [q_n, qd_n, self._filtered, tip_local, depth, fill_ratio, phase, dh], dim=-1
        )
        if self.cfg.debug_vis:
            self._draw_markers(tip_w)
        return {"policy": obs.clamp(-10.0, 10.0)}

    # ------------------------------------------------------------------ #
    # reward
    # ------------------------------------------------------------------ #
    def _update_phase(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Refresh the dig/lift phase flag and the success flag.

        Idempotent, so it is safe to call from both ``_get_dones`` and
        ``_get_rewards`` regardless of which Isaac Lab calls first.
        """
        pos = getattr(self.robot.data, self._pos_attr)[:, self._bucket_id]
        height = (pos - self.scene.env_origins)[:, 2]
        fill_ratio = self.soil.fill_ratio

        self._back |= fill_ratio >= self.cfg.target_fill
        self._success = (
            self._back
            & (height >= self.cfg.lift_height)
            & (fill_ratio >= self.cfg.min_fill_at_success)
        )
        return height, fill_ratio

    def _get_rewards(self) -> torch.Tensor:
        height, fill_ratio = self._update_phase()

        if self.cfg.reward_mode == "vortex":
            rew = self._reward_vortex(height, fill_ratio)
        else:
            rew = self._reward_dense(height, fill_ratio)

        self._prev_fill_ratio = fill_ratio.clone()
        self._prev_height = height.clone()
        self._prev_fill_mass = self.soil.fill_mass.clone()
        return rew

    def _reward_dense(self, height: torch.Tensor, fill_ratio: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        d_fill = (fill_ratio - self._prev_fill_ratio).clamp(min=0.0)
        d_height = height - self._prev_height

        # phase 1: fill the bucket.  phase 2: lift it.
        dig = (~self._back).float()
        lift = self._back.float()

        r = cfg.w_fill * d_fill * dig
        r = r + cfg.w_lift * d_height.clamp(-0.5, 0.5) * lift * fill_ratio
        r = r + cfg.w_success * self._success.float()

        # gentle pull towards the soil before the first bite, otherwise the
        # policy has no gradient to follow at all
        tip_depth = self._soil_out["depth"]
        r = r + cfg.w_reach * (tip_depth > 0).float() * dig * (fill_ratio < 1e-3).float()

        # costs
        r = r - cfg.w_time
        r = r - cfg.w_spill * self._soil_out["spilled"] / max(cfg.soil.capacity, 1e-6)
        tau = getattr(self.robot.data, "applied_torque", None)
        if tau is not None:
            tau = tau[:, self._act_joint_ids]
            r = r - cfg.w_effort * (tau / self._effort_limit).pow(2).mean(dim=1)
        r = r - cfg.w_action_rate * (self._filtered - self._prev_filtered).pow(2).sum(dim=1)
        return r

    def _reward_vortex(self, height: torch.Tensor, fill_ratio: torch.Tensor) -> torch.Tensor:
        """Faithful port of ``Reward/RewardDDPG.py::get_score_height``.

        The original is written as a chain of ``if`` statements (not ``elif``),
        so the "M < M_old" branch is immediately overwritten by the trailing
        ``else``.  The intended semantics are implemented here; the mass and
        height thresholds (1200 / 1000 / 500 units, 4.2 m) become fractions of
        the configured target fill.
        """
        m = self.soil.fill_mass
        m_old = self._prev_fill_mass
        h_old = self._prev_height
        target_m = self.cfg.target_fill * self.cfg.soil.capacity * self.cfg.soil.density

        r = torch.full_like(m, -1.2)

        # ---- digging phase
        dig = ~self._back
        r = torch.where(dig & (m > m_old), torch.full_like(r, -0.5), r)
        r = torch.where(dig & (m == m_old) & (m != 0), torch.full_like(r, -0.8), r)
        r = torch.where(dig & (m < m_old) & (m != 0), torch.full_like(r, -1.5), r)

        # ---- lifting phase
        lift = self._back
        gain = (m - m_old) / max(0.6 * target_m, 1.0)
        up, same, down = height > h_old, height == h_old, height < h_old
        keep = m >= m_old
        r = torch.where(lift & down & keep, torch.full_like(r, -1.2), r)
        r = torch.where(lift & down & ~keep, gain - 1.5, r)
        r = torch.where(lift & same & keep, torch.full_like(r, -0.8), r)
        r = torch.where(lift & same & ~keep, gain - 1.1, r)
        r = torch.where(lift & up & keep, torch.full_like(r, -0.5), r)
        r = torch.where(lift & up & ~keep, gain - 0.8, r)

        # ---- finish
        done = lift & (height >= self.cfg.lift_height)
        r = torch.where(done & (m > 0.85 * target_m), torch.full_like(r, 3.0), r)
        r = torch.where(done & (m > 0.40 * target_m) & (m <= 0.85 * target_m),
                        torch.full_like(r, 2.0), r)
        r = torch.where(done & (m <= 0.40 * target_m), torch.full_like(r, -0.3), r)
        return r

    # ------------------------------------------------------------------ #
    # termination
    # ------------------------------------------------------------------ #
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_phase()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self._success.clone(), time_out

    # ------------------------------------------------------------------ #
    # reset
    # ------------------------------------------------------------------ #
    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        noise = 0.05 * (torch.rand_like(joint_pos) * 2.0 - 1.0)
        joint_pos = (joint_pos + noise).clamp(
            self.robot.data.soft_joint_pos_limits[env_ids, :, 0],
            self.robot.data.soft_joint_pos_limits[env_ids, :, 1],
        )
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids)

        self.soil.reset(env_ids)
        self._back[env_ids] = False
        self._success[env_ids] = False
        self._raw_actions[env_ids] = 0.0
        self._filtered[env_ids] = 0.0
        self._prev_filtered[env_ids] = 0.0
        self._prev_fill_ratio[env_ids] = 0.0
        self._prev_fill_mass[env_ids] = 0.0

        pos = getattr(self.robot.data, self._pos_attr)[:, self._bucket_id]
        self._prev_height[env_ids] = (pos - self.scene.env_origins)[env_ids, 2]

        zero = torch.zeros(len(env_ids), 3, device=self.device)
        self._set_bucket_wrench(zero, zero, env_ids=env_ids)

    # ------------------------------------------------------------------ #
    # visualisation (optional, never fatal)
    # ------------------------------------------------------------------ #
    def _init_markers(self) -> None:
        try:
            cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/soil",
                markers={
                    "cell": sim_utils.CuboidCfg(
                        size=(self.cfg.soil.bucket_width, self.soil.dr, 0.08),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.32, 0.18)),
                    ),
                    "tip": sim_utils.SphereCfg(
                        radius=0.18,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.1)),
                    ),
                },
            )
            self._soil_markers = VisualizationMarkers(cfg)
        except Exception as exc:  # pragma: no cover
            print(f"[excavator_rl] soil markers disabled: {exc}")
            self._soil_markers = None

    def _draw_markers(self, tip_w: torch.Tensor) -> None:
        if self._soil_markers is None:
            return
        try:
            n = min(self.cfg.vis_num_envs, self.num_envs)
            step = max(1, self.cfg.soil.num_cells // 60)
            r = self.soil.r_centers[::step]
            h = self.soil.height[:n, ::step]
            origins = self.scene.env_origins[:n]

            cells = torch.zeros(n, r.numel(), 3, device=self.device)
            cells[..., 1] = r.unsqueeze(0)
            cells[..., 2] = h
            cells = cells + origins.unsqueeze(1)
            cells = cells.reshape(-1, 3)

            pts = torch.cat([cells, tip_w[:n]], dim=0)
            idx = torch.cat(
                [
                    torch.zeros(cells.shape[0], dtype=torch.long, device=self.device),
                    torch.ones(n, dtype=torch.long, device=self.device),
                ]
            )
            self._soil_markers.visualize(translations=pts, marker_indices=idx)
        except Exception:  # pragma: no cover
            self._soil_markers = None
