"""Configuration for the excavator digging task."""

from __future__ import annotations

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .excavator_cfg import excavator_cfg
from .excavator_params import (
    ARM_JOINTS,
    BUCKET_OPEN_DIR,
    BUCKET_TIP_OFFSET,
    BUCKET_WIDTH,
    SWING_JOINT,
    VELOCITY_LIMITS,
)
from .soil import SoilCfg

#: Repository root (the folder that contains ``excavator_rl/``).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Location of the excavator USD.  ``scripts/setup_assets.sh`` writes it to
#: ``assets/usd/excavator.usd`` inside this repository.  Override with the
#: EXCAVATOR_USD environment variable, ``--usd`` on the scripts, or by setting
#: ``cfg.usd_path`` directly.
DEFAULT_USD = os.path.expanduser(
    os.environ.get("EXCAVATOR_USD", os.path.join(REPO_ROOT, "assets", "usd", "excavator.usd"))
)

#: bucket-link centre of mass, from the USD mass properties [m, bucket frame]
BUCKET_COM_OFFSET = (0.0037527457, -1.0269979, -0.24527328)


@configclass
class DiggingEnvCfg(DirectRLEnvCfg):
    """Dig a bucket of soil and lift it to a target height.

    Mirrors Task 1 of hanzunye/vortexRL ("digging and lifting"): the agent
    commands boom / stick / bucket cylinder velocities, must load the bucket to
    a target fill, then raise it above a target height without spilling.
    """

    # ---------------------------------------------------------------- core --
    decimation: int = 4
    """Physics steps per control step.  dt = 1/120 -> 30 Hz control."""
    episode_length_s: float = 20.0
    """vortexRL used 50-180 steps of 0.5 s; 20 s at 30 Hz = 600 steps."""

    action_space: int = 3
    observation_space: int = 16
    state_space: int = 0

    # ----------------------------------------------------------- simulation -
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.9,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=30.0,          # 11.1 m reach + soil bed -> keep them apart
        replicate_physics=True,
    )

    # ---------------------------------------------------------------- asset -
    usd_path: str = DEFAULT_USD
    """Path to the excavator USD (see scripts/convert_urdf.py)."""

    control_swing: bool = False
    """Add the slew joint to the action space.  vortexRL controls 3 cylinders
    only; enable this once digging works if you want a full dig-and-swing
    cycle (the 1-D soil bed then becomes an approximation)."""

    ground_z: float = -2.0
    """Height of the collision ground plane [m].  It sits below the deepest
    trench so the bucket can penetrate the (analytic) soil freely; the machine
    itself is root-fixed and needs no ground contact."""

    # ----------------------------------------------------------------- soil -
    soil: SoilCfg = SoilCfg(bucket_width=BUCKET_WIDTH)

    # ----------------------------------------------------------------- task -
    target_fill: float = 0.60
    """Bucket fill ratio that ends the digging phase [-]."""
    lift_height: float = 4.20
    """Bucket pivot height that completes the episode [m] (vortexRL used 4.2)."""
    min_fill_at_success: float = 0.40
    """Payload that must still be in the bucket when the height is reached."""

    action_scale: tuple[float, ...] = tuple(VELOCITY_LIMITS[j] for j in ARM_JOINTS)
    """Joint velocity commanded by an action of +-1 [rad/s]."""
    action_lowpass: float = 0.25
    """First-order filter coefficient on the velocity command, 0 < a <= 1.
    vortexRL low-pass-filters the cylinder command to mimic spool dynamics;
    a = 0.25 at 30 Hz is a ~40 ms time constant."""

    # -------------------------------------------------------------- rewards -
    reward_mode: str = "dense"
    """``"dense"`` (recommended for large-scale PPO) or ``"vortex"`` (faithful
    port of vortexRL's segmented, all-negative reward)."""

    w_fill: float = 60.0
    """Reward per unit of fill-ratio gained while digging."""
    w_lift: float = 12.0
    """Reward per metre of height gained while lifting (payload retained)."""
    w_success: float = 200.0
    w_time: float = 0.05
    w_spill: float = 30.0
    w_effort: float = 0.02
    w_action_rate: float = 0.01
    w_reach: float = 0.5
    """Shaping that pulls the tip towards the soil at the start of an episode."""

    # ------------------------------------------------------------- viz/debug -
    debug_vis: bool = True
    vis_num_envs: int = 4
    """How many environments get soil markers drawn (markers are not free)."""

    # ------------------------------------------------------------ internals -
    tip_offset: tuple[float, float, float] = BUCKET_TIP_OFFSET
    bucket_com_offset: tuple[float, float, float] = BUCKET_COM_OFFSET
    bucket_open_dir: tuple[float, float, float] = BUCKET_OPEN_DIR

    robot: ArticulationCfg = MISSING
    """Filled in by :meth:`__post_init__` from :data:`usd_path`."""

    def __post_init__(self):
        parent_post = getattr(super(), "__post_init__", None)
        if callable(parent_post):
            parent_post()
        self.robot = excavator_cfg(self.usd_path)
        n_act = 4 if self.control_swing else 3
        self.action_space = n_act
        # joint pos + joint vel + last action + tip xyz + depth + fill + phase + dh
        self.observation_space = 3 * n_act + 3 + 4
        if self.control_swing:
            self.action_scale = tuple(
                [VELOCITY_LIMITS[SWING_JOINT]] + [VELOCITY_LIMITS[j] for j in ARM_JOINTS]
            )


@configclass
class DiggingEnvCfg_PLAY(DiggingEnvCfg):
    """Small, visualised variant for inspecting a trained policy."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.soil.height_range = (0.0, 0.0)
        self.soil.slope_range = (0.0, 0.0)
        self.debug_vis = True
