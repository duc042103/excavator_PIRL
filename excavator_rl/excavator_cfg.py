"""Articulation declaration for the MathScavator9000 hydraulic excavator.

Every number below was read out of the CAD-derived assets, not guessed:

  * joint names / axes / limits  -> MathScavator9000_flat.SLDASM_physics.usd
                                    (and MathScavator9000_flat.SLDASM.urdf)
  * link masses / inertias       -> same physics layer (SolidWorks mass properties)
  * link geometry (lengths,      -> meshes/*.STL vertex analysis
    bucket width, tip offset)

Machine class
-------------
boom 6.24 m, stick 2.98 m, bucket pivot->tooth tips 2.18 m, bucket width
1.148 m, max tooth reach 11.1 m, moving mass 33.1 t (chassis 17.4 + boom 7.2 +
stick 3.3 + bucket 5.1).  That is a ~36 t class machine -- the same class as
the Vortex Studio model in hanzunye/vortexRL, which is why the RL setup ports
over with only unit changes.

What is WRONG in the shipped USD and fixed here
-----------------------------------------------
The Isaac Sim URDF importer wrote auto-tuned *position* drives:

    stiffness 5.4e7 .. 2.9e8,  damping 2.2e4 .. 1.1e5,
    maxForce  3.4e38 (FLT_MAX, i.e. unlimited),
    physxJoint:maxJointVelocity 3.4e38 (unlimited)

so every joint behaves as an infinitely strong position servo.  That is the
reason a velocity/torque command appears to do "nothing": the servo fights it.
A hydraulic excavator is velocity-commanded through its spool valves, so here
the drives are redeclared as pure velocity drives (stiffness = 0) with finite
damping and *realistic* effort ceilings derived from 36 t-class digging forces.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .excavator_params import (
    ALL_JOINTS,
    BASE_HEIGHT,
    BOOM_JOINT,
    BUCKET_JOINT,
    DEFAULT_JOINT_POS,
    EFFORT_LIMITS,
    STICK_JOINT,
    SWING_JOINT,
    VELOCITY_DRIVE_DAMPING,
    VELOCITY_LIMITS,
)


def excavator_cfg(
    usd_path: str,
    prim_path: str = "{ENV_REGEX_NS}/Excavator",
    fix_base: bool = True,
    self_collision: bool = False,
) -> ArticulationCfg:
    """Build the excavator :class:`ArticulationCfg`.

    Args:
        usd_path: path to the converted excavator USD.  Either the importer
            output ``MathScavator9000_flat.SLDASM.usd`` or the USD produced by
            ``scripts/convert_urdf.py``.
        prim_path: prim path pattern for the articulation.
        fix_base: keep the undercarriage welded to the world.  True for digging
            (the machine does not travel); set False only if you add track
            joints, which this CAD does not have.
        self_collision: excavator links overlap by design at the joints, so
            self-collision stays off.
    """
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=self_collision,
                # heavy links + large lever arms: the solver needs the iterations
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                fix_root_link=fix_base,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, BASE_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=dict(DEFAULT_JOINT_POS),
            joint_vel={j: 0.0 for j in ALL_JOINTS},
        ),
        actuators={
            # one actuator group per joint so each keeps its own hydraulic
            # ceiling -- a single shared group would flatten them.
            "swing": ImplicitActuatorCfg(
                joint_names_expr=[SWING_JOINT],
                effort_limit_sim=EFFORT_LIMITS[SWING_JOINT],
                velocity_limit_sim=VELOCITY_LIMITS[SWING_JOINT],
                stiffness=0.0,                                   # velocity drive
                damping=VELOCITY_DRIVE_DAMPING[SWING_JOINT],
            ),
            "boom": ImplicitActuatorCfg(
                joint_names_expr=[BOOM_JOINT],
                effort_limit_sim=EFFORT_LIMITS[BOOM_JOINT],
                velocity_limit_sim=VELOCITY_LIMITS[BOOM_JOINT],
                stiffness=0.0,
                damping=VELOCITY_DRIVE_DAMPING[BOOM_JOINT],
            ),
            "stick": ImplicitActuatorCfg(
                joint_names_expr=[STICK_JOINT],
                effort_limit_sim=EFFORT_LIMITS[STICK_JOINT],
                velocity_limit_sim=VELOCITY_LIMITS[STICK_JOINT],
                stiffness=0.0,
                damping=VELOCITY_DRIVE_DAMPING[STICK_JOINT],
            ),
            "bucket": ImplicitActuatorCfg(
                joint_names_expr=[BUCKET_JOINT],
                effort_limit_sim=EFFORT_LIMITS[BUCKET_JOINT],
                velocity_limit_sim=VELOCITY_LIMITS[BUCKET_JOINT],
                stiffness=0.0,
                damping=VELOCITY_DRIVE_DAMPING[BUCKET_JOINT],
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
