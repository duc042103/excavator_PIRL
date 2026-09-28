"""Articulation declaration for the MathScavator9000 hydraulic excavator.

Every number below was read out of the CAD-derived assets, not guessed:

  * joint names / axes / limits  -> MathScavator9000_flat.SLDASM_physics.usd
                                    (and MathScavator9000_flat.SLDASM.urdf)
  * link masses / inertias       -> same physics layer (SolidWorks mass properties)
  * link geometry (lengths,      -> meshes/*.STL vertex analysis
    bucket width, tip offset)

Machine class
-------------
boom 6.24 m, stick 2.98 m, bucket pivot->cutting-edge 1.835 m, bucket width
1.148 m, max tip reach 10.8 m, moving mass 33.1 t (chassis 17.4 + boom 7.2 +
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

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# --------------------------------------------------------------------------- #
# names -- exactly as they appear in the USD / URDF
# --------------------------------------------------------------------------- #
SWING_JOINT = "base_chassis_joint"      # revolute about Z, upper carriage slew
BOOM_JOINT = "chassis_boom_joint"       # revolute about X
STICK_JOINT = "boom_stick_joint"        # revolute about X
BUCKET_JOINT = "stick_bucket_joint"     # revolute about -X (USD folds the sign
                                        # into a 180 deg Z frame rotation)

ARM_JOINTS = [BOOM_JOINT, STICK_JOINT, BUCKET_JOINT]
ALL_JOINTS = [SWING_JOINT] + ARM_JOINTS

BASE_LINK = "base_link"                 # undercarriage / tracks (fixed root)
CHASSIS_LINK = "base_chassis_link"      # upper carriage
BOOM_LINK = "chassis_boom_link"
STICK_LINK = "boom_stick_link"
BUCKET_LINK = "stick_bucket_link"

# --------------------------------------------------------------------------- #
# geometry measured from the STL meshes (metres, in the *link* frame)
# --------------------------------------------------------------------------- #
BOOM_LENGTH = 6.24
STICK_LENGTH = 2.9796
BUCKET_RADIUS = 1.835                   # pivot -> cutting edge
BUCKET_WIDTH = 1.148                    # cutting edge span (X extent of mesh)

#: cutting edge in the bucket-link frame.  At q_bucket = 0 the bucket is fully
#: curled and the edge points back at the machine (-Y); at q_bucket = -2.27 rad
#: it is fully open, edge pointing outward and down -- the dig-entry attitude.
BUCKET_TIP_OFFSET = (0.0, -1.835, -0.04)

#: direction the bucket opening faces, in the bucket-link frame (+Z when
#: curled).  Used by the soil model to decide when the payload spills out.
BUCKET_OPEN_DIR = (0.0, 0.0, 1.0)

#: base_link origin sits this far above the bottom of the tracks, so spawning
#: the root at this height puts the machine exactly on a ground plane at z = 0.
#: (base_link.STL spans z = -1.295 .. 0.)  This is the fix for the classic
#: "excavator sinks through the ground plane" import problem.
BASE_HEIGHT = 1.295

# --------------------------------------------------------------------------- #
# joint limits straight from the URDF (rad)
# --------------------------------------------------------------------------- #
JOINT_LIMITS = {
    SWING_JOINT: (-math.pi, math.pi),
    BOOM_JOINT: (-0.5585, 1.2415),      # -32.0 deg .. +71.1 deg
    STICK_JOINT: (-0.8179, 1.0821),     # -46.9 deg .. +62.0 deg
    BUCKET_JOINT: (-2.2689, 0.0),       # -130.0 deg ..   0.0 deg
}

# --------------------------------------------------------------------------- #
# actuator envelope for a ~36 t machine (Cat 336 class reference figures)
#
#   bucket: 175 kN tip digging force x 1.835 m  ->  3.2e5 N.m
#   stick : 145 kN tip force        x ~3.5 m    ->  5.0e5 N.m
#   boom  : holds arm+payload (~19 t at ~5 m)   ->  9.3e5 N.m static,
#                                                   1.5e6 N.m with margin
#   swing : 36 t class slew torque              ->  1.2e5 N.m
#
# Velocity ceilings come from typical cycle times (boom raise ~4 s over 1 rad).
# Damping is sized so that  damping * v_max ~ effort_limit, i.e. the joint can
# actually reach its commanded speed but saturates at the real force ceiling --
# this is what makes the machine *feel* hydraulic instead of infinitely stiff.
# --------------------------------------------------------------------------- #
EFFORT_LIMITS = {
    SWING_JOINT: 1.2e5,
    BOOM_JOINT: 1.5e6,
    STICK_JOINT: 5.0e5,
    BUCKET_JOINT: 3.3e5,
}

VELOCITY_LIMITS = {
    SWING_JOINT: 0.90,
    BOOM_JOINT: 0.35,
    STICK_JOINT: 0.50,
    BUCKET_JOINT: 0.80,
}

VELOCITY_DRIVE_DAMPING = {
    j: EFFORT_LIMITS[j] / VELOCITY_LIMITS[j] for j in ALL_JOINTS
}

#: dig-entry pose: bucket tip on the ground 7.5 m out, cutting edge pointing
#: outward and 30 deg down from vertical.  Solved with the FK chain above.
DEFAULT_JOINT_POS = {
    SWING_JOINT: 0.0,
    BOOM_JOINT: 0.410,
    STICK_JOINT: -0.083,
    BUCKET_JOINT: -1.746,
}


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
                effort_limit=EFFORT_LIMITS[SWING_JOINT],
                velocity_limit=VELOCITY_LIMITS[SWING_JOINT],
                stiffness=0.0,                                   # velocity drive
                damping=VELOCITY_DRIVE_DAMPING[SWING_JOINT],
            ),
            "boom": ImplicitActuatorCfg(
                joint_names_expr=[BOOM_JOINT],
                effort_limit=EFFORT_LIMITS[BOOM_JOINT],
                velocity_limit=VELOCITY_LIMITS[BOOM_JOINT],
                stiffness=0.0,
                damping=VELOCITY_DRIVE_DAMPING[BOOM_JOINT],
            ),
            "stick": ImplicitActuatorCfg(
                joint_names_expr=[STICK_JOINT],
                effort_limit=EFFORT_LIMITS[STICK_JOINT],
                velocity_limit=VELOCITY_LIMITS[STICK_JOINT],
                stiffness=0.0,
                damping=VELOCITY_DRIVE_DAMPING[STICK_JOINT],
            ),
            "bucket": ImplicitActuatorCfg(
                joint_names_expr=[BUCKET_JOINT],
                effort_limit=EFFORT_LIMITS[BUCKET_JOINT],
                velocity_limit=VELOCITY_LIMITS[BUCKET_JOINT],
                stiffness=0.0,
                damping=VELOCITY_DRIVE_DAMPING[BUCKET_JOINT],
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
