"""Excavator constants: joint names, limits, actuator envelope, geometry.

Pure python on purpose -- no Isaac Lab import -- so that
``scripts/convert_urdf.py --fix-only`` and the unit tests can use these values
on a machine without Isaac Sim, and so that importing them never happens
"too early" (Isaac Lab modules may only be imported after the app launched).

See :mod:`excavator_rl.excavator_cfg` for where every number comes from.
"""

from __future__ import annotations

import math

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
#
# The drives are pure velocity drives (stiffness 0), so PhysX applies
#     tau = damping * (v_cmd - v),   clipped to +-effort_limit.
# Under a steady load tau_load the joint therefore creeps at
#     v_err = tau_load / damping.
# With damping = effort / v_max (the first version of this file) the boom,
# which carries ~60 % of its effort ceiling in gravity load alone, sagged at
# ~60 % of its top speed while commanded to hold still.  A real spool valve
# holds the load, so the damping is raised by VELOCITY_TRACKING_GAIN: the
# creep drops to a few percent of v_max, while the effort ceiling still caps
# the force -- the machine stalls against hard soil instead of being
# infinitely strong.
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

VELOCITY_TRACKING_GAIN = 10.0
"""How many times stiffer than ``effort / v_max`` the velocity loop is."""

VELOCITY_DRIVE_DAMPING = {
    j: VELOCITY_TRACKING_GAIN * EFFORT_LIMITS[j] / VELOCITY_LIMITS[j] for j in ALL_JOINTS
}

#: dig-entry pose: bucket tip on the ground 7.5 m out, cutting edge pointing
#: outward and 30 deg down from vertical.  Solved with the FK chain above.
DEFAULT_JOINT_POS = {
    SWING_JOINT: 0.0,
    BOOM_JOINT: 0.410,
    STICK_JOINT: -0.083,
    BUCKET_JOINT: -1.746,
}
