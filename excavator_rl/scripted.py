"""Scripted digging "expert": planar kinematics of the arm + resolved-rate IK.

Used by ``scripts/check_model.py`` (Isaac Sim) and ``tools/pybullet_twin.py``
(PyBullet) to prove the task is solvable before any RL: it bites into the
soil, drags the bucket towards the cab at constant depth, curls it and lifts
it past the target height.  Pure torch, no Isaac Lab import.
"""

from __future__ import annotations

import math
import os

import torch

from .excavator_params import BASE_HEIGHT, BUCKET_TIP_OFFSET

#: the URDF shipped in assets/, used for the joint origins
DEFAULT_URDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "MathScavator9000_flat", "urdf", "MathScavator9000_flat.SLDASM.urdf",
)


class ArmKinematics:
    """Boom / stick / bucket in the digging (y, z) plane, from the URDF joint origins."""

    def __init__(self, urdf: str = DEFAULT_URDF):
        import xml.etree.ElementTree as ET

        joints = {j.get("name"): j for j in ET.parse(urdf).getroot().iter("joint")}

        def origin(name):
            return torch.tensor([float(v) for v in joints[name].find("origin").get("xyz").split()])

        self.o_boom = origin("chassis_boom_joint") + torch.tensor([0.0, 0.0, BASE_HEIGHT])
        self.o_stick, self.o_bucket = origin("boom_stick_joint"), origin("stick_bucket_joint")
        self.axis_bucket = float(joints["stick_bucket_joint"].find("axis").get("xyz").split()[0])
        self.tip = torch.tensor(BUCKET_TIP_OFFSET)

    @staticmethod
    def _rx(a):
        c, s = math.cos(a), math.sin(a)
        return torch.tensor([[1.0, 0, 0], [0, c, -s], [0, s, c]])

    def forward(self, q) -> torch.Tensor:
        """q = (boom, stick, bucket) -> (tooth y, tooth z, bucket pitch phi)."""
        qb, qs, qk = (float(v) for v in q)
        R = self._rx(qb)
        p = self.o_boom + R @ self.o_stick
        R = R @ self._rx(qs)
        p = p + R @ self.o_bucket
        phi = qb + qs + self.axis_bucket * qk
        tip = p + self._rx(phi) @ self.tip
        return torch.tensor([tip[1], tip[2], phi])

    def jacobian(self, q, eps=1e-4) -> torch.Tensor:
        q = torch.as_tensor(q, dtype=torch.float64)
        f0 = self.forward(q)
        J = torch.zeros(3, 3)
        for i in range(3):
            dq = q.clone()
            dq[i] += eps
            J[:, i] = (self.forward(dq) - f0) / eps
        return J


class ScriptedDigger:
    """Tracks a tooth-tip / bucket-pitch trajectory with resolved-rate IK.

    Waypoints (time [s], tooth y [m], tooth z [m], bucket pitch phi [rad]):
    bite in, drag towards the cab at constant depth while curling, close the
    bucket (opening up), then lift it.
    """

    WAYPOINTS = [
        (1.5, 7.40, -0.55, 2.25, "penetrate"),
        (5.5, 5.20, -0.55, 1.50, "drag"),
        (7.5, 5.40, 0.30, 0.59, "curl"),
        (11.5, 5.20, 4.90, 0.59, "lift"),
    ]

    def __init__(self, kin: ArmKinematics, vmax: torch.Tensor, gain: float = 2.0):
        self.kin, self.vmax, self.gain = kin, vmax, gain
        self.start = None

    def phase(self, t: float) -> str:
        for wp in self.WAYPOINTS:
            if t < wp[0]:
                return wp[4]
        return "hold"

    def reference(self, t: float) -> torch.Tensor:
        pts = [(0.0, *self.start.tolist())] + [wp[:4] for wp in self.WAYPOINTS]
        for (t0, *a), (t1, *b) in zip(pts, pts[1:]):
            if t <= t1:
                w = (t - t0) / (t1 - t0)
                return torch.tensor([ai + w * (bi - ai) for ai, bi in zip(a, b)])
        return torch.tensor(pts[-1][1:])

    def action(self, t: float, q_arm: torch.Tensor) -> torch.Tensor:
        x = self.kin.forward(q_arm)
        if self.start is None:
            self.start = x.clone()
        err = self.reference(t) - x
        qd = torch.linalg.solve(self.kin.jacobian(q_arm), self.gain * err)
        a = qd / self.vmax
        return a / max(1.0, float(a.abs().max()))  # keep the direction when saturating
