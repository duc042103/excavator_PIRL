#!/usr/bin/env python3
"""Dump the physics content of a USD -- joints, drives, limits, masses.

Runs on plain ``usd-core`` (``pip install usd-core``); no Isaac Sim needed, so
you can check an asset on any machine.

    python tools/inspect_usd.py excavator.usd
    python tools/inspect_usd.py MathScavator9000_flat.SLDASM/configuration/*_physics.usd
"""

from __future__ import annotations

import math
import sys

from pxr import Usd, UsdGeom, UsdPhysics

FLT_MAX = 3.4028235e38


def dump(path: str) -> None:
    print("=" * 78)
    print("FILE:", path)
    stage = Usd.Stage.Open(path)
    if stage is None:
        print("  cannot open")
        return

    default = stage.GetDefaultPrim()
    print(f"  defaultPrim = {default.GetPath() if default else None}")
    print(f"  upAxis = {UsdGeom.GetStageUpAxis(stage)}   metersPerUnit = {UsdGeom.GetStageMetersPerUnit(stage)}")

    bodies, joints = [], []
    # TraverseAll(), not Traverse(): the Isaac Sim importer writes its physics
    # into a layer of `over` prims, and the default predicate refuses to
    # descend into those when the base layer is not alongside.
    for prim in stage.TraverseAll():
        if prim.HasAPI(UsdPhysics.MassAPI):
            bodies.append(prim)
        if prim.IsA(UsdPhysics.Joint):
            joints.append(prim)

    if bodies:
        print("\n  LINKS")
        total = 0.0
        for prim in bodies:
            m = UsdPhysics.MassAPI(prim)
            mass = m.GetMassAttr().Get() or 0.0
            total += mass
            print(f"    {prim.GetName():<24} mass = {mass:>12.1f} kg   "
                  f"inertia = {m.GetDiagonalInertiaAttr().Get()}")
        print(f"    {'TOTAL':<24}        {total:>12.1f} kg")

    if joints:
        print("\n  JOINTS")
        for prim in joints:
            j = UsdPhysics.Joint(prim)
            kind = prim.GetTypeName()
            print(f"    {prim.GetName():<24} <{kind}>")
            b0 = j.GetBody0Rel().GetTargets()
            b1 = j.GetBody1Rel().GetTargets()
            print(f"      parent={[str(p) for p in b0]}  child={[str(p) for p in b1]}")

            rj = UsdPhysics.RevoluteJoint(prim)
            if rj and rj.GetAxisAttr().Get():
                lo, hi = rj.GetLowerLimitAttr().Get(), rj.GetUpperLimitAttr().Get()
                if lo is not None and hi is not None:
                    print(f"      axis={rj.GetAxisAttr().Get()}  "
                          f"limits = {lo:.2f} .. {hi:.2f} deg "
                          f"({math.radians(lo):.3f} .. {math.radians(hi):.3f} rad)")

            for dof in ("angular", "linear"):
                d = UsdPhysics.DriveAPI.Get(prim, dof)
                if not d:
                    continue
                k = d.GetStiffnessAttr().Get()
                c = d.GetDampingAttr().Get()
                fmax = d.GetMaxForceAttr().Get()
                note = ""
                if k and k > 1e6:
                    note += "  <-- huge stiffness: this is a rigid POSITION servo"
                if fmax and fmax >= FLT_MAX * 0.99:
                    note += "  <-- unlimited force"
                print(f"      drive[{dof}] type={d.GetTypeAttr().Get()} "
                      f"stiffness={k:.4g} damping={c:.4g} maxForce={fmax:.4g}{note}")

            v = prim.GetAttribute("physxJoint:maxJointVelocity")
            if v and v.Get() is not None:
                val = v.Get()
                tag = "  <-- unlimited" if val >= FLT_MAX * 0.99 else ""
                print(f"      maxJointVelocity = {val:.4g}{tag}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    for p in sys.argv[1:]:
        dump(p)
