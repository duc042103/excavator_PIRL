#!/usr/bin/env python3
"""Repair the SolidWorks-exported excavator URDF and convert it to USD.

Three things are wrong with ``MathScavator9000_flat.SLDASM.urdf`` as exported:

1. mesh paths use ``package://MathScavator9000_flat.SLDASM/meshes/...`` which
   resolves only inside a ROS workspace;
2. every joint has ``effort="0" velocity="0"`` -- the exporter never fills
   these in, and Isaac Sim then invents its own (unlimited) values;
3. no joint damping/friction at all.

This script rewrites those, then runs the Isaac Lab URDF converter with
*velocity* drives, so the resulting articulation is commanded the way a
hydraulic machine actually is.

Usage
-----
    # full conversion (needs Isaac Sim)
    ./isaaclab.sh -p scripts/convert_urdf.py \
        --input  /path/to/MathScavator9000_flat.SLDASM.urdf \
        --output ~/excavator_assets/excavator.usd

    # only rewrite the URDF (plain python, no Isaac Sim needed)
    python scripts/convert_urdf.py --input ... --output ... --fix-only
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from excavator_rl.excavator_cfg import (  # noqa: E402
    EFFORT_LIMITS,
    JOINT_LIMITS,
    VELOCITY_DRIVE_DAMPING,
    VELOCITY_LIMITS,
)

#: viscous damping written into the URDF, ~2 % of the drive damping: enough to
#: keep the solver calm without making the joints feel like treacle.
JOINT_DAMPING_FRACTION = 0.02
JOINT_FRICTION = {
    "base_chassis_joint": 5.0e3,
    "chassis_boom_joint": 8.0e3,
    "boom_stick_joint": 4.0e3,
    "bucket_joint": 2.0e3,
    "stick_bucket_joint": 2.0e3,
}


def fix_urdf(src: str, dst_dir: str) -> str:
    """Rewrite mesh URIs and joint limits.  Returns the path of the new URDF."""
    src = os.path.abspath(src)
    src_dir = os.path.dirname(src)
    os.makedirs(dst_dir, exist_ok=True)

    tree = ET.parse(src)
    root = tree.getroot()

    # ---- 1. mesh paths -------------------------------------------------- #
    mesh_dir_candidates = [
        os.path.join(src_dir, "meshes"),
        os.path.join(os.path.dirname(src_dir), "meshes"),
    ]
    mesh_dir = next((d for d in mesh_dir_candidates if os.path.isdir(d)), None)
    if mesh_dir is None:
        raise FileNotFoundError(
            f"no meshes/ directory next to {src} (looked in {mesh_dir_candidates})"
        )

    n_mesh = 0
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        if fn.startswith("package://"):
            base = os.path.basename(fn)
            abs_path = os.path.join(mesh_dir, base)
            if not os.path.isfile(abs_path):
                raise FileNotFoundError(f"{abs_path} referenced by the URDF is missing")
            mesh.set("filename", abs_path)
            n_mesh += 1

    # ---- 2. joint limits + dynamics ------------------------------------- #
    n_joint = 0
    for joint in root.iter("joint"):
        name = joint.get("name")
        if joint.get("type") not in ("revolute", "prismatic", "continuous"):
            continue
        if name not in EFFORT_LIMITS:
            print(f"  ! joint '{name}' is not in the actuator table, left untouched")
            continue

        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        limit.set("effort", f"{EFFORT_LIMITS[name]:.6g}")
        limit.set("velocity", f"{VELOCITY_LIMITS[name]:.6g}")
        lo, hi = JOINT_LIMITS[name]
        limit.set("lower", f"{lo:.6g}")
        limit.set("upper", f"{hi:.6g}")

        dyn = joint.find("dynamics")
        if dyn is None:
            dyn = ET.SubElement(joint, "dynamics")
        dyn.set("damping", f"{VELOCITY_DRIVE_DAMPING[name] * JOINT_DAMPING_FRACTION:.6g}")
        dyn.set("friction", f"{JOINT_FRICTION.get(name, 1.0e3):.6g}")
        n_joint += 1

    out = os.path.join(dst_dir, "excavator_fixed.urdf")
    tree.write(out, encoding="utf-8", xml_declaration=True)

    # keep a copy of the meshes next to the fixed URDF so the asset is portable
    local_meshes = os.path.join(dst_dir, "meshes")
    if os.path.abspath(mesh_dir) != os.path.abspath(local_meshes):
        os.makedirs(local_meshes, exist_ok=True)
        for f in os.listdir(mesh_dir):
            if f.lower().endswith((".stl", ".obj", ".dae")):
                shutil.copy2(os.path.join(mesh_dir, f), os.path.join(local_meshes, f))

    print(f"  fixed {n_mesh} mesh paths and {n_joint} joints -> {out}")
    return out


def convert(fixed_urdf: str, output_usd: str) -> None:
    """Run the Isaac Lab URDF -> USD converter (import after app launch)."""
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    usd_dir = os.path.dirname(os.path.abspath(output_usd))
    usd_name = os.path.basename(output_usd)
    os.makedirs(usd_dir, exist_ok=True)

    common = dict(
        asset_path=fixed_urdf,
        usd_dir=usd_dir,
        usd_file_name=usd_name,
        fix_base=True,
        merge_fixed_joints=False,
        force_usd_conversion=True,
    )

    try:
        # Isaac Lab >= 2.0: structured joint-drive config
        drive = UrdfConverterCfg.JointDriveCfg(
            target_type="velocity",
            drive_type="force",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=None),
        )
        cfg = UrdfConverterCfg(joint_drive=drive, **common)
    except AttributeError:
        # Isaac Lab 1.x
        cfg = UrdfConverterCfg(
            default_drive_type="velocity",
            default_drive_stiffness=0.0,
            default_drive_damping=1.0e5,
            **common,
        )

    converter = UrdfConverter(cfg)
    print(f"  USD written to {converter.usd_path}")
    print(
        "\nNote: the actuator ceilings that matter at runtime come from "
        "excavator_rl/excavator_cfg.py (ImplicitActuatorCfg), which overrides "
        "whatever the converter wrote into the USD drives."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="path to the exported .urdf")
    parser.add_argument("--output", required=True, help="path of the .usd to write")
    parser.add_argument("--fix-only", action="store_true", help="skip the USD conversion")
    parser.add_argument("--headless", action="store_true", default=True)
    args, _ = parser.parse_known_args()

    work_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)), "urdf_fixed")
    print(f"[1/2] repairing URDF\n  source: {args.input}")
    fixed = fix_urdf(args.input, work_dir)

    if args.fix_only:
        print("[2/2] skipped (--fix-only)")
        return

    print("[2/2] converting to USD (starting Isaac Sim, this takes a minute)")
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        convert(fixed, args.output)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
