#!/usr/bin/env python3
"""Soil-model tests.  Plain torch -- no Isaac Sim needed.

    python tests/test_soil.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from excavator_rl.soil import SoilCfg, SoilModel  # noqa: E402

DT = 1.0 / 30.0
OK, FAIL = "  ok  ", " FAIL "
_failures = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{OK if cond else FAIL}] {name}{('  -- ' + detail) if detail else ''}")
    if not cond:
        _failures.append(name)


def make(n: int = 4) -> SoilModel:
    soil = SoilModel(SoilCfg(), n, "cpu")
    soil.reset(randomize=False)
    return soil


def up(n: int) -> torch.Tensor:
    d = torch.zeros(n, 3)
    d[:, 2] = 1.0
    return d


def test_no_contact_above_surface():
    soil = make()
    tip = torch.tensor([[0.0, 7.0, 0.5]]).repeat(4, 1)
    vel = torch.tensor([[0.0, -1.0, 0.0]]).repeat(4, 1)
    out = soil.step(tip, vel, up(4), DT)
    check("no force above the surface", torch.allclose(out["force"], torch.zeros(4, 3)))
    check("no fill above the surface", float(soil.fill.max()) == 0.0)


def test_resistance_grows_with_depth():
    mags = []
    for z in (-0.05, -0.2, -0.5):
        soil = make(1)
        tip = torch.tensor([[0.0, 7.0, z]])
        vel = torch.tensor([[0.0, -0.5, 0.0]])
        out = soil.step(tip, vel, up(1), DT)
        mags.append(float(torch.linalg.norm(out["force"][0])))
    check("resistance increases with depth", mags[0] < mags[1] < mags[2], f"{[round(m) for m in mags]} N")
    check("resistance stays in a sane range", 1e3 < mags[-1] < 6.1e5, f"{mags[-1]:.0f} N")


def test_force_opposes_motion():
    soil = make(1)
    tip = torch.tensor([[0.0, 7.0, -0.3]])
    vel = torch.tensor([[0.0, -1.0, -0.2]])
    out = soil.step(tip, vel, up(1), DT)
    f = out["force"][0]
    check("force opposes the tool velocity", float(torch.dot(f, vel[0])) < 0.0)


def test_drag_fills_the_bucket():
    soil = make(1)
    z, fill_hist = -0.35, []
    for k in range(120):                      # 4 s of dragging inwards
        r = 8.0 - 0.04 * k
        tip = torch.tensor([[0.0, r, z]])
        vel = torch.tensor([[0.0, -1.2, 0.0]])
        soil.step(tip, vel, up(1), DT)
        fill_hist.append(float(soil.fill[0]))
    check("bucket fills while dragging", fill_hist[-1] > 0.3, f"{fill_hist[-1]:.2f} m^3")
    check("fill is monotone", all(b >= a - 1e-9 for a, b in zip(fill_hist, fill_hist[1:])))
    check("fill never exceeds capacity", fill_hist[-1] <= soil.cfg.capacity + 1e-9)
    check("soil was removed from the bed", float(soil.moved_volume()[0]) > 0.3,
          f"{float(soil.moved_volume()[0]):.2f} m^3")


def test_carving_is_idempotent():
    """Re-sweeping the same trench must not create material out of nothing.

    Slumping is switched off here: with it, the trench walls collapse into the
    trench and a second pass legitimately picks that material up (covered by
    test_volume_is_conserved_with_slumping)."""
    soil = SoilModel(SoilCfg(repose_iters=0), 1, "cpu")
    soil.reset(randomize=False)
    for _ in range(2):
        for k in range(60):
            tip = torch.tensor([[0.0, 8.0 - 0.05 * k, -0.3]])
            soil.step(tip, torch.tensor([[0.0, -1.0, 0.0]]), up(1), DT)
    first_pass_volume = float(soil.moved_volume()[0])
    before = float(soil.fill[0])
    for k in range(60):                        # third identical pass
        tip = torch.tensor([[0.0, 8.0 - 0.05 * k, -0.3]])
        soil.step(tip, torch.tensor([[0.0, -1.0, 0.0]]), up(1), DT)
    check("an already-cut trench yields nothing more", abs(float(soil.fill[0]) - before) < 1e-6)
    check("moved volume is conserved", abs(float(soil.moved_volume()[0]) - first_pass_volume) < 1e-6)


def test_volume_is_conserved_with_slumping():
    """Bed + bucket volume stays constant through digging, slumping and spilling."""
    cfg = SoilCfg(fill_efficiency=1.0, capacity=50.0)
    soil = SoilModel(cfg, 1, "cpu")
    soil.reset(randomize=False)
    bed = lambda: float(soil.height.sum()) * soil.dr * cfg.bucket_width  # noqa: E731
    total0 = bed()
    for _ in range(3):
        for k in range(60):
            soil.step(torch.tensor([[0.0, 8.0 - 0.05 * k, -0.4]]), torch.tensor([[0.0, -1.5, 0.0]]), up(1), DT)
    tipped = torch.tensor([[0.0, -1.0, -0.2]]) / torch.linalg.norm(torch.tensor([0.0, -1.0, -0.2]))
    for _ in range(20):                                    # dump part of it back, out of the soil
        soil.step(torch.tensor([[0.0, 7.0, 2.0]]), torch.zeros(1, 3), tipped, DT)
    err = abs(bed() + float(soil.fill[0]) - total0)
    check("bed + bucket volume conserved", err < 1e-3, f"error {err:.2e} m^3, fill {float(soil.fill[0]):.2f} m^3")


def test_cut_depth_is_measured_against_undisturbed_soil():
    """Dragging at constant depth keeps meeting the full cut thickness."""
    soil = make(1)
    depths = []
    for k in range(60):
        out = soil.step(torch.tensor([[0.0, 8.0 - 0.02 * k, -0.5]]), torch.tensor([[0.0, -0.6, 0.0]]), up(1), DT)
        depths.append(float(out["depth"][0]))
    check("cut depth stays at the blade depth while dragging", min(depths[5:]) > 0.45,
          f"min {min(depths[5:]):.2f} m (blade 0.50 m deep)")


def test_no_spill_while_cutting():
    soil = make(1)
    soil.fill[0] = 0.8
    facing_cab = torch.tensor([[0.0, -1.0, 0.0]])                 # opening towards the machine
    for k in range(30):
        soil.step(torch.tensor([[0.0, 8.0 - 0.02 * k, -0.4]]), torch.tensor([[0.0, -0.6, 0.0]]), facing_cab, DT)
    check("payload stays in while the teeth are in the soil", float(soil.fill[0]) >= 0.8)


def test_spill():
    soil = make(1)
    soil.fill[0] = 1.0
    tipped = torch.tensor([[0.0, -0.95, -0.3]])       # opening rotated ~108 deg from up
    tipped = tipped / torch.linalg.norm(tipped)
    for _ in range(30):
        soil.step(torch.tensor([[0.0, 7.0, 3.0]]), torch.zeros(1, 3), tipped, DT)
    check("payload spills when the bucket is rolled out", float(soil.fill[0]) < 0.4,
          f"{float(soil.fill[0]):.3f} m^3 left")

    soil2 = make(1)
    soil2.fill[0] = 1.0
    for _ in range(30):
        soil2.step(torch.tensor([[0.0, 7.0, 3.0]]), torch.zeros(1, 3), up(1), DT)
    check("payload is kept when the bucket is curled", abs(float(soil2.fill[0]) - 1.0) < 1e-6)


def test_batched_envs_are_independent():
    soil = make(8)
    tip = torch.zeros(8, 3)
    tip[:, 1] = 7.0
    tip[:, 2] = -0.3
    tip[4:, 2] = 2.0                                  # half of them in the air
    vel = torch.zeros(8, 3)
    vel[:, 1] = -1.0
    for _ in range(40):
        soil.step(tip, vel, up(8), DT)
    check("digging envs filled", bool((soil.fill[:4] > 0.05).all()))
    check("airborne envs stayed empty", float(soil.fill[4:].max()) == 0.0)

    soil.reset(torch.tensor([0, 1]), randomize=False)
    check("reset clears only the selected envs",
          float(soil.fill[0]) == 0.0 and float(soil.fill[2]) > 0.0)


def test_reset_randomisation():
    soil = SoilModel(SoilCfg(), 64, "cpu")
    soil.reset(randomize=True)
    check("randomised beds differ", float(soil.height[:, 0].std()) > 1e-3)
    check("cohesion randomised", float(soil.cohesion.std()) > 1.0)
    check("heights stay in range",
          bool((soil.height > -1.0).all() and (soil.height < 1.5).all()))


def test_no_nans():
    soil = make(16)
    torch.manual_seed(0)
    for _ in range(200):
        tip = torch.randn(16, 3) * torch.tensor([1.0, 3.0, 1.0]) + torch.tensor([0.0, 7.0, -0.2])
        vel = torch.randn(16, 3) * 2.0
        d = torch.randn(16, 3)
        d = d / torch.linalg.norm(d, dim=1, keepdim=True)
        out = soil.step(tip, vel, d, DT)
        if not torch.isfinite(out["force"]).all() or not torch.isfinite(soil.fill).all():
            check("random inputs stay finite", False)
            return
    check("random inputs stay finite", True)
    check("fill stays inside [0, capacity]",
          bool((soil.fill >= -1e-9).all() and (soil.fill <= soil.cfg.capacity + 1e-9).all()))


if __name__ == "__main__":
    for fn in [
        test_no_contact_above_surface,
        test_resistance_grows_with_depth,
        test_force_opposes_motion,
        test_drag_fills_the_bucket,
        test_carving_is_idempotent,
        test_volume_is_conserved_with_slumping,
        test_cut_depth_is_measured_against_undisturbed_soil,
        test_no_spill_while_cutting,
        test_spill,
        test_batched_envs_are_independent,
        test_reset_randomisation,
        test_no_nans,
    ]:
        print(f"\n{fn.__name__}")
        fn()

    print("\n" + "=" * 52)
    if _failures:
        print(f"{len(_failures)} FAILED: {_failures}")
        sys.exit(1)
    print("all soil-model tests passed")
