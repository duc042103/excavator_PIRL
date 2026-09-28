"""Analytic soil / digging model, fully vectorised in torch.

Isaac Sim has no built-in soil.  Particle (PBD) or deformable-body soil exists
but costs far too much to run thousands of environments in parallel, and the
imported bucket is a *convex hull* collider anyway, so it could never scoop
particles geometrically.  vortexRL got its soil from Vortex Studio's hybrid
particle/mesh solver; the equivalent here is an analytic model, which is what
almost every large-scale excavation-RL paper uses:

1.  **Terrain** -- a 1-D height profile h(r) along the digging plane, r being
    the radial distance from the swing axis.  Cheap, and the arm only ever digs
    in that plane (swing is locked for the digging task, exactly as in
    vortexRL, whose agent commands only boom / stick / bucket).

2.  **Resistance** -- the Fundamental Earthmoving Equation (Reece 1964),

        F = w * ( rho * g * d^2 * N_gamma  +  c * d * N_c  +  q * d * N_q )

    with an added viscous term proportional to tool speed.  `d` is the blade
    penetration depth, `w` the bucket width.  The force opposes the tip
    velocity and is applied at the cutting edge, so it produces the correct
    reaction torque on stick and boom as well.

    The cutting depth ``d`` is measured against the *undisturbed* soil around
    the blade (the highest cell within ``carve_halfwidth``), i.e. the soil the
    edge is about to cut, not the trench it has already left behind.

3.  **Material transfer** -- swept-min carving.  Cells the cutting edge swept
    through since the previous step are lowered to the edge's height; the
    displaced volume goes into the bucket up to its rated capacity.
    Geometrically exact and unconditionally stable (no CFL condition, no
    explosion at large dt).

4.  **Spillage** -- when the teeth are out of the soil and the bucket opening
    tips more than `spill_angle` away from vertical, the payload drains out
    with a first-order time constant and is returned to the bed.  While the
    edge is cutting, material is being pressed into the bucket and does not
    flow out (a backhoe drags with its opening facing the cab).

The model is deliberately free of Isaac Lab imports so it can be unit-tested on
a laptop with nothing but torch -- see ``tests/test_soil.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SoilCfg:
    """Soil bed and material parameters (SI units)."""

    # ---- bed geometry -----------------------------------------------------
    r_min: float = 3.0
    """Inner radius of the soil bed, measured from the swing axis [m]."""
    r_max: float = 11.0
    """Outer radius of the soil bed [m].  The teeth reach 10.9 m at ground level."""
    num_cells: int = 160
    """Number of radial cells.  (r_max - r_min) / num_cells = 5 cm here."""
    base_height: float = 0.0
    """Nominal soil surface height [m], world z (ground plane is z = 0)."""

    # ---- randomisation ----------------------------------------------------
    height_range: tuple[float, float] = (-0.10, 0.35)
    """Per-episode uniform offset applied to the whole bed [m]."""
    slope_range: tuple[float, float] = (-0.04, 0.04)
    """Per-episode surface slope, dz/dr [-]."""

    # ---- material ---------------------------------------------------------
    density: float = 1800.0
    """Bulk density [kg/m^3].  Typical excavated soil: 1500 - 2000."""
    cohesion: float = 8.0e3
    """Cohesion c [Pa].  Sand ~0, stiff clay ~40 kPa."""
    cohesion_range: tuple[float, float] = (2.0e3, 20.0e3)
    """Per-episode randomisation range for cohesion [Pa]."""
    n_gamma: float = 4.0
    """Gravity/weight factor N_gamma of the earthmoving equation [-]."""
    n_c: float = 6.0
    """Cohesion factor N_c [-]."""
    n_q: float = 1.5
    """Surcharge factor N_q [-] (applied to the payload already in the bucket)."""
    viscous: float = 2.0e4
    """Viscous term [N.s/m per m^2 of blade cross-section]."""
    max_force: float = 6.0e5
    """Hard ceiling on the resistance force [N] -- keeps the solver sane."""

    # ---- tool -------------------------------------------------------------
    bucket_width: float = 1.148
    """Cutting edge width [m] (measured from the bucket STL)."""
    capacity: float = 1.5
    """Heaped bucket capacity [m^3].  ~36 t class."""
    fill_efficiency: float = 0.75
    """Fraction of the carved volume that ends up inside the bucket [-]."""
    carve_halfwidth: float = 0.30
    """Radial half-width of the carving footprint around the tip [m]."""
    max_depth: float = 1.20
    """Depth beyond which resistance saturates [m] (blade fully buried)."""

    # ---- spillage ---------------------------------------------------------
    spill_angle: float = 1.05
    """Tilt of the bucket opening from vertical beyond which soil spills [rad]."""
    spill_tau: float = 0.35
    """Time constant of the spill [s]."""

    # ---- pile stability ---------------------------------------------------
    repose_angle: float = 1.05
    """Steepest slope the bed can hold [rad].  60 deg by default: cohesive soil
    stands in a steep face, but dumped material still spreads into a heap
    instead of stacking into a one-cell spike."""
    repose_iters: int = 2
    """Slope-limiter sweeps per control step.  0 disables slumping."""


class SoilModel:
    """Batched soil bed.  One height profile per environment."""

    def __init__(self, cfg: SoilCfg, num_envs: int, device: str | torch.device):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = torch.device(device)

        self.dr = (cfg.r_max - cfg.r_min) / cfg.num_cells
        #: radial centre of every cell, shape (num_cells,)
        self.r_centers = (
            cfg.r_min + (torch.arange(cfg.num_cells, device=self.device) + 0.5) * self.dr
        )

        #: (num_envs, num_cells) soil surface height, world z
        self.height = torch.full(
            (num_envs, cfg.num_cells), cfg.base_height, device=self.device
        )
        #: (num_envs, num_cells) untouched reference profile, for "how much was moved"
        self.height0 = self.height.clone()
        #: (num_envs,) volume currently inside the bucket [m^3]
        self.fill = torch.zeros(num_envs, device=self.device)
        #: (num_envs,) per-episode cohesion [Pa]
        self.cohesion = torch.full((num_envs,), cfg.cohesion, device=self.device)
        #: (num_envs,) penetration depth of the last step [m]
        self.depth = torch.zeros(num_envs, device=self.device)
        #: (num_envs,) radius of the cutting edge at the previous step (NaN = none yet)
        self.prev_r = torch.full((num_envs,), float("nan"), device=self.device)

    # ------------------------------------------------------------------ #
    # episode handling
    # ------------------------------------------------------------------ #
    def reset(self, env_ids: torch.Tensor | None = None, randomize: bool = True) -> None:
        """Rebuild the bed for the given environments."""
        cfg = self.cfg
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = env_ids.numel()
        if n == 0:
            return

        if randomize:
            off = _uniform((n, 1), cfg.height_range, self.device)
            slope = _uniform((n, 1), cfg.slope_range, self.device)
            coh = _uniform((n,), cfg.cohesion_range, self.device)
        else:
            off = torch.zeros((n, 1), device=self.device)
            slope = torch.zeros((n, 1), device=self.device)
            coh = torch.full((n,), cfg.cohesion, device=self.device)

        r = self.r_centers.unsqueeze(0)                       # (1, C)
        profile = cfg.base_height + off + slope * (r - cfg.r_min)

        self.height[env_ids] = profile
        self.height0[env_ids] = profile
        self.fill[env_ids] = 0.0
        self.cohesion[env_ids] = coh
        self.depth[env_ids] = 0.0
        self.prev_r[env_ids] = float("nan")

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    def _cell_index(self, r: torch.Tensor) -> torch.Tensor:
        """Cell index containing radius ``r``, clamped to the bed."""
        idx = ((r - self.cfg.r_min) / self.dr).long()
        return idx.clamp(0, self.cfg.num_cells - 1)

    def surface_height(self, r: torch.Tensor) -> torch.Tensor:
        """Soil surface height at radius ``r``.  Shape (num_envs,)."""
        idx = self._cell_index(r)
        return self.height.gather(1, idx.unsqueeze(1)).squeeze(1)

    @property
    def fill_mass(self) -> torch.Tensor:
        """Payload mass currently in the bucket [kg]."""
        return self.fill * self.cfg.density

    @property
    def fill_ratio(self) -> torch.Tensor:
        return self.fill / self.cfg.capacity

    def moved_volume(self) -> torch.Tensor:
        """Total soil volume displaced from the original profile [m^3]."""
        return ((self.height0 - self.height).clamp(min=0.0)
                * self.dr * self.cfg.bucket_width).sum(dim=1)

    # ------------------------------------------------------------------ #
    # main update
    # ------------------------------------------------------------------ #
    def step(
        self,
        tip_pos: torch.Tensor,
        tip_vel: torch.Tensor,
        open_dir_w: torch.Tensor,
        dt: float,
        in_bed: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Advance the soil one control step.

        Args:
            tip_pos: (N, 3) cutting-edge position in the *environment-local*
                frame (env origin subtracted).
            tip_vel: (N, 3) cutting-edge linear velocity [m/s].
            open_dir_w: (N, 3) unit vector of the bucket opening, world frame.
            dt: control step [s].
            in_bed: (N,) bool -- environments whose bucket is inside the dig
                sector.  ``None`` means all of them (swing locked).

        Returns:
            dict with ``force`` (N, 3) resistance force in world frame to be
            applied at the cutting edge, ``depth`` (N,), ``d_fill`` (N,) volume
            picked up this step, and ``spilled`` (N,).
        """
        cfg = self.cfg
        r = torch.linalg.norm(tip_pos[:, :2], dim=1)
        z = tip_pos[:, 2]

        inside = (r > cfg.r_min) & (r < cfg.r_max)
        if in_bed is not None:
            inside = inside & in_bed

        # cut thickness: undisturbed soil around the blade (the highest cell
        # within carve_halfwidth), not the trench already carved behind it
        near = (self.r_centers.unsqueeze(0) - r.unsqueeze(1)).abs() <= max(cfg.carve_halfwidth, self.dr)
        surf = torch.where(near, self.height, torch.full_like(self.height, -1e9)).amax(dim=1)
        surf = torch.maximum(surf, self.surface_height(r))
        depth = (surf - z).clamp(min=0.0, max=cfg.max_depth)
        depth = torch.where(inside, depth, torch.zeros_like(depth))
        self.depth = depth

        # ---------------- resistance force (Reece / FEE) -------------------
        w = cfg.bucket_width
        g = 9.81
        surcharge = self.fill_mass * g / (w * max(cfg.capacity ** (2 / 3), 1e-6))
        speed = torch.linalg.norm(tip_vel, dim=1)

        f_grav = cfg.density * g * depth.pow(2) * cfg.n_gamma
        f_coh = self.cohesion * depth * cfg.n_c
        f_surch = surcharge * depth * cfg.n_q
        f_visc = cfg.viscous * depth * speed
        f_mag = (w * (f_grav + f_coh + f_surch) + f_visc).clamp(max=cfg.max_force)
        f_mag = torch.where(depth > 0.0, f_mag, torch.zeros_like(f_mag))

        # opposing the tip motion; if the tool is nearly still, push straight up
        dir_ = -tip_vel / speed.clamp(min=1e-3).unsqueeze(1)
        up = torch.zeros_like(dir_)
        up[:, 2] = 1.0
        dir_ = torch.where((speed > 1e-3).unsqueeze(1), dir_, up)
        force = dir_ * f_mag.unsqueeze(1)

        # ---------------- carve + fill ------------------------------------
        d_fill = torch.zeros_like(depth)
        digging = depth > 1e-4
        prev_r = torch.where(torch.isnan(self.prev_r), r, self.prev_r)
        if digging.any():
            # cells the edge swept through since the previous step get cut
            # down to the edge height
            lo = torch.minimum(prev_r, r) - 0.5 * self.dr
            hi = torch.maximum(prev_r, r) + 0.5 * self.dr
            rc = self.r_centers.unsqueeze(0)
            mask = (rc >= lo.unsqueeze(1)) & (rc <= hi.unsqueeze(1)) & digging.unsqueeze(1)
            target = z.unsqueeze(1).expand_as(self.height)
            new_h = torch.where(mask, torch.minimum(self.height, target), self.height)
            carved = (self.height - new_h) * self.dr * w        # (N, C) volume
            self.height = new_h

            room = (cfg.capacity - self.fill).clamp(min=0.0)
            gained = (carved.sum(dim=1) * cfg.fill_efficiency).minimum(room)
            self.fill = self.fill + gained
            d_fill = gained

        self.prev_r = r.clone()

        # ---------------- spillage ----------------------------------------
        # only once the teeth are out of the soil: while cutting, material is
        # pressed into the bucket (which drags with its opening facing the cab)
        tilt = torch.acos(open_dir_w[:, 2].clamp(-1.0, 1.0))
        spilling = (tilt > cfg.spill_angle) & (self.fill > 0.0) & ~digging
        spilled = torch.zeros_like(self.fill)
        if spilling.any():
            keep = torch.exp(torch.tensor(-dt / cfg.spill_tau, device=self.device))
            lost = torch.where(spilling, self.fill * (1.0 - keep), torch.zeros_like(self.fill))
            self.fill = self.fill - lost
            spilled = lost
            # give the material back to the bed, spread over the bucket
            # footprint (dumping into a single cell would build a spike that
            # the bucket immediately re-digs)
            valid = spilling & inside
            if valid.any():
                dist = (self.r_centers.unsqueeze(0) - r.unsqueeze(1)).abs()
                foot = (dist <= cfg.carve_halfwidth) & valid.unsqueeze(1)
                n_cells = foot.sum(dim=1).clamp(min=1)
                dh = lost / (self.dr * w * n_cells)
                self.height = self.height + foot * dh.unsqueeze(1)

        self._relax_repose()
        return {"force": force, "depth": depth, "d_fill": d_fill, "spilled": spilled}

    # ------------------------------------------------------------------ #
    def _relax_repose(self) -> None:
        """Slump any slope steeper than the angle of repose.  Volume conserving."""
        cfg = self.cfg
        if cfg.repose_iters <= 0:
            return
        max_dh = float(torch.tan(torch.tensor(cfg.repose_angle))) * self.dr
        for _ in range(cfg.repose_iters):
            dh = self.height[:, 1:] - self.height[:, :-1]
            over = dh.abs() > max_dh
            if not bool(over.any()):
                break
            t = torch.where(
                dh > max_dh,
                (dh - max_dh) * 0.5,
                torch.where(dh < -max_dh, (dh + max_dh) * 0.5, torch.zeros_like(dh)),
            )
            pad = torch.zeros(self.height.shape[0], 1, device=self.device)
            self.height = (
                self.height
                + torch.cat([t, pad], dim=1)      # cell i receives from pair (i, i+1)
                - torch.cat([pad, t], dim=1)      # cell i gives to   pair (i-1, i)
            )


def _uniform(shape, rng: tuple[float, float], device) -> torch.Tensor:
    lo, hi = rng
    return torch.rand(shape, device=device) * (hi - lo) + lo
