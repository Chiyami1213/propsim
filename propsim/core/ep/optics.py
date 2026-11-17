r"""Gridded-ion optics, one aperture cell, axisymmetric (r, z) (MATHKIT.md Section 5, particle pushing).

Field: Poisson :math:`\frac1r\partial_r(r\partial_r\phi)+\partial_z^2\phi=-\rho/\varepsilon_0` on a uniform grid (5-point
stencil, SciPy sparse LU), electrodes as Dirichlet cells, the upstream emission plane at the discharge potential, the
downstream boundary at the neutralized-beam potential, symmetry at the axis and at the cell edge.  Ions start at the
emission plane with the Bohm speed and a uniform current density (the plasma meniscus is not modelled: the emission
plane is fixed), are pushed with the Boris scheme (energy-conserving; :math:`\vec B=0` here so it is leapfrog), and
deposit their space charge; field and trajectories are iterated with under-relaxation, so the beamlet focus depends on
the perveance.  Charge-exchange ions are born along the downstream beam (seeded RNG) and those that fall back onto the
accelerator grid give the erosion trend.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu

from . import AMU, E, EPS0, KB


@dataclass(frozen=True)
class OpticsGeom:
    r_s: float = 0.95e-3     # screen aperture radius (m)
    t_s: float = 0.38e-3     # screen thickness
    gap: float = 0.66e-3     # screen-accel gap
    r_a: float = 0.57e-3     # accel aperture radius
    t_a: float = 0.50e-3     # accel thickness
    R_cell: float = 1.25e-3  # cell radius (half the hole pitch)
    z_up: float = 0.40e-3    # emission plane to the screen grid
    L_down: float = 4.0e-3   # accel grid exit to the downstream boundary
    V_s: float = 1100.0      # screen (discharge-plasma) potential (V)
    V_a: float = -180.0      # accel potential (V)
    Te: float = 5.0          # discharge electron temperature (eV) -> Bohm speed
    M_amu: float = 131.293
    n: int = 48              # radial cells (z cells follow from the same spacing)

    @property
    def z_s0(self) -> float:
        return self.z_up

    @property
    def z_a0(self) -> float:
        return self.z_up + self.t_s + self.gap

    @property
    def L(self) -> float:
        return self.z_a0 + self.t_a + self.L_down


class Optics:
    def __init__(self, g: OpticsGeom) -> None:
        self.g = g
        self.h = g.R_cell / g.n
        self.nr = g.n + 1
        self.nz = int(round(g.L / self.h)) + 1
        self.r = np.arange(self.nr) * self.h
        self.z = np.arange(self.nz) * self.h
        Z, R = np.meshgrid(self.z, self.r, indexing="ij")              # arrays [iz, ir]
        self.screen = (Z >= g.z_s0) & (Z <= g.z_s0 + g.t_s) & (R >= g.r_s)
        self.accel = (Z >= g.z_a0) & (Z <= g.z_a0 + g.t_a) & (R >= g.r_a)
        fixed = self.screen | self.accel
        fixed[0, :] = True                                             # emission plane
        fixed[-1, :] = True                                            # downstream neutralized beam
        self.fixed = fixed
        self.phi_fix = np.where(self.screen, g.V_s, 0.0) + np.where(self.accel, g.V_a, 0.0)
        self.phi_fix[0, :] = g.V_s + 2.0 * g.Te                        # plasma slightly above the screen grid
        self.phi_fix[-1, :] = 0.0
        self._lu = self._factorize()
        self.M = g.M_amu * AMU
        self.v_bohm = float(np.sqrt(E * g.Te / self.M))

    # ---------------------------------------------------------------- Poisson
    def _idx(self, iz, ir):
        return iz * self.nr + ir

    def _factorize(self):
        nz, nr, h = self.nz, self.nr, self.h
        A = lil_matrix((nz * nr, nz * nr))
        for iz in range(nz):
            for ir in range(nr):
                k = self._idx(iz, ir)
                if self.fixed[iz, ir]:
                    A[k, k] = 1.0
                    continue
                r = ir * h
                # z neighbours
                A[k, self._idx(iz - 1, ir)] += 1.0
                A[k, self._idx(iz + 1, ir)] += 1.0
                if ir == 0:                                            # axis: lim (1/r) d(r dphi/dr) = 2 d2phi/dr2
                    A[k, self._idx(iz, 1)] += 4.0
                    A[k, k] += -6.0
                elif ir == nr - 1:                                     # symmetry at the cell edge (mirror)
                    A[k, self._idx(iz, ir - 1)] += 2.0
                    A[k, k] += -4.0
                else:
                    A[k, self._idx(iz, ir + 1)] += 1.0 + h / (2 * r)
                    A[k, self._idx(iz, ir - 1)] += 1.0 - h / (2 * r)
                    A[k, k] += -4.0
        return splu(A.tocsc())

    def solve(self, rho: np.ndarray | None = None) -> np.ndarray:
        b = np.where(self.fixed, self.phi_fix, 0.0)
        if rho is not None:
            b = np.where(self.fixed, b, -rho / EPS0 * self.h**2)
        return self._lu.solve(b.ravel()).reshape(self.nz, self.nr)

    # ---------------------------------------------------------------- particles
    def _field(self, phi: np.ndarray):
        Ez, Er = np.gradient(-phi, self.h, self.h)
        return Ez, Er

    def _interp(self, F: np.ndarray, z: np.ndarray, r: np.ndarray) -> np.ndarray:
        fz = np.clip(z / self.h, 0, self.nz - 1.001)
        fr = np.clip(r / self.h, 0, self.nr - 1.001)
        iz, ir = fz.astype(int), fr.astype(int)
        a, b = fz - iz, fr - ir
        return ((1 - a) * (1 - b) * F[iz, ir] + a * (1 - b) * F[iz + 1, ir] + (1 - a) * b * F[iz, ir + 1]
                + a * b * F[iz + 1, ir + 1])

    def trace(self, phi: np.ndarray, I_beamlet: float, n_rays: int = 60, keep: int = 18):
        """Push rays with the Boris/leapfrog scheme; returns tracks, impingement fraction, exit angles and rho."""
        g, h = self.g, self.h
        Ez, Er = self._field(phi)
        q_m = E / self.M
        r0 = g.r_s * 0.98 * np.sqrt((np.arange(n_rays) + 0.5) / n_rays)    # uniform current density at emission
        I_k = I_beamlet / n_rays
        x, y, z = r0.copy(), np.zeros(n_rays), np.full(n_rays, 1e-9)
        vx, vy, vz = np.zeros(n_rays), np.zeros(n_rays), np.full(n_rays, self.v_bohm)
        alive = np.ones(n_rays, bool)
        hit = np.zeros(n_rays, bool)
        hit_s = np.zeros(n_rays, bool)
        rho = np.zeros((self.nz, self.nr))
        vmax = np.sqrt(2 * q_m * (g.V_s - g.V_a + 4 * g.Te))
        dt = 0.2 * h / vmax                                             # <= 0.2 cell-crossing time
        sel = np.linspace(0, n_rays - 1, min(keep, n_rays)).astype(int)
        tracks = [[(float(z[i]), float(r0[i]))] for i in sel]
        for step in range(200000):
            if not alive.any():
                break
            r = np.hypot(x, y)
            ez = self._interp(Ez, z, r)
            er = self._interp(Er, z, r)
            with np.errstate(invalid="ignore", divide="ignore"):
                cx, cy = np.where(r > 0, x / r, 1.0), np.where(r > 0, y / r, 0.0)
            vx = np.where(alive, vx + q_m * er * cx * dt, vx)
            vy = np.where(alive, vy + q_m * er * cy * dt, vy)
            vz = np.where(alive, vz + q_m * ez * dt, vz)
            x, y, z = np.where(alive, x + vx * dt, x), np.where(alive, y + vy * dt, y), np.where(alive, z + vz * dt, z)
            r = np.hypot(x, y)
            iz = np.clip((z / h).astype(int), 0, self.nz - 1)
            ir = np.clip((r / h).astype(int), 0, self.nr - 1)
            vol = 2 * np.pi * np.maximum(ir * h, 0.5 * h) * h * h
            dep = alive & (z <= g.z_a0 + g.t_a)                         # downstream the beam is electron-neutralized
            np.add.at(rho, (iz[dep], ir[dep]), I_k * dt / vol[dep])
            on = (self.screen | self.accel)[iz, ir] | (r > g.R_cell)
            newly = alive & on
            hit |= newly
            hit_s |= newly & self.screen[iz, ir]
            alive &= ~on
            gone = alive & ((z >= g.L) | (z < 0))
            alive &= ~gone
            if step % 4 == 0:
                for k, i in enumerate(sel):
                    if alive[i] or newly[i] or gone[i]:
                        tracks[k].append((float(z[i]), float(r[i])))
        vr = (vx * x + vy * y) / np.maximum(np.hypot(x, y), 1e-30)
        theta = np.arctan2(vr, vz)
        out = ~hit
        # screen interception (upstream transparency loss, from the fixed emission plane) is reported apart from the
        # accelerator-grid / neighbour-cell impingement that erodes the accel grid
        return dict(tracks=[np.array(t) for t in tracks], dt_track=4.0 * dt, impinge=float((hit & ~hit_s).mean()),
                    screen=float(hit_s.mean()), theta=theta[out],
                    theta_rms=float(np.sqrt(np.mean(theta[out] ** 2))) if out.any() else float("nan"), rho=rho)

    def electron_rho(self, phi: np.ndarray, I_beamlet: float) -> np.ndarray:
        """Boltzmann discharge electrons upstream of the screen exit: n_e = n_0 exp((phi - phi_p)/T_e), with n_0 set by
        the Bohm flux of the extracted current (quasi-neutral plasma at the emission plane)."""
        g = self.g
        J = I_beamlet / (np.pi * g.r_s**2)
        n0 = J / (E * self.v_bohm)
        Z = self.z[:, None] * np.ones((1, self.nr))
        phi_p = self.phi_fix[0, 0]
        ne = n0 * np.exp(np.clip((phi - phi_p) / g.Te, -60.0, 0.0))
        return np.where(Z <= g.z_s0 + g.t_s, -E * ne, 0.0)

    def run(self, I_beamlet: float, iters: int = 10, relax: float = 0.3, n_rays: int = 60):
        """Self-consistent beamlet: alternate trajectory traces (ion space charge) and field solves with Boltzmann
        discharge electrons upstream (the sheath adjusts), under-relaxed Picard iteration."""
        phi = self.solve()
        rho = np.zeros((self.nz, self.nr))
        hist = []
        for _ in range(iters):
            tr = self.trace(phi, I_beamlet, n_rays)
            target = tr["rho"] + self.electron_rho(phi, I_beamlet)
            rho = (1 - relax) * rho + relax * target
            phi = self.solve(rho)
            hist.append(tr["theta_rms"])
        tr = self.trace(phi, I_beamlet, n_rays)
        tail = np.array(hist[-4:] + [tr["theta_rms"]])
        tr["phi"] = phi
        tr["theta_hist"] = hist
        tr["theta_med"] = float(np.median(tail))
        # near / above the perveance limit the space-charge-limited iteration does not settle: report it honestly
        tr["converged"] = bool(np.nanstd(tail) < np.radians(2.0))
        return tr

    def cex(self, phi: np.ndarray, I_beamlet: float, n0: float, sigma: float = 4.0e-19, n_ions: int = 300, seed: int = 1):
        """Charge-exchange ions born at rest along the downstream beam column; fraction that falls onto the accel grid
        (``n0`` neutral density, m⁻³; ``sigma`` CEX cross-section, m²).  CEX current ~ I_b n0 sigma L."""
        rng = np.random.default_rng(seed)
        g, h = self.g, self.h
        Ez, Er = self._field(phi)
        q_m = E / self.M
        z = rng.uniform(g.z_a0, g.L, n_ions)
        r = g.r_a * np.sqrt(rng.uniform(0, 1, n_ions)) * 0.8
        vz, vr = np.zeros(n_ions), np.zeros(n_ions)
        vth = np.sqrt(KB * 300.0 / self.M)
        vz += rng.normal(0, vth, n_ions)
        vr += rng.normal(0, vth, n_ions)
        alive = np.ones(n_ions, bool)
        back = np.zeros(n_ions, bool)
        dt = 0.2 * h / np.sqrt(2 * q_m * (abs(g.V_a) + 50.0))
        for _ in range(100000):
            if not alive.any():
                break
            ez, er = self._interp(Ez, z, np.abs(r)), self._interp(Er, z, np.abs(r))
            vz = np.where(alive, vz + q_m * ez * dt, vz)
            vr = np.where(alive, vr + q_m * er * np.sign(r + 1e-30) * dt, vr)
            z, r = np.where(alive, z + vz * dt, z), np.where(alive, r + vr * dt, r)
            iz = np.clip((z / h).astype(int), 0, self.nz - 1)
            ir = np.clip((np.abs(r) / h).astype(int), 0, self.nr - 1)
            hit = alive & self.accel[iz, ir]
            back |= hit
            alive &= ~hit & (z < g.L) & (z > g.z_a0 - 0.5 * g.gap) & (np.abs(r) <= g.R_cell)
        L_col = g.L - g.z_a0
        I_cex = I_beamlet * n0 * sigma * L_col                       # CEX production in the downstream column
        return dict(frac_back=float(back.mean()), I_cex=float(I_cex), I_accel=float(I_cex * back.mean()))


def perveance_max(g: OpticsGeom) -> float:
    """Child-Langmuir perveance limit of the screen aperture over the effective gap (A·V⁻³ᐟ²)."""
    le = np.hypot(g.gap + g.t_s, g.r_s)
    return float(4 * EPS0 / 9 * np.sqrt(2 * E / (g.M_amu * AMU)) * np.pi * g.r_s**2 / le**2)
