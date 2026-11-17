r"""3-D (and 2-D) level-set grain burnback (MATHKIT.md Section 2.4).

Convention: :math:`\phi<0` is gas (port), :math:`\phi>0` propellant; the front :math:`\phi=0` moves toward the propellant
at speed :math:`F=r>0`: :math:`\phi_t+F|\nabla\phi|=0`.  For :math:`F>0` the Godunov (Osher-Sethian) upwind gradient is

.. math::  |\nabla\phi|^2\approx\sum_{axes}\max(D^-\phi,0)^2+\min(D^+\phi,0)^2 .

CFL :math:`\Delta t\le0.5\,\Delta x/F_{max}`.  Cells outside the propellant mask are inert (F = 0), so the front stops
at the case wall and inhibited faces.  Burning area: :math:`A_b=\int\delta_\epsilon(\phi)|\nabla\phi|\,dV`,
:math:`\delta_\epsilon=\frac{1+\cos(\pi\phi/\epsilon)}{2\epsilon}`, :math:`\epsilon=1.5\Delta x`.

The kernels use Numba when available; a vectorized NumPy fallback gives the same numbers.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import ndimage as ndi

try:                                   # optional accelerator (builds without numba fall back to NumPy)
    import numba as nb
    HAVE_NUMBA = True
except Exception:                      # noqa: BLE001
    HAVE_NUMBA = False


# ------------------------------------------------------------------------------------------- gradient kernels
def godunov_np(phi: np.ndarray, h: float) -> np.ndarray:
    """Godunov upwind |grad phi| for F > 0 (any dimension), edge-replicated boundaries."""
    g2 = np.zeros_like(phi)
    for ax in range(phi.ndim):
        p = np.pad(phi, [(1, 1) if a == ax else (0, 0) for a in range(phi.ndim)], mode="edge")
        sl = lambda s: tuple(s if a == ax else slice(None) for a in range(phi.ndim))
        dm = (p[sl(slice(1, -1))] - p[sl(slice(0, -2))]) / h
        dp = (p[sl(slice(2, None))] - p[sl(slice(1, -1))]) / h
        g2 += np.maximum(dm, 0.0) ** 2 + np.minimum(dp, 0.0) ** 2
    return np.sqrt(g2)


if HAVE_NUMBA:
    @nb.njit(parallel=True, fastmath=True, cache=False)
    def _step3_nb(phi, active, dt, h, out):
        nx, ny, nz = phi.shape
        for i in nb.prange(nx):
            for j in range(ny):
                for k in range(nz):
                    if not active[i, j, k]:
                        out[i, j, k] = phi[i, j, k]
                        continue
                    c = phi[i, j, k]
                    g2 = 0.0
                    # x
                    dm = (c - phi[max(i - 1, 0), j, k]) / h
                    dp = (phi[min(i + 1, nx - 1), j, k] - c) / h
                    g2 += max(dm, 0.0) ** 2 + min(dp, 0.0) ** 2
                    dm = (c - phi[i, max(j - 1, 0), k]) / h
                    dp = (phi[i, min(j + 1, ny - 1), k] - c) / h
                    g2 += max(dm, 0.0) ** 2 + min(dp, 0.0) ** 2
                    dm = (c - phi[i, j, max(k - 1, 0)]) / h
                    dp = (phi[i, j, min(k + 1, nz - 1)] - c) / h
                    g2 += max(dm, 0.0) ** 2 + min(dp, 0.0) ** 2
                    out[i, j, k] = c - dt * np.sqrt(g2)


class LevelSet:
    """Unit-speed burnback of ``phi0`` (negative = port) inside the boolean ``active`` propellant mask.

    ``h`` is the cell size.  Burn distance ``w`` equals time for unit speed, so the tables are functions of ``w``."""

    def __init__(self, phi0: np.ndarray, active: np.ndarray, h: float, use_numba: bool = True,
                 evolve: np.ndarray | None = None) -> None:
        self.phi = phi0.astype(np.float32 if phi0.ndim == 3 else np.float64)
        self.active = active.astype(bool)             # where propellant exists: measurements are taken here
        # The front is advanced everywhere (a frozen layer outside the case would create a jump in phi at the wall that
        # the delta-function integral counts as burning surface); measurements are restricted to the propellant mask.
        self.evolve = np.ones_like(self.active) if evolve is None else evolve
        self.h = float(h)
        self.w = 0.0
        self.use_nb = bool(use_numba and HAVE_NUMBA and phi0.ndim == 3)

    # ---------------------------------------------------------------- time stepping
    def step(self, dt: float) -> None:
        if self.use_nb:
            out = np.empty_like(self.phi)
            _step3_nb(self.phi, self.evolve, np.float32(dt), np.float32(self.h), out)
            self.phi = out
        else:
            g = godunov_np(self.phi, self.h)
            self.phi = np.where(self.evolve, self.phi - dt * g, self.phi)
        self.w += dt

    def reinit(self, n_iter: int = 8) -> None:
        """Sussman PDE reinitialization toward a signed distance (keeps the front, restores |grad phi| = 1)."""
        phi0 = self.phi.copy()
        s = phi0 / np.sqrt(phi0**2 + self.h**2)
        dt = 0.3 * self.h
        for _ in range(n_iter):
            gp = godunov_np(self.phi, self.h)                          # sign > 0 branch
            gm = godunov_np(-self.phi, self.h)                         # sign < 0 branch (mirror)
            g = np.where(s > 0, gp, gm)
            self.phi = np.where(self.evolve, self.phi - dt * s * (g - 1.0), self.phi)

    # ---------------------------------------------------------------- measurements
    def gas_coverage(self) -> np.ndarray:
        """Anti-aliased gas fraction per cell (1 inside the port, 0 in the propellant), inert cells excluded."""
        return np.clip(0.5 - self.phi / self.h, 0.0, 1.0) * self.active

    def burning_area(self) -> float:
        """:math:`A_b=\\int\\delta_\\epsilon(\\phi)|\\nabla\\phi|\\,dV` over the propellant mask."""
        eps = 1.5 * self.h
        g = np.sqrt(sum(a**2 for a in np.gradient(self.phi.astype(np.float64), self.h)))
        d = np.where(np.abs(self.phi) < eps, 0.5 * (1.0 + np.cos(np.pi * self.phi / eps)) / eps, 0.0)
        return float(np.sum(d * g * self.active) * self.h**self.phi.ndim)

    def gas_volume(self) -> float:
        return float(self.gas_coverage().sum() * self.h**self.phi.ndim)

    def port_area_profile(self) -> np.ndarray:
        """Port cross-section area per axial slice (3-D, axis 0 is x)."""
        return self.gas_coverage().sum(axis=tuple(range(1, self.phi.ndim))) * self.h ** (self.phi.ndim - 1)

    def surface_area_marching_cubes(self) -> float:
        """Independent 3-D area check: marching-cubes triangles whose centroid lies in the propellant mask."""
        from skimage import measure
        if not (self.phi.min() < 0.0 < self.phi.max()):
            return 0.0
        verts, faces, _, _ = measure.marching_cubes(self.phi, 0.0, spacing=(self.h,) * 3)
        tri = verts[faces]
        cen = (tri.mean(axis=1) / self.h).round().astype(int)
        cen = np.clip(cen, 0, np.array(self.phi.shape) - 1)
        keep = self.active[cen[:, 0], cen[:, 1], cen[:, 2]]
        a = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        return float(0.5 * np.linalg.norm(a, axis=1)[keep].sum())

    # ---------------------------------------------------------------- run
    def area(self) -> float:
        """Burning area: marching-cubes triangles inside the propellant (3-D, no smoothing band that could leak across
        the case wall); the delta-function integral in 2-D or when scikit-image is missing."""
        if self.phi.ndim == 3:
            try:
                return self.surface_area_marching_cubes()
            except ImportError:
                pass
            except ValueError:                       # no zero level left: fully burnt through
                return 0.0
        return self.burning_area()

    def run(self, w_max: float, n_samples: int = 31, reinit_every: int = 0,
            progress: Callable[[float], None] | None = None, cancel=None,
            on_sample: Callable[["LevelSet"], None] | None = None) -> dict[str, np.ndarray]:
        """Burn to ``w_max`` and return tables ``w, A_b, V_gas, A_port(x)``; stops early at burn-through."""
        dt = 0.5 * self.h
        n = int(np.ceil(w_max / dt))
        ws = np.linspace(0.0, w_max, n_samples)
        w_tab, Ab, Vg, Ap = [], [], [], []
        k_next, k = 0, 0

        def sample():
            w_tab.append(self.w)
            Ab.append(self.area())
            Vg.append(self.gas_volume())
            Ap.append(self.port_area_profile())
            if on_sample is not None:
                on_sample(self)

        sample()
        k_next = 1
        for k in range(1, n + 1):
            if cancel is not None and cancel.is_set():
                raise InterruptedError("cancelled")
            self.step(min(dt, w_max - self.w))
            if reinit_every and k % reinit_every == 0:
                self.reinit()
            if k_next < n_samples and self.w >= ws[k_next] - 1e-12:
                sample()
                k_next += 1
                if progress:
                    progress(self.w / w_max)
            if k > 2 and not (self.phi[self.active] > 0).any():
                break
        if len(w_tab) < n_samples and (w_tab[-1] < w_max - 1e-12 or len(w_tab) < n_samples):
            sample()
        if progress:
            progress(1.0)
        return dict(w=np.array(w_tab), A_b=np.array(Ab), V_gas=np.array(Vg), A_port=np.array(Ap))


# ------------------------------------------------------------------------------------------- initial fields
def box_grid(R_out: float, L: float, h: float, pad: int = 2):
    """Cell-centred 3-D grid covering the cylinder (x along the axis, y and z across), plus masks."""
    nx = int(round(L / h)) + 2 * pad
    ny = nz = int(round(2 * R_out / h)) + 2 * pad
    x = (np.arange(nx) - pad + 0.5) * h
    y = (np.arange(ny) - (ny - 1) / 2) * h
    z = (np.arange(nz) - (nz - 1) / 2) * h
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    active = (np.hypot(Y, Z) <= R_out) & (X >= 0.0) & (X <= L)
    return X, Y, Z, active


def phi_extruded(section_sdf: Callable[[np.ndarray, np.ndarray], np.ndarray], R_out: float, L: float, h: float):
    """Extrude a 2-D signed-distance function ``f(Y, Z)`` along x; returns ``(phi, active, h)``."""
    X, Y, Z, active = box_grid(R_out, L, h)
    sd2 = section_sdf(Y[0], Z[0])                # evaluate on one 2-D slice, then extrude along x
    return np.broadcast_to(sd2[None], X.shape).copy(), active, h   # not clipped outside the mask: no artificial end caps


def phi_partial_fins(R_out: float, L: float, h: float, R_p: float, n_fin: int, w_fin: float, depth: float,
                     fin_frac: float):
    """Finocyl whose fins exist only in the head-end fraction ``fin_frac`` of the length (a genuinely 3-D grain)."""
    from .section2d import rot, sd_circle, sd_segment
    X, Y, Z, active = box_grid(R_out, L, h)
    bore = sd_circle(Y, Z, R_p)
    fins = np.full_like(bore, 1e3)
    seg = max(depth * R_out - w_fin / 2, 0.0)
    for j in range(n_fin):
        u, v = rot(Y, Z, 2 * np.pi * j / n_fin)
        fins = np.minimum(fins, sd_segment(u, v, 0.0, seg) - w_fin / 2)
    fins = np.maximum(fins, X - fin_frac * L)                               # CSG intersection with the head-end slab
    phi = np.minimum(bore, fins)
    return phi, active, h


def phi_from_stl(path: str, R_out: float, L: float, h: float):
    """Level set from an STL of the *port* (gas) volume, axis along x, placed with its head end at x = 0."""
    import trimesh
    mesh = trimesh.load(path, force="mesh")
    X, Y, Z, active = box_grid(R_out, L, h)
    vox = mesh.voxelized(h).fill()
    origin = vox.transform[:3, 3]
    M = vox.matrix
    idx = np.floor((np.stack([X, Y, Z], axis=-1) - origin) / h + 0.5).astype(int)
    ok = np.all((idx >= 0) & (idx < np.array(M.shape)), axis=-1)
    port = np.zeros(X.shape, bool)
    port[ok] = M[idx[ok][:, 0], idx[ok][:, 1], idx[ok][:, 2]]
    phi = (ndi.distance_transform_edt(~port) - ndi.distance_transform_edt(port)) * h
    return phi, active, h
