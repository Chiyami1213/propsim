r"""One-group reactor criticality (Emrich Ch. 8).

Material buckling :math:`B_m^2=\frac{\nu\Sigma_f-\Sigma_a}{D}` (Eq. 8.8), extrapolation distance :math:`d=2D`
(Eq. 8.16), :math:`k_{eff}=\frac{k_\infty}{1+L^2B_g^2}` (Eq. 8.21) with the geometric bucklings of Table 8.2, and the
one-group reflected slab :math:`D_c\alpha\tan\alpha a=\frac{D_r}{L_r}\coth\frac{b}{L_r}` (Eq. 8.41).

Reflected spheres and cylinders use the reflector savings :math:`\delta` of the equivalent slab: the extrapolated core
dimension :math:`R+2D` of the bare core is replaced by :math:`R+\delta`, the usual one-group approximation. Lengths in
m and cross sections in m⁻¹ (multiply Emrich's cm⁻¹ by 100).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

J0_ZERO = 2.404825557695773      # first zero of the Bessel function J0


def material_buckling(nuSf: float, Sa: float, D: float) -> float:
    return (nuSf - Sa) / D


def geometric_buckling(geom: str, R: float, H: float, ext: float) -> float:
    """``geom`` in ``sphere`` / ``cylinder`` / ``slab``; ``R`` radius (half-thickness for a slab), ``H`` cylinder
    height, ``ext`` the extrapolation added to each dimension (2D bare, the reflector savings when reflected)."""
    if geom == "sphere":
        return (np.pi / (R + ext)) ** 2
    if geom == "slab":
        return (np.pi / (2.0 * (R + ext))) ** 2
    return (J0_ZERO / (R + ext)) ** 2 + (np.pi / (H + 2.0 * ext)) ** 2


def k_eff(nuSf: float, Sa: float, D: float, Bg2: float) -> float:
    """Emrich Eq. 8.21."""
    return nuSf / (Sa + D * Bg2)


def reflected_slab_half_thickness(Bm2: float, Dc: float, Dr: float, Lr: float, b: float) -> float:
    """Critical core half-thickness ``a`` of a slab with a reflector of thickness ``b`` on each face (Emrich Eq. 8.41,
    with the vacuum extrapolation 2 D_r added to ``b``). First root, ``0 < alpha a < pi/2``."""
    al = np.sqrt(Bm2)
    rhs = Dr / Lr / np.tanh((b + 2.0 * Dr) / Lr)
    f = lambda a: Dc * al * np.tan(al * a) - rhs
    hi = (0.5 * np.pi - 1e-9) / al
    return float(brentq(f, 1e-12, hi))


def reflector_savings(Bm2: float, Dc: float, Dr: float, Lr: float, b: float) -> float:
    r"""Reflector savings :math:`\delta=\frac{\pi}{2B_m}-a_{refl}`, so the reflected core behaves like a bare core
    extrapolated by :math:`\delta` instead of :math:`2D`."""
    if b <= 0:
        return 2.0 * Dc
    return 0.5 * np.pi / np.sqrt(Bm2) - reflected_slab_half_thickness(Bm2, Dc, Dr, Lr, b)


def critical_size(geom: str, Bm2: float, ext: float, HD: float = 1.0) -> float:
    """Critical radius (half-thickness for a slab) for buckling ``Bm2``; a cylinder keeps ``H = HD * 2R``."""
    if Bm2 <= 0:
        return float("nan")
    if geom == "sphere":
        return np.pi / np.sqrt(Bm2) - ext
    if geom == "slab":
        return 0.5 * np.pi / np.sqrt(Bm2) - ext
    f = lambda R: geometric_buckling("cylinder", R, 2.0 * HD * R, ext) - Bm2
    lo = 1e-6 + 0.0
    hi = 1e3
    if f(lo + 1e-9) < 0:
        return float("nan")
    return float(brentq(f, lo + 1e-9, hi))


def volume(geom: str, R: float, H: float) -> float:
    """Core volume (m³); a slab is per unit face area (m)."""
    if geom == "sphere":
        return 4.0 / 3.0 * np.pi * R**3
    if geom == "slab":
        return 2.0 * R
    return np.pi * R**2 * H


def flux_shape(geom: str, r: np.ndarray, z: np.ndarray, R: float, H: float, ext: float) -> np.ndarray:
    r"""Normalised fundamental-mode flux on a grid of radius ``r`` and axial position ``z`` (from the mid-plane):
    :math:`J_0\!\left(\frac{2.405r}{\tilde R}\right)\cos\frac{\pi z}{\tilde H}` for a cylinder,
    :math:`\frac{\sin(\pi r/\tilde R)}{\pi r/\tilde R}` for a sphere (``r`` the radial distance) and
    :math:`\cos\frac{\pi x}{2\tilde a}` for a slab."""
    from scipy.special import j0
    r = np.asarray(r, float)
    z = np.asarray(z, float)
    if geom == "cylinder":
        return j0(J0_ZERO * r / (R + ext)) * np.cos(np.pi * z / (H + 2.0 * ext))
    if geom == "sphere":
        rho = np.sqrt(r**2 + z**2)
        x = np.pi * rho / (R + ext)
        return np.where(x > 1e-9, np.sin(x) / np.maximum(x, 1e-12), 1.0)
    return np.cos(0.5 * np.pi * r / (R + ext)) + 0.0 * z


def reflector_flux(x: np.ndarray, a: float, b: float, Dr: float, Lr: float) -> np.ndarray:
    r"""Reflector flux shape :math:`\frac{\sinh\left((a+b+2D_r-x)/L_r\right)}{\sinh\left((b+2D_r)/L_r\right)}`,
    equal to 1 at the core surface ``x = a`` (Emrich Eq. 8.33); multiply by the core-surface flux."""
    bt = b + 2.0 * Dr
    return np.sinh(np.clip(a + bt - np.asarray(x, float), 0.0, None) / Lr) / np.sinh(bt / Lr)


def analyse(nuSf: float, Sa: float, D: float, geom: str, R: float, HD: float, b: float, Dr: float, Lr: float,
            rho_fuel: float) -> dict:
    """Everything the Live model shows: k_inf, L², B_m², savings, critical size and mass, and k_eff of the core of
    radius ``R`` (``H = 2 R HD`` for a cylinder). ``rho_fuel`` is the fissile-bearing core density (kg·m⁻³)."""
    Bm2 = material_buckling(nuSf, Sa, D)
    kinf = nuSf / Sa
    L2 = D / Sa
    ext = reflector_savings(Bm2, D, Dr, Lr, b) if Bm2 > 0 else 2.0 * D
    ext_bare = 2.0 * D
    H = 2.0 * HD * R
    Bg2 = geometric_buckling(geom, R, H, ext)
    Rc = critical_size(geom, Bm2, ext, HD)
    Rc_bare = critical_size(geom, Bm2, ext_bare, HD)
    Hc = 2.0 * HD * Rc
    Vc = volume(geom, Rc, Hc) if np.isfinite(Rc) else float("nan")
    return dict(Bm2=Bm2, kinf=kinf, L2=L2, ext=ext, Bg2=Bg2, keff=k_eff(nuSf, Sa, D, Bg2), Rc=Rc, Rc_bare=Rc_bare,
                Hc=Hc, Vc=Vc, Mc=rho_fuel * Vc, H=H, rho=1.0 - 1.0 / k_eff(nuSf, Sa, D, Bg2))
