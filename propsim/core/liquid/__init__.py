r"""Liquid-engine sizing (MATHKIT.md Section 4.1): injector orifices and chamber volume from :math:`L^*`.

Orifice: :math:`\dot m=C_dA\sqrt{2\rho\Delta p}`.  Momentum ratio of the two propellants
:math:`J=\frac{(\rho u^2)_1}{(\rho u^2)_2}` (for like-on-unlike doublets a ratio near 1 gives a mixing-optimal impingement).
Chamber: :math:`V_c=L^*A_t` with a cylindrical section plus a 45° convergent frustum, gas residence time
:math:`\theta_g=\frac{L^*c^*}{RT_c}`.
"""
from __future__ import annotations

import numpy as np

# Handbook liquid densities at storage state (kg·m⁻³)
RHO_OX = {"LOX": 1141.0, "N2O4": 1443.0}
RHO_FUEL = {"RP-1": 810.0, "H2": 70.8, "CH4": 423.0, "Ethanol": 789.0, "MMH": 874.0}


def orifice_mdot(Cd: float, A, rho: float, dp):
    return Cd * np.asarray(A, float) * np.sqrt(2.0 * rho * np.maximum(dp, 0.0))


def orifice_area(mdot: float, Cd: float, rho: float, dp: float) -> float:
    return mdot / (Cd * np.sqrt(2.0 * rho * dp))


def injector_design(mdot: float, rho: float, dp: float, Cd: float, d: float) -> dict[str, float]:
    """Orifice count and velocity for a total flow ``mdot`` through holes of diameter ``d`` (SI)."""
    A_tot = orifice_area(mdot, Cd, rho, dp)
    a1 = np.pi * d**2 / 4.0
    n = A_tot / a1
    u = mdot / (rho * A_tot)                          # velocity in the orifice plane (continuity)
    return dict(A_tot=A_tot, n_exact=n, n=int(max(1, round(n))), u=u, dp=dp, d=d, jet_u_ideal=Cd * np.sqrt(2 * dp / rho))


def momentum_ratio(mdot_1: float, rho_1: float, A_1: float, mdot_2: float, rho_2: float, A_2: float) -> float:
    """:math:`J=\\frac{\\rho_1u_1^2}{\\rho_2u_2^2}` with :math:`u=\\frac{\\dot m}{\\rho A}`."""
    u1, u2 = mdot_1 / (rho_1 * A_1), mdot_2 / (rho_2 * A_2)
    return float(rho_1 * u1**2 / (rho_2 * u2**2))


def chamber_volume(Lstar: float, At: float) -> float:
    return Lstar * At


def chamber_geometry(Lstar: float, Dt: float, contraction: float, theta_conv_deg: float = 45.0) -> dict[str, float]:
    """Cylinder length ``Lc`` and diameter ``Dc`` such that cylinder + convergent frustum hold ``L* A_t``."""
    At = np.pi * Dt**2 / 4.0
    Vc = Lstar * At
    Dc = Dt * np.sqrt(contraction)
    Rc, Rt = Dc / 2.0, Dt / 2.0
    Lconv = (Rc - Rt) / np.tan(np.radians(theta_conv_deg))
    Vconv = np.pi * Lconv / 3.0 * (Rc**2 + Rc * Rt + Rt**2)
    Ac = np.pi * Rc**2
    Lc = max((Vc - Vconv) / Ac, 0.0)
    return dict(Vc=Vc, Dc=Dc, Lc=Lc, Lconv=Lconv, Vconv=Vconv, feasible=bool(Vc > Vconv))


def residence_time(Lstar: float, cstar: float, RT: float) -> float:
    """Gas residence time :math:`\\theta_g=\\frac{L^*c^*}{RT_c}` (s)."""
    return Lstar * cstar / RT


def injector_stiffness_ok(dp: float, pc: float) -> tuple[bool, float]:
    """Common rule of thumb: injector drop of at least 15 % of chamber pressure (decouples feed and chamber)."""
    return bool(dp >= 0.15 * pc), float(dp / pc)
