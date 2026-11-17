r"""Electric propulsion relations (MATHKIT.md Section 5; AE 6451).

Rocket equation :math:`\Delta v=v_e\ln\frac{m_0}{m_f}`; jet power and efficiency :math:`\frac F{P_{in}}=\frac{2\eta_T}{v_e}`;
Stuhlinger payload fraction :math:`\frac{m_{pl}}{m_0}=e^{-\Delta v/v_e}-\big(\frac{v_e}{v_{ch}}\big)^2(1-e^{-\Delta v/v_e})`,
:math:`v_{ch}^2=\frac{2\eta t_b}{\alpha}`; Child-Langmuir :math:`j=\frac{4\varepsilon_0}{9}\sqrt{\frac{2q}{M}}\frac{V^{3/2}}{d^2}`;
ion thrust :math:`F=I_b\sqrt{2MV_b/e}`; :math:`\vec E\times\vec B` drift :math:`v=E/B`; Edelbaum circle-to-circle
:math:`\Delta v=\sqrt{v_0^2-2v_0v_1\cos(\frac\pi2\Delta i)+v_1^2}` (as given in MathKit).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from ..constants import G0

E = 1.602176634e-19          # (C)
EPS0 = 8.8541878128e-12      # (F·m⁻¹)
AMU = 1.66053906660e-27      # (kg)
KB = 1.380649e-23            # (J·K⁻¹)
ME = 9.1093837015e-31        # (kg)
MU_EARTH = 3.986004418e14    # (m³·s⁻²)
R_EARTH = 6378.137e3         # (m)
PROPELLANTS = {"Xe": 131.293, "Kr": 83.798, "Ar": 39.948, "I": 126.904}


def rocket_dv(ve, m0, mf):
    return ve * np.log(m0 / mf)


def mass_ratio(dv, ve):
    return np.exp(dv / ve)


def thrust_per_power(eta, ve):
    """:math:`\\frac{F}{P_{in}}=\\frac{2\\eta}{v_e}` (N·W⁻¹)."""
    return 2.0 * eta / ve


def stuhlinger_payload(dv, ve, vch):
    x = np.exp(-dv / np.asarray(ve, float))
    return x - (np.asarray(ve, float) / vch) ** 2 * (1 - x)


def v_char(eta: float, t_b: float, alpha: float) -> float:
    """Characteristic velocity :math:`v_{ch}=\\sqrt{2\\eta t_b/\\alpha}` (``alpha`` specific mass, kg·W⁻¹)."""
    return float(np.sqrt(2.0 * eta * t_b / alpha))


def optimum_ve(dv: float, vch: float) -> tuple[float, float]:
    """Exhaust velocity that maximizes the Stuhlinger payload fraction (numerical, bounded)."""
    res = minimize_scalar(lambda ve: -stuhlinger_payload(dv, ve, vch), bounds=(0.05 * vch, 5.0 * vch), method="bounded",
                          options={"xatol": 1e-6 * vch})
    return float(res.x), float(-res.fun)


def child_langmuir_j(V, d, M):
    return 4 * EPS0 / 9 * np.sqrt(2 * E / M) * np.asarray(V, float) ** 1.5 / d**2


def perveance(I, V):
    """Beamlet perveance :math:`\\mathcal P=I/V^{3/2}` (A·V⁻³ᐟ²)."""
    return I / np.asarray(V, float) ** 1.5


def ion_thrust(Ib, Vb, M):
    """Singly charged, no divergence: :math:`F=I_b\\sqrt{2MV_b/e}`."""
    return Ib * np.sqrt(2 * M * np.asarray(Vb, float) / E)


def mass_utilization(Ib, M, mdot):
    return Ib * M / (E * mdot)


def exb_drift(Ef, B):
    return np.asarray(Ef, float) / np.asarray(B, float)


def edelbaum_dv(v0, v1, di):
    """MathKit form (``di`` in radians)."""
    return np.sqrt(v0**2 - 2 * v0 * v1 * np.cos(0.5 * np.pi * di) + v1**2)


def circular_v(r):
    return np.sqrt(MU_EARTH / np.asarray(r, float))


def thrust_efficiency(eta_b: float, eta_m: float, eta_div: float, eta_q: float) -> float:
    """Efficiency decomposition :math:`\\eta=\\eta_b\\eta_m\\eta_{div}\\eta_q` (beam/electrical, utilization, divergence, charge)."""
    return eta_b * eta_m * eta_div * eta_q


def divergence_eff(theta_rad: float) -> float:
    """Thrust loss from the beam divergence half-angle: :math:`\\eta_{div}=\\cos^2\\bar\\theta`."""
    return float(np.cos(theta_rad) ** 2)


def doubles_factor(f2: float) -> float:
    """Thrust correction for a doubly-charged current fraction ``f2`` = I⁺⁺/I⁺ (Goebel & Katz, Ch. 2):
    :math:`\\alpha=\\frac{1+f_2/\\sqrt2}{1+f_2}`."""
    return (1.0 + f2 / np.sqrt(2.0)) / (1.0 + f2)


def isp_from_voltage(V, M, eta_v: float = 1.0):
    """Ideal beam Isp for accelerating potential ``V`` (s)."""
    return np.sqrt(2 * E * eta_v * np.asarray(V, float) / M) / G0


def electrothermal(P: float, eta_heat: float, mdot: float, cp: float, T0: float, eta_noz: float = 0.9) -> dict:
    """Resistojet/arcjet energy balance: :math:`\\eta P=\\dot mc_p(T_c-T_0)`, :math:`v_e=\\sqrt{2\\eta_n c_pT_c}`."""
    Tc = T0 + eta_heat * P / (mdot * cp)
    ve = np.sqrt(2.0 * eta_noz * cp * Tc)
    F = mdot * ve
    return dict(Tc=float(Tc), ve=float(ve), F=float(F), Isp=float(ve / G0), eta_T=float(F**2 / (2 * mdot * P)))
