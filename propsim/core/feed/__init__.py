r"""Nitrous-oxide properties, self-pressurized equilibrium tank and injector models (MATHKIT.md Section 3.6).

Saturation pressure and densities use the ESDU 91022 correlations; CoolProp (``NitrousOxide``) supplies the
internal energy, enthalpy and entropy that ESDU lacks, tabulated once on a fine temperature grid so tank solves stay
fast.  Injector flow: SPI, HEM (isentropic flash) and the Dyer NHNE blend.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

TC, PC, RHOC = 309.57, 7.251e6, 452.0            # (K), (Pa), (kg·m⁻³)  ESDU 91022
R_N2O = 8.314462618 / 0.044013                   # (J·kg⁻¹·K⁻¹)
GAMMA_VAP = 1.27                                 # vapour isentropic exponent (orifice flow of gas)
T_MIN, T_MAX = 200.0, 309.4


def psat(T):
    t = 1 - np.asarray(T, float) / TC
    return PC * np.exp(TC / np.asarray(T, float) * (-6.71893 * t + 1.35966 * t**1.5 - 1.3779 * t**2.5 - 4.051 * t**5))


def rho_l(T):
    t = 1 - np.asarray(T, float) / TC
    return RHOC * np.exp(1.72328 * t ** (1 / 3) - 0.8395 * t ** (2 / 3) + 0.5106 * t - 0.10412 * t ** (4 / 3))


def rho_v(T):
    x = TC / np.asarray(T, float) - 1
    return RHOC * np.exp(-1.009 * x ** (1 / 3) - 6.28792 * x ** (2 / 3) + 7.50332 * x - 7.90463 * x ** (4 / 3)
                         + 0.629427 * x ** (5 / 3))


@lru_cache(maxsize=1)
def _tab():
    """Saturated-liquid/vapour u, h, s from CoolProp on a 0.2 K grid (one-off, ~0.3 s)."""
    from CoolProp.CoolProp import PropsSI
    T = np.arange(T_MIN, T_MAX + 1e-9, 0.2)
    d = {"T": T}
    for key, q, name in (("u_l", 0, "U"), ("u_v", 1, "U"), ("h_l", 0, "H"), ("h_v", 1, "H"), ("s_l", 0, "S"),
                         ("s_v", 1, "S")):
        d[key] = np.array([PropsSI(name, "T", float(t), "Q", q, "NitrousOxide") for t in T])
    d["ps"] = psat(T)
    return d


def _sat(name: str, T):
    t = _tab()
    return np.interp(T, t["T"], t[name])


def u_l(T): return _sat("u_l", T)
def u_v(T): return _sat("u_v", T)
def h_l(T): return _sat("h_l", T)
def h_v(T): return _sat("h_v", T)


def T_of_psat(p):
    """Saturation temperature from pressure (inverse of the ESDU p_sat on the table grid)."""
    t = _tab()
    return np.interp(p, t["ps"], t["T"])


# ------------------------------------------------------------------------------------------- injectors
def mdot_spi(Cd, A, rho, dp):
    r""":math:`\dot m=C_dA\sqrt{2\rho\Delta p}`."""
    return Cd * A * np.sqrt(2.0 * rho * np.maximum(dp, 0.0))


def flash(T1: float, p2: float) -> tuple[float, float]:
    """Isentropic flash of saturated liquid at ``T1`` to ``p2`` (< p_sat): returns (h2, rho2)."""
    T2 = float(T_of_psat(p2))
    s1, sl, sv = _sat("s_l", T1), _sat("s_l", T2), _sat("s_v", T2)
    x = float(np.clip((s1 - sl) / (sv - sl), 0.0, 1.0))
    h2 = (1 - x) * _sat("h_l", T2) + x * _sat("h_v", T2)
    rho2 = 1.0 / ((1 - x) / rho_l(T2) + x / rho_v(T2))
    return float(h2), float(rho2)


def mdot_dyer(Cd, A, T1: float, p1: float, p2: float, pv: float | None = None) -> float:
    r"""Dyer NHNE: :math:`\dot m=\frac{k}{1+k}\dot m_{SPI}+\frac{1}{1+k}\dot m_{HEM}`, :math:`k=\sqrt{\frac{p_1-p_2}{p_v-p_2}}`.
    Liquid at tank temperature ``T1`` upstream pressure ``p1``; ``pv`` defaults to p_sat(T1)."""
    pv = float(psat(T1)) if pv is None else pv
    if p1 <= p2:
        return 0.0
    rho1, h1 = float(rho_l(T1)), float(_sat("h_l", T1))
    spi = mdot_spi(Cd, A, rho1, p1 - p2)
    if p2 >= pv:                          # no flashing: pure single-phase liquid flow
        return float(spi)
    k = np.sqrt((p1 - p2) / (pv - p2))
    h2, rho2 = flash(T1, p2)
    hem = Cd * A * rho2 * np.sqrt(2.0 * max(h1 - h2, 0.0))
    return float(k / (1 + k) * spi + 1 / (1 + k) * hem)


def mdot_gas(Cd, A, p1, T1, p2) -> float:
    """Ideal-gas orifice flow (choked or subsonic) for the vapour phase."""
    g = GAMMA_VAP
    pr = max(p2 / max(p1, 1.0), 1e-9)
    crit = (2.0 / (g + 1.0)) ** (g / (g - 1.0))
    if pr <= crit:
        return float(Cd * A * p1 * np.sqrt(g / (R_N2O * T1)) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0))))
    return float(Cd * A * p1 * np.sqrt(2.0 * g / ((g - 1.0) * R_N2O * T1) * (pr ** (2.0 / g) - pr ** ((g + 1.0) / g))))


# ------------------------------------------------------------------------------------------- tank
@dataclass
class TankProps:
    T: float
    p: float
    m_liq: float
    m_vap: float
    liquid: bool        # False once the liquid is depleted (vapour-only phase)


def tank_state(m: float, U: float, V: float) -> TankProps:
    """Equilibrium state of total mass ``m`` and internal energy ``U`` in volume ``V``.

    Two-phase: solve T (brentq) from ``U = m_l u_l + m_v u_v`` with ``m_l`` from the volume constraint
    :math:`V=m_l(\\frac{1}{\\rho_l}-\\frac{1}{\\rho_v})+\\frac{m}{\\rho_v}`.  When the liquid is depleted the vapour is
    treated as a real gas through CoolProp (density and energy fixed)."""
    def ml_of(T):
        return (V - m / rho_v(T)) / (1.0 / rho_l(T) - 1.0 / rho_v(T))

    def f(T):
        ml = ml_of(T)
        return ml * u_l(T) + (m - ml) * u_v(T) - U

    # Two-phase states need  rho_v(T) <= m/V <= rho_l(T).  m_l(T) >= 0 only up to the temperature where the vapour
    # density reaches the mean density (otherwise the whole content is vapour / supercritical-like).
    rho_avg = m / V
    lo = T_MIN
    hi = T_MAX if float(rho_v(T_MAX)) <= rho_avg else brentq(lambda T: float(rho_v(T)) - rho_avg, T_MIN, T_MAX,
                                                              xtol=1e-9)
    if rho_avg >= float(rho_l(lo)):
        raise ValueError("tank overfilled: mean density exceeds the liquid density")
    flo, fhi = f(lo), f(hi)
    if fhi < 0.0:                       # more energy than the all-vapour limit: superheated vapour only
        return _vapour_state(m, U, V)
    if flo > 0.0:
        raise ValueError("tank state bracket failed: energy below the two-phase range")
    T = brentq(f, lo, hi, xtol=1e-9)
    ml = ml_of(T)
    if ml <= 1e-6 * m:
        return _vapour_state(m, U, V)
    return TankProps(T, float(psat(T)), float(ml), float(m - ml), True)


def _vapour_state(m: float, U: float, V: float) -> TankProps:
    from CoolProp.CoolProp import PropsSI
    T = PropsSI("T", "D", m / V, "U", U / m, "NitrousOxide")
    p = PropsSI("P", "D", m / V, "T", T, "NitrousOxide")
    return TankProps(float(T), float(p), 0.0, float(m), False)


def tank_initial(T0: float, V: float, fill: float) -> tuple[float, float]:
    """Total mass and internal energy for liquid fill fraction ``fill`` (of V, by volume) at saturation T0."""
    ml = fill * V * float(rho_l(T0))
    mv = (1 - fill) * V * float(rho_v(T0))
    return ml + mv, ml * float(u_l(T0)) + mv * float(u_v(T0))


def tank_rates(m: float, U: float, V: float, Cd: float, A_inj: float, p2: float, loss_frac: float = 0.0):
    """``(dm/dt, dU/dt, state, mdot)`` for the draining tank at downstream pressure ``p2``."""
    st = tank_state(m, U, V)
    if st.liquid:
        p1 = (1.0 - loss_frac) * st.p                 # line / valve loss lowers the injector inlet pressure
        md = mdot_dyer(Cd, A_inj, st.T, p1, p2, pv=st.p)
        dU = -md * float(h_l(st.T))
    else:
        from CoolProp.CoolProp import PropsSI
        md = mdot_gas(Cd, A_inj, st.p, st.T, p2)
        h = PropsSI("H", "D", m / V, "T", st.T, "NitrousOxide")
        dU = -md * float(h)
    return -md, dU, st, md
