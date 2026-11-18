r"""NTR performance with a frozen, calorically perfect propellant (Emrich Ch. 2).

Thermodynamic specific impulse (Emrich Eq. 2.19):
:math:`I_{sp}=\frac{1}{g_c}\sqrt{\frac{2\gamma}{\gamma-1}\frac{R_uT_0}{m_w}\left[1-\left(\frac{P_e}{P_0}\right)^{\frac{\gamma-1}{\gamma}}\right]}`.
The nozzle uses the same isentropic relations as the rest of PropSim (:mod:`propsim.core.nozzle`), so the vacuum Isp
here equals :math:`c^*C_F/g_0` with zero ambient pressure. The reactor power is the enthalpy rise of the propellant from
the core inlet to the chamber, :math:`P_R=\dot m\,[h(T_0)-h(T_{in})]`. For hydrogen the enthalpy comes from CoolProp
(real gas; its ideal-gas part is valid to high temperature), for the other propellants from the frozen :math:`c_p`.
Dissociation is neglected (small for hydrogen below about 3000 K at NTR chamber pressures).
"""
from __future__ import annotations

import numpy as np

from ..constants import G0
from ..nozzle import CF, Gamma, exit_state
from . import R_U

# Frozen-flow propellant models: (molar mass kg·kmol⁻¹, gamma, label).
# H2: m_w = 2.016 (IUPAC), gamma = 1.41 (Emrich 2e Ch. 2 uses 1.4-1.41 for the NTR estimates).
# NH3: fully decomposed to N2 + 3 H2 (m_w = 8.51, gamma = 1.38), the same model the Electric module's resistojet uses.
# He: m_w = 4.003, gamma = 5/3 (monatomic).
PROPELLANTS = {
    "H2": (2.016, 1.41, "Hydrogen"),
    "NH3": (8.51, 1.38, "Ammonia (decomposed, N₂ + 3H₂)"),
    "He": (4.003, 5.0 / 3.0, "Helium"),
}


def cp_frozen(mw: float, g: float) -> float:
    r""":math:`c_p=\frac{\gamma}{\gamma-1}\frac{R_u}{m_w}` (J·kg⁻¹·K⁻¹)."""
    return g / (g - 1.0) * R_U / mw


def isp_thermo(T0, mw, g, pe_p0=0.0):
    """Emrich Eq. 2.19 (``pe_p0 = 0`` gives Eq. 2.20, the infinite-expansion maximum). Vectorised over ``T0``."""
    T0 = np.asarray(T0, float)
    return np.sqrt(2 * g / (g - 1) * R_U * T0 / mw * (1.0 - np.asarray(pe_p0, float) ** ((g - 1) / g))) / G0


def performance(T0: float, pc: float, eps: float, mw: float, g: float, F: float, pa: float = 0.0, T_in: float = 300.0,
                eta_n: float = 1.0, power_density: float = 1.3e9, LD: float = 1.0, fluid: str | None = None) -> dict:
    r"""Operating point of an NTR sized for vacuum thrust ``F`` (N).

    ``T0`` chamber temperature (K), ``pc`` chamber pressure (Pa), ``eps`` nozzle area ratio, ``pa`` ambient pressure
    (Pa), ``T_in`` core inlet temperature (K), ``eta_n`` thrust-coefficient efficiency, ``power_density`` average core
    power density (W·m⁻³), ``LD`` core length over diameter, ``fluid`` a CoolProp name for the real-gas enthalpy rise
    (``None`` uses the frozen :math:`c_p`). Returns Isp (vacuum and at ``pa``), :math:`c^*`,
    :math:`C_F`, flow, throat and exit radii, reactor power, core size and the power-to-thrust ratio.
    """
    R = R_U / mw
    cs = np.sqrt(R * T0) / Gamma(g)
    Me, pe_pc = exit_state(eps, g)
    cf_v = float(CF(g, pe_pc, 0.0, eps)) * eta_n
    cf_a = float(CF(g, pe_pc, pa / pc, eps)) * eta_n
    isp_v = cs * cf_v / G0
    isp_a = cs * cf_a / G0
    mdot = F / (isp_v * G0)
    At = mdot * cs / pc
    cp = cp_frozen(mw, g)
    dh = enthalpy_rise(T_in, T0, pc, fluid) if fluid else cp * (T0 - T_in)
    P_R = mdot * dh
    V_core = P_R / power_density
    D_core = (4.0 * V_core / (np.pi * LD)) ** (1.0 / 3.0)
    return dict(isp_v=float(isp_v), isp_a=float(isp_a), isp_max=float(isp_thermo(T0, mw, g)), cstar=float(cs),
                cf_v=cf_v, cf_a=cf_a, F_a=float(mdot * isp_a * G0), mdot=float(mdot), At=float(At),
                Rt=float(np.sqrt(At / np.pi)), Re=float(np.sqrt(eps * At / np.pi)), Me=float(Me),
                pe=float(pe_pc * pc), Te=float(T0 / (1 + 0.5 * (g - 1) * Me**2)), cp=float(cp), P_R=float(P_R),
                P_per_F=float(P_R / F), V_core=float(V_core), D_core=float(D_core), L_core=float(LD * D_core))


def enthalpy_rise(T_in: float, T0: float, p: float, fluid: str = "Hydrogen") -> float:
    """Real-gas enthalpy rise from the core inlet to the chamber at pressure ``p`` (J·kg⁻¹), CoolProp."""
    from CoolProp.CoolProp import PropsSI
    return float(PropsSI("H", "T", T0, "P", p, fluid) - PropsSI("H", "T", T_in, "P", p, fluid))


def isp_vs_eps(T0: float, mw: float, g: float, eps: np.ndarray, eta_n: float = 1.0) -> np.ndarray:
    """Vacuum Isp against area ratio (vectorised through the supersonic Mach inversion)."""
    from ..nozzle import mach_from_area_array, p_ratio
    Me = mach_from_area_array(np.asarray(eps, float), g, True)
    pe = p_ratio(Me, g)
    cs = np.sqrt(R_U / mw * T0) / Gamma(g)
    return cs * CF(g, pe, 0.0, np.asarray(eps, float)) * eta_n / G0


def payload_fraction(dv: float, isp, f_inert: float) -> np.ndarray:
    r"""Single-stage payload fraction :math:`\frac{m_{pl}}{m_0}=\frac{e^{-\Delta v/v_e}-f_{inert}}{1-f_{inert}}`
    where the inert (tank + engine) mass is a fixed fraction :math:`f_{inert}=\frac{m_{inert}}{m_{inert}+m_p}` of the
    loaded stage (Emrich Eq. 1.5 with a structural fraction)."""
    mf = np.exp(-dv / (np.asarray(isp, float) * G0))
    # m0 = m_pl + m_s + m_p, m_s = f/(1-f) m_p  ->  m_pl/m0 = mf - (1 - mf) f/(1-f)
    return mf - (1.0 - mf) * f_inert / (1.0 - f_inert)
