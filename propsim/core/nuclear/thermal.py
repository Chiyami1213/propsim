r"""Axial-flow fuel element thermal hydraulics (Emrich Ch. 9).

Each hexagonal element is replaced by equivalent cells: a coolant channel of radius :math:`r_i` inside a fuel annulus
out to :math:`r_o` with an adiabatic outer boundary (Emrich Eqs. 9.10-9.17). With a chopped-cosine power shape
:math:`q'''(z)=q_0\cos az` (``z`` from the core mid-plane) the propellant, wall and peak fuel temperatures are

.. math::
    T_p(z)=T_{in}+\frac{q_0A_f}{a\dot mc_p}\left[\sin az+\sin\frac{aL}{2}\right],\qquad
    T_{max}(z)=T_p(z)+q_0\cos(az)\,R_{tot},

    R_{tot}=\frac{r_o^2-r_i^2}{2r_ih_c}+\frac{1}{2k}\left[r_o^2\ln\frac{r_o}{r_i}-\frac{r_o^2-r_i^2}{2}\right]

(Emrich Eqs. 9.35-9.40). When :math:`h_c` comes from Dittus-Boelter (Eq. 9.21) it varies along the channel and the
peak is found numerically instead of from Eq. 9.41. The pressure drop marches the Darcy friction term plus flow
acceleration of an ideal gas, the approach of Emrich Eqs. 9.43-9.50.

Hydrogen transport properties for Dittus-Boelter come from CoolProp (Leachman et al. 2009 equation of state, Muzny
et al. 2013 viscosity, Assael et al. 2011 conductivity) at the local bulk temperature and the inlet pressure. Those
correlations are fitted up to 1000 K; above that CoolProp extrapolates them, which is adequate for preliminary design
(the ideal-gas heat capacity it returns at 3000 K, 18.4 kJ·kg⁻¹·K⁻¹, matches the JANAF value).
"""
from __future__ import annotations

import numpy as np

from . import R_U



def h2_props(T, p: float):
    """CoolProp hydrogen ``(mu Pa·s, k W·m⁻¹·K⁻¹, cp J·kg⁻¹·K⁻¹)`` at temperatures ``T`` (K) and pressure ``p`` (Pa)."""
    from CoolProp.CoolProp import PropsSI
    T = np.atleast_1d(np.asarray(T, float))
    P = np.full_like(T, p)
    return (PropsSI("V", "T", T, "P", P, "Hydrogen"), PropsSI("L", "T", T, "P", P, "Hydrogen"),
            PropsSI("CPMASS", "T", T, "P", P, "Hydrogen"))


def cell_resistance(ri: float, ro: float, k: float, hc) -> np.ndarray:
    """Film plus conduction resistance per unit volumetric heat source, ``R_tot`` (K·m³·W⁻¹), Emrich Eq. 9.39."""
    film = (ro**2 - ri**2) / (2.0 * ri * np.asarray(hc, float))
    cond = (ro**2 * np.log(ro / ri) - 0.5 * (ro**2 - ri**2)) / (2.0 * k)
    return film + cond


def q0_from_average(P_ave: float, a: float, L: float) -> float:
    r""":math:`q_0=\frac{P_{ave}aL}{2\sin\frac{aL}{2}}` (Emrich Eq. 9.34)."""
    return P_ave * a * L / (2.0 * np.sin(0.5 * a * L))


def q0_for_peak(T_lim: float, T_in: float, ri: float, ro: float, L: float, a: float, mdot: float, cp: float, k: float,
                hc: float) -> float:
    """Peak power density that puts the hottest fuel at ``T_lim`` (constant ``hc``; Emrich Eq. 9.41 solved for q0)."""
    Af = np.pi * (ro**2 - ri**2)
    c1 = Af / (a * mdot * cp)
    c2 = float(cell_resistance(ri, ro, k, hc))
    return (T_lim - T_in) / (c1 * np.sin(0.5 * a * L) + np.hypot(c1, c2))


def dittus_boelter(T, p: float, mdot: float, ri: float):
    r""":math:`h_c=0.023\,Re^{0.8}Pr^{0.4}\,\frac{k}{D}` with CoolProp hydrogen properties at ``T`` (Emrich Eq. 9.21).
    Returns ``(hc, Re, Pr)``."""
    D = 2.0 * ri
    G = mdot / (np.pi * ri**2)
    mu, kf, cpf = h2_props(T, p)
    Re = G * D / mu
    Pr = cpf * mu / kf
    return 0.023 * Re**0.8 * Pr**0.4 * kf / D, Re, Pr


def friction_factor(Re):
    """Darcy friction factor for a smooth tube: 64/Re (laminar) or McAdams 0.184 Re^-0.2 (turbulent)."""
    Re = np.asarray(Re, float)
    return np.where(Re < 2300.0, 64.0 / np.maximum(Re, 1.0), 0.184 * np.maximum(Re, 2300.0) ** -0.2)


def axial_cell(ri: float, ro: float, L: float, a: float, q0: float, mdot: float, cp: float, k: float, T_in: float,
               hc: float | None = None, p_in: float = 7e6, mw: float = 2.016, n: int = 161) -> dict:
    r"""Temperatures and pressure along one fuel cell. ``hc=None`` uses Dittus-Boelter at the local propellant
    temperature; a number fixes it (W·m⁻²·K⁻¹). ``z`` runs from the inlet (0) to the outlet (``L``).

    Returns profiles ``z, q, Tp, Tw, Tmax, hc, p, Re, Ma`` and the scalars ``Tout, Tpk, zpk, dp, Af, Rtot``.
    """
    zc = np.linspace(-0.5 * L, 0.5 * L, n)                    # from the mid-plane, Emrich's coordinate
    Af = np.pi * (ro**2 - ri**2)
    q = q0 * np.cos(a * zc)
    Tp = T_in + q0 * Af / (a * mdot * cp) * (np.sin(a * zc) + np.sin(0.5 * a * L))
    if hc is None:
        h, Re, _ = dittus_boelter(Tp, p_in, mdot, ri)
    else:
        h = np.full(n, float(hc))
        Re = 2.0 * ri * mdot / (np.pi * ri**2) / h2_props(Tp, p_in)[0]
    Tw = Tp + q * (ro**2 - ri**2) / (2.0 * ri * h)
    Tmax = Tp + q * cell_resistance(ri, ro, k, h)
    # pressure: dp = -(f/D) G^2/(2 rho) dz - G^2 d(1/rho), rho = p / (R T)  (ideal gas)
    R = R_U / mw
    D = 2.0 * ri
    G = mdot / (np.pi * ri**2)
    f = friction_factor(Re)
    p = np.empty(n)
    p[0] = p_in
    dz = np.diff(zc)
    for i in range(n - 1):
        Tm = 0.5 * (Tp[i] + Tp[i + 1])
        fm = 0.5 * (f[i] + f[i + 1])
        pg = p[i]
        for _ in range(3):                                      # trapezoid in density, fixed-point on p[i+1]
            pm = 0.5 * (p[i] + pg)
            dpf = fm / D * G**2 * R * Tm / (2.0 * pm) * dz[i]
            dpa = G**2 * R * (Tp[i + 1] / pg - Tp[i] / p[i])
            pg = max(p[i] - dpf - dpa, 1e3)
        p[i + 1] = pg
    rho = p / (R * Tp)
    gam = cp / (cp - R)
    Ma = (G / rho) / np.sqrt(gam * R * Tp)
    i_pk = int(np.argmax(Tmax))
    return dict(z=zc + 0.5 * L, q=q, Tp=Tp, Tw=Tw, Tmax=Tmax, hc=h, p=p, Re=Re, Ma=Ma, Tout=float(Tp[-1]),
                Tpk=float(Tmax[i_pk]), zpk=float(zc[i_pk] + 0.5 * L), Twpk=float(Tw.max()), dp=float(p_in - p[-1]),
                Af=float(Af), Rtot=float(np.mean(cell_resistance(ri, ro, k, h))),
                P_cell=float(mdot * cp * (Tp[-1] - T_in)))


def peak_closed_form(ri: float, ro: float, L: float, a: float, q0: float, mdot: float, cp: float, k: float,
                     T_in: float, hc: float) -> tuple[float, float]:
    r"""Emrich Eq. 9.41: :math:`\tan(az_{pk})=\frac{C_1}{C_2}`, :math:`T_{max,pk}=T_{in}+C_1\sin\frac{aL}{2}+\sqrt{C_1^2+C_2^2}`.
    Returns ``(z_pk from the inlet, T_max_pk)``."""
    c1 = q0 * np.pi * (ro**2 - ri**2) / (a * mdot * cp)
    c2 = q0 * float(cell_resistance(ri, ro, k, hc))
    return float(0.5 * L + np.arctan2(c1, c2) / a), float(T_in + c1 * np.sin(0.5 * a * L) + np.hypot(c1, c2))
