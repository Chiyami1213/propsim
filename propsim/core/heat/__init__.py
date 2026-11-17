r"""Gas-side heat transfer (Bartz) and regenerative-cooling channel march (MATHKIT.md Section 4.2).

.. math::
    h_g=\frac{0.026}{D_t^{0.2}}\Big(\frac{\mu^{0.2}c_p}{Pr^{0.6}}\Big)\Big(\frac{p_0}{c^*}\Big)^{0.8}
    \Big(\frac{D_t}{R_c}\Big)^{0.1}\Big(\frac{A_t}{A}\Big)^{0.9}\sigma,\qquad
    \sigma=\Big[\Big(\tfrac12\tfrac{T_w}{T_0}(1+\tfrac{\gamma-1}2M^2)+\tfrac12\Big)^{0.68}(1+\tfrac{\gamma-1}2M^2)^{0.12}\Big]^{-1}

Per axial station the hot-gas wall, the wall conduction and the coolant film form a series resistance,
:math:`q=\frac{T_{aw}-T_{cool}}{1/h_g+t_w/k_w+1/h_c}`; the coolant (Dittus-Boelter, :math:`Nu=0.023Re^{0.8}Pr^{0.4}`) heats
as :math:`\dot m_c c_p\,dT=q\,2\pi r\,ds` and loses pressure by Darcy friction.  Gas transport properties use the
standard rocket estimates :math:`\mu=1.184\times10^{-7}\bar M^{1/2}T^{0.6}` and :math:`Pr=\frac{4\gamma}{9\gamma-5}`
(Huzel & Huang Ch. 4; Sutton Ch. 8).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

R_U = 8.314462618


def sigma(Tw_T0, g, M, w: float = 0.6):
    """Bartz property-variation factor (viscosity exponent ``w``; 0.68 and 0.12 correspond to w = 0.6)."""
    m = 1.0 + 0.5 * (g - 1.0) * np.asarray(M, float) ** 2
    return 1.0 / ((0.5 * Tw_T0 * m + 0.5) ** (0.8 - w / 5.0) * m ** (w / 5.0))


def bartz_hg(Dt, mu, cp, Pr, p0, cstar, Rc, A_At, sig):
    """Gas-side film coefficient (W·m⁻²·K⁻¹); ``A_At`` is the local area ratio A/A_t."""
    return (0.026 / Dt**0.2 * (mu**0.2 * cp / Pr**0.6) * (p0 / cstar) ** 0.8 * (Dt / Rc) ** 0.1
            * np.asarray(A_At, float) ** -0.9 * sig)


def gas_transport(Tc: float, Mw: float, gamma: float) -> tuple[float, float, float]:
    """Stagnation ``(cp, mu, Pr)`` of the combustion gas: ``Mw`` in kg·mol⁻¹."""
    cp = gamma / (gamma - 1.0) * R_U / Mw
    mu = 1.184e-7 * np.sqrt(Mw * 1e3) * Tc**0.6
    return cp, mu, 4.0 * gamma / (9.0 * gamma - 5.0)


def adiabatic_wall_T(T0, g, M, Pr):
    """Turbulent recovery temperature, recovery factor :math:`Pr^{1/3}`."""
    r = Pr ** (1.0 / 3.0)
    m = 0.5 * (g - 1.0) * np.asarray(M, float) ** 2
    return T0 * (1.0 + r * m) / (1.0 + m)


# ------------------------------------------------------------------------------------------- coolants
COOLANTS = {"Water": (280.0, 640.0), "Ethanol": (250.0, 500.0), "n-Dodecane": (270.0, 700.0),
            "Methane": (110.0, 600.0), "Hydrogen": (21.0, 800.0)}
COOLANT_LABELS = {"Water": "Water", "Ethanol": "Ethanol", "n-Dodecane": "Kerosene surrogate (n-dodecane)",
                  "Methane": "Methane", "Hydrogen": "Hydrogen"}


@dataclass
class Coolant:
    fluid: str
    T: np.ndarray
    rho: np.ndarray
    cp: np.ndarray
    mu: np.ndarray
    k: np.ndarray

    def at(self, T: float) -> tuple[float, float, float, float]:
        return tuple(float(np.interp(T, self.T, a)) for a in (self.rho, self.cp, self.mu, self.k))


@lru_cache(maxsize=24)
def _table(fluid: str, p_q: float) -> Coolant:
    from CoolProp.CoolProp import PropsSI
    lo, hi = COOLANTS[fluid]
    T = np.linspace(lo, hi, 70)
    p = p_q * 0.25e6
    f = lambda key: np.array([PropsSI(key, "T", float(t), "P", p, fluid) for t in T])
    return Coolant(fluid, T, f("D"), f("C"), f("V"), f("L"))


def coolant(fluid: str, p: float) -> Coolant:
    """Property table at pressure ``p`` (Pa), quantized to 0.25 MPa so slider moves reuse the cache."""
    return _table(fluid, round(p / 0.25e6))


# ------------------------------------------------------------------------------------------- regenerative channel march
@dataclass(frozen=True)
class RegenDesign:
    fluid: str
    mdot_c: float            # total coolant flow (kg·s⁻¹)
    T_in: float              # coolant inlet temperature at the nozzle exit (K)
    p_in: float              # coolant inlet pressure (Pa)
    n_ch: int                # number of channels
    h_ch: float              # channel height (m)
    fill: float              # channel width as a fraction of the pitch at each station
    t_w: float               # hot-wall thickness (m)
    k_w: float               # wall conductivity (W·m⁻¹·K⁻¹)
    T_w_max: float           # allowable hot-wall temperature (K)
    fin: float = 1.0         # rib/fin enhancement of the coolant-side coefficient (1 = smooth channel)


def regen_march(x: np.ndarray, r: np.ndarray, Rt: float, gas: dict, M: np.ndarray, des: RegenDesign,
                Rc_curv: float | None = None) -> dict[str, np.ndarray]:
    """March the coolant from the last station (nozzle exit) to the first (injector end).

    ``gas``: ``Tc, p0, cstar, gamma, Mw``.  Returns per-station arrays in the order of ``x`` (head -> exit)."""
    Tc, p0, cs, g, Mw = gas["Tc"], gas["p0"], gas["cstar"], gas["gamma"], gas["Mw"]
    cpg, mug, Pr = gas_transport(Tc, Mw, g)
    Dt = 2.0 * Rt
    Rc = Rc_curv if Rc_curv is not None else 0.382 * Rt * 2.0
    A_At = (r / Rt) ** 2
    Taw = adiabatic_wall_T(Tc, g, M, Pr)
    n = len(x)
    ds = np.hypot(np.diff(x), np.diff(r))                         # segment lengths between stations
    q, hg, Twg, Twc, Tcl, pcl, hc, vel = (np.zeros(n) for _ in range(8))
    T, p = des.T_in, des.p_in
    cool = coolant(des.fluid, des.p_in)
    for i in range(n - 1, -1, -1):
        rho, cp, mu, k = cool.at(T)
        pitch = 2.0 * np.pi * (r[i] + des.t_w + 0.5 * des.h_ch) / des.n_ch
        wch = des.fill * pitch
        Aflow, Dh = wch * des.h_ch, 2.0 * wch * des.h_ch / (wch + des.h_ch)
        v = des.mdot_c / (des.n_ch * rho * Aflow)
        Re = rho * v * Dh / mu
        Prc = mu * cp / k
        hci = des.fin * 0.023 * Re**0.8 * Prc**0.4 * k / Dh
        Tw = 0.5 * (T + Taw[i])
        for _ in range(8):                                        # fixed point on the wall temperature (sigma, h_g)
            sig = float(sigma(Tw / Tc, g, M[i]))
            h = float(bartz_hg(Dt, mug, cpg, Pr, p0, cs, Rc, A_At[i], sig))
            qi = (Taw[i] - T) / (1.0 / h + des.t_w / des.k_w + 1.0 / hci)
            Tw = Taw[i] - qi / h
        q[i], hg[i], Twg[i], hc[i], vel[i] = qi, h, Tw, hci, v
        Twc[i], Tcl[i], pcl[i] = Tw - qi * des.t_w / des.k_w, T, p
        j = i - 1 if i > 0 else None
        if j is not None:                                         # advance to the next station (toward the head end)
            dA = 2.0 * np.pi * 0.5 * (r[i] + r[j]) * ds[j]
            T = T + qi * dA / (des.mdot_c * cp)
            f = 0.184 * Re**-0.2
            p = p - f * ds[j] / Dh * rho * v**2 / 2.0
    dA_all = 2.0 * np.pi * 0.5 * (r[1:] + r[:-1]) * ds
    Q = float(np.sum(0.5 * (q[1:] + q[:-1]) * dA_all))
    return dict(q=q, h_g=hg, T_wg=Twg, T_wc=Twc, T_cool=Tcl, p_cool=pcl, h_c=hc, v=vel, T_aw=Taw, Q=Q,
                dT_cool=float(Tcl[0] - des.T_in) if len(Tcl) else 0.0, dp=float(des.p_in - pcl[0]),
                T_wg_max=float(Twg.max()), q_max=float(q.max()), margin=float(des.T_w_max - Twg.max()))
