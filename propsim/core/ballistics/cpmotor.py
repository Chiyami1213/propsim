r"""Quasi-steady cylindrical-perforated (CP) solid motor marched in web distance, with optional end-face burning and
throat erosion.

For a hollow cylinder (outer radius :math:`r_o`, bore :math:`r_i`, length :math:`L`, outer surface inhibited) burnt
through a web :math:`w`:

.. math::
    A_b=2\pi(r_i+w)(L-2\epsilon_ew)+\epsilon_e\,2\pi\big(r_o^2-(r_i+w)^2\big),\qquad
    V_p=\pi\big(r_o^2-(r_i+w)^2\big)(L-2\epsilon_ew),

with :math:`\epsilon_e=1` when both end faces burn (they recede at the burn rate, so the length shrinks by :math:`2w`)
and 0 when they are inhibited.  The chamber pressure is the equilibrium of the burn law :math:`\dot r=ap_c^n` and the
nozzle flow,

.. math::
    p_c=\Big(a\rho_pc^*\frac{A_b}{A_t}\Big)^{\frac1{1-n}},\qquad dt=\frac{dw}{\dot r},\qquad
    F=C_F(\gamma,\varepsilon,p_c)\,A_tp_c,\qquad \varepsilon=\frac{A_e}{A_t}

(:math:`C_F` from the ideal-gas nozzle relations).  Throat erosion grows the radius per time step by
:math:`\dot e_{ref}\,dt` (constant rate) or :math:`\dot e_{ref}(p_c/p_{ref})^{0.8}dt` (pressure dependent), using the
step's own :math:`p_c` and :math:`dt`.  The march ends when the web is consumed (bore meets the outer radius, or the
end faces meet at :math:`L/2`); there is no tail-off.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import G0, P_SEA_LEVEL
from ..nozzle import CF, exit_state

EROSION = ("none", "const", "pressure")
EROSION_LABELS = ("Constant throat area", "Throat erosion (constant rate)", "Throat erosion (pressure dependent)")


@dataclass(frozen=True)
class CPMotor:
    r_o: float                 # grain outer radius (m)
    r_i: float                 # initial bore radius (m)
    L: float                   # grain length (m)
    rho_p: float               # propellant density (kg·m⁻³)
    cstar: float               # characteristic velocity (m·s⁻¹)
    n: float                   # burn-rate pressure exponent
    r_ref: float               # reference burn rate (m·s⁻¹) at ``p_ref``
    p_ref: float               # reference chamber pressure (Pa)
    R_t: float                 # initial throat radius (m)
    R_e: float                 # nozzle exit radius (m)
    gamma: float = 1.2
    ends: bool = False         # both end faces burn
    erosion: str = "none"      # "none" | "const" | "pressure"
    edot_ref: float = 0.0      # throat erosion rate at ``p_ref`` (m·s⁻¹)
    dw: float = 2e-4           # web step (m)

    @property
    def a(self) -> float:
        """Burn-rate coefficient (m·s⁻¹·Pa⁻ⁿ) from the reference point."""
        return self.r_ref / self.p_ref**self.n

    @property
    def web(self) -> float:
        return min(self.r_o - self.r_i, 0.5 * self.L) if self.ends else self.r_o - self.r_i

    def check(self) -> None:
        if not (0.0 < self.r_i < self.r_o):
            raise ValueError("the bore radius must be positive and smaller than the outer radius")
        if self.n >= 1.0:
            raise ValueError("pressure exponent n must be below 1 for a stable equilibrium pressure")
        if self.R_e <= self.R_t:
            raise ValueError("the nozzle exit radius must exceed the throat radius")
        if self.erosion not in EROSION:
            raise ValueError(f"erosion must be one of {EROSION}")


def burn_area(m: CPMotor, w):
    w = np.asarray(w, float)
    e = 1.0 if m.ends else 0.0
    ri = m.r_i + w
    return 2 * np.pi * ri * (m.L - 2 * e * w) + e * 2 * np.pi * (m.r_o**2 - ri**2)


def propellant_volume(m: CPMotor, w):
    w = np.asarray(w, float)
    e = 1.0 if m.ends else 0.0
    return np.pi * (m.r_o**2 - (m.r_i + w) ** 2) * (m.L - 2 * e * w)


def cf_pair(gamma: float, eps: np.ndarray, pc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(C_F vacuum, C_F at sea level, p_e/p_c)`` for expansion ratios ``eps`` (interpolated from an exact table)."""
    lo, hi = max(float(eps.min()) * 0.999, 1.0001), float(eps.max()) * 1.001
    grid = np.geomspace(lo, hi, 80)
    pe = np.array([exit_state(e, gamma)[1] for e in grid])
    pe_pc = np.exp(np.interp(np.log(eps), np.log(grid), np.log(pe)))
    cf_v = CF(gamma, pe_pc, 0.0, eps)
    cf_sl = CF(gamma, pe_pc, P_SEA_LEVEL / np.maximum(pc, 1.0), eps)
    return cf_v, cf_sl, pe_pc


def march(m: CPMotor) -> dict[str, np.ndarray]:
    """March the web; every array has one entry per web station (SI units)."""
    m.check()
    n_pts = int(np.ceil(m.web / m.dw)) + 1
    w = np.linspace(0.0, m.web, n_pts)
    dw = w[1] - w[0] if n_pts > 1 else m.dw
    Ab = burn_area(m, w)
    a, n = m.a, m.n
    Rt = np.empty(n_pts)
    Pc, rdot, dt = np.empty(n_pts), np.empty(n_pts), np.empty(n_pts)
    Rt[0] = m.R_t
    for i in range(n_pts):
        At = np.pi * Rt[i] ** 2
        Pc[i] = (a * m.rho_p * m.cstar * Ab[i] / At) ** (1.0 / (1.0 - n))
        rdot[i] = a * Pc[i] ** n
        dt[i] = dw / rdot[i]
        if i + 1 < n_pts:
            if m.erosion == "const":
                Rt[i + 1] = Rt[i] + dt[i] * m.edot_ref
            elif m.erosion == "pressure":
                Rt[i + 1] = Rt[i] + dt[i] * m.edot_ref * (Pc[i] / m.p_ref) ** 0.8
            else:
                Rt[i + 1] = Rt[i]
    t = np.concatenate([[0.0], np.cumsum(dt[:-1])])
    At = np.pi * Rt**2
    eps = np.pi * m.R_e**2 / At
    cf_v, cf_sl, pe_pc = cf_pair(m.gamma, eps, Pc)
    mdot = At * Pc / m.cstar
    Vp = propellant_volume(m, w)
    return dict(w=w, t=t, dt=dt, Pc=Pc, Ab=Ab, rdot=rdot, Rt=Rt, At=At, eps=eps, Kn=Ab / At, CFv=cf_v, CFsl=cf_sl,
                F_vac=cf_v * At * Pc, F_sl=cf_sl * At * Pc, mdot=mdot, m_p=m.rho_p * Vp, pe_pc=pe_pc,
                mdot_in=m.rho_p * Ab * rdot)


def summary(m: CPMotor, d: dict[str, np.ndarray]) -> dict[str, float]:
    t_b = float(d["t"][-1] + d["dt"][-1])
    t = np.concatenate([d["t"], [t_b]])
    ext = lambda y: np.concatenate([y, [y[-1]]])
    It_vac = float(np.trapezoid(ext(d["F_vac"]), t))
    It_sl = float(np.trapezoid(ext(d["F_sl"]), t))
    m_burn = float(d["m_p"][0] - d["m_p"][-1])
    return dict(t_burn=t_b, I_vac=It_vac, I_sl=It_sl, m_propellant=float(d["m_p"][0]), m_burned=m_burn,
                Pc_max=float(d["Pc"].max()), Pc_avg=float(np.trapezoid(ext(d["Pc"]), t) / t_b),
                F_vac_max=float(d["F_vac"].max()), F_sl_max=float(d["F_sl"].max()),
                Isp_vac=It_vac / (max(m_burn, 1e-12) * G0), Isp_sl=It_sl / (max(m_burn, 1e-12) * G0),
                Kn_max=float(d["Kn"].max()), Rt_end=float(d["Rt"][-1]), web=m.web)
