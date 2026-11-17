r"""Solid-rocket internal ballistics (MATHKIT.md Sections 2.1-2.3): burn law, equilibrium, 0-D and spatial (1-D) transient.

Burn law :math:`r=a(T_i)\,p_c^n`, :math:`a(T_i)=a_0e^{\sigma_p(T_i-T_0)}` (``a`` in m·s⁻¹·Pa⁻ⁿ); equilibrium
:math:`p_c=(a\rho_pc^*K_n)^{\frac1{1-n}}`, :math:`K_n=A_b/A_t`.

Chamber filling with a growing free volume (state :math:`[p_c,\ w,\ V_c]`):

.. math::  \frac{dp_c}{dt}=\frac{RT_c}{V_c}\Big[(\rho_p-\rho_c)A_b r+\dot m_{ig}-\frac{p_cA_t}{c^*}\Big],\quad
           \frac{dV_c}{dt}=A_b r

Spatial form: ``N`` axial cells, each with its own burn distance :math:`w_i`; local port area, perimeter and hydraulic
diameter come from the section burnback tables.  Erosive burning (Lenoir-Robillard), solved per cell:

.. math::  r=a p^n+\alpha\frac{G^{0.8}}{D^{0.2}}\exp\Big(-\frac{\beta r\rho_p}{G}\Big)

with :math:`G` the core mass flux at the cell (igniter gas plus upstream propellant gas).  Ignition spreads from the
head end at ``v_fs``; after burnout the chamber empties through the nozzle (tail-off).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from ..constants import P_SEA_LEVEL
from ..nozzle import CF, Gamma, exit_state

R_U = 8.314462618


def a_from_mm_MPa(a_prime: float, n: float) -> float:
    r"""Unit trap: :math:`r` (mm/s) :math:`=a'p` (MPa)\ :sup:`n` gives :math:`a=10^{-3}a'(10^6)^{-n}` (m·s⁻¹·Pa⁻ⁿ)."""
    return 1e-3 * a_prime * (1e6) ** (-n)


@dataclass(frozen=True)
class SolidProp:
    rho_p: float = 1750.0       # (kg·m⁻³)
    a: float = a_from_mm_MPa(5.0, 0.35)     # reference coefficient at T0 (m·s⁻¹·Pa⁻ⁿ)
    n: float = 0.35
    sigma_p: float = 0.002      # temperature sensitivity (K⁻¹)
    T0: float = 293.15          # reference grain temperature (K)
    Ti: float = 293.15          # initial grain temperature (K)
    cstar: float = 1500.0       # (m·s⁻¹)
    Tc: float = 3000.0          # (K)
    gamma: float = 1.2
    alpha: float = 6.0e-6       # Lenoir-Robillard coefficient (SI); 0 disables erosive burning
    beta: float = 53.0

    @property
    def a_eff(self) -> float:
        return self.a * float(np.exp(self.sigma_p * (self.Ti - self.T0)))

    @property
    def RT(self) -> float:
        return (self.cstar * float(Gamma(self.gamma))) ** 2

    def rate(self, p):
        """Base burn rate (m·s⁻¹) at chamber pressure ``p`` (Pa), including the temperature sensitivity."""
        return self.a_eff * np.maximum(p, 1.0) ** self.n


def p_equilibrium(prop: SolidProp, Kn: float) -> float:
    """:math:`p_c=(a\\rho_pc^*K_n)^{1/(1-n)}` (Pa); only meaningful for n < 1."""
    if prop.n >= 1.0:
        raise ValueError("equilibrium pressure does not exist for n >= 1 (unstable)")
    return (prop.a_eff * prop.rho_p * prop.cstar * Kn) ** (1.0 / (1.0 - prop.n))


def stability_note(n: float) -> str:
    if n >= 1.0:
        return "UNSTABLE: n >= 1, no stable equilibrium pressure"
    if n >= 0.7:
        return "WARNING: n >= 0.7, pressure is very sensitive to K_n and grain temperature"
    return "stable (n < 0.7)"


def lenoir_robillard(r0: np.ndarray, G: np.ndarray, D: np.ndarray, prop: SolidProp) -> np.ndarray:
    """Solve :math:`r=r_0+\\alpha G^{0.8}D^{-0.2}e^{-\\beta r\\rho_p/G}` per cell (``brentq``, explicit bracket)."""
    r0, G, D = np.broadcast_arrays(np.asarray(r0, float), np.asarray(G, float), np.asarray(D, float))
    out = r0.copy()
    if prop.alpha <= 0.0:
        return out
    for i in np.ndindex(r0.shape):
        g, d = G[i], D[i]
        if g <= 1e-9 or d <= 1e-9:
            continue
        hi = r0[i] + prop.alpha * g**0.8 * d**-0.2
        f = lambda r, i=i, g=g, d=d: r - r0[i] - prop.alpha * g**0.8 * d**-0.2 * np.exp(-prop.beta * r * prop.rho_p / g)
        out[i] = brentq(f, r0[i], hi, xtol=1e-14, rtol=1e-12)
    return out


# ------------------------------------------------------------------------------------------- motor definition
@dataclass(frozen=True)
class SectionTable:
    """Port geometry vs burn distance for an extruded grain section (SI)."""

    d: np.ndarray               # burn distance (m)
    A_p: np.ndarray             # port area (m²)
    P: np.ndarray               # burning perimeter (m)

    @property
    def web(self) -> float:
        return float(self.d[-1])

    def area(self, w):
        return np.interp(w, self.d, self.A_p)

    def perim(self, w):
        return np.where(np.asarray(w) >= self.web, 0.0, np.interp(w, self.d, self.P))


@dataclass(frozen=True)
class SolidMotor:
    prop: SolidProp
    table: SectionTable
    L: float                    # grain length (m)
    At: float                   # throat area (m²)
    eps: float                  # nozzle expansion ratio
    N: int = 20                 # axial cells (1 = 0-D)
    V_extra: float = 5e-4       # head-end and aft free volume (m³)
    m_ig: float = 0.05          # igniter gas mass (kg)
    t_ig: float = 0.1           # igniter burn time (s)
    v_fs: float = 200.0         # flame-spreading speed (m·s⁻¹); large = instantaneous
    p_ign: float = 0.3e6        # pressure below which the propellant does not sustain burning (Pa)
    p0: float = P_SEA_LEVEL
    R_case: float = 0.06        # grain outer radius (m)

    @property
    def R_out(self) -> float:
        return self.R_case

    @property
    def R0(self) -> float:
        """Equivalent circular port radius at ignition (m)."""
        return float(np.sqrt(self.table.A_p[0] / np.pi))

    @property
    def Dt(self) -> float:
        return float(2.0 * np.sqrt(self.At / np.pi))

    @property
    def dx(self) -> float:
        return self.L / self.N

    def ignition_times(self) -> np.ndarray:
        return (np.arange(self.N) + 0.5) * self.dx / self.v_fs


# ------------------------------------------------------------------------------------------- physics
def evaluate(m: SolidMotor, t: float, y: np.ndarray) -> dict:
    """Instantaneous quantities for state ``y = [w_1..w_N, p_c, V_c]``."""
    N, prop = m.N, m.prop
    w, p, Vc = y[:N], y[N], y[N + 1]
    tab = m.table
    Ap, Pb = tab.area(w), tab.perim(w)
    D = 4.0 * Ap / np.maximum(Pb, 1e-12)
    sustain = (t < m.t_ig) or (p >= m.p_ign)                  # the igniter lights it; afterwards the pressure must sustain it
    lit = (t >= m.ignition_times()) & (Pb > 0.0) & sustain
    r0 = np.where(lit, prop.rate(p), 0.0)
    mig = m.m_ig / m.t_ig if t < m.t_ig else 0.0
    # head-to-aft march: core flux includes the igniter gas and the propellant gas from upstream cells
    r = np.zeros(N)
    G = np.zeros(N)
    cum = mig
    for i in range(N):
        if not lit[i]:
            G[i] = cum / max(Ap[i], 1e-12)
            continue
        g_in = cum / max(Ap[i], 1e-12)
        ri = float(lenoir_robillard(r0[i], max(g_in, 1e-9), D[i], prop)) if prop.alpha > 0 and g_in > 1e-9 else r0[i]
        # iterate once with mid-cell flux (cell's own gas contributes half)
        gen = prop.rho_p * Pb[i] * m.dx * ri
        gmid = (cum + 0.5 * gen) / max(Ap[i], 1e-12)
        ri = float(lenoir_robillard(r0[i], max(gmid, 1e-9), D[i], prop)) if prop.alpha > 0 and gmid > 1e-9 else r0[i]
        r[i], G[i] = ri, gmid
        cum += prop.rho_p * Pb[i] * m.dx * ri
    dVc = float(np.sum(Pb * m.dx * r))                       # = A_b r summed
    Ab = float(np.sum(Pb * m.dx))
    mgen = prop.rho_p * dVc
    rho_c = p / prop.RT
    dp = prop.RT / Vc * ((prop.rho_p - rho_c) * dVc + mig - p * m.At / prop.cstar)
    return dict(w=w, p=p, Vc=Vc, r=r, r0=r0, G=G, Ab=Ab, Ap=Ap, Pb=Pb, D=D, mgen=mgen, mig=mig, dp=dp, dVc=dVc,
                dw=r, lit=lit)


def _rhs(t: float, y, m: SolidMotor):
    e = evaluate(m, t, y)
    return np.concatenate([e["dw"], [e["dp"], e["dVc"]]])


def integrate(m: SolidMotor, t_end: float, n_frames: int = 601, progress: Callable[[float], None] | None = None,
              cancel=None, n_chunks: int = 40) -> dict[str, np.ndarray]:
    t = np.linspace(0.0, t_end, n_frames)
    N = m.N
    Vc0 = float(np.sum(m.table.area(np.zeros(N))) * m.dx) + m.V_extra
    y = np.concatenate([np.zeros(N), [m.p0, Vc0]])
    Y = np.empty((len(y), n_frames))
    Y[:, 0] = y
    edges = np.linspace(0, n_frames - 1, n_chunks + 1).astype(int)
    atol = np.concatenate([np.full(N, 1e-9), [1e-3, 1e-9]])
    for k in range(n_chunks):
        i0, i1 = edges[k], edges[k + 1]
        if i1 <= i0:
            continue
        if cancel is not None and cancel.is_set():
            raise InterruptedError("cancelled")
        sol = solve_ivp(_rhs, (t[i0], t[i1]), y, args=(m,), method="LSODA", t_eval=t[i0:i1 + 1], rtol=1e-8, atol=atol,
                        max_step=max((t[i1] - t[i0]) / 2.0, 1e-4))
        if not sol.success:
            raise RuntimeError(f"integrator failed: {sol.message}")
        Y[:, i0:i1 + 1] = sol.y
        y = sol.y[:, -1]
        y[:N] = np.minimum(y[:N], m.table.web)            # burnt-out cells stay at the web
        if progress:
            progress((k + 1) / n_chunks)
    return derived(m, t, Y)


def derived(m: SolidMotor, t, Y) -> dict[str, np.ndarray]:
    N, prop = m.N, m.prop
    n = len(t)
    keys = ["Ab", "mgen", "mig", "r_mean", "r_head", "r_aft", "eros_aft"]
    d = {k: np.empty(n) for k in keys}
    W = np.minimum(Y[:N], m.table.web)
    r_x, G_x = np.empty((N, n)), np.empty((N, n))
    lit_frac = np.empty(n)
    for i in range(n):
        e = evaluate(m, t[i], np.concatenate([W[:, i], Y[N:, i]]))
        d["Ab"][i], d["mgen"][i], d["mig"][i] = e["Ab"], e["mgen"], e["mig"]
        burning = e["r"] > 0
        d["r_mean"][i] = e["r"][burning].mean() * 1e3 if burning.any() else 0.0
        d["r_head"][i], d["r_aft"][i] = e["r"][0] * 1e3, e["r"][-1] * 1e3
        d["eros_aft"][i] = e["r"][-1] / e["r0"][-1] if e["r0"][-1] > 0 else 1.0
        r_x[:, i], G_x[:, i] = e["r"] * 1e3, e["G"]
        lit_frac[i] = float(np.mean(e["r"] > 0))
    p = np.maximum(Y[N], 0.0)                                 # solver noise can leave -1e-3 Pa after tail-off
    ge, pe_pc = prop.gamma, exit_state(m.eps, prop.gamma)[1]
    cf = np.array([CF(ge, pe_pc, P_SEA_LEVEL / max(pi, 1.0), m.eps) for pi in p])
    F = np.maximum(cf * p * m.At, 0.0)
    out = {"Pc": p / 1e6, "F": F / 1e3, "Ab": d["Ab"], "mdot": p * m.At / prop.cstar, "mdot_gen": d["mgen"],
           "Vc": Y[N + 1], "w_mean": W.mean(axis=0) * 1e3, "rdot": d["r_mean"], "rdot_head": d["r_head"],
           "rdot_aft": d["r_aft"], "erosive_aft": d["eros_aft"], "CF": cf}
    out["Kn"] = d["Ab"] / m.At
    out["lit_frac"] = lit_frac
    for i in range(N):
        out[f"w_{i:02d}"] = W[i] * 1e3
        out[f"rd_{i:02d}"] = r_x[i]
        out[f"G_{i:02d}"] = G_x[i]
    return out


def summary(m: SolidMotor, t, ch: dict[str, np.ndarray]) -> dict[str, float]:
    F = ch["F"] * 1e3
    It = float(np.trapezoid(F, t))
    mprop = float(np.trapezoid(ch["mdot_gen"], t) + m.m_ig)
    above = np.where(ch["Pc"] >= 0.1 * ch["Pc"].max())[0]
    tb = float(t[above[-1]] - t[above[0]]) if above.size > 1 else 0.0
    return {"I_t_Ns": It, "Isp_avg": It / (mprop * 9.80665) if mprop > 0 else float("nan"), "m_prop": mprop,
            "Pc_max_MPa": float(ch["Pc"].max()), "F_max_kN": float(ch["F"].max()), "t_burn": tb,
            "Pc_eq_MPa": float(p_equilibrium(m.prop, float(ch["Kn"].max())) / 1e6) if m.prop.n < 1 else float("nan"),
            "Kn_max": float(ch["Kn"].max()), "web_mm": m.table.web * 1e3,
            "erosive_peak": float(np.max(ch["erosive_aft"])), "note": stability_note(m.prop.n)}
