r"""Solid-rocket internal ballistics (MATHKIT.md Sections 2.1-2.3): burn law, equilibrium, 0-D and spatial (1-D) transient.

Burn law :math:`r=a(T_i)\,p_c^n`, :math:`a(T_i)=a_0e^{\sigma_p(T_i-T_0)}` (``a`` in m·s⁻¹·Pa⁻ⁿ); equilibrium
:math:`p_c=(a\rho_pc^*K_n)^{\frac1{1-n}}`, :math:`K_n=A_b/A_t`.

Chamber filling with a growing free volume (state :math:`[p_c,\ w,\ V_c]`):

.. math::  \frac{dp_c}{dt}=\frac{RT_c}{V_c}\Big[(\rho_p-\rho_c)A_b r+\dot m_{ig}-\frac{p_cA_t}{c^*}\Big],\quad
           \frac{dV_c}{dt}=A_b r

Spatial form: ``N`` axial cells, each with its own burn distance :math:`w_i`; local port area, perimeter and hydraulic
diameter come from the section burnback tables.  The motor can be a *stack of grains* (:class:`Grain`), each with its
own section table, length, propellant and burning or inhibited end faces; the cells are shared out among the grains
in proportion to their length.  An uninhibited end face regresses axially at the grain's burn rate, and the grain's
cells shrink with it, so with :math:`e_h, e_a` the burnt head and aft depths the cell length is
:math:`\Delta x_i\,(L_g-e_h-e_a)/L_g`.  With several propellants the chamber gas is the mass-flow-weighted mean of the
propellants' :math:`RT` and :math:`\gamma` (frozen composition), and :math:`c^*=\sqrt{RT}/\Gamma(\gamma)`.  Erosive burning (Lenoir-Robillard), solved per cell:

.. math::  r=a p^n+\alpha\frac{G^{0.8}}{D^{0.2}}\exp\Big(-\frac{\beta r\rho_p}{G}\Big)

with :math:`G` the core mass flux at the cell (igniter gas plus upstream propellant gas).  Ignition spreads from the
head end at ``v_fs``; after burnout the chamber empties through the nozzle (tail-off).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
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
    A_case: float = 0.0         # grain outer area (m²); needed for burning end faces (0 = unknown)

    @property
    def web(self) -> float:
        return float(self.d[-1])

    def area(self, w):
        return np.interp(w, self.d, self.A_p)

    def perim(self, w):
        return np.where(np.asarray(w) >= self.web, 0.0, np.interp(w, self.d, self.P))


@dataclass(frozen=True)
class Grain:
    """One grain of a stack: its own propellant, section table, length and end-face inhibition."""

    prop: SolidProp
    table: SectionTable
    L: float                    # (m)
    burn_head: bool = False     # head-end face uninhibited (burns axially)
    burn_aft: bool = False      # aft-end face uninhibited


@dataclass(frozen=True)
class SolidMotor:
    prop: SolidProp
    table: SectionTable
    L: float                    # grain length (m); total length of the stack when ``grains`` is given
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
    grains: tuple = ()          # optional stack of :class:`Grain`; empty = one grain from ``prop``/``table``/``L``

    @classmethod
    def from_grains(cls, grains, **kw) -> "SolidMotor":
        grains = tuple(grains)
        return cls(prop=grains[0].prop, table=grains[0].table, L=sum(g.L for g in grains), grains=grains, **kw)

    @property
    def stack(self) -> tuple:
        return self.grains or (Grain(self.prop, self.table, self.L),)

    @cached_property
    def cells(self) -> tuple[np.ndarray, np.ndarray]:
        """``(grain index, nominal length)`` of every axial cell; at least one cell per grain."""
        st = self.stack
        if len(st) == 1:
            return np.zeros(self.N, int), np.full(self.N, st[0].L / self.N)
        n = [max(1, int(round(self.N * g.L / self.L))) for g in st]
        return (np.repeat(np.arange(len(st)), n), np.concatenate([np.full(k, g.L / k) for k, g in zip(n, st)]))

    @property
    def nc(self) -> int:
        return len(self.cells[0])

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
        return self.L / self.nc

    def cell_x(self) -> np.ndarray:
        """Cell-centre positions along the axis from the head end (m), at ignition."""
        dx = self.cells[1]
        return np.cumsum(dx) - 0.5 * dx

    def ignition_times(self) -> np.ndarray:
        return self.cell_x() / self.v_fs


def _grain_slices(m: SolidMotor) -> list[slice]:
    gi = m.cells[0]
    return [slice(int(np.argmax(gi == g)), int(np.argmax(gi == g) + np.sum(gi == g))) for g in range(len(m.stack))]


def _grain_mass0(g: Grain) -> float:
    """Propellant mass of a grain at ignition (kg); used only as the fallback mixing weight when nothing burns."""
    vol = (g.table.A_case - float(g.table.A_p[0])) * g.L if g.table.A_case > 0 else g.L
    return g.prop.rho_p * vol


# ------------------------------------------------------------------------------------------- physics
def evaluate(m: SolidMotor, t: float, y: np.ndarray) -> dict:
    """Instantaneous quantities for state ``y = [w_1..w_nc, p_c, V_c, e_head_1..e_head_G, e_aft_1..e_aft_G]``."""
    nc, st = m.nc, m.stack
    G = len(st)
    gi, dx0 = m.cells
    sl = _grain_slices(m)
    w, p, Vc = y[:nc], y[nc], y[nc + 1]
    e_h, e_a = y[nc + 2:nc + 2 + G], y[nc + 2 + G:nc + 2 + 2 * G]
    Ap, Pb = np.empty(nc), np.empty(nc)
    for g, gr in enumerate(st):
        Ap[sl[g]], Pb[sl[g]] = gr.table.area(w[sl[g]]), gr.table.perim(w[sl[g]])
    Lg = np.array([gr.L for gr in st])
    f = np.clip((Lg - e_h - e_a) / Lg, 0.0, 1.0)
    dx = dx0 * f[gi]
    D = 4.0 * Ap / np.maximum(Pb, 1e-12)
    sustain = (t < m.t_ig) or (p >= m.p_ign)                  # the igniter lights it; afterwards the pressure must sustain it
    ign = m.ignition_times()
    lit = (t >= ign) & (Pb > 0.0) & sustain
    r0 = np.array([st[gi[i]].prop.rate(p) for i in range(nc)]) if G > 1 else st[0].prop.rate(p) * np.ones(nc)
    r0 = np.where(lit, r0, 0.0)
    mig = m.m_ig / m.t_ig if t < m.t_ig else 0.0
    # head-to-aft march: core flux includes the igniter gas and the propellant gas from upstream cells
    r = np.zeros(nc)
    Gf = np.zeros(nc)
    cum = mig
    gen_g = np.zeros(G)                                       # propellant gas mass flow by grain (kg/s)
    for i in range(nc):
        prop = st[gi[i]].prop
        if not lit[i]:
            Gf[i] = cum / max(Ap[i], 1e-12)
            continue
        g_in = cum / max(Ap[i], 1e-12)
        ri = float(lenoir_robillard(r0[i], max(g_in, 1e-9), D[i], prop)) if prop.alpha > 0 and g_in > 1e-9 else r0[i]
        # iterate once with mid-cell flux (cell's own gas contributes half)
        gen = prop.rho_p * Pb[i] * dx[i] * ri
        gmid = (cum + 0.5 * gen) / max(Ap[i], 1e-12)
        ri = float(lenoir_robillard(r0[i], max(gmid, 1e-9), D[i], prop)) if prop.alpha > 0 and gmid > 1e-9 else r0[i]
        r[i], Gf[i] = ri, gmid
        gen = prop.rho_p * Pb[i] * dx[i] * ri
        cum += gen
        gen_g[gi[i]] += gen
    dV_side = float(np.sum(Pb * dx * r))
    dVc = dV_side
    Ab = float(np.sum(Pb * dx))
    # end faces: an uninhibited face burns axially at the base rate (no erosive augmentation), over A_case - A_p
    de_h, de_a = np.zeros(G), np.zeros(G)
    for g, gr in enumerate(st):
        if f[g] <= 0.0 or not (gr.burn_head or gr.burn_aft) or gr.table.A_case <= 0.0:
            continue
        rb = float(gr.prop.rate(p)) if sustain else 0.0
        for flag, cell, de in ((gr.burn_head, sl[g].start, de_h), (gr.burn_aft, sl[g].stop - 1, de_a)):
            if flag and t >= ign[cell]:
                A_face = max(gr.table.A_case - float(Ap[cell]), 0.0)
                de[g] = rb if A_face > 0.0 else 0.0
                dVc += A_face * de[g]
                gen_g[g] += gr.prop.rho_p * A_face * de[g]
                Ab += A_face * (1.0 if de[g] > 0.0 else 0.0)
    mgen = float(gen_g.sum())
    if G == 1:
        RT, cstar, gam = st[0].prop.RT, st[0].prop.cstar, st[0].prop.gamma
    else:
        wts = gen_g.copy()
        wts[0] += mig
        if wts.sum() <= 1e-12:
            wts = np.array([_grain_mass0(g_) for g_ in st])
        wts = wts / wts.sum()
        RT = float(sum(wts[g] * st[g].prop.RT for g in range(G)))
        gam = float(sum(wts[g] * st[g].prop.gamma for g in range(G)))
        cstar = float(np.sqrt(RT) / Gamma(gam))
    rho_c = p / RT
    dp = RT / Vc * (mgen - rho_c * dVc + mig - p * m.At / cstar)
    return dict(w=w, p=p, Vc=Vc, r=r, r0=r0, G=Gf, Ab=Ab, Ap=Ap, Pb=Pb, D=D, mgen=mgen, mig=mig, dp=dp, dVc=dVc,
                dw=r, lit=lit, RT=RT, cstar=cstar, gamma=gam, de_head=de_h, de_aft=de_a, gen_g=gen_g, f=f)


def _rhs(t: float, y, m: SolidMotor):
    e = evaluate(m, t, y)
    return np.concatenate([e["dw"], [e["dp"], e["dVc"]], e["de_head"], e["de_aft"]])


def _clip_state(m: SolidMotor, y: np.ndarray) -> None:
    """Burnt-out cells stay at their web; the axial burn of a grain cannot exceed its length."""
    nc, st = m.nc, m.stack
    G = len(st)
    gi = m.cells[0]
    y[:nc] = np.minimum(y[:nc], np.array([st[g].table.web for g in gi]))
    for g, gr in enumerate(st):
        eh, ea = y[nc + 2 + g], y[nc + 2 + G + g]
        if eh + ea > gr.L:
            s = gr.L / (eh + ea)
            y[nc + 2 + g], y[nc + 2 + G + g] = eh * s, ea * s


def initial_state(m: SolidMotor) -> np.ndarray:
    nc, G = m.nc, len(m.stack)
    gi, dx0 = m.cells
    Vc0 = float(sum(np.sum(gr.table.area(np.zeros(1)) * dx0[gi == g]) for g, gr in enumerate(m.stack))) + m.V_extra
    return np.concatenate([np.zeros(nc), [m.p0, Vc0], np.zeros(2 * G)])


def integrate(m: SolidMotor, t_end: float, n_frames: int = 601, progress: Callable[[float], None] | None = None,
              cancel=None, n_chunks: int = 40) -> dict[str, np.ndarray]:
    t = np.linspace(0.0, t_end, n_frames)
    nc, G = m.nc, len(m.stack)
    y = initial_state(m)
    Y = np.empty((len(y), n_frames))
    Y[:, 0] = y
    edges = np.linspace(0, n_frames - 1, n_chunks + 1).astype(int)
    atol = np.concatenate([np.full(nc, 1e-9), [1e-3, 1e-9], np.full(2 * G, 1e-9)])
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
        _clip_state(m, y)
        if progress:
            progress((k + 1) / n_chunks)
    return derived(m, t, Y)


def derived(m: SolidMotor, t, Y) -> dict[str, np.ndarray]:
    nc, st = m.nc, m.stack
    n = len(t)
    keys = ["Ab", "mgen", "mig", "r_mean", "r_head", "r_aft", "eros_aft", "cstar", "gamma"]
    d = {k: np.empty(n) for k in keys}
    webs = np.array([st[g].table.web for g in m.cells[0]])
    W = np.minimum(Y[:nc], webs[:, None])
    r_x, G_x = np.empty((nc, n)), np.empty((nc, n))
    lit_frac = np.empty(n)
    for i in range(n):
        e = evaluate(m, t[i], np.concatenate([W[:, i], Y[nc:, i]]))
        d["Ab"][i], d["mgen"][i], d["mig"][i] = e["Ab"], e["mgen"], e["mig"]
        d["cstar"][i], d["gamma"][i] = e["cstar"], e["gamma"]
        burning = e["r"] > 0
        d["r_mean"][i] = e["r"][burning].mean() * 1e3 if burning.any() else 0.0
        d["r_head"][i], d["r_aft"][i] = e["r"][0] * 1e3, e["r"][-1] * 1e3
        d["eros_aft"][i] = e["r"][-1] / e["r0"][-1] if e["r0"][-1] > 0 else 1.0
        r_x[:, i], G_x[:, i] = e["r"] * 1e3, e["G"]
        lit_frac[i] = float(np.mean(e["r"] > 0))
    p = np.maximum(Y[nc], 0.0)                                # solver noise can leave -1e-3 Pa after tail-off
    pe_pc = np.array([exit_state(m.eps, g)[1] for g in d["gamma"]])
    cf = np.array([CF(g, pe, P_SEA_LEVEL / max(pi, 1.0), m.eps) for g, pe, pi in zip(d["gamma"], pe_pc, p)])
    F = np.maximum(cf * p * m.At, 0.0)
    out = {"Pc": p / 1e6, "F": F / 1e3, "Ab": d["Ab"], "mdot": p * m.At / d["cstar"], "mdot_gen": d["mgen"],
           "Vc": Y[nc + 1], "w_mean": W.mean(axis=0) * 1e3, "rdot": d["r_mean"], "rdot_head": d["r_head"],
           "rdot_aft": d["r_aft"], "erosive_aft": d["eros_aft"], "CF": cf}
    out["Kn"] = d["Ab"] / m.At
    out["lit_frac"] = lit_frac
    if len(st) > 1:
        out["cstar_eff"], out["gamma_eff"] = d["cstar"], d["gamma"]
    G = len(st)
    for g in range(G):
        if st[g].burn_head or st[g].burn_aft:
            out[f"L_eff_{g}"] = (st[g].L - Y[nc + 2 + g] - Y[nc + 2 + G + g]) * 1e3
    for i in range(nc):
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
    st = m.stack
    same = all(g.prop == st[0].prop for g in st)
    n_max = max(g.prop.n for g in st)
    return {"I_t_Ns": It, "Isp_avg": It / (mprop * 9.80665) if mprop > 0 else float("nan"), "m_prop": mprop,
            "Pc_max_MPa": float(ch["Pc"].max()), "F_max_kN": float(ch["F"].max()), "t_burn": tb,
            "Pc_eq_MPa": (float(p_equilibrium(m.prop, float(ch["Kn"].max())) / 1e6)
                          if same and m.prop.n < 1 else float("nan")),
            "Kn_max": float(ch["Kn"].max()), "web_mm": max(g.table.web for g in st) * 1e3,
            "erosive_peak": float(np.max(ch["erosive_aft"])), "note": stability_note(n_max),
            "n_grains": len(st), "mixed": not same}
