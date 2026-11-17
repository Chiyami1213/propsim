r"""Liquid feed-network transient (MATHKIT.md Section 4.3): tank -> line -> valve -> manifold -> injector -> chamber.

Nodes carry pressure with liquid capacitance :math:`C=\frac{V}{\rho a^2}`; branches carry mass flow with inertance
:math:`\mathcal L=\frac lA` and resistance :math:`K\frac{\dot m|\dot m|}{2\rho A^2}`:

.. math::
    \mathcal L\dot m=p_{up}-p_{down}-K\frac{\dot m|\dot m|}{2\rho A^2},\qquad C\dot p=\sum\dot m_{in}-\sum\dot m_{out}

Per propellant chain the states are ``[m1, p1, m2, p2, dV]`` (line flow, valve-inlet pressure, valve flow, manifold
pressure, liquid volume expelled for the ullage); the chamber has ``p_c`` and one first-order combustion lag per
propellant.  The valve resistance is :math:`K_v(t)=K_{v,0}/f(t)^2` with the opening schedule ``f(t)``.  Pressurant is
either regulated (constant tank pressure) or blowdown (polytropic ullage).  The instantaneous closure limit reproduces
the Joukowsky surge :math:`\Delta p=\rho a v`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from ..constants import P_SEA_LEVEL
from ..nozzle import CF, Gamma, exit_state
from ..thermo.cea import chamber_lut

P_AMB = P_SEA_LEVEL
LP_TAU = 0.003         # (s) low-pass time constant: line damping acts on flow deviations faster than this


def valve_fraction(t: float, t_open: float, ramp_open: float, t_close: float, ramp_close: float,
                   floor: float = 1e-4) -> float:
    """Opening fraction: smooth 0 -> 1 over ``ramp_open`` from ``t_open``; 1 -> 0 over ``ramp_close`` from ``t_close``."""
    smooth = lambda x: x * x * (3.0 - 2.0 * x)                     # smooth-step: continuous slope at both ends
    up = smooth(float(np.clip((t - t_open) / max(ramp_open, 1e-9), 0.0, 1.0)))
    dn = 1.0 - smooth(float(np.clip((t - t_close) / max(ramp_close, 1e-9), 0.0, 1.0)))
    return float(max(up * dn, floor))


@dataclass(frozen=True)
class Chain:
    rho: float                  # (kg·m⁻³)
    a_snd: float                # speed of sound in the liquid (m·s⁻¹)
    p_tank: float               # tank pressure setting / initial (Pa)
    V_ullage: float             # initial ullage volume (m³) (blowdown)
    gamma_g: float              # pressurant polytropic exponent
    regulated: bool
    l1: float                   # line length (m)
    A1: float                   # line flow area (m²)
    K1: float                   # line loss coefficient
    Kv0: float                  # valve loss coefficient when fully open
    l2: float                   # valve-to-manifold passage length (m)
    A2: float                   # passage area (m²)
    V_m: float                  # manifold volume (m³)
    Cd: float                   # injector discharge coefficient
    A_inj: float                # total injector area (m²)
    t_open: float
    ramp_open: float
    t_close: float
    ramp_close: float
    zeta: float = 0.05          # damping ratio of the line / passage oscillations (wall friction, compliance)

    @property
    def R1(self) -> float:
        return 2.0 * self.zeta * float(np.sqrt(self.L1 / self.C1))

    @property
    def R2(self) -> float:
        return 2.0 * self.zeta * float(np.sqrt(self.L2 / self.C2))

    @property
    def L1(self) -> float:
        return self.l1 / self.A1

    @property
    def L2(self) -> float:
        return self.l2 / self.A2

    @property
    def C1(self) -> float:
        return self.A1 * self.l1 / self.a_snd**2          # mass-flow basis: dm/dp = V/a^2 (V/(rho a^2) is per volume flow)

    @property
    def C2(self) -> float:
        return self.V_m / self.a_snd**2

    def tank_p(self, dV: float) -> float:
        if self.regulated:
            return self.p_tank
        return self.p_tank * (self.V_ullage / (self.V_ullage + max(dV, 0.0))) ** self.gamma_g


@dataclass(frozen=True)
class FeedSystem:
    pair: str
    ox: Chain
    fu: Chain
    Vc: float                   # chamber free volume (m³)
    At: float                   # throat area (m²)
    eps: float
    tau_c: float                # combustion lag (s)
    eta_cstar: float = 0.97
    MR_start: float = 2.4       # mixture ratio used for c* before both gas flows exist


def injector_flow(ch: Chain, p2: float, pc: float) -> float:
    return ch.Cd * ch.A_inj * np.sqrt(2.0 * ch.rho * max(p2 - pc, 0.0))


def _chain_rhs(ch: Chain, t: float, y: np.ndarray, pc: float):
    m1, p1, m2, p2, dV, lp1, lp2 = y
    pt = ch.tank_p(dV)
    f = valve_fraction(t, ch.t_open, ch.ramp_open, ch.t_close, ch.ramp_close)
    Kv = ch.Kv0 / f**2
    m3 = injector_flow(ch, p2, pc)
    dm1 = (pt - p1 - ch.K1 * m1 * abs(m1) / (2.0 * ch.rho * ch.A1**2) - ch.R1 * (m1 - lp1)) / ch.L1
    dp1 = (m1 - m2) / ch.C1
    dm2 = (p1 - p2 - Kv * m2 * abs(m2) / (2.0 * ch.rho * ch.A2**2) - ch.R2 * (m2 - lp2)) / ch.L2
    dp2 = (m2 - m3) / ch.C2
    ddV = m1 / ch.rho
    return np.array([dm1, dp1, dm2, dp2, ddV, (m1 - lp1) / LP_TAU, (m2 - lp2) / LP_TAU]), m3, f, pt


def _cstar_RT(sys: FeedSystem, mgo: float, mgf: float, pc: float):
    mr = mgo / mgf if mgf > 1e-9 and mgo > 1e-9 else sys.MR_start
    cs0, Tc, gam, _ = chamber_lut(sys.pair)(mr, max(pc, 0.2e6))
    cs = sys.eta_cstar * cs0
    return cs, (cs * float(Gamma(gam))) ** 2, gam, Tc


def rhs(t: float, y: np.ndarray, sys: FeedSystem) -> np.ndarray:
    yo, yf, pc, mgo, mgf = y[0:7], y[7:14], y[14], y[15], y[16]
    do, m3o, *_ = _chain_rhs(sys.ox, t, yo, pc)
    df, m3f, *_ = _chain_rhs(sys.fu, t, yf, pc)
    cs, RT, gam, _ = _cstar_RT(sys, mgo, mgf, pc)
    dmgo = (m3o - mgo) / sys.tau_c
    dmgf = (m3f - mgf) / sys.tau_c
    dpc = RT / sys.Vc * (mgo + mgf - pc * sys.At / cs)
    if pc <= P_AMB and dpc < 0.0:
        dpc = 0.0                                      # after shutdown the chamber settles at ambient pressure
    return np.concatenate([do, df, [dpc, dmgo, dmgf]])


def initial_state(sys: FeedSystem) -> np.ndarray:
    def chain0(ch: Chain):
        return [0.0, ch.p_tank, 0.0, P_AMB, 0.0, 0.0, 0.0]
    return np.array(chain0(sys.ox) + chain0(sys.fu) + [P_AMB, 0.0, 0.0])


def steady_state(sys: FeedSystem, open_valves: bool = True) -> dict[str, float]:
    """Steady flows and chamber pressure with the valves fully open (regulated tank pressure)."""
    def R(ch: Chain) -> float:
        return (ch.K1 / (2 * ch.rho * ch.A1**2) + ch.Kv0 / (2 * ch.rho * ch.A2**2)
                + 1.0 / (2 * ch.rho * (ch.Cd * ch.A_inj) ** 2))

    def flows(pc: float):
        mo = np.sqrt(max(sys.ox.p_tank - pc, 0.0) / R(sys.ox))
        mf = np.sqrt(max(sys.fu.p_tank - pc, 0.0) / R(sys.fu))
        return mo, mf

    def resid(pc: float) -> float:
        mo, mf = flows(pc)
        cs, _, _, _ = _cstar_RT(sys, mo, mf, pc)
        return pc - cs * (mo + mf) / sys.At

    pmax = min(sys.ox.p_tank, sys.fu.p_tank)
    pc = brentq(resid, 0.25e6, pmax * (1 - 1e-9), xtol=1e-3)
    mo, mf = flows(pc)
    return dict(pc=pc, mdot_o=mo, mdot_f=mf, OF=mo / mf, dp_inj_o=(mo / (sys.ox.Cd * sys.ox.A_inj)) ** 2 / (2 * sys.ox.rho),
                dp_inj_f=(mf / (sys.fu.Cd * sys.fu.A_inj)) ** 2 / (2 * sys.fu.rho))


def integrate(sys: FeedSystem, t_end: float, n_frames: int = 601, progress: Callable[[float], None] | None = None,
              cancel=None, n_chunks: int = 40) -> dict[str, np.ndarray]:
    t = np.linspace(0.0, t_end, n_frames)
    y = initial_state(sys)
    Y = np.empty((len(y), n_frames))
    Y[:, 0] = y
    edges = np.linspace(0, n_frames - 1, n_chunks + 1).astype(int)
    scale = np.array([1.0, 1e4, 1.0, 1e4, 1e-6, 1.0, 1.0] * 2 + [1e4, 1e-3, 1e-3])
    for k in range(n_chunks):
        i0, i1 = edges[k], edges[k + 1]
        if i1 <= i0:
            continue
        if cancel is not None and cancel.is_set():
            raise InterruptedError("cancelled")
        sol = solve_ivp(rhs, (t[i0], t[i1]), y, args=(sys,), method="BDF", t_eval=t[i0:i1 + 1], rtol=1e-6,
                        atol=1e-6 * scale + 1e-9, first_step=1e-6)
        if not sol.success:
            raise RuntimeError(f"integrator failed: {sol.message}")
        Y[:, i0:i1 + 1] = sol.y
        y = sol.y[:, -1]
        if progress:
            progress((k + 1) / n_chunks)
    return derived(sys, t, Y)


def derived(sys: FeedSystem, t, Y) -> dict[str, np.ndarray]:
    n = len(t)
    out = {k: np.empty(n) for k in ("mdot_o", "mdot_f", "f_o", "f_f", "ptank_o", "ptank_f", "cstar", "gamma")}
    for i in range(n):
        _, m3o, fo, pto = _chain_rhs(sys.ox, t[i], Y[0:7, i], Y[14, i])
        _, m3f, ff, ptf = _chain_rhs(sys.fu, t[i], Y[7:14, i], Y[14, i])
        cs, _, gam, _ = _cstar_RT(sys, Y[15, i], Y[16, i], Y[14, i])
        out["mdot_o"][i], out["mdot_f"][i], out["f_o"][i], out["f_f"][i] = m3o, m3f, fo, ff
        out["ptank_o"][i], out["ptank_f"][i], out["cstar"][i], out["gamma"][i] = pto, ptf, cs, gam
    pc = np.maximum(Y[14], 0.0)
    pe_pc = {}
    cf = np.empty(n)
    for i in range(n):
        g = round(out["gamma"][i], 3)
        if g not in pe_pc:
            pe_pc[g] = exit_state(sys.eps, g)[1]
        cf[i] = CF(g, pe_pc[g], P_AMB / max(pc[i], 1.0), sys.eps)
    mg = Y[15] + Y[16]
    ch = {"Pc": pc / 1e6, "F": np.maximum(cf * pc * sys.At, 0.0) / 1e3, "CF": cf,
          "mdot_o": out["mdot_o"], "mdot_f": out["mdot_f"], "mdot_in": out["mdot_o"] + out["mdot_f"],
          "OF": np.where(Y[16] > 1e-6, Y[15] / np.maximum(Y[16], 1e-9), np.nan), "cstar": out["cstar"],
          "p1_o": Y[1] / 1e6, "p2_o": Y[3] / 1e6, "p1_f": Y[8] / 1e6, "p2_f": Y[10] / 1e6,
          "ptank_o": out["ptank_o"] / 1e6, "ptank_f": out["ptank_f"] / 1e6, "valve_o": out["f_o"],
          "valve_f": out["f_f"], "m1_o": Y[0], "m1_f": Y[7], "mdot_gas": mg}
    return ch


def summary(sys: FeedSystem, t, ch: dict[str, np.ndarray]) -> dict[str, float]:
    st = steady_state(sys)
    It = float(np.trapezoid(ch["F"] * 1e3, t))
    mo = float(np.trapezoid(ch["mdot_o"], t))
    mf = float(np.trapezoid(ch["mdot_f"], t))
    pc_max = float(ch["Pc"].max())
    return {"I_t_Ns": It, "Pc_max_MPa": pc_max, "Pc_ss_MPa": st["pc"] / 1e6, "OF_ss": st["OF"], "m_ox": mo, "m_fuel": mf,
            "overshoot": pc_max / (st["pc"] / 1e6) - 1.0,
            "surge_o_MPa": float(ch["p1_o"].max() - ch["ptank_o"][0]), "surge_f_MPa": float(ch["p1_f"].max() - ch["ptank_f"][0]),
            "dp_inj_o_frac": st["dp_inj_o"] / st["pc"], "dp_inj_f_frac": st["dp_inj_f"] / st["pc"],
            "OF_min_start": float(np.nanmin(ch["OF"][ch["Pc"] > 0.3 * st["pc"] / 1e6])) if (ch["Pc"] > 0.3 * st["pc"] / 1e6).any() else float("nan")}


_ = field
