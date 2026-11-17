r"""Quasi-1D hybrid burn (MATHKIT.md Section 3.3): axial port march, filling-ODE chamber, throat erosion, throttling,
and an oxidizer *supply* that is either a prescribed valve profile or a self-pressurized N₂O tank (blowdown).

State :math:`y=[R_1..R_N,\;P_c/10^6,\;R_t,\;\text{supply state}]`.  At each instant, march down the port:

.. math::
    G_i=\frac{\dot m_{ox}+\sum_{j<i}\dot m_{f,j}}{\pi R_i^2},\quad \dot r_i=f(G_i,x_i),\quad
    \dot m_{f,i}=\rho_f\,2\pi R_i\,\Delta x\,\dot r_i,\quad \dot R_i=\dot r_i

.. math::
    \frac{dP_c}{dt}=\frac{RT_c}{V_c}\Big(\dot m_{ox}+\sum\dot m_{f,i}-\frac{P_c A_t}{\eta\,c^*(O/F,P_c)}\Big),\quad
    \frac{dR_t}{dt}=k_t\,(P_c/\text{MPa})^{m_t}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from ..constants import G0, P_SEA_LEVEL
from ..nozzle import CF, Gamma, exit_state
from ..regression import marxman_rdot
from ..thermo.cea import chamber_lut


@dataclass(frozen=True)
class Motor1D:
    pair: str                 # CEA pair: "GOX/HTPB" | "LOX/HTPB" | "N2O/HTPB"
    L: float                  # (m) grain length
    R0: float                 # (m) initial port radius
    R_out: float              # (m) outer radius (web = R_out - R0)
    rho_f: float              # (kg·m⁻³)
    law: str                  # "power" | "marxman"
    a: float                  # power law: (m·s⁻¹) with G in (kg·m⁻²·s⁻¹)
    n: float
    B: float                  # blowing number (Marxman)
    mu: float                 # (Pa·s) (Marxman)
    Dt: float                 # (m) initial throat diameter
    eps: float                # exit area ratio (A_e / initial A_t)
    eta_cstar: float
    kt: float = 0.0           # throat erosion: dR_t/dt = kt (Pc/MPa)^mt, (m·s⁻¹)
    mt: float = 1.0
    N: int = 20
    V_extra: float = 2.5e-4   # (m³)

    @property
    def dx(self) -> float:
        return self.L / self.N


# ------------------------------------------------------------------------------------------- supplies
def valve_factor(t, tau_open: float, tau_close: float, t_burn: float):
    t = np.asarray(t, float)
    return (1 - np.exp(-t / tau_open)) * (1 - np.exp(-np.maximum(t_burn - t, 0.0) / tau_close))


def throttle_factor(t: float, level: float, t1: float, t2: float, ramp: float = 0.5) -> float:
    """1 -> ``level`` between ``t1`` and ``t1+ramp``, back to 1 between ``t2`` and ``t2+ramp`` (piecewise linear)."""
    return float(np.interp(t, [0.0, t1, t1 + ramp, t2, t2 + ramp], [1.0, 1.0, level, level, 1.0]))


@dataclass(frozen=True)
class ValveSupply:
    """Prescribed oxidizer flow: ``mdot`` x throttle profile x first-order valve opening/closing."""

    mdot: float
    t_burn: float
    tau_open: float = 0.25
    tau_close: float = 0.15
    throttle: float = 1.0
    t_th1: float = 3.0
    t_th2: float = 6.0
    n_state = 0

    def y0(self) -> list[float]:
        return []

    def flow(self, t: float, ys, Pc: float) -> tuple[float, list[float], dict]:
        m = self.mdot * throttle_factor(t, self.throttle, self.t_th1, self.t_th2) * float(
            valve_factor(t, self.tau_open, self.tau_close, self.t_burn))
        return m, [], {}


@dataclass(frozen=True)
class N2OSupply:
    """Self-pressurized N₂O tank (equilibrium model) feeding the motor through a Dyer injector."""

    V: float                  # (m³) tank volume
    T0: float                 # (K) initial temperature
    fill: float               # liquid volume fraction
    Cd: float
    A_inj: float              # (m²) total injector area
    t_burn: float
    loss_frac: float = 0.05   # fraction of tank pressure lost in feed line before the injector
    tau_open: float = 0.25
    tau_close: float = 0.15
    n_state = 2

    def y0(self) -> list[float]:
        from ..feed import tank_initial
        return list(tank_initial(self.T0, self.V, self.fill))

    def flow(self, t: float, ys, Pc: float) -> tuple[float, list[float], dict]:
        from ..feed import tank_rates
        v = float(valve_factor(t, self.tau_open, self.tau_close, self.t_burn))
        dm, dU, st, md = tank_rates(ys[0], ys[1], self.V, self.Cd, self.A_inj, Pc, self.loss_frac)
        return v * md, [v * dm, v * dU], {"p_tank": st.p, "T_tank": st.T, "m_liq": st.m_liq, "m_tank": ys[0]}


# ------------------------------------------------------------------------------------------- physics
def _march(p: Motor1D, R: np.ndarray, mox: float):
    """Axial march: returns (rdot_i, mf_i, G_i)."""
    N, dx = p.N, p.dx
    rdot, mf, G = np.zeros(N), np.zeros(N), np.zeros(N)
    cum = 0.0
    for i in range(N):
        Ri = max(R[i], 1e-6)
        G[i] = (mox + cum) / (np.pi * Ri**2)
        if Ri < p.R_out:
            if p.law == "marxman":
                rdot[i] = float(marxman_rdot(G[i], (i + 0.5) * dx, p.B, p.mu, p.rho_f))
            else:
                rdot[i] = p.a * G[i] ** p.n
        mf[i] = p.rho_f * 2 * np.pi * Ri * dx * rdot[i]
        cum += mf[i]
    return rdot, mf, G


def evaluate(p: Motor1D, supply, t: float, y: np.ndarray) -> dict:
    """All instantaneous quantities for state ``y`` (one evaluation of the same physics the ODE uses)."""
    N = p.N
    R, Pc, Rt, ys = y[:N], y[N] * 1e6, y[N + 1], y[N + 2:]
    mox, dys, extra = supply.flow(t, ys, Pc)
    rdot, mf, G = _march(p, R, mox)
    mft = float(mf.sum())
    OF = mox / mft if mft > 1e-12 else np.nan
    cs0, Tc, gam, _ = chamber_lut(p.pair)(OF if np.isfinite(OF) else 5.0, Pc)
    cs = p.eta_cstar * cs0
    At = np.pi * Rt**2
    Vc = float(np.sum(np.pi * np.maximum(R, 1e-6) ** 2) * p.dx) + p.V_extra
    RT = (cs * Gamma(gam)) ** 2
    dPc = RT / Vc * (mox + mft - Pc * At / cs)
    dR = rdot
    dRt = p.kt * (max(Pc, 0.0) / 1e6) ** p.mt
    return dict(mox=mox, rdot=rdot, mf=mf, G=G, mft=mft, OF=OF, cs=cs, Tc=Tc, gam=gam, At=At, Vc=Vc, dPc=dPc,
                dR=dR, dRt=dRt, dys=dys, extra=extra)


def _rhs(t: float, y, p: Motor1D, supply):
    e = evaluate(p, supply, t, y)
    return np.concatenate([e["dR"], [e["dPc"] / 1e6, e["dRt"]], e["dys"]])


def integrate(p: Motor1D, supply, t_end: float, n_frames: int = 601, progress: Callable[[float], None] | None = None,
              cancel=None, n_chunks: int = 40) -> dict[str, np.ndarray]:
    """Integrate with LSODA in chunks (stiff chamber filling) so progress/cancel are serviced between chunks."""
    t = np.linspace(0.0, t_end, n_frames)
    N = p.N
    y = np.concatenate([np.full(N, p.R0), [P_SEA_LEVEL / 1e6, p.Dt / 2], supply.y0()])
    Y = np.empty((len(y), n_frames))
    Y[:, 0] = y
    edges = np.linspace(0, n_frames - 1, n_chunks + 1).astype(int)
    atol = np.concatenate([np.full(N, 1e-9), [1e-9, 1e-9], np.full(supply.n_state, 1e-6)])
    for k in range(n_chunks):
        i0, i1 = edges[k], edges[k + 1]
        if i1 <= i0:
            continue
        if cancel is not None and cancel.is_set():
            raise InterruptedError("cancelled")
        sol = solve_ivp(_rhs, (t[i0], t[i1]), y, args=(p, supply), method="LSODA", t_eval=t[i0:i1 + 1],
                        rtol=1e-8, atol=atol)
        if not sol.success:
            raise RuntimeError(f"integrator failed: {sol.message}")
        Y[:, i0:i1 + 1] = sol.y
        y = sol.y[:, -1]
        if progress:
            progress((k + 1) / n_chunks)
    return derived_channels(p, supply, t, Y)


def derived_channels(p: Motor1D, supply, t, Y) -> dict[str, np.ndarray]:
    n, N = len(t), p.N
    keys = ["mox", "mft", "OF", "cs", "Tc", "gam", "rdot_mean", "rdot_head", "rdot_aft", "G_head", "At"]
    d = {k: np.empty(n) for k in keys}
    extra_keys = ("p_tank", "T_tank", "m_liq", "m_tank")
    ex = {k: np.full(n, np.nan) for k in extra_keys}
    rdot_x = np.empty((N, n))
    G_x = np.empty((N, n))
    for i in range(n):
        e = evaluate(p, supply, t[i], Y[:, i])
        d["mox"][i], d["mft"][i], d["OF"][i], d["cs"][i], d["Tc"][i], d["gam"][i], d["At"][i] = (
            e["mox"], e["mft"], e["OF"], e["cs"], e["Tc"], e["gam"], e["At"])
        r = e["rdot"]
        d["rdot_mean"][i], d["rdot_head"][i], d["rdot_aft"][i] = r.mean() * 1e3, r[0] * 1e3, r[-1] * 1e3
        d["G_head"][i] = e["G"][0]
        rdot_x[:, i], G_x[:, i] = r * 1e3, e["G"]
        for k, v in e["extra"].items():
            ex[k][i] = v
    Pc = Y[N] * 1e6
    cf = np.array([CF(g, exit_state(p.eps * (p.Dt / 2) ** 2 / (rt**2), g)[1], P_SEA_LEVEL / max(pc, 1.0),
                      p.eps * (p.Dt / 2) ** 2 / rt**2) for g, pc, rt in zip(d["gam"], Pc, Y[N + 1])])
    F = np.maximum(cf * Pc * d["At"], 0.0)
    mdot = d["mox"] + d["mft"]
    Isp = np.where(mdot > 1e-3 * max(d["mox"].max(), 1e-12), F / np.maximum(mdot, 1e-12) / G0, np.nan)
    out = {"Pc": Pc / 1e6, "F": F / 1e3, "OF": np.where(d["mox"] > 1e-3 * max(d["mox"].max(), 1e-12), d["OF"], np.nan),
           "mdot_ox": d["mox"], "mdot_f": d["mft"], "rdot": d["rdot_mean"], "rdot_head": d["rdot_head"],
           "rdot_aft": d["rdot_aft"], "Isp": Isp, "Tc": d["Tc"], "gamma": d["gam"], "cstar": d["cs"], "CF": cf,
           "Rt": Y[N + 1] * 1e3, "G_head": d["G_head"], "R": Y[:N].mean(axis=0) * 1e3}
    for i in range(N):
        out[f"Rp_{i:02d}"] = Y[i] * 1e3
        out[f"rd_{i:02d}"] = rdot_x[i]
        out[f"G_{i:02d}"] = G_x[i]
    out.update({k: v for k, v in ex.items() if np.isfinite(v).any()})
    return out


def summary(p: Motor1D, t, ch: dict[str, np.ndarray]) -> dict[str, float]:
    It = float(np.trapezoid(ch["F"] * 1e3, t))
    mprop = float(np.trapezoid(ch["mdot_ox"] + ch["mdot_f"], t))
    mox = float(np.trapezoid(ch["mdot_ox"], t))
    mf = float(np.trapezoid(ch["mdot_f"], t))
    R_end = np.array([ch[f"Rp_{i:02d}"][-1] for i in range(p.N)]) * 1e-3
    web = float((R_end.max() - p.R0) / (p.R_out - p.R0))
    return {"I_t_kNs": It / 1e3, "Isp_avg": It / (mprop * G0) if mprop > 0 else float("nan"), "m_prop": mprop,
            "m_ox": mox, "m_fuel": mf, "OF_avg": mox / mf if mf > 0 else float("nan"),
            "Pc_max_MPa": float(np.nanmax(ch["Pc"])), "F_max_kN": float(np.nanmax(ch["F"])),
            "R_end_mm": float(R_end.mean() * 1e3), "web_burned": web,
            "gamma_mean": float(np.nanmean(ch["gamma"])), "Rt_end_mm": float(ch["Rt"][-1]),
            "Rt_growth_pct": float(100 * (ch["Rt"][-1] / ch["Rt"][0] - 1))}
