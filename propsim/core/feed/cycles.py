r"""Pump-fed engine cycles (MATHKIT.md Section 4.3): gas generator, expander and staged combustion.

Pump (affinity-scaled quadratic curve through the design point): :math:`\Delta p=\Delta p_{so}\big(\frac N{N_0}\big)^2-
(\Delta p_{so}-\Delta p_d)\big(\frac{\dot mN_0}{N\dot m_d}\big)^2`, power :math:`P=\frac{\dot m\Delta p}{\rho\eta}`.
Turbine: :math:`P=\eta\,\dot m_tc_pT_1\big[1-\big(\frac{p_2}{p_1}\big)^{\frac{\gamma-1}\gamma}\big]`.

Steady balance: unknowns ``(N, P_c)`` with (i) chamber mass balance and (ii) turbine power = pump power / mechanical
efficiency; the pump outlet pressures follow from matching the pump curve to the downstream resistances.  The
transient integrates the shaft speed with the chamber quasi-steady (its time constant is milliseconds against seconds of
spool-up).  Turbine drive: GG bleeds propellant (exhaust overboard); expander drives with all heated fuel (closed);
staged combustion burns all fuel plus a share of oxidizer in a preburner (closed).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from ..nozzle import CF, exit_state
from ..thermo.cea import chamber_lut

CYCLES = ["gg", "expander", "staged"]
CYCLE_LABELS = ["Gas generator (open)", "Expander (closed)", "Staged combustion (closed, fuel-rich)"]


@dataclass(frozen=True)
class Pump:
    rho: float
    dp_d: float              # design pressure rise (Pa)
    mdot_d: float            # design flow (kg·s⁻¹)
    N0: float                # design speed (rpm)
    eta: float
    shutoff: float = 1.25    # shut-off head / design head
    p_in: float = 0.35e6     # inlet (tank + NPSH) pressure (Pa)

    def dp(self, mdot: float, N: float) -> float:
        """Affinity-scaled curve through the design point: shut-off rise at zero flow, quadratic droop with flow."""
        so, x = self.shutoff * self.dp_d, N / self.N0
        return so * x**2 - (so - self.dp_d) * (mdot / self.mdot_d) ** 2

    def power(self, mdot: float, dp: float) -> float:
        return mdot * dp / (self.rho * self.eta)


@dataclass(frozen=True)
class Turbine:
    eta: float
    cp: float                # (J·kg⁻¹·K⁻¹)
    gamma: float
    T_in: float              # (K)
    p_ex: float = 0.12e6     # exhaust pressure for the open (GG) cycle (Pa)

    def power(self, mdot: float, p_in: float, p_out: float) -> float:
        pr = max(min(p_out / p_in, 1.0), 1e-6)
        return self.eta * mdot * self.cp * self.T_in * (1.0 - pr ** ((self.gamma - 1.0) / self.gamma))


@dataclass(frozen=True)
class Engine:
    pair: str
    cycle: str
    At: float
    eps: float
    eta_cstar: float
    pump_o: Pump
    pump_f: Pump
    turb: Turbine
    K_o: float               # oxidizer line + injector: dp = K m^2 (Pa per (kg/s)^2)
    K_f: float               # fuel injector (after the turbine for closed cycles)
    K_jac: float             # fuel jacket / preburner resistance before the turbine
    A_gg: float              # GG bleed orifice (Cd*A, m²) per propellant, scaled by the valve schedule
    MR_gg: float             # GG / preburner mixture ratio (oxidizer per fuel)
    k_gg: float              # turbine inlet pressure as a fraction of P_c (GG) / turbine outlet (staged)
    eta_mech: float = 0.98
    K_turb: float = 0.0      # turbine as a flow restriction: dp = K m^2 (closed cycles)
    I_shaft: float = 0.05    # (kg·m²)
    P_start: float = 40e3    # starter power (W)
    t_start: float = 1.0     # starter burn time (s)


def _outlet(pump: Pump, N: float, p_down: float, K: float, bleed_fn: Callable[[float], float] | None, rho: float):
    """Pump outlet pressure and flows ``(p, main, bleed)`` where the pump curve meets main + bleed demand;
    ``p_down`` is the pressure the main line discharges into."""
    pc = p_down

    def flows(p: float):
        main = np.sqrt(max(p - pc, 0.0) / K)
        bleed = bleed_fn(p) if bleed_fn is not None else 0.0
        return main, bleed

    def g(p: float) -> float:
        main, bleed = flows(p)
        return pump.p_in + pump.dp(main + bleed, N) - p

    lo = pc * (1.0 + 1e-9)
    hi = pump.p_in + pump.dp(0.0, N)
    if hi <= lo or g(lo) <= 0.0:                      # pump cannot lift the outlet above the chamber pressure
        return None
    p = brentq(g, lo, hi, xtol=1e-3)
    main, bleed = flows(p)
    return p, main, bleed


def balance_point(e: Engine, N: float, pc: float, gg_open: float = 1.0) -> dict | None:
    """All flows, pressures and powers for given shaft speed ``N`` and chamber pressure ``pc``."""
    cd_gg = e.A_gg * gg_open
    bleed = lambda rho, frac: (lambda p: cd_gg * np.sqrt(2.0 * rho * max(p - frac * pc, 0.0)))
    if e.cycle == "gg":
        of_ = _outlet(e.pump_f, N, pc, e.K_f, bleed(e.pump_f.rho, e.k_gg), e.pump_f.rho)
        if of_ is None:
            return None
        p_f, m_f, m_ggf = of_
        m_ggo = e.MR_gg * m_ggf                                   # oxidizer bleed follows the GG mixture ratio
        oo = _outlet(e.pump_o, N, pc, e.K_o, lambda p: m_ggo, e.pump_o.rho)
        if oo is None:
            return None
        p_o, m_o, _ = oo
        m_t, p_t_in, p_t_out = m_ggf + m_ggo, e.k_gg * pc, e.turb.p_ex
        burn_o, burn_f = m_o, m_f                                  # the GG exhaust leaves overboard
        pump_o_flow, pump_f_flow = m_o + m_ggo, m_f + m_ggf
        m_gg = m_t
    elif e.cycle == "expander":
        # fuel: pump -> jacket (K_jac) -> turbine -> injector (K_f); oxidizer straight to the injector
        of_ = _outlet(e.pump_f, N, pc, e.K_f + e.K_jac + e.K_turb, None, e.pump_f.rho)
        oo = _outlet(e.pump_o, N, pc, e.K_o, None, e.pump_o.rho)
        if of_ is None or oo is None:
            return None
        (p_f, m_f, _), (p_o, m_o, _) = of_, oo
        p_t_out = pc + e.K_f * m_f**2
        m_t, p_t_in = m_f, p_t_out + e.K_turb * m_f**2
        burn_o, burn_f, pump_o_flow, pump_f_flow, m_gg = m_o, m_f, m_o, m_f, 0.0
    else:                                                          # staged combustion, fuel-rich preburner
        k_line = e.K_jac + e.K_turb * (1.0 + e.MR_gg) ** 2        # jacket/preburner + turbine (flow m_f (1+MR_gg))
        of_ = _outlet(e.pump_f, N, e.k_gg * pc, k_line, None, e.pump_f.rho)      # discharges at the turbine exit
        if of_ is None:
            return None
        p_f, m_f, _ = of_
        m_pb_o = e.MR_gg * m_f                                    # preburner oxidizer
        oo = _outlet(e.pump_o, N, pc, e.K_o, lambda p: m_pb_o, e.pump_o.rho)
        if oo is None:
            return None
        p_o, m_o_main, _ = oo
        m_t, p_t_out = m_f + m_pb_o, e.k_gg * pc
        p_t_in = p_t_out + e.K_turb * m_t**2
        burn_o, burn_f = m_o_main + m_pb_o, m_f                    # preburner exhaust joins the main chamber
        pump_o_flow, pump_f_flow, m_gg, m_o = burn_o, m_f, m_pb_o, burn_o
    P_t = e.turb.power(m_t, max(p_t_in, 1.0), p_t_out)
    P_pump = e.pump_o.power(pump_o_flow, p_o - e.pump_o.p_in) + e.pump_f.power(pump_f_flow, p_f - e.pump_f.p_in)
    mr_main = burn_o / max(burn_f, 1e-9)
    cs0, Tc, gam, _ = chamber_lut(e.pair)(mr_main, max(pc, 0.3e6))
    cs = e.eta_cstar * cs0
    return dict(p_o=p_o, p_f=p_f, m_o=burn_o, m_f=burn_f, m_gg=m_gg, m_t=m_t, P_t=P_t, P_pump=P_pump, p_t_in=p_t_in,
                p_t_out=p_t_out, mr=mr_main, cstar=cs, gamma=gam, Tc=Tc, m_chamber=burn_o + burn_f,
                pc_calc=cs * (burn_o + burn_f) / e.At, dp_o=p_o - e.pump_o.p_in, dp_f=p_f - e.pump_f.p_in)


def chamber_pressure(e: Engine, N: float, gg_open: float = 1.0, pc0: float = 4e6) -> float | None:
    """Quasi-steady chamber pressure at shaft speed ``N``: root of ``pc - c* m_chamber(pc)/A_t``."""
    def r(pc: float) -> float:
        b = balance_point(e, N, pc, gg_open)
        return pc - (b["pc_calc"] if b else 0.0)

    lo, hi = 0.12e6, max(e.pump_o.p_in + e.pump_o.dp(0.0, N), e.pump_f.p_in + e.pump_f.dp(0.0, N))
    grid = np.geomspace(lo, hi * 0.999, 14)
    vals = [r(p) for p in grid]
    for i in range(len(grid) - 1):
        if vals[i] * vals[i + 1] < 0:
            pc = brentq(r, grid[i], grid[i + 1], xtol=1.0)
            return pc if balance_point(e, N, pc, gg_open) is not None else None
    return None


def _power_gap(e: Engine, N: float, gg_open: float):
    pc = chamber_pressure(e, N, gg_open)
    if pc is None:
        return None, None
    b = balance_point(e, N, pc, gg_open)
    return (e.eta_mech * b["P_t"] - b["P_pump"]), pc


def steady(e: Engine, gg_open: float = 1.0) -> dict:
    """Steady balance: shaft speed ``N`` where turbine power = pump power / eta_mech, with the chamber quasi-steady.

    Scans ``N`` for the sign change of the power gap (excess power below, deficit above the balanced speed)."""
    Ns = np.linspace(0.35, 1.9, 32) * e.pump_o.N0
    gaps = [_power_gap(e, n, gg_open)[0] for n in Ns]
    root = None
    for i in range(len(Ns) - 1):
        if gaps[i] is not None and gaps[i + 1] is not None and gaps[i] > 0 >= gaps[i + 1]:
            root = (Ns[i], Ns[i + 1])
    if root is None:
        raise RuntimeError("cycle balance not found: no shaft speed where turbine power equals pump power "
                           "(increase the turbine drive or reduce the pump demand)")
    N = brentq(lambda n: _power_gap(e, n, gg_open)[0], root[0], root[1], xtol=1e-3)
    pc = chamber_pressure(e, N, gg_open)
    b = balance_point(e, N, pc, gg_open)
    b.update(N=N, pc=pc)
    pe_pc = exit_state(e.eps, b["gamma"])[1]
    b["CF"] = float(CF(b["gamma"], pe_pc, 101325.0 / pc, e.eps))
    b["F"] = b["CF"] * pc * e.At
    return b


@dataclass(frozen=True)
class CycleDesign:
    pair: str
    cycle: str
    Pc: float                 # design chamber pressure (Pa)
    Dt: float                 # throat diameter (m)
    eps: float
    MR: float                 # main-chamber mixture ratio
    eta_cstar: float
    dp_inj_o: float           # injector drop as a fraction of Pc
    dp_inj_f: float
    dp_jac: float             # jacket / preburner drop as a fraction of Pc (expander, staged)
    eta_po: float
    eta_pf: float
    eta_t: float
    N0: float                 # design shaft speed (rpm)
    T_t: float                # turbine inlet temperature (K); expander/staged: *required* value is reported
    cp_t: float
    gamma_t: float
    MR_gg: float              # GG / preburner mixture ratio
    PR_t: float               # turbine pressure ratio p_in/p_out (expander, staged)
    p_in: float = 0.35e6
    p_ex: float = 0.12e6
    k_gg: float = 0.9


def design_engine(d: CycleDesign, I_shaft: float = 0.05, P_start: float = 40e3, t_start: float = 1.0) -> tuple[Engine, dict]:
    """Size pumps, resistances and the turbine drive so the cycle balances at ``(N0, Pc)`` by construction."""
    At = np.pi * d.Dt**2 / 4.0
    cs, Tc, gam, _ = chamber_lut(d.pair)(d.MR, d.Pc)
    cs *= d.eta_cstar
    mdot = d.Pc * At / cs
    m_f, m_o = mdot / (1.0 + d.MR), mdot * d.MR / (1.0 + d.MR)
    rho_o = {"LOX": 1141.0, "N2O4": 1443.0}.get(d.pair.split("/")[0], 1141.0)
    rho_f = {"RP-1": 810.0, "H2": 70.8, "CH4": 423.0, "Ethanol": 789.0, "MMH": 874.0}.get(d.pair.split("/")[1], 810.0)
    m_o_inj = m_o - (d.MR_gg * m_f if d.cycle == "staged" else 0.0)      # staged: the preburner takes part of the oxidizer
    K_o = d.dp_inj_o * d.Pc / m_o_inj**2
    K_f = d.dp_inj_f * d.Pc / m_f**2
    K_jac = d.dp_jac * d.Pc / m_f**2 if d.cycle != "gg" else 0.0
    # design pressures (the turbine is a flow restriction: dp = K_turb m^2 gives the pressure ratio)
    K_turb = 0.0
    if d.cycle == "gg":
        p_t_in, p_t_out = d.k_gg * d.Pc, d.p_ex
        p_f = d.Pc + K_f * m_f**2
    elif d.cycle == "expander":
        p_t_out = d.Pc + K_f * m_f**2
        p_t_in = d.PR_t * p_t_out
        K_turb = (p_t_in - p_t_out) / m_f**2
        p_f = p_t_in + K_jac * m_f**2
    else:
        p_t_out = d.k_gg * d.Pc
        p_t_in = d.PR_t * p_t_out
        m_tt = m_f * (1.0 + d.MR_gg)
        K_turb = (p_t_in - p_t_out) / m_tt**2
        p_f = p_t_in + K_jac * m_f**2
    p_o = d.Pc + K_o * m_o_inj**2
    # fixed point on the bleed flows (pump flows include them)
    m_gg_f = 0.0
    for _ in range(25):
        m_gg_o = d.MR_gg * m_gg_f if d.cycle in ("gg", "staged") else 0.0
        if d.cycle == "staged":
            m_gg_f = 0.0
            m_gg_o = d.MR_gg * m_f
        flow_o = m_o + (m_gg_o if d.cycle == "gg" else 0.0)
        flow_f = m_f + (m_gg_f if d.cycle == "gg" else 0.0)
        if d.cycle == "staged":
            flow_o = m_o + m_gg_o * 0.0 + (d.MR_gg * m_f) * 0.0
        pump_o = Pump(rho_o, p_o - d.p_in, flow_o, d.N0, d.eta_po, p_in=d.p_in)
        pump_f = Pump(rho_f, p_f - d.p_in, flow_f, d.N0, d.eta_pf, p_in=d.p_in)
        P_pump = pump_o.power(flow_o, p_o - d.p_in) + pump_f.power(flow_f, p_f - d.p_in)
        P_req = P_pump / 0.98
        if d.cycle == "gg":
            m_t = P_req / (d.eta_t * d.cp_t * d.T_t * (1.0 - (p_t_out / p_t_in) ** ((d.gamma_t - 1.0) / d.gamma_t)))
            new = m_t / (1.0 + d.MR_gg)
            if abs(new - m_gg_f) < 1e-12:
                break
            m_gg_f = new
        else:
            break
    if d.cycle == "gg":
        m_t = m_gg_f * (1.0 + d.MR_gg)
        A_gg = m_gg_f / np.sqrt(2.0 * rho_f * max(p_f - p_t_in, 1.0))
        T_req = d.T_t
        turb = Turbine(d.eta_t, d.cp_t, d.gamma_t, d.T_t, d.p_ex)
    else:
        m_t = m_f if d.cycle == "expander" else m_f * (1.0 + d.MR_gg)
        T_req = P_req / (d.eta_t * m_t * d.cp_t * (1.0 - (1.0 / d.PR_t) ** ((d.gamma_t - 1.0) / d.gamma_t)))
        A_gg = 1e-6
        turb = Turbine(d.eta_t, d.cp_t, d.gamma_t, T_req, d.p_ex)
    eng = Engine(d.pair, d.cycle, At, d.eps, d.eta_cstar, pump_o, pump_f, turb, K_o, K_f, K_jac, A_gg, d.MR_gg,
                 d.k_gg, K_turb=K_turb, I_shaft=I_shaft, P_start=P_start, t_start=t_start)
    info = dict(mdot=mdot, m_o=m_o, m_f=m_f, m_t=m_t, P_pump=P_pump, P_t=P_req, T_req=T_req, p_o=p_o, p_f=p_f,
                p_t_in=p_t_in, p_t_out=p_t_out, cstar=cs, Tc=Tc, gamma=gam, A_gg=A_gg, bleed=(m_t if d.cycle == "gg" else 0.0))
    return eng, info


def transient(e: Engine, t_end: float, gg_ramp: float = 0.5, n_frames: int = 201, N_start: float = 300.0,
              progress: Callable[[float], None] | None = None, cancel=None) -> dict[str, np.ndarray]:
    """Spool-up: shaft speed ODE with starter assist and a GG/preburner valve opening over ``gg_ramp``."""
    t = np.linspace(0.0, t_end, n_frames)

    def valve(tt: float) -> float:
        x = float(np.clip(tt / max(gg_ramp, 1e-6), 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def torque_power(tt: float, N: float):
        g = max(valve(tt), 1e-3)
        start = e.P_start * max(1.0 - tt / e.t_start, 0.0)
        pc = chamber_pressure(e, max(N, 50.0), g)
        if pc is None:                                   # pumps cannot yet lift the outlet: no flow, the starter spins up
            return start, 0.0, None
        b = balance_point(e, max(N, 50.0), pc, g)
        if b is None:                                    # root on the boundary of the pump's operating range
            return start, 0.0, None
        return e.eta_mech * b["P_t"] - b["P_pump"] + start, pc, b

    def rhs(tt, y):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("cancelled")
        N = y[0]
        dP, _, _ = torque_power(tt, N)
        om = 2.0 * np.pi * max(N, 100.0) / 60.0
        return [dP / (e.I_shaft * om) * 60.0 / (2.0 * np.pi)]

    sol = solve_ivp(rhs, (0.0, t_end), [N_start], t_eval=t, method="RK45", rtol=1e-5, atol=1e-2, max_step=t_end / 100)
    N = sol.y[0]
    out = {k: np.zeros(len(t)) for k in ("pc", "F", "m_o", "m_f", "m_gg", "P_t", "P_pump", "mr", "p_o", "p_f", "P_net")}
    for i, (ti, Ni) in enumerate(zip(t, N)):
        P, pc, b = torque_power(ti, Ni)
        if b is None:
            continue
        pe_pc = exit_state(e.eps, b["gamma"])[1]
        cf = float(CF(b["gamma"], pe_pc, 101325.0 / max(pc, 1.0), e.eps))
        for k, v in (("pc", pc), ("F", max(cf * pc * e.At, 0.0)), ("m_o", b["m_o"]), ("m_f", b["m_f"]),
                     ("m_gg", b["m_gg"]), ("P_t", b["P_t"]), ("P_pump", b["P_pump"]), ("mr", b["mr"]),
                     ("p_o", b["p_o"]), ("p_f", b["p_f"]), ("P_net", P)):
            out[k][i] = v
        if progress and i % 10 == 0:
            progress(i / len(t))
    out["N"] = N
    out["t"] = t
    return out

