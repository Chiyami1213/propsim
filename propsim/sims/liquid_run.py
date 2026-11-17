"""Run drivers for liquid engines: feed-network transient, pump-fed cycles, chugging / L* stability."""
from __future__ import annotations

import numpy as np

from ..core.feed import cycles as cy
from ..core.feed import network as nw
from ..core.result import Result
from ..core.schema import C, P, ParamModel
from ..core.stability import chugging as ch
from ..core.thermo.cea import chamber_lut
from .base import Simulation

PAIRS = ["LOX/RP-1", "LOX/CH4", "LOX/Ethanol", "N2O4/MMH"]
PAIR_LABELS = ["LOX / RP-1", "LOX / CH₄", "LOX / ethanol", "N₂O₄ / MMH"]
MR_DEFAULT = {"LOX/RP-1": 2.6, "LOX/CH4": 3.2, "LOX/Ethanol": 1.5, "N2O4/MMH": 2.0}
# Handbook-level liquid properties: (density kg/m3, speed of sound m/s)
OX_PROPS = {"LOX": (1141.0, 900.0), "N2O4": (1443.0, 1000.0)}
FUEL_PROPS = {"RP-1": (810.0, 1300.0), "CH4": (423.0, 1400.0), "Ethanol": (789.0, 1180.0), "MMH": (874.0, 1300.0)}


def props(pair: str):
    ox, fu = pair.split("/")
    return OX_PROPS[ox], FUEL_PROPS[fu]


class EngineGeometry(ParamModel):
    """Thrust chamber geometry shared by the liquid Run models (also used by their 3D scene)."""

    pair: str = C("LOX/RP-1", choices=PAIRS, labels=PAIR_LABELS, symbol=r"\mathrm{propellants}",
                  desc="oxidizer / fuel pair (CEA tables)", group="Engine")
    D_t: float = P(50.0, symbol=r"D_t", desc="throat diameter", units="mm", dims="L", lo=15.0, hi=200.0, step=0.5,
                   group="Engine")
    eps: float = P(12.0, symbol=r"\varepsilon=\frac{A_e}{A_t}", desc="nozzle expansion ratio", units="", dims="1",
                   lo=2.0, hi=60.0, step=0.1, group="Engine")
    L_star: float = P(1.0, symbol=r"L^*", desc="characteristic length (chamber volume per throat area)", units="m",
                      dims="L", lo=0.4, hi=3.0, step=0.01, group="Engine")
    contraction: float = P(3.0, symbol=r"\frac{A_c}{A_t}", desc="chamber contraction ratio", units="", dims="1",
                           lo=1.5, hi=8.0, step=0.1, group="Engine")


# ====================================================================================== feed transient
class FeedParams(EngineGeometry):
    Pc: float = P(5.0, symbol=r"P_c", desc="design chamber pressure used to size the injectors", units="MPa",
                  dims="M L⁻¹ T⁻²", lo=1.0, hi=15.0, step=0.05, group="Sizing")
    MR: float = P(2.6, symbol=r"\frac{O}{F}", desc="design mixture ratio", units="", dims="1", lo=0.8, hi=6.0,
                  step=0.05, group="Sizing")
    dp_inj: float = P(25.0, symbol=r"\frac{\Delta p_{inj}}{P_c}", desc="injector pressure drop at design", units="%",
                      dims="1", lo=5.0, hi=60.0, step=0.5, group="Sizing")
    p_tank: float = P(7.5, symbol=r"P_{tank}", desc="tank pressure (regulated setting / initial blowdown value)",
                      units="MPa", dims="M L⁻¹ T⁻²", lo=2.0, hi=20.0, step=0.05, group="Tanks")
    pressurant: str = C("regulated", choices=["regulated", "blowdown"],
                        labels=["Regulated pressurant (constant tank pressure)", "Blowdown (polytropic ullage)"],
                        symbol=r"\mathrm{pressurant}", desc="how the tanks are pressurized", group="Tanks")
    V_ull: float = P(15.0, symbol=r"V_{ull}", desc="initial ullage volume per tank (blowdown)", units="L", dims="L³",
                     lo=2.0, hi=80.0, step=0.5, group="Tanks")
    l1: float = P(3.0, symbol=r"l_1", desc="feed-line length", units="m", dims="L", lo=0.3, hi=15.0, step=0.1,
                  group="Lines")
    d1: float = P(40.0, symbol=r"d_1", desc="feed-line diameter", units="mm", dims="L", lo=10.0, hi=120.0, step=0.5,
                  group="Lines")
    K1: float = P(2.0, symbol=r"K_1", desc="feed-line loss coefficient", units="", dims="1", lo=0.0, hi=20.0,
                  step=0.1, group="Lines")
    Kv: float = P(3.0, symbol=r"K_v", desc="fully-open valve loss coefficient", units="", dims="1", lo=0.1, hi=30.0,
                  step=0.1, group="Lines")
    V_m: float = P(0.2, symbol=r"V_m", desc="injector manifold volume", units="L", dims="L³", lo=0.02, hi=5.0,
                   step=0.01, group="Lines")
    zeta: float = P(0.05, symbol=r"\zeta", desc="damping ratio of the line oscillations", units="", dims="1", lo=0.0,
                    hi=0.5, step=0.005, group="Lines")
    tau_c: float = P(2.0, symbol=r"\tau_c", desc="combustion lag", units="ms", dims="T", lo=0.2, hi=20.0, step=0.1,
                     group="Chamber")
    t_open_f: float = P(0.05, symbol=r"t_{o,f}", desc="fuel valve opening start", units="s", dims="T", lo=0.0, hi=0.5,
                        step=0.005, group="Valve timing")
    t_open_o: float = P(0.07, symbol=r"t_{o,ox}", desc="oxidizer valve opening start", units="s", dims="T", lo=0.0,
                        hi=0.5, step=0.005, group="Valve timing")
    ramp_open: float = P(100.0, symbol=r"\Delta t_{open}", desc="valve opening time", units="ms", dims="T", lo=5.0,
                         hi=500.0, step=1.0, group="Valve timing")
    t_close: float = P(0.8, symbol=r"t_{close}", desc="valve closing start", units="s", dims="T", lo=0.2, hi=3.0,
                       step=0.01, group="Valve timing")
    ramp_close: float = P(50.0, symbol=r"\Delta t_{close}", desc="valve closing time (short = water hammer)",
                          units="ms", dims="T", lo=0.2, hi=500.0, step=0.2, group="Valve timing")
    t_end: float = P(1.2, symbol=r"t_{end}", desc="simulated time", units="s", dims="T", lo=0.4, hi=5.0, step=0.05,
                     group="Simulation")

    def system(self) -> nw.FeedSystem:
        (rho_o, a_o), (rho_f, a_f) = props(self.pair)
        At = np.pi * (self.D_t * 1e-3) ** 2 / 4.0
        cs0, _, _, _ = chamber_lut(self.pair)(self.MR, self.Pc * 1e6)
        cs = 0.97 * cs0
        mdot = self.Pc * 1e6 * At / cs
        m_f, m_o = mdot / (1.0 + self.MR), mdot * self.MR / (1.0 + self.MR)
        A1 = np.pi * (self.d1 * 1e-3) ** 2 / 4.0
        dp = self.dp_inj / 100.0 * self.Pc * 1e6
        Cd = 0.7

        def chain(rho, a, m, t_open):
            return nw.Chain(rho=rho, a_snd=a, p_tank=self.p_tank * 1e6, V_ullage=self.V_ull * 1e-3, gamma_g=1.3,
                            regulated=self.pressurant == "regulated", l1=self.l1, A1=A1, K1=self.K1, Kv0=self.Kv,
                            l2=0.3, A2=A1, V_m=self.V_m * 1e-3, Cd=Cd, A_inj=m / (Cd * np.sqrt(2 * rho * dp)),
                            t_open=t_open, ramp_open=self.ramp_open * 1e-3, t_close=self.t_close,
                            ramp_close=self.ramp_close * 1e-3, zeta=self.zeta)
        Vc = self.L_star * At
        return nw.FeedSystem(self.pair, chain(rho_o, a_o, m_o, self.t_open_o), chain(rho_f, a_f, m_f, self.t_open_f),
                             Vc=Vc, At=At, eps=self.eps, tau_c=self.tau_c * 1e-3)


class FeedTransient(Simulation):
    N_FRAMES = 601

    def run(self, p: FeedParams, progress, cancel) -> Result:
        sys = p.system()
        ch_ = nw.integrate(sys, p.t_end, self.N_FRAMES, progress, cancel)
        t = np.linspace(0.0, p.t_end, self.N_FRAMES)
        meta = nw.summary(sys, t, ch_)
        ss = nw.steady_state(sys)
        meta["Pc_target"] = p.Pc
        log = [f"setup feed network: {p.pair}, tanks {p.p_tank:.1f} (MPa) {p.pressurant}, lines {p.l1:.1f} (m) x "
               f"{p.d1:.0f} (mm), injector drop {100 * meta['dp_inj_o_frac']:.0f} / {100 * meta['dp_inj_f_frac']:.0f} (%) of P_c",
               f"steady state with open valves: P_c {ss['pc'] / 1e6:.2f} (MPa), O/F {ss['OF']:.2f}"]
        if p.p_tank * 1e6 < ss["pc"] * 1.1:
            log.append("WARNING tank pressure is close to the chamber pressure: no feed margin")
        if meta["dp_inj_o_frac"] < 0.15 or meta["dp_inj_f_frac"] < 0.15:
            log.append("WARNING injector drop below 15 % of P_c (feed-coupled instability risk)")
        a_o = props(p.pair)[0][1]
        v = ss["mdot_o"] / (props(p.pair)[0][0] * np.pi * (p.d1 * 1e-3) ** 2 / 4.0)
        jk = props(p.pair)[0][0] * a_o * v / 1e6
        log.append(f"surge upper bound (Joukowsky, oxidizer line) {jk:.2f} (MPa); simulated rise "
                   f"{meta['surge_o_MPa']:.2f} (MPa) for a {p.ramp_close:.1f} (ms) closure")
        if meta["OF_min_start"] < 0.7 * ss["OF"]:
            log.append(f"start-up runs fuel-rich down to O/F {meta['OF_min_start']:.2f} (valve lead)")
        log.append(f"t = {p.t_end:.2f} (s), {self.N_FRAMES} frames")
        return Result(t, ch_, meta, log)


# ====================================================================================== pump-fed cycles
class CycleParams(EngineGeometry):
    cycle: str = C("gg", choices=cy.CYCLES, labels=cy.CYCLE_LABELS, symbol=r"\mathrm{cycle}",
                   desc="turbopump drive cycle", group="Cycle")
    Pc: float = P(6.0, symbol=r"P_c", desc="design chamber pressure", units="MPa", dims="M L⁻¹ T⁻²", lo=2.0, hi=20.0,
                  step=0.05, group="Design point")
    MR: float = P(2.6, symbol=r"\frac{O}{F}", desc="main-chamber mixture ratio", units="", dims="1", lo=0.8, hi=6.0,
                  step=0.05, group="Design point")
    N0: float = P(20000.0, symbol=r"N_0", desc="design shaft speed", units="rpm", dims="T⁻¹", lo=5000.0, hi=60000.0,
                  step=100.0, group="Design point")
    dp_inj: float = P(20.0, symbol=r"\frac{\Delta p_{inj}}{P_c}", desc="injector drop (both propellants)", units="%",
                      dims="1", lo=8.0, hi=40.0, step=0.5, group="Design point")
    dp_jac: float = P(10.0, symbol=r"\frac{\Delta p_{jac}}{P_c}", desc="jacket / preburner drop (closed cycles)",
                      units="%", dims="1", lo=0.0, hi=30.0, step=0.5, group="Design point")
    eta_p: float = P(0.68, symbol=r"\eta_p", desc="pump efficiency (both pumps)", units="", dims="1", lo=0.4, hi=0.85,
                     step=0.01, group="Turbomachinery")
    eta_t: float = P(0.6, symbol=r"\eta_t", desc="turbine efficiency", units="", dims="1", lo=0.3, hi=0.85, step=0.01,
                     group="Turbomachinery")
    T_t: float = P(900.0, symbol=r"T_t", desc="turbine inlet temperature (GG: drive gas; closed cycles: reported "
                   "as required)", units="K", dims="Θ", lo=500.0, hi=1400.0, step=5.0, group="Turbomachinery")
    cp_t: float = P(2700.0, symbol=r"c_{p,t}", desc="turbine gas specific heat", units="J·kg⁻¹·K⁻¹", dims="L² T⁻² Θ⁻¹",
                    lo=1200.0, hi=6000.0, step=50.0, group="Turbomachinery")
    MR_gg: float = P(0.35, symbol=r"\frac{O}{F}_{gg}", desc="gas-generator / preburner mixture ratio", units="",
                     dims="1", lo=0.1, hi=1.0, step=0.01, group="Turbomachinery")
    PR_t: float = P(1.5, symbol=r"\Pi_t", desc="turbine pressure ratio (closed cycles)", units="", dims="1", lo=1.1,
                    hi=3.0, step=0.01, group="Turbomachinery")
    I_shaft: float = P(0.05, symbol=r"I", desc="shaft inertia", units="kg·m²", dims="M L²", lo=0.005, hi=1.0,
                       step=0.005, group="Start-up")
    P_start: float = P(40.0, symbol=r"P_{start}", desc="starter power", units="kW", dims="M L² T⁻³", lo=5.0, hi=300.0,
                       step=1.0, group="Start-up")
    t_start: float = P(1.0, symbol=r"t_{start}", desc="starter burn time", units="s", dims="T", lo=0.2, hi=4.0,
                       step=0.05, group="Start-up")
    gg_ramp: float = P(0.8, symbol=r"t_{gg}", desc="drive-valve opening time", units="s", dims="T", lo=0.1, hi=3.0,
                       step=0.05, group="Start-up")
    t_end: float = P(8.0, symbol=r"t_{end}", desc="simulated time", units="s", dims="T", lo=2.0, hi=30.0, step=0.5,
                     group="Simulation")

    def design(self) -> cy.CycleDesign:
        closed = self.cycle != "gg"
        gamma_t = 1.2 if self.cycle == "expander" else 1.15
        return cy.CycleDesign(pair=self.pair, cycle=self.cycle, Pc=self.Pc * 1e6, Dt=self.D_t * 1e-3, eps=self.eps,
                              MR=self.MR, eta_cstar=0.97, dp_inj_o=self.dp_inj / 100, dp_inj_f=self.dp_inj / 100,
                              dp_jac=self.dp_jac / 100 if closed else 0.0, eta_po=self.eta_p, eta_pf=self.eta_p - 0.03,
                              eta_t=self.eta_t, N0=self.N0, T_t=self.T_t, cp_t=self.cp_t, gamma_t=gamma_t,
                              MR_gg=self.MR_gg, PR_t=self.PR_t, k_gg=1.25 if self.cycle == "staged" else 0.9)


class PumpFedCycle(Simulation):
    N_FRAMES = 121

    def run(self, p: CycleParams, progress, cancel) -> Result:
        eng, info = cy.design_engine(p.design(), p.I_shaft, p.P_start * 1e3, p.t_start)
        st = cy.steady(eng)
        tr = cy.transient(eng, p.t_end, p.gg_ramp, self.N_FRAMES, progress=progress, cancel=cancel)
        t = tr.pop("t")
        for k in ("pc", "F", "m_o", "m_f", "m_gg", "P_t", "P_pump", "P_net", "p_o", "p_f"):
            tr[k] = tr[k] / (1e6 if k in ("pc", "p_o", "p_f") else 1e3 if k in ("F", "P_t", "P_pump", "P_net") else 1.0)
        tr["Pc"] = tr.pop("pc")
        tr["mdot"] = tr["m_o"] + tr["m_f"] + 0.0
        tr["N_ratio"] = tr["N"] / p.N0
        isp_loss = float(info["bleed"] / max(info["mdot"] + info["bleed"], 1e-9)) if p.cycle == "gg" else 0.0
        meta = dict(N_ss=st["N"], Pc_ss=st["pc"] / 1e6, F_ss=st["F"] / 1e3, T_req=info["T_req"], m_turb=info["m_t"],
                    bleed_frac=isp_loss, P_pump_kW=st["P_pump"] / 1e3, mdot=info["mdot"],
                    PR_turb=st["p_t_in"] / st["p_t_out"], settle=float(tr["N"][-1] / st["N"]),
                    Isp_ss=st["F"] / ((st["m_o"] + st["m_f"] + (st["m_gg"] if p.cycle == "gg" else 0.0)) * 9.80665))
        log = [f"design: {p.cycle} cycle, {p.pair}, P_c {p.Pc:.1f} (MPa), N0 {p.N0:.0f} (rpm), flow {info['mdot']:.2f} (kg/s)",
               f"steady balance: N {st['N']:.0f} (rpm), P_c {st['pc'] / 1e6:.3f} (MPa), turbine {st['P_t'] * 0.98 / 1e3:.0f} = pumps "
               f"{st['P_pump'] / 1e3:.0f} (kW), turbine flow {info['m_t']:.2f} (kg/s)"]
        if p.cycle == "gg":
            log.append(f"GG bleed {100 * isp_loss:.1f} % of the propellant flow is lost overboard (Isp penalty)")
        else:
            log.append(f"required turbine inlet temperature {info['T_req']:.0f} (K)")
            if info["T_req"] > 900.0:
                log.append("WARNING required turbine inlet temperature above 900 K: more heat is needed than a jacket "
                           "or fuel-rich preburner normally provides at this pressure ratio")
        log.append(f"spool-up reaches {100 * meta['settle']:.1f} % of the steady speed at t = {p.t_end:.1f} (s)")
        return Result(t, tr, meta, log)


# ====================================================================================== chugging / L* stability
class ChuggingParams(EngineGeometry):
    Pc: float = P(5.0, symbol=r"P_c", desc="chamber pressure", units="MPa", dims="M L⁻¹ T⁻²", lo=1.0, hi=15.0,
                  step=0.05, group="Operating point")
    MR: float = P(2.6, symbol=r"\frac{O}{F}", desc="mixture ratio", units="", dims="1", lo=0.8, hi=6.0, step=0.05,
                  group="Operating point")
    dp_inj: float = P(20.0, symbol=r"\frac{\Delta p_{inj}}{P_c}", desc="injector drop", units="%", dims="1", lo=3.0,
                      hi=60.0, step=0.5, group="Feed system")
    l_feed: float = P(3.0, symbol=r"l_f", desc="feed-line length (inertance)", units="m", dims="L", lo=0.0, hi=15.0,
                      step=0.1, group="Feed system")
    d_feed: float = P(40.0, symbol=r"d_f", desc="feed-line diameter", units="mm", dims="L", lo=10.0, hi=120.0, step=0.5,
                      group="Feed system")
    tau_c: float = P(1.5, symbol=r"\tau_c", desc="combustion delay", units="ms", dims="T", lo=0.1, hi=20.0, step=0.1,
                     group="Combustion")
    pi0: float = P(1.0, symbol=r"\pi_0", desc="initial chamber-pressure perturbation", units="%", dims="1", lo=0.1,
                   hi=10.0, step=0.1, group="Simulation")
    t_end: float = P(100.0, symbol=r"t_{end}", desc="simulated time", units="ms", dims="T", lo=20.0, hi=1000.0,
                     step=5.0, group="Simulation")

    def terms(self) -> dict:
        (rho_o, _), (rho_f, _) = props(self.pair)
        cs0, Tc, gam, _ = chamber_lut(self.pair)(self.MR, self.Pc * 1e6)
        cs = 0.97 * cs0
        RT = (cs * float(np.sqrt(gam) * (2 / (gam + 1)) ** ((gam + 1) / (2 * (gam - 1))))) ** 2
        At = np.pi * (self.D_t * 1e-3) ** 2 / 4.0
        mdot = self.Pc * 1e6 * At / cs
        dp = self.dp_inj / 100.0 * self.Pc * 1e6
        kappa = self.Pc * 1e6 / (2.0 * dp)
        theta = self.L_star * cs / RT
        A1 = np.pi * (self.d_feed * 1e-3) ** 2 / 4.0
        tau_f = (self.l_feed / A1) * (mdot / (1.0 + self.MR)) / (2.0 * dp)             # fuel line, representative
        return dict(kappa=kappa, theta=theta, tau_f=tau_f, tau_c=self.tau_c * 1e-3, cs=cs, RT=RT, mdot=mdot, Tc=Tc,
                    gamma=gam, dp=dp)


class Chugging(Simulation):
    def run(self, p: ChuggingParams, progress, cancel) -> Result:
        tm = p.terms()
        mg = ch.stability_margins(tm["kappa"], tm["tau_c"], tm["theta"], tm["tau_f"])
        w = np.geomspace(1.0, 4.0e4, 2000)
        G = ch.open_loop(w, tm["kappa"], tm["tau_c"], tm["theta"], tm["tau_f"])
        t, pi, m = ch.simulate(tm["kappa"], tm["tau_c"], tm["theta"], tm["tau_f"], p.t_end * 1e-3, p.pi0 / 100.0)
        n = 601
        tt = np.linspace(0.0, p.t_end * 1e-3, n)
        gr = ch.growth_rate(t, pi)
        meta = dict(mg, kappa=tm["kappa"], theta_g_ms=tm["theta"] * 1e3, tau_f_ms=tm["tau_f"] * 1e3,
                    tau_c_ms=tm["tau_c"] * 1e3, growth=gr, dp_frac=p.dp_inj / 100.0, f_nominal=1.0 / (2 * np.pi * np.sqrt(
                        max(tm["theta"] * tm["tau_f"], 1e-12))) if tm["tau_f"] > 0 else float("nan"))
        res_ch = {"pi": np.interp(tt, t, pi) * 100.0, "m": np.interp(tt, t, m) * 100.0,
                  "Pc": p.Pc * (1.0 + np.interp(tt, t, pi))}
        arrays = {"nyq_w": w, "nyq_re": G.real, "nyq_im": G.imag}
        # minimum stable injector drop at this L*: kappa_crit = Pc / (2 dp_min)
        dp_min = p.Pc * 1e6 / (2.0 * mg["kappa_crit"]) if np.isfinite(mg["kappa_crit"]) else 0.0
        meta["dp_min_frac"] = dp_min / (p.Pc * 1e6)
        log = [f"setup time-lag model: kappa {tm['kappa']:.2f}, theta_g {tm['theta'] * 1e3:.2f} (ms), tau_c {tm['tau_c'] * 1e3:.2f} (ms), "
               f"tau_f {tm['tau_f'] * 1e3:.3f} (ms)",
               f"Nyquist: {'STABLE' if mg['stable'] else 'UNSTABLE'}, gain margin {mg['gain_margin']:.2f}, crossover "
               f"{mg['f_pc']:.0f} (Hz), critical injector drop >= {100 * meta['dp_min_frac']:.1f} (%) of P_c"]
        if not mg["stable"]:
            log.append("WARNING unstable: raise the injector pressure drop or shorten the combustion delay / L*")
        log.append(f"time trace growth rate {gr:.1f} (1/s)")
        return Result(tt, res_ch, meta, log, arrays)
