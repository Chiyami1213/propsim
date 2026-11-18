"""Run driver for the Nuclear module: reactor kinetics with thermal feedback (startup, reactivity step, shutdown)."""
from __future__ import annotations

import numpy as np

from ..core.nuclear import kinetics as kn
from ..core.result import Result
from ..core.schema import C, P, ParamModel
from .base import Simulation

SCENARIOS = ["startup", "step", "shutdown"]
SCENARIO_LABELS = ["Startup with outlet temperature control", "Reactivity step at full power",
                   "Scram from full power with decay heat"]


class KineticsParams(ParamModel):
    scenario: str = C("startup", choices=SCENARIOS, labels=SCENARIO_LABELS, symbol=r"\mathrm{scenario}",
                      desc="what the drums and the turbopump do", group="Scenario")
    t_end: float = P(40.0, symbol=r"t_{end}", desc="simulated time", units="s", dims="T", lo=2.0, hi=600.0, step=1.0,
                     group="Scenario")
    n0: float = P(-2.0, symbol=r"\log_{10}\frac{P_{start}}{P_0}", desc="startup: initial power (log10 of the fraction)",
                  units="", dims="1", lo=-8.0, hi=-1.0, step=0.1, group="Startup")
    m0: float = P(5.0, symbol=r"\frac{\dot m_{start}}{\dot m_0}", desc="startup: initial propellant flow", units="%",
                  dims="1", lo=1.0, hi=50.0, step=0.5, group="Startup")
    t_flow: float = P(20.0, symbol=r"t_{ramp}", desc="startup: flow and set-point ramp time", units="s", dims="T",
                      lo=1.0, hi=120.0, step=0.5, group="Startup")
    T_set: float = P(2845.0, symbol=r"T_{set}", desc="startup: outlet temperature set point", units="K", dims="Θ",
                     lo=500.0, hi=3200.0, step=5.0, group="Startup")
    gain: float = P(10.0, symbol=r"K_d", desc="startup: drum rate per kelvin of outlet error (x 1e-6)",
                    units="s⁻¹·K⁻¹", dims="T⁻¹ Θ⁻¹", lo=0.5, hi=100.0, step=0.5, group="Startup")
    rate: float = P(1.0, symbol=r"\frac{\dot\rho_{max}}{\beta}", desc="startup: maximum drum reactivity rate (dollars per second)",
                    units="s⁻¹", dims="T⁻¹", lo=0.05, hi=5.0, step=0.05, group="Startup")
    rho1: float = P(0.3, symbol=r"\rho_1", desc="step: reactivity inserted by the drums", units="$", dims="1", lo=-5.0,
                    hi=1.5, step=0.01, group="Step")
    t_ramp: float = P(0.0, symbol=r"t_{ins}", desc="step: insertion time (0 for a step)", units="s", dims="T", lo=0.0,
                      hi=60.0, step=0.1, group="Step")
    t_sd: float = P(5.0, symbol=r"t_{sd}", desc="shutdown: scram time", units="s", dims="T", lo=0.5, hi=300.0,
                    step=0.5, group="Shutdown")
    rho_sd: float = P(-30.0, symbol=r"\rho_{sd}", desc="shutdown: drum reactivity at the scram", units="$", dims="1",
                      lo=-60.0, hi=-0.5, step=0.5, group="Shutdown")
    t_fp: float = P(600.0, symbol=r"t_{fp}", desc="shutdown: full-power time before the scram (decay heat)", units="s",
                    dims="T", lo=10.0, hi=10000.0, step=10.0, group="Shutdown")
    m_cool: float = P(3.0, symbol=r"\frac{\dot m_{cool}}{\dot m_0}", desc="shutdown: minimum cooldown flow", units="%",
                      dims="1", lo=0.5, hi=30.0, step=0.5, group="Shutdown")
    tau_flow: float = P(5.0, symbol=r"\tau_{\dot m}", desc="shutdown: lag of the flow behind the shutdown power",
                        units="s", dims="T", lo=0.2, hi=60.0, step=0.1, group="Shutdown")
    q0: float = P(3750.0, symbol=r"q_0", desc="full-power density", units="W·cm⁻³", dims="M L⁻¹ T⁻³", lo=100.0,
                  hi=10000.0, step=10.0, group="Core")
    Lam: float = P(100.0, symbol=r"\Lambda", desc="prompt neutron generation time", units="μs", dims="T", lo=1.0,
                   hi=1000.0, step=1.0, group="Core")
    alpha_f: float = P(2.0, symbol=r"\alpha_f", desc="magnitude of the negative fuel coefficient (x 1e-5)",
                       units="K⁻¹", dims="Θ⁻¹", lo=-2.0, hi=10.0, step=0.1, group="Feedback")
    alpha_m: float = P(4.5, symbol=r"\alpha_m", desc="magnitude of the negative moderator coefficient (x 1e-5)",
                       units="K⁻¹", dims="Θ⁻¹", lo=-2.0, hi=10.0, step=0.1, group="Feedback")
    beta_p: float = P(100.0, symbol=r"\beta_p", desc="propellant density coefficient", units="cm³·g⁻¹",
                      dims="L³ M⁻¹", lo=0.0, hi=300.0, step=1.0, group="Feedback")
    U_fm: float = P(32.3, symbol=r"U_{fm}", desc="fuel to moderator conductance per unit volume",
                    units="W·cm⁻³·K⁻¹", dims="M L⁻¹ T⁻³ Θ⁻¹", lo=1.0, hi=200.0, step=0.1, group="Thermal")
    U_mp: float = P(3.14, symbol=r"U_{mp}", desc="moderator to propellant conductance per unit volume",
                    units="W·cm⁻³·K⁻¹", dims="M L⁻¹ T⁻³ Θ⁻¹", lo=0.2, hi=50.0, step=0.01, group="Thermal")
    mdot: float = P(0.083, symbol=r"\dot m_0", desc="full-power propellant flow per unit core volume",
                    units="g·s⁻¹·cm⁻³", dims="M L⁻³ T⁻¹", lo=0.01, hi=0.5, step=0.001, group="Thermal")
    T_in: float = P(155.0, symbol=r"T_{in}", desc="core inlet temperature", units="K", dims="Θ", lo=20.0, hi=800.0,
                    step=1.0, group="Thermal")
    P: float = P(7.0, symbol=r"P_c", desc="full-power core pressure", units="MPa", dims="M L⁻¹ T⁻²", lo=1.0, hi=15.0,
                 step=0.1, group="Thermal")

    def core(self) -> kn.Core:
        return kn.Core(q0=self.q0 * 1e6, Lam=self.Lam * 1e-6, U_fm=self.U_fm * 1e6, U_mp=self.U_mp * 1e6,
                       P=self.P * 1e6, mdot=self.mdot * 1e3, T_in=self.T_in, alpha_f=self.alpha_f * 1e-5,
                       alpha_m=self.alpha_m * 1e-5, beta_p=self.beta_p * 1e-3)

    def program(self, core: kn.Core) -> kn.Program:
        b = core.beta
        if self.scenario == "startup":
            return kn.Program(n0=10.0**self.n0, m0=self.m0 / 100, m1=1.0, t_flow=self.t_flow, control="temperature",
                              t_ramp=self.t_flow, T_set=self.T_set, gain=self.gain * 1e-6 * b / 0.0065,
                              rate_max=self.rate * b)
        if self.scenario == "step":
            return kn.Program(n0=1.0, rho1=self.rho1 * b, t_ramp=self.t_ramp)
        return kn.Program(n0=1.0, t_sd=self.t_sd, rho_sd=self.rho_sd * b, t_fp=self.t_fp, m_cool=self.m_cool / 100,
                          tau_flow=self.tau_flow)


class ReactorKinetics(Simulation):
    N_FRAMES = 401

    def run(self, p: KineticsParams, progress, cancel) -> Result:
        core = p.core()
        prog = p.program(core)
        progress(0.02)
        o = kn.simulate(core, prog, p.t_end, self.N_FRAMES, progress, cancel)
        b = core.beta
        ch = {k: o[k] for k in ("n", "P_tot", "decay", "Tf", "Tm", "Tp", "Tout", "mflow", "T_set", "pc")}
        for k in ("rho", "rho_ext", "rho_f", "rho_m", "rho_pr"):
            ch[k] = o[k] / b
        cst = core.constants()
        w = np.logspace(-4, 3, 300)
        H = core.transfer(w) / 1e6                                        # W·cm⁻³ per unit reactivity
        f_sd = kn.cooldown_fraction(p.t_fp, min(p.rho_sd, -0.01) * b, core.Lam)
        meta = dict(Tf_max=float(o["Tf"].max()), P_max=float(o["P_tot"].max()), net=float(cst["net"]),
                    beta_p_crit=float(cst["beta_p_crit"] * 1e3), f_sd=float(f_sd), tau_f=cst["tau_f"],
                    tau_m=cst["tau_m"], tau_p=cst["tau_p"], t_unit="s")
        progress(1.0)
        Tf0, Tm0, Tp0 = o["steady0"]
        log = [f"setup {p.scenario}: initial steady state T_f {Tf0:.0f} (K), T_m {Tm0:.0f} (K), T_p {Tp0:.0f} (K)",
               f"thermal time constants τ_f {cst['tau_f'] * 1e3:.1f} (ms), τ_m {cst['tau_m'] * 1e3:.1f} (ms), "
               f"τ_p {cst['tau_p'] * 1e3:.2f} (ms)",
               f"low-frequency feedback α_f A_f + α_m A_m − β_p B_p = {cst['net'] * 1e6:.3g} (cm³·W⁻¹) "
               f"({'stable' if cst['net'] > 0 else 'UNSTABLE'}), critical β_p {cst['beta_p_crit'] * 1e3:.0f} (cm³·g⁻¹)",
               f"peak fuel temperature {meta['Tf_max']:.0f} (K), peak power {meta['P_max']:.3g} of full power",
               f"cooldown propellant after {p.t_fp:.0f} (s) at full power: {f_sd * 100:.2f} (%) of the burn propellant"]
        return Result(o["t"], ch, meta, log, arrays={"bode_w": w, "bode_gain": 20 * np.log10(np.abs(H)),
                                                     "bode_phase": np.degrees(np.angle(H))})
