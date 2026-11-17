"""Run driver: 0-D hybrid burn (HTPB/GOX power law), see ``core.ballistics.hybrid0d``."""
from __future__ import annotations

import numpy as np

from ..core.ballistics import hybrid0d as h0
from ..core.result import Result
from ..core.schema import P, ParamModel
from .base import Simulation


class HybridBurnParams(ParamModel):
    # Oxidizer
    mdot_ox: float = P(1.80, symbol=r"\dot{m}_{ox}", desc="oxidizer mass flow", units="kg·s⁻¹", dims="M T⁻¹",
                       lo=0.2, hi=5.0, step=0.01, group="Oxidizer")
    p_tank: float = P(5.50, symbol=r"P_{tank}", desc="oxidizer tank pressure", units="MPa", dims="M L⁻¹ T⁻²",
                      lo=2.0, hi=10.0, step=0.05, group="Oxidizer")
    # Fuel grain
    R_p0: float = P(25.0, symbol=r"R_{p,0}", desc="initial port radius", units="mm", dims="L",
                    lo=10.0, hi=45.0, step=0.5, group="Fuel grain")
    web: float = P(35.0, symbol=r"w_0", desc="initial web thickness (outer radius minus port radius)", units="mm",
                   dims="L", lo=5.0, hi=60.0, step=0.5, group="Fuel grain")
    L_g: float = P(0.75, symbol=r"L_g", desc="grain length", units="m", dims="L", lo=0.2, hi=2.0, step=0.01,
                   group="Fuel grain")
    rho_f: float = P(920.0, symbol=r"\rho_f", desc="fuel density", units="kg·m⁻³", dims="M L⁻³", lo=700.0,
                     hi=1200.0, step=5.0, group="Fuel grain")
    a: float = P(0.0541, symbol=r"a", desc=r"regression coefficient (rdot in mm/s, G_ox in kg/m²/s)",
                 units="mm·s⁻¹", dims="L T⁻¹ (M L⁻² T⁻¹)⁻ⁿ", lo=0.01, hi=0.2, step=0.0001, group="Fuel grain")
    n: float = P(0.647, symbol=r"n", desc="regression exponent", units="", dims="1", lo=0.3, hi=1.0, step=0.001,
                 group="Fuel grain")
    # Nozzle
    D_t: float = P(32.0, symbol=r"D_t", desc="throat diameter", units="mm", dims="L", lo=10.0, hi=100.0, step=0.5,
                   group="Nozzle")
    eps: float = P(6.0, symbol=r"\varepsilon=\frac{A_e}{A_t}", desc="nozzle expansion ratio", units="", dims="1",
                   lo=2.0, hi=40.0, step=0.1, group="Nozzle")
    # Simulation
    eta_cstar: float = P(0.95, symbol=r"\eta_{c^*}", desc="combustion efficiency on c*", units="", dims="1",
                         lo=0.8, hi=1.0, step=0.005, group="Simulation")
    t_burn: float = P(10.0, symbol=r"t_b", desc="burn duration (oxidizer valve closes at this time)", units="s",
                      dims="T", lo=1.0, hi=30.0, step=0.5, group="Simulation")

    def to_si(self) -> h0.HybridBurnSI:
        return h0.HybridBurnSI(
            mdot_ox=self.mdot_ox, p_tank=self.p_tank * 1e6, R0=self.R_p0 * 1e-3,
            L=self.L_g, R_out=(self.R_p0 + self.web) * 1e-3, rho_f=self.rho_f, a=self.a * 1e-3, n=self.n,
            Dt=self.D_t * 1e-3, eps=self.eps, t_burn=self.t_burn, eta_cstar=self.eta_cstar)


class HybridBurn0D(Simulation):
    N_FRAMES = 601

    def run(self, params: HybridBurnParams, progress, cancel) -> Result:
        p = params.to_si()
        ch = h0.integrate(p, self.N_FRAMES, progress, cancel)
        t = np.linspace(0.0, p.t_burn, self.N_FRAMES)
        meta = h0.summary(p, t, ch)
        log = [f"setup 0-D hybrid burn, {self.N_FRAMES} frames", "LSODA rtol 1e-8 (stiff chamber filling)"]
        if meta["dp_inj_margin"] < 0.2:
            log.append(f"WARNING injector margin (P_tank-P_c)/P_c = {meta['dp_inj_margin']*100:.0f} % < 20 % "
                       "(feed-coupled instability risk)")
        if meta["web_burned"] >= 0.999:
            log.append("NOTE web fully consumed before t_burn (burnout)")
        log.append(f"t = {p.t_burn:.1f} (s) burnout, {self.N_FRAMES} frames")
        return Result(t, ch, meta, log)
