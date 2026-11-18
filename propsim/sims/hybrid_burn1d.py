"""Run drivers: quasi-1D hybrid burn (valve-fed, with throttling and throat erosion) and N₂O blowdown motor."""
from __future__ import annotations

import numpy as np

from dataclasses import replace

from ..core.ballistics import hybrid1d as h1
from ..core.ballistics import solid as sol
from ..core.grain import builder as gb
from ..core.grain import stack as gstack
from ..core.regression import POWER_LAWS
from ..core.result import Result
from ..core.schema import C, P, ParamModel, T
from ..core.thermo.cea import PAIRS
from .base import Simulation

LAW_LABELS = [r"Power law $\dot r=aG_{ox}^{n}$", "Marxman"]


class _MotorFields(ParamModel):
    """Grain, regression and nozzle fields shared by both models."""

    R_p0: float = P(25.0, symbol=r"R_{p,0}", desc="initial port radius", units="mm", dims="L", lo=10.0, hi=45.0,
                    step=0.5, group="Fuel grain")
    web: float = P(35.0, symbol=r"w_0", desc="initial web thickness (outer radius minus port radius)", units="mm",
                   dims="L", lo=5.0, hi=60.0, step=0.5, group="Fuel grain")
    L_g: float = P(0.75, symbol=r"L_g", desc="grain length", units="m", dims="L", lo=0.2, hi=2.0, step=0.01,
                   group="Fuel grain")
    rho_f: float = P(920.0, symbol=r"\rho_f", desc="fuel density", units="kg·m⁻³", dims="M L⁻³", lo=700.0, hi=1200.0,
                     step=5.0, group="Fuel grain")
    kind: str = C("circle", choices=["circle", "custom"], labels=["Circular port", "Custom (shape list)"],
                  symbol=r"\mathrm{port}", desc="port cross-section (custom: drawn as shapes, outer radius = port "
                  "radius + web)", group="Fuel grain")
    section_json: str = T("", label="custom port", desc="custom port drawn as a list of shapes, with optional "
                          "inhibited surfaces (used when the port is custom)", editor="section", group="Fuel grain")
    stack_json: str = T("", label="fuel grain stack", desc="optional stack of fuel grains, each with its own length, "
                        "port, fuel, regression law and chemistry (empty = one grain from the parameters above)",
                        editor="hstack", group="Fuel grain")
    law: str = C("power", choices=["power", "marxman"], labels=LAW_LABELS, symbol=r"\mathrm{regression\ law}",
                 desc="local regression law applied at every axial station (total mass flux G)", group="Regression")
    a: float = P(0.0541, symbol=r"a", desc="power-law coefficient (rdot in mm/s, G in kg/m²/s)", units="mm·s⁻¹",
                 dims="L T⁻¹ (M L⁻² T⁻¹)⁻ⁿ", lo=0.01, hi=0.3, step=0.0001, group="Regression")
    n: float = P(0.647, symbol=r"n", desc="power-law exponent", units="", dims="1", lo=0.3, hi=1.0, step=0.001,
                 group="Regression")
    B: float = P(6.0, symbol=r"B", desc="blowing number (Marxman law)", units="", dims="1", lo=1.0, hi=20.0,
                 step=0.1, group="Regression")
    D_t: float = P(32.0, symbol=r"D_t", desc="initial throat diameter", units="mm", dims="L", lo=10.0, hi=100.0,
                   step=0.5, group="Nozzle")
    eps: float = P(6.0, symbol=r"\varepsilon=\frac{A_e}{A_t}", desc="nozzle expansion ratio (of the initial throat)",
                   units="", dims="1", lo=2.0, hi=40.0, step=0.1, group="Nozzle")
    kt: float = P(0.0, symbol=r"k_t", desc="throat erosion rate per MPa of chamber pressure", units="mm·s⁻¹",
                  dims="L T⁻¹", lo=0.0, hi=0.2, step=0.002, group="Nozzle")
    eta_cstar: float = P(0.95, symbol=r"\eta_{c^*}", desc="combustion efficiency on c*", units="", dims="1", lo=0.8,
                         hi=1.0, step=0.005, group="Simulation")

    @property
    def R_out(self) -> float:
        """Grain outer radius (mm): port radius plus web (the section editor draws inside it)."""
        return self.R_p0 + self.web

    def _table(self, spec: gb.SectionSpec) -> sol.SectionTable:
        b = gb.cached_section(replace(spec, R_out=self.R_out).to_json())
        return sol.SectionTable(b.d * 1e-3, b.area * 1e-6, b.perim * 1e-3, b.A_case * 1e-6)

    def _default_spec(self) -> gb.SectionSpec:
        if self.kind == "custom" and self.section_json.strip():
            return gb.spec_from_json(self.section_json)
        return gb.spec_from_dict({"R_out": self.R_out, "shapes": [{"kind": "circle", "r": self.R_p0}]})

    def _fuel(self, table: sol.SectionTable, L: float, over: dict) -> h1.HGrain:
        f = {"rho_f": self.rho_f, "law": self.law, "a": self.a, "n": self.n, "B": self.B, **over}
        if f["law"] not in ("power", "marxman"):
            raise ValueError(f"regression law {f['law']!r}: use 'power' or 'marxman'")
        chem = None
        if "chem" in over:
            c = gstack._thaw(over["chem"])
            chem = h1.ChemTable(*(tuple(float(x) for x in c[k]) for k in ("of", "cstar", "Tc", "gamma")))
        elif "pair" in over:
            if over["pair"] not in PAIRS:
                raise ValueError(f"unknown CEA pair {over['pair']!r}; use one of {sorted(PAIRS)}")
            chem = over["pair"]
        return h1.HGrain(table, L, float(f["rho_f"]), f["law"], float(f["a"]) * 1e-3, float(f["n"]), float(f["B"]),
                         chem)

    def _motor(self, pair: str) -> h1.Motor1D:
        defs = gstack.parse_stack(self.stack_json, gstack.FUEL_KEYS)
        if self.kind == "custom" or defs:
            kw = dict(Dt=self.D_t * 1e-3, eps=self.eps, eta_cstar=self.eta_cstar, kt=self.kt * 1e-3, N=20)
            if not defs:
                defs = [gstack.GrainDef(self.L_g)]
            grains = [self._fuel(self._table(gb.spec_from_dict(d.section) if d.section else self._default_spec()),
                                 d.L, dict(d.prop)) for d in defs]
            return h1.Motor1D.from_grains(grains, pair, **kw)
        return h1.Motor1D(pair=pair, L=self.L_g, R0=self.R_p0 * 1e-3, R_out=(self.R_p0 + self.web) * 1e-3,
                          rho_f=self.rho_f, law=self.law, a=self.a * 1e-3, n=self.n, B=self.B, mu=8e-5,
                          Dt=self.D_t * 1e-3, eps=self.eps, eta_cstar=self.eta_cstar, kt=self.kt * 1e-3, N=20)


class HybridBurn1DParams(_MotorFields):
    pair: str = C("GOX/HTPB", choices=["GOX/HTPB", "LOX/HTPB"], labels=["GOX / HTPB", "LOX / HTPB"],
                  symbol=r"\mathrm{propellants}", desc="oxidizer / fuel pair (NASA CEA)", group="Oxidizer")
    mdot_ox: float = P(1.80, symbol=r"\dot{m}_{ox}", desc="oxidizer mass flow (before throttling)", units="kg·s⁻¹",
                       dims="M T⁻¹", lo=0.2, hi=5.0, step=0.01, group="Oxidizer")
    throttle: float = P(1.0, symbol=r"\vartheta", desc="throttle level during the throttle window (1 = none)",
                        units="", dims="1", lo=0.3, hi=1.0, step=0.01, group="Throttling")
    t_th1: float = P(3.0, symbol=r"t_{th,1}", desc="throttle-down start time", units="s", dims="T", lo=0.5, hi=20.0,
                     step=0.1, group="Throttling")
    t_th2: float = P(6.0, symbol=r"t_{th,2}", desc="throttle-up start time", units="s", dims="T", lo=1.0, hi=28.0,
                     step=0.1, group="Throttling")
    t_burn: float = P(10.0, symbol=r"t_b", desc="burn duration (oxidizer valve closes at this time)", units="s",
                      dims="T", lo=1.0, hi=30.0, step=0.5, group="Simulation")

    def to_si(self) -> h1.Motor1D:
        return self._motor(self.pair)

    def supply(self) -> h1.ValveSupply:
        return h1.ValveSupply(self.mdot_ox, self.t_burn, throttle=self.throttle, t_th1=self.t_th1,
                              t_th2=max(self.t_th2, self.t_th1 + 0.6))


class HybridBurn1D(Simulation):
    N_FRAMES = 601

    def run(self, params: HybridBurn1DParams, progress, cancel) -> Result:
        p, sup = params.to_si(), params.supply()
        ch = h1.integrate(p, sup, params.t_burn, self.N_FRAMES, progress, cancel)
        t = np.linspace(0.0, params.t_burn, self.N_FRAMES)
        meta = h1.summary(p, t, ch)
        meta["x_cells"] = p.cell_x().tolist()
        log = [f"setup quasi-1D hybrid burn, {p.nc} axial cells, {self.N_FRAMES} frames, {params.pair} (CEA tables)",
               "LSODA rtol 1e-8 (stiff chamber filling)"]
        if params.throttle < 0.999:
            log.append(f"throttle to {params.throttle:.2f} between t = {params.t_th1:.1f} and "
                       f"{max(params.t_th2, params.t_th1 + 0.6):.1f} (s)")
        if meta["web_burned"] >= 0.999:
            log.append("NOTE web fully consumed at the aft end before t_burn (burnout / sliver-free grain)")
        log.append(f"t = {params.t_burn:.1f} (s) burnout; web burned {100 * meta['web_burned']:.0f} %")
        return Result(t, ch, meta, log)


class N2OBlowdownParams(_MotorFields):
    L_g: float = P(0.5, symbol=r"L_g", desc="grain length", units="m", dims="L", lo=0.2, hi=2.0, step=0.01,
                   group="Fuel grain")
    D_t: float = P(28.0, symbol=r"D_t", desc="initial throat diameter", units="mm", dims="L", lo=10.0, hi=100.0,
                   step=0.5, group="Nozzle")
    V_tank: float = P(20.0, symbol=r"V_{tank}", desc="oxidizer tank volume", units="L", dims="L³", lo=5.0, hi=100.0,
                      step=0.5, group="Self-pressurized N₂O tank")
    T0: float = P(293.15, symbol=r"T_0", desc="initial tank temperature (saturated)", units="K", dims="Θ", lo=250.0,
                  hi=303.0, step=0.5, group="Self-pressurized N₂O tank")
    fill: float = P(0.80, symbol=r"\phi_{ℓ}", desc="initial liquid volume fraction", units="", dims="1", lo=0.3,
                    hi=0.95, step=0.01, group="Self-pressurized N₂O tank")
    Cd: float = P(0.65, symbol=r"C_d", desc="injector discharge coefficient", units="", dims="1", lo=0.3, hi=1.0,
                  step=0.01, group="Injector")
    d_inj: float = P(2.5, symbol=r"d_{inj}", desc="injector hole diameter", units="mm", dims="L", lo=0.5, hi=4.0,
                     step=0.05, group="Injector")
    n_inj: float = P(8.0, symbol=r"N_{inj}", desc="number of injector holes", units="", dims="1", lo=1.0, hi=40.0,
                     step=1.0, group="Injector")
    loss: float = P(0.05, symbol=r"\Delta_{feed}", desc="fraction of tank pressure lost in the feed line", units="",
                    dims="1", lo=0.0, hi=0.4, step=0.01, group="Injector")
    t_burn: float = P(12.0, symbol=r"t_b", desc="run time (valve closes at this time)", units="s", dims="T", lo=2.0,
                      hi=40.0, step=0.5, group="Simulation")

    def to_si(self) -> h1.Motor1D:
        return self._motor("N2O/HTPB")

    def supply(self) -> h1.N2OSupply:
        return h1.N2OSupply(V=self.V_tank * 1e-3, T0=self.T0, fill=self.fill, Cd=self.Cd,
                            A_inj=self.n_inj * np.pi * (self.d_inj * 1e-3 / 2) ** 2, t_burn=self.t_burn,
                            loss_frac=self.loss)


class N2OBlowdown(Simulation):
    N_FRAMES = 601

    def run(self, params: N2OBlowdownParams, progress, cancel) -> Result:
        p, sup = params.to_si(), params.supply()
        ch = h1.integrate(p, sup, params.t_burn, self.N_FRAMES, progress, cancel)
        t = np.linspace(0.0, params.t_burn, self.N_FRAMES)
        meta = h1.summary(p, t, ch)
        meta["x_cells"] = p.cell_x().tolist()
        m0 = ch["m_tank"][0]
        meta["m_ox_tank0"] = float(m0)
        meta["m_ox_used"] = float(m0 - ch["m_tank"][-1])
        dry = np.where(ch["m_liq"] <= 1e-6 * m0)[0]
        meta["t_liquid_depleted"] = float(t[dry[0]]) if dry.size else float("nan")
        meta["p_tank0_MPa"] = float(ch["p_tank"][0] / 1e6)
        log = [f"setup N₂O blowdown: tank {params.V_tank:.0f} L at {params.T0:.1f} K, {m0:.2f} kg N₂O, "
               f"p_sat {meta['p_tank0_MPa']:.2f} MPa", f"quasi-1D motor, {p.nc} cells; Dyer injector, "
               f"Cd {params.Cd:.2f}, {params.n_inj:.0f} x {params.d_inj:.2f} mm",
               "regression law defaults to the HTPB/GOX power law — set your own for N₂O/HTPB"]
        if np.isfinite(meta["t_liquid_depleted"]):
            log.append(f"liquid N₂O depleted at t = {meta['t_liquid_depleted']:.2f} (s): vapour-only blowdown")
        if meta["web_burned"] >= 0.999:
            log.append("NOTE web fully consumed before the end of the run")
        margin = float(np.nanmin((ch["p_tank"][5:-20] * 1e-6 - ch["Pc"][5:-20]) / np.maximum(ch["Pc"][5:-20], 1e-9)))
        if margin < 0.2:
            log.append(f"WARNING injector margin (P_tank-P_c)/P_c min {margin * 100:.0f} % < 20 % "
                       "(feed-coupled instability risk)")
        log.append(f"t = {params.t_burn:.1f} (s), {self.N_FRAMES} frames")
        return Result(t, ch, meta, log)


_ = POWER_LAWS   # re-exported for the Live regression model
