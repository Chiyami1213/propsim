r"""Steady models of the library component types for the engine-graph solver.

Every model owns a few unknowns (through-flows, the shaft speed lives on the shaft group) and the same number of
residual equations, except the chamber (one more equation: the throat) and the nozzle (one fewer: it only receives the
throat flow), so any graph that obeys the port rules gives a square system.  Equations (steady, SI):

* passive loss  :math:`\Delta p = K\frac{\dot m|\dot m|}{2\rho A^2}`, orifice/injector :math:`\dot m = C_dA\sqrt{2\rho\Delta p}`
  (Huzel & Huang eqs. 7-2, 7-3, 8-1), line friction by Haaland's explicit Colebrook fit;
* pump  :math:`\Delta p = \rho\psi U^2\big[s(\tfrac{N}{N_d})^2-(s-1)(\tfrac{Q}{Q_d})^2\big]` (affinity-scaled curve through
  the design point, H&H eq. 6-4 head coefficient, shut-off ratio ``s``), :math:`P=\frac{\dot m\Delta p}{\rho\eta}`;
* turbine  gas :math:`P=\eta\dot m c_pT_0[1-R_t^{-(\gamma-1)/\gamma}]` (H&H eq. 6-20), pure fluid
  :math:`P=\eta\dot m(h_0-h_s)`, liquid (hydraulic) :math:`P=\eta\dot m\Delta p/\rho`; flow either at a fixed pressure
  ratio or through a choked nozzle :math:`\dot m=A_n p_0\Gamma/\sqrt{RT_0}`;
* chamber  :math:`p_cA_t=\eta_{c^*}c^*(O/F,p_c)\dot m` with the shipped CEA tables, thrust
  :math:`F=\eta_{C_F}C_Fp_cA_t` (Sutton eq. 3-30);
* regenerative jacket  heat load and friction from a Bartz march along the chamber contour (``jacket.py``).
"""
from __future__ import annotations

import numpy as np

from ..constants import G0
from ..nozzle import CF, Gamma, exit_state
from .drive_gas import drive_gas
from .props import State, density, gas_state, liquid_state, props

PAIR_KEYS = {("LOX", "RP1"): "LOX/RP-1", ("LOX", "RP1_SURR"): "LOX/RP-1", ("LOX", "LH2"): "LOX/LH2",
             ("LOX", "LCH4"): "LOX/CH4", ("LOX", "ETOH"): "LOX/Ethanol", ("LOX", "ETOH95"): "LOX/Ethanol",
             ("LOX", "ETOH75"): "LOX/Ethanol", ("N2O4", "MMH"): "N2O4/MMH"}
VALVE_TYPES = ("valve_ball", "valve_butterfly", "valve_poppet")


def area(d: float) -> float:
    return np.pi * d * d / 4.0


def sq(m: float) -> float:
    return m * abs(m)


class Model:
    """Base: a component instance. ``unknowns`` names its solver unknowns; see :class:`~.network.Network`."""

    unknowns: tuple[str, ...] = ()
    shaft_role = None                  # "pump" | "turbine" | None
    signs: tuple[int, ...] = ()        # d(unknown)/d(pseudo-time) = sign * residual (relaxation toward steady state)

    def __init__(self, net, nid: str) -> None:
        self.net, self.id = net, nid
        self.node = net.nodes[nid]
        self.type = self.node["type"]
        self.opt = dict(self.node.get("options", {}))
        self.slow: dict = {}           # quantities frozen between outer iterations (heat loads, friction factors)

    def par(self, name: str, default=None):
        return self.net.param(self.id, name, default)

    # interface ---------------------------------------------------------------------------------------------------
    def flows(self, x) -> dict[str, float]:
        """Mass flow leaving the component through each fluid port (negative = entering)."""
        return {}

    def residuals(self, x, ev) -> list[float]:
        return []

    def outlets(self, x, ev) -> dict[str, State]:
        return {}

    def shaft_power(self, x, ev) -> float:
        """Power delivered to the shaft (turbines positive, pumps negative)."""
        return 0.0

    def outputs(self, x, ev) -> dict[str, float]:
        return {}

    def update_slow(self, x, ev) -> None:
        pass

    def guess(self, g) -> list[float]:
        return [g.flow(self.id)] * len(self.unknowns)


# ======================================================================================================== through-flow
class Through(Model):
    """Two fluid ports, one through-flow ``m`` from ``inlet`` to ``outlet``."""

    unknowns = ("m",)
    signs = (1,)
    pin, pout = "inlet", "outlet"

    def flows(self, x):
        return {self.pin: -x[0], self.pout: x[0]}

    def dp(self, m: float, ev) -> float:
        return 0.0

    def residuals(self, x, ev):
        return [(ev.p(self.id, self.pin) - ev.p(self.id, self.pout) - self.dp(x[0], ev)) / ev.p_ref]

    def outlets(self, x, ev):
        s = ev.inlet(self.id, self.pin)
        return {self.pout: s}

    def rho_in(self, ev) -> float:
        return density(ev.inlet(self.id, self.pin), ev.p(self.id, self.pin))

    def outputs(self, x, ev):
        return dict(mdot=x[0], dp=ev.p(self.id, self.pin) - ev.p(self.id, self.pout))


class Line(Through):
    def update_slow(self, x, ev):
        s = ev.inlet(self.id, self.pin)
        p = ev.p(self.id, self.pin)
        D = self.par("diameter")
        mu = 1e-5 if s.kind == "gas" else props(s.fluid).at(p, s.h, s.T)["mu"]
        Re = max(abs(x[0]) * D / (area(D) * mu), 2300.0)
        eps = (self.par("roughness", 1.5e-5) or 1.5e-5) / D
        self.slow["f"] = (-1.8 * np.log10(6.9 / Re + (eps / 3.7) ** 1.11)) ** -2

    def dp(self, m, ev):
        D, L = self.par("diameter"), self.par("length")
        K = (self.par("K_fittings", 0.0) or 0.0) + self.slow.get("f", 0.02) * L / D
        return K * sq(m) / (2.0 * self.rho_in(ev) * area(D) ** 2)

    def outputs(self, x, ev):
        o = super().outputs(x, ev)
        o["velocity"] = x[0] / (self.rho_in(ev) * area(self.par("diameter")))
        return o


class Valve(Through):
    def position(self) -> float:
        v = self.net._free_val.get((self.id, "position"))
        return float(v) if v is not None else float(self.opt.get("position", 1.0))

    def dp(self, m, ev):
        K = self.par("K_open") / max(self.position(), 1e-3) ** 2
        return K * sq(m) / (2.0 * self.rho_in(ev) * area(self.par("diameter")) ** 2)

    def outputs(self, x, ev):
        o = super().outputs(x, ev)
        o["position"] = self.position()
        return o


class CheckValve(Through):
    def dp(self, m, ev):
        return (self.par("K_open") * sq(m) / (2.0 * self.rho_in(ev) * area(self.par("diameter")) ** 2)
                + (self.par("dp_crack", 0.0) or 0.0) * np.tanh(m / max(1e-3 * self.net.m_ref, 1e-6)))


class Orifice(Through):
    def cda(self) -> float:
        return self.par("Cd") * area(self.par("diameter"))

    def dp(self, m, ev):
        return sq(m) / (2.0 * self.rho_in(ev) * self.cda() ** 2)


class Injector(Orifice):
    pin, pout = "inlet", "outlet"

    def cda(self) -> float:
        return self.par("Cd") * self.par("A_total")

    def outputs(self, x, ev):
        o = super().outputs(x, ev)
        p_down = ev.p(self.id, self.pout)
        o["stiffness"] = o["dp"] / max(p_down, 1.0)
        o["velocity"] = x[0] / (self.rho_in(ev) * self.cda())
        return o


class Manifold(Through):
    pass


class Venturi(Through):
    """Cavitating venturi: the flow is fixed by the inlet and vapour pressures while the throat cavitates."""

    signs = (-1,)

    def residuals(self, x, ev):
        s = ev.inlet(self.id, self.pin)
        p1, p2 = ev.p(self.id, self.pin), ev.p(self.id, self.pout)
        pv = 0.0 if s.kind == "gas" else props(s.fluid).p_vap(s.T)
        p_th = max(pv, p2)                         # unchoked when the outlet is above the vapour pressure
        cda = self.par("Cd") * area(self.par("throat_diameter"))
        m_cav = cda * np.sqrt(2.0 * self.rho_in(ev) * max(p1 - p_th, 0.0))
        return [(x[0] - m_cav) / ev.m_ref]


class Inducer(Through):
    """Inducer: suction-performance check only (its head is small and not modelled); no pressure change."""


class Pump(Through):
    shaft_role = "pump"
    signs = (-1,)

    def design(self, rho: float) -> tuple[float, float]:
        U = self.par("N_design") * self.par("D_tip") / 2.0
        return rho * self.par("psi_design") * U * U, self.par("Q_design")

    def rise(self, m, ev):
        rho = self.rho_in(ev)
        dp_d, Q_d = self.design(rho)
        s = float(self.opt.get("shutoff_ratio", 1.25))
        N = ev.speed(self.id) / self.par("N_design")
        Q = m / rho
        return dp_d * (s * N * abs(N) - (s - 1.0) * (Q / Q_d) * abs(Q / Q_d))

    def residuals(self, x, ev):
        return [(ev.p(self.id, self.pout) - ev.p(self.id, self.pin) - self.rise(x[0], ev)) / ev.p_ref]

    def eta(self) -> float:
        return self.par("eta_design")

    def power(self, x, ev) -> float:
        dp = ev.p(self.id, self.pout) - ev.p(self.id, self.pin)
        return x[0] * dp / (self.rho_in(ev) * self.eta())

    def shaft_power(self, x, ev):
        return -self.power(x, ev)

    def outlets(self, x, ev):
        s = ev.inlet(self.id, self.pin)
        if s.kind == "gas":
            return {self.pout: s}
        dh = self.power(x, ev) / max(abs(x[0]), 1e-9)
        p2 = ev.p(self.id, self.pout)
        h = s.h + dh
        return {self.pout: s.with_(h=h, T=props(s.fluid).T_ph(p2, h, s.T))}

    def outputs(self, x, ev):
        s = ev.inlet(self.id, self.pin)
        p1 = ev.p(self.id, self.pin)
        rho = self.rho_in(ev)
        dp = ev.p(self.id, self.pout) - p1
        pv = 0.0 if s.kind == "gas" else props(s.fluid).p_vap(s.T)
        N = ev.speed(self.id)
        Q = x[0] / rho
        H = dp / (rho * G0)
        return dict(mdot=x[0], dp=dp, head=H, power=self.power(x, ev), efficiency=self.eta(), speed=N,
                    npsh_available=(p1 - pv) / (rho * G0), flow_ratio=Q / self.par("Q_design"),
                    specific_speed=N * np.sqrt(max(Q, 0.0)) / max(G0 * H, 1e-9) ** 0.75)


class Turbine(Through):
    shaft_role = "turbine"

    @property
    def signs(self):
        return (-1,) if self.model() == "choked" else (1,)

    def model(self) -> str:
        m = self.opt.get("flow_model", "auto")
        if m == "auto":
            return "choked" if self.par("A_nozzle") else "pressure_ratio"
        return m

    def residuals(self, x, ev):
        p1, p2 = ev.p(self.id, self.pin), ev.p(self.id, self.pout)
        if self.model() == "choked":
            s = ev.inlet(self.id, self.pin)
            if s.kind == "gas":
                cp, R, T, g = s.cp, s.R, s.T, s.gamma
            else:
                a = props(s.fluid).at(p1, s.h, s.T)
                g, T = 1.4, a["T"]
                R = p1 / (a["rho"] * T)
            m_n = self.par("A_nozzle") * p1 * Gamma(g) / np.sqrt(R * T)
            return [(x[0] - m_n) / ev.m_ref]
        return [(p1 - self.par("pressure_ratio") * p2) / ev.p_ref]

    def eta(self) -> float:
        return self.par("eta_design")

    def specific_work(self, ev) -> tuple[float, float]:
        """Isentropic and actual specific work (J/kg) from inlet to outlet pressure."""
        s = ev.inlet(self.id, self.pin)
        p1, p2 = ev.p(self.id, self.pin), ev.p(self.id, self.pout)
        pr = max(min(p2 / max(p1, 1.0), 1.0), 1e-6)
        if s.kind == "gas":
            w_s = s.cp * s.T * (1.0 - pr ** ((s.gamma - 1.0) / s.gamma))
        else:
            pp = props(s.fluid)
            a = pp.at(p1, s.h, s.T)
            if a["phase"] == "liquid" or pp.incompressible:
                w_s = max(p1 - p2, 0.0) / a["rho"]
            else:
                try:
                    w_s = s.h - pp.h_ps(p2, a["s"])
                except ValueError:
                    w_s = max(p1 - p2, 0.0) / a["rho"]
        return w_s, self.eta() * w_s

    def shaft_power(self, x, ev):
        return x[0] * self.specific_work(ev)[1]

    def outlets(self, x, ev):
        s = ev.inlet(self.id, self.pin)
        w = self.specific_work(ev)[1]
        p2 = ev.p(self.id, self.pout)
        if s.kind == "gas":
            T = s.T - w / s.cp
            return {self.pout: s.with_(T=T, h=s.cp * T)}
        h = s.h - w
        return {self.pout: s.with_(h=h, T=props(s.fluid).T_ph(p2, h, s.T))}

    def outputs(self, x, ev):
        s = ev.inlet(self.id, self.pin)
        p1, p2 = ev.p(self.id, self.pin), ev.p(self.id, self.pout)
        w_s, w = self.specific_work(ev)
        N = ev.speed(self.id)
        P = x[0] * w
        o = dict(mdot=x[0], power=P, efficiency=self.eta(), pressure_ratio=p1 / max(p2, 1.0), speed=N,
                 torque=P / max(N, 1e-9), T_inlet=s.T)
        if self.par("D_mean"):
            o["u_over_c0"] = N * self.par("D_mean") / 2.0 / np.sqrt(2.0 * max(w_s, 1e-9))
        return o


class Jacket(Through):
    """Regenerative cooling jacket on the wall of the chamber it is joined to by a thermal edge."""

    pin, pout = "coolant_in", "coolant_out"

    def dp(self, m, ev):
        if self.opt.get("hydraulics", "friction") == "none":         # loss carried by a separate orifice
            return 0.0
        return self.slow.get("K", 0.0) * sq(m) / self.rho_in(ev)

    def outlets(self, x, ev):
        s = ev.inlet(self.id, self.pin)
        Q = self.slow.get("Q", 0.0)
        p2 = ev.p(self.id, self.pout)
        m = max(abs(x[0]), 1e-9)
        if s.kind == "gas":
            T = s.T + Q / (m * s.cp)
            return {self.pout: s.with_(T=T, h=s.cp * T)}
        h = s.h + Q / m
        return {self.pout: s.with_(h=h, T=props(s.fluid).T_ph(p2, h, s.T))}

    def update_slow(self, x, ev):
        from .jacket import march
        ch = self.net.wall_source(self.id)
        if ch is None:
            self.slow.update(Q=0.0, K=0.0)
            return
        gas = ev.model(ch).gas(ev)
        if gas is None:
            return
        res = march(self, ev, gas, abs(x[0]))
        self.slow.update(res)

    def outputs(self, x, ev):
        o = super().outputs(x, ev)
        for k in ("Q", "T_wg_max", "q_max", "T_wc_max", "section"):
            if k in self.slow:
                o[k] = self.slow[k]
        s_in, s_out = ev.inlet(self.id, self.pin), ev.outlet_state(self.id, self.pout)
        if s_out is not None:
            o["T_out"] = s_out.T
            o["dT"] = s_out.T - s_in.T
        return o


# ======================================================================================================== sources/sinks
class Tank(Model):
    unknowns = ("m",)
    signs = (-1,)

    def fluid(self) -> str:
        return self.net.port_fluid(self.id, "outlet")

    def p_tank(self) -> float:
        return self.par("p_set") if self.type == "tank_regulated" else self.par("p0")

    def flows(self, x):
        return {"outlet": x[0]}

    def residuals(self, x, ev):
        return [(ev.p(self.id, "outlet") - self.p_tank()) / ev.p_ref]

    def outlets(self, x, ev):
        T = self.par("T_prop") if self.type == "tank_regulated" else None
        fid = self.fluid()
        if T is None:
            T = float(self.opt.get("T_prop", 293.15))
        return {"outlet": liquid_state(fid, self.p_tank(), T, self.net.ox_fraction(fid))}

    def outputs(self, x, ev):
        return dict(mdot=x[0], p=self.p_tank())


class Chamber(Model):
    unknowns = ("m_ox", "m_fuel")
    signs = (1, 1)

    def flows(self, x):
        return {"ox_in": -x[0], "fuel_in": -x[1], "throat": x[0] + x[1]}

    def lut_key(self) -> str:
        key = PAIR_KEYS.get((self.net.ox, self.net.fuel))
        if key is None:
            raise ValueError(f"no CEA table for {self.net.ox}/{self.net.fuel}: tables cover "
                             + ", ".join(sorted(set(PAIR_KEYS.values()))))
        return key

    def lut(self):
        from ..thermo.cea import chamber_lut
        return chamber_lut(self.lut_key())

    def split(self, x, ev) -> tuple[float, float]:
        """Oxidizer-origin and fuel-origin mass flow into the chamber (preburner gas counted by its origin)."""
        so, sf = ev.inlet(self.id, "ox_in"), ev.inlet(self.id, "fuel_in")
        mo = x[0] * so.ox_frac + x[1] * sf.ox_frac
        mf = x[0] * (1 - so.ox_frac) + x[1] * (1 - sf.ox_frac)
        return mo, mf

    def film(self) -> float:
        return (self.par("film_fraction", 0.0) or 0.0) if self.opt.get("cooling") == "film" else 0.0

    def gas(self, ev, x=None) -> dict | None:
        x = ev.local(self.id) if x is None else x
        mo, mf = self.split(x, ev)
        f = self.film()
        mf_core = mf * (1.0 - f)
        mr = mo / max(mf_core, 1e-12)
        pc = ev.p(self.id, "throat")
        cs, Tc, g, Mw = self.lut()(mr, max(pc, 2e5))
        return dict(mr=mr, mr_overall=mo / max(mf, 1e-12), cstar=cs * self.par("eta_cstar"), cstar_ideal=cs,
                    Tc=Tc, gamma=g, Mw=Mw, p0=pc, At=self.par("A_t"), m_core=mo + mf_core, m_film=mf * f)

    def injector_end_ratio(self, gamma: float) -> float:
        """Injector-end static over nozzle-stagnation pressure for heat release in the constant-area chamber
        (Rayleigh line from M = 0 to the Mach number of the contraction ratio)."""
        cr = self.par("contraction_ratio", None)
        if not cr or self.opt.get("rayleigh_loss", "on") == "off":
            return 1.0
        from ..nozzle import mach_from_area
        M = float(mach_from_area(cr, gamma, False))
        return (1.0 + gamma * M * M) / (1.0 + 0.5 * (gamma - 1.0) * M * M) ** (gamma / (gamma - 1.0))

    def residuals(self, x, ev):
        g = self.gas(ev, x)
        p_inj = g["p0"] * self.injector_end_ratio(g["gamma"])
        return [(ev.p(self.id, "ox_in") - p_inj) / ev.p_ref, (ev.p(self.id, "fuel_in") - p_inj) / ev.p_ref]

    def throat_flow(self, ev) -> float:
        """Choked throat flow :math:`p_cA_t/c^*`; film coolant passes the throat without adding to c*."""
        g = self.gas(ev)
        cs_eff = g["cstar"] * g["m_core"] / max(g["m_core"] + g["m_film"], 1e-12)
        return g["p0"] * g["At"] / cs_eff

    def outlets(self, x, ev):
        g = self.gas(ev, x)
        R = 8.314462618 / g["Mw"]
        cp = g["gamma"] * R / (g["gamma"] - 1.0)
        return {"throat": gas_state(g["Tc"], cp, g["gamma"], g["mr_overall"] / (1 + g["mr_overall"]))}

    def outputs(self, x, ev):
        g = self.gas(ev, x)
        return dict(p_c=g["p0"], p_injector_end=g["p0"] * self.injector_end_ratio(g["gamma"]), mixture_ratio=g["mr"], mixture_ratio_overall=g["mr_overall"], T_c=g["Tc"],
                    cstar=g["cstar"], gamma=g["gamma"], mdot=x[0] + x[1], film_flow=g["m_film"],
                    cooling=self.opt.get("cooling", "regenerative"))

    def guess(self, g):
        return [g.flow(self.id, "ox"), g.flow(self.id, "fuel")]


class Nozzle(Model):
    unknowns = ("m",)
    signs = (1,)

    def flows(self, x):
        return {"inlet": -x[0]}

    def residuals(self, x, ev):
        ch = ev.model(self.net.upstream_chamber(self.id))
        return [(ch.throat_flow(ev) - x[0]) / ev.m_ref]

    def performance(self, x, ev) -> dict:
        """Thrust from the CEA vacuum thrust coefficient of the shipped tables (shifting equilibrium at the
        nozzle's area ratio) less the ambient term, times the nozzle efficiency ``eta_CF``."""
        up = self.net.upstream_chamber(self.id)
        ch = ev.model(up)
        g = ch.gas(ev)
        pc = g["p0"]
        eps = self.par("expansion_ratio")
        pa = self.net.ambient(self.id)
        try:
            from ..thermo.cea import Thermo
            pt = Thermo(ch.lut_key(), "tables").point(g["mr"], max(pc, 2e5), eps, pa)
            cf_vac = pt["isp_vac"] * G0 / pt["cstar"]
            pe_pc = float(exit_state(eps, g["gamma"])[1])
        except (ValueError, KeyError):
            pe_pc = float(exit_state(eps, g["gamma"])[1])
            cf_vac = float(CF(g["gamma"], pe_pc, 0.0, eps))
        cf = (cf_vac - pa / pc * eps) * self.par("eta_CF")
        F = cf * pc * g["At"]
        m_tot = max(x[0], 1e-9)
        return dict(thrust=F, isp=F / (m_tot * G0), CF=cf, p_exit=pe_pc * pc, p_ambient=pa, mdot=x[0],
                    cstar=g["cstar"], isp_vac=(cf_vac * self.par("eta_CF")) * pc * g["At"] / (m_tot * G0))

    def outputs(self, x, ev):
        o = self.performance(x, ev)
        o.update(self.wall(ev))
        return o

    def wall(self, ev) -> dict:
        """Wall state of an uncooled extension: radiation-equilibrium temperature (radiative), or the adiabatic
        wall temperature and cold-wall heat flux an ablative or heat-sink liner must take, from the split area
        ratio to the exit.  Regenerative walls report through their jackets."""
        cooling = self.opt.get("cooling", "regenerative")
        if cooling == "regenerative":
            return {}
        from ..heat import adiabatic_wall_T, bartz_hg, gas_transport, sigma
        from ..nozzle import mach_from_area
        ch = ev.model(self.net.upstream_chamber(self.id))
        g = ch.gas(ev)
        Tc, gam = g["Tc"], g["gamma"]
        cpg, mug, Pr = gas_transport(Tc, g["Mw"], gam)
        Dt = float(np.sqrt(4.0 * g["At"] / np.pi))
        eps = self.par("expansion_ratio")
        a0 = min(float(self.opt.get("split_area_ratio", 5.0)), eps)
        ar = np.geomspace(max(a0, 1.0001), eps, 12)
        em = float(self.opt.get("emissivity", 0.8))
        SB = 5.670374419e-8
        T_max, q_max, Taw_max = 0.0, 0.0, 0.0
        for a in ar:
            M = float(mach_from_area(a, gam, True))
            Taw = float(adiabatic_wall_T(Tc, gam, M, Pr))
            if cooling == "radiative":
                Tw = 0.5 * Taw
                for _ in range(30):                        # h_g (T_aw - T_w) = eps sigma T_w^4
                    h = float(bartz_hg(Dt, mug, cpg, Pr, g["p0"], g["cstar"], 0.382 * Dt, a,
                                       float(sigma(Tw / Tc, gam, M))))
                    f = h * (Taw - Tw) - em * SB * Tw**4
                    Tw = float(np.clip(Tw + f / (h + 4 * em * SB * Tw**3), 200.0, Taw))
                T_max = max(T_max, Tw)
            else:
                h = float(bartz_hg(Dt, mug, cpg, Pr, g["p0"], g["cstar"], 0.382 * Dt, a,
                                   float(sigma(300.0 / Tc, gam, M))))
                q_max = max(q_max, h * (Taw - 300.0))
            Taw_max = max(Taw_max, Taw)
        o = dict(cooling=cooling, T_aw_extension=Taw_max)
        if cooling == "radiative":
            o["T_wall_radiative"] = T_max
        else:
            o["q_cold_wall"] = q_max
        return o

    def guess(self, g):
        return [g.flow(self.id, "total")]


class GasGenerator(Model):
    """Gas generator or preburner: oxidizer and fuel in, drive gas out at the outlet-station pressure."""

    unknowns = ("m_ox", "m_fuel")
    signs = (1, 1)

    def flows(self, x):
        return {"ox_in": -x[0], "fuel_in": -x[1], "outlet": x[0] + x[1]}

    def inj(self) -> float:
        return float(self.opt.get("injector_dp_fraction", 0.0))

    def residuals(self, x, ev):
        p = ev.p(self.id, "outlet")
        k = 1.0 + self.inj()
        return [(ev.p(self.id, "ox_in") - k * p) / ev.p_ref, (ev.p(self.id, "fuel_in") - k * p) / ev.p_ref]

    def mr(self, x, ev) -> float:
        so, sf = ev.inlet(self.id, "ox_in"), ev.inlet(self.id, "fuel_in")
        mo = x[0] * so.ox_frac + x[1] * sf.ox_frac
        mf = x[0] * (1 - so.ox_frac) + x[1] * (1 - sf.ox_frac)
        return mo / max(mf, 1e-12)

    def curve(self, ev):
        """Drive-gas properties tabulated over O/F once per outer iteration (smooth for the Newton solve)."""
        if "curve" not in self.slow:
            so, sf = ev.inlet(self.id, "ox_in"), ev.inlet(self.id, "fuel_in")
            p = ev.p(self.id, "outlet")
            user = (self.par("T_out"), self.par("cp_gas"), self.par("gamma_gas"))
            model = self.opt.get("gas_model", "auto")
            grid = np.concatenate([np.linspace(0.05, 2.0, 40), np.linspace(2.2, 80.0, 40)])
            rows = []
            for mr in grid:
                try:
                    d = drive_gas(self.net.ox, self.net.fuel, float(mr), so.T, sf.T, p, model, user)
                    rows.append((mr, d["T"], d["cp"], d["gamma"], d["model"]))
                except ValueError:
                    continue
            if not rows:
                raise ValueError(f"{self.id}: no drive-gas model applies; choose the user gas model and set "
                                 "T_out, cp_gas and gamma_gas")
            a = np.array([r[:4] for r in rows])
            self.slow["curve"] = (a, rows[0][4], {round(r[0], 6): r[4] for r in rows})
        return self.slow["curve"]

    def update_slow(self, x, ev):
        self.slow.pop("curve", None)

    def gas(self, x, ev) -> dict:
        a, _, names = self.curve(ev)
        mr = self.mr(x, ev)
        f = lambda j: float(np.interp(mr, a[:, 0], a[:, j]))
        i = int(np.argmin(np.abs(a[:, 0] - mr)))
        out_of_range = not (a[0, 0] <= mr <= a[-1, 0])
        return dict(mr=mr, T=f(1), cp=f(2), gamma=f(3), model=names[round(a[i, 0], 6)], clamped=out_of_range)

    def outlets(self, x, ev):
        g = self.gas(x, ev)
        return {"outlet": gas_state(g["T"], g["cp"], g["gamma"], g["mr"] / (1.0 + g["mr"]))}

    def outputs(self, x, ev):
        g = self.gas(x, ev)
        return dict(p_gg=ev.p(self.id, "outlet"), mixture_ratio=g["mr"], T_out=g["T"], cp=g["cp"],
                    gamma=g["gamma"], mdot=x[0] + x[1], gas_model=g["model"])

    def guess(self, g):
        return [g.flow(self.id, "ox"), g.flow(self.id, "fuel")]


class Shaft(Model):
    """Shaft component: joins machines; adds its auxiliary (gearbox, accessory) power to the shaft balance."""

    def shaft_power(self, x, ev):
        return -(self.par("aux_power", 0.0) or 0.0)


MODELS = {"line": Line, "valve_ball": Valve, "valve_butterfly": Valve, "valve_poppet": Valve, "check_valve": CheckValve,
          "orifice": Orifice, "injector": Injector, "manifold": Manifold, "cavitating_venturi": Venturi,
          "inducer": Inducer, "pump_centrifugal": Pump, "turbine_impulse": Turbine, "regen_jacket": Jacket,
          "tank_regulated": Tank, "tank_blowdown": Tank, "chamber": Chamber, "nozzle_bell": Nozzle,
          "gas_generator": GasGenerator, "shaft": Shaft}
