"""Electric module: mission & power trades, electrothermal, gridded ion and Hall thrusters (Live); ion optics, Hall
discharge and low-thrust spiral (Run)."""
from __future__ import annotations

import numpy as np
import pyvista as pv

from ..core import ep
from ..core.constants import G0
from ..core.ep import hall as hl
from ..core.nozzle import T_ratio, contour, mach_along_contour, p_ratio
from ..core.result import Curve, PlotSpec, TabSpec
from ..core.schema import C, P, ParamModel
from ..sims.ep_run import (PROP_CHOICES, PROP_LABELS, HallDischarge, HallRunParams, IonOptics, IonOpticsParams,
                           LowThrustSpiral, SpiralParams)
from ..ui.theme import CMAPS
from .base import ModelController
from .geometry import broadcast_scalar, rev_grid, thetas

DAY = 86400.0
TT_DAYS = r"$t$ (days)"


def polylines(P3: np.ndarray) -> pv.PolyData:
    """``(n_lines, n_points, 3)`` array -> one PolyData with ``n_lines`` polylines."""
    n, k, _ = P3.shape
    poly = pv.PolyData(P3.reshape(-1, 3))
    poly.lines = np.hstack([np.full((n, 1), k), np.arange(n * k).reshape(n, k)]).ravel()
    return poly


def _cut_ok(phi: np.ndarray, cut: float) -> np.ndarray:
    return phi <= np.radians(360.0 - cut) if cut > 0 else np.ones_like(phi, bool)


# ====================================================================================== mission & power trades (Live)
class TradesParams(ParamModel):
    dv: float = P(5.9, symbol=r"\Delta v", desc="mission velocity increment", units="km·s⁻¹", dims="L T⁻¹", lo=0.5,
                  hi=20.0, step=0.05, group="Mission")
    t_b: float = P(180.0, symbol=r"t_b", desc="thrusting time", units="days", dims="T", lo=10.0, hi=3000.0, step=5.0,
                   group="Mission")
    isp: float = P(1800.0, symbol=r"I_{sp}", desc="specific impulse", units="s", dims="T", lo=200.0, hi=10000.0,
                   step=10.0, group="Thruster")
    eta: float = P(0.6, symbol=r"\eta_T", desc="thruster total efficiency", units="", dims="1", lo=0.1, hi=0.85,
                   step=0.01, group="Thruster")
    alpha: float = P(20.0, symbol=r"\alpha", desc="specific mass of the power system (array, PPU, radiators)",
                     units="kg·kW⁻¹", dims="T³ L⁻²", lo=2.0, hi=200.0, step=0.5, group="Power system")
    m0: float = P(2000.0, symbol=r"m_0", desc="initial spacecraft mass", units="kg", dims="M", lo=10.0, hi=50000.0,
                  step=10.0, group="Spacecraft")


class TradesLiveModel(ModelController):
    key = "ep.trades"
    title = "Mission & power trades"
    module = "Electric"
    mode = "live"
    Params = TradesParams
    fields = [r"$\frac{m}{m_0}$ (–)"]

    def __init__(self) -> None:
        self._m: dict = {}

    def evaluate(self, p: TradesParams) -> dict:
        dv = p.dv * 1e3
        tb = p.t_b * DAY
        a = p.alpha / 1e3
        vch = ep.v_char(p.eta, tb, a)
        ve = p.isp * G0
        x = float(np.exp(-dv / ve))
        f_pl = float(ep.stuhlinger_payload(dv, ve, vch))
        f_pr = 1.0 - x
        f_ps = (ve / vch) ** 2 * (1.0 - x)
        ve_opt, pl_opt = ep.optimum_ve(dv, vch)
        m_p = p.m0 * f_pr
        P_jet = 0.5 * (m_p / tb) * ve**2
        P_in = P_jet / p.eta
        F = float(ep.thrust_per_power(p.eta, ve)) * P_in
        isp_g = np.geomspace(200.0, 10000.0, 160)
        ve_g = isp_g * G0
        xg = np.exp(-dv / ve_g)
        return dict(vch=vch, ve=ve, f_pl=f_pl, f_pr=f_pr, f_ps=f_ps, isp_opt=ve_opt / G0, pl_opt=pl_opt, P_in=P_in, F=F,
                    m_p=m_p, isp_g=isp_g, pl_g=xg - (ve_g / vch) ** 2 * (1 - xg), pr_g=1 - xg,
                    ps_g=(ve_g / vch) ** 2 * (1 - xg), fp_g=2 * p.eta / ve_g * 1e6, mr_g=1 / xg,
                    dv_set=[0.5 * dv, dv, 2 * dv])

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        xi = r["isp_g"]
        xl = r"$I_{sp}$ (s)"
        frac = PlotSpec("Stuhlinger mass budget", xl, r"$\frac{m}{m_0}$ (–)",
                        [Curve(xi, np.clip(r["pl_g"], -0.2, 1), "thrust", "payload"),
                         Curve(xi, r["pr_g"], "fuel", "propellant"), Curve(xi, np.clip(r["ps_g"], 0, 1.2), "hot", "power system")],
                        markers=[(p.isp, r["f_pl"], "thrust"), (r["isp_opt"], r["pl_opt"], "text")])
        fam = []
        cols = ["dim", "thrust", "pressure"]
        for k, dv in enumerate(r["dv_set"]):
            ve_g = xi * G0
            xg = np.exp(-dv / ve_g)
            fam.append(Curve(xi, np.clip(xg - (ve_g / r["vch"]) ** 2 * (1 - xg), -0.2, 1), cols[k],
                             rf"$\Delta v$ = {dv / 1e3:.1f} (km·s⁻¹)"))
        payload = PlotSpec("Payload fraction vs velocity increment", xl, r"$\frac{m_{pl}}{m_0}$ (–)", fam,
                           markers=[(p.isp, r["f_pl"], "thrust")])
        fp = PlotSpec("Thrust per unit power", xl, r"$\frac{F}{P_{in}}$ (mN·kW⁻¹)", [Curve(xi, r["fp_g"], "pressure")],
                      markers=[(p.isp, 2 * p.eta / r["ve"] * 1e6, "pressure")])
        mr = PlotSpec("Rocket equation mass ratio", xl, r"$\frac{m_0}{m_f}$ (–)", [Curve(xi, np.minimum(r["mr_g"], 20), "fuel")],
                      markers=[(p.isp, float(np.exp(p.dv * 1e3 / r["ve"])), "fuel")])
        return [TabSpec("Mass budget", [frac, payload]), TabSpec("Thrust & power", [fp, mr])]

    def readout(self, p, r, frame) -> list[str]:
        return [rf"$v_{{ch}}$ = {r['vch'] / 1e3:.2f} (km·s⁻¹)", rf"$I_{{sp,opt}}$ = {r['isp_opt']:.0f} (s)",
                rf"$\frac{{m_{{pl}}}}{{m_0}}$ = {r['f_pl']:.3f}", rf"$P_{{in}}$ = {r['P_in'] / 1e3:.1f} (kW)"]

    def caption(self, p, r, frame) -> str:
        warn = " · payload fraction negative: mission infeasible at this $I_{sp}$" if r["f_pl"] < 0 else ""
        return (rf"propellant {r['m_p']:.0f} (kg) · $F$ = {r['F'] * 1e3:.0f} (mN) · optimum payload fraction "
                rf"{r['pl_opt']:.3f}{warn}")

    def export_data(self, p, r):
        return {"Isp_s": r["isp_g"], "payload_frac": r["pl_g"], "prop_frac": r["pr_g"], "power_frac": r["ps_g"],
                "F_per_P_mN_per_kW": r["fp_g"]}

    # 3D: stacked spacecraft blocks with heights set by the mass fractions and an array sized by the power
    def build_scene(self, vp, p, r=None) -> None:
        vp.plotter.clear()
        self._m = {"built": True}
        self.update_scene(vp, p, r or self.evaluate(p), None)
        vp.reset_view(zoom=1.0)

    def update_scene(self, vp, p, r, frame) -> None:
        if not self._m:
            return self.build_scene(vp, p, r)
        pl = vp.plotter
        fr = [max(r["f_pl"], 0.0), max(min(r["f_ps"], 1.0), 0.0), r["f_pr"]]
        cols = ["#5cc97a", "#ff9d3c", "#4ea1ff"]
        z = 0.0
        for k, (f, c) in enumerate(zip(fr, cols)):
            h = max(f, 1e-3) * 2.0
            pl.add_mesh(pv.Cylinder(center=(0, 0, z + h / 2), direction=(0, 0, 1), radius=0.5, height=h, resolution=48),
                        color=c, name=f"block{k}", smooth_shading=True)
            z += h
        span = 0.6 + 2.5 * np.sqrt(max(r["P_in"], 1.0) / 2e4)
        for s, nm in ((1, "wing_r"), (-1, "wing_l")):
            pl.add_mesh(pv.Box(bounds=(s * 0.55, s * (0.55 + span), -0.35, 0.35, z * 0.45 - 0.01, z * 0.45 + 0.01)),
                        color="#2b4a7a", name=nm)
        vp.set_colorbar("viridis", 0.0, 1.0, r"$\frac{m}{m_0}$ (–)")
        vp.render()


# ====================================================================================== electrothermal (Live)
GASES = {"H2": (14300.0, 1.40, 2.016), "He": (5193.0, 1.667, 4.003), "NH3": (3374.0, 1.38, 8.51), "N2": (1040.0, 1.40, 28.01)}
GAS_LABELS = ["Hydrogen", "Helium", "Ammonia (decomposed, N₂ + 3H₂)", "Nitrogen"]


class ThermalParams(ParamModel):
    gas: str = C("NH3", choices=list(GASES), labels=GAS_LABELS, symbol=r"\mathrm{propellant}",
                 desc="working gas (frozen, constant specific heat)", group="Propellant")
    power: float = P(1500.0, symbol=r"P", desc="electric input power", units="W", dims="M L² T⁻³", lo=10.0, hi=50000.0,
                 step=10.0, group="Heater")
    eta_h: float = P(0.8, symbol=r"\eta_h", desc="fraction of the power that heats the gas", units="", dims="1", lo=0.2,
                     hi=0.98, step=0.01, group="Heater")
    mdot: float = P(40.0, symbol=r"\dot m", desc="propellant mass flow", units="mg·s⁻¹", dims="M T⁻¹", lo=1.0,
                    hi=2000.0, step=1.0, group="Heater")
    T_lim: float = P(2500.0, symbol=r"T_{lim}", desc="heater / electrode material temperature limit", units="K",
                     dims="Θ", lo=1000.0, hi=4000.0, step=10.0, group="Heater")
    Pc: float = P(200.0, symbol=r"P_c", desc="chamber pressure (sets the throat size)", units="kPa", dims="M L⁻¹ T⁻²",
                  lo=20.0, hi=2000.0, step=5.0, group="Nozzle")
    eps: float = P(50.0, symbol=r"\varepsilon=\frac{A_e}{A_t}", desc="nozzle expansion ratio", units="", dims="1",
                   lo=5.0, hi=200.0, step=1.0, group="Nozzle")
    eta_n: float = P(0.9, symbol=r"\eta_n", desc="nozzle efficiency (frozen flow and viscous loss)", units="", dims="1",
                     lo=0.5, hi=1.0, step=0.01, group="Nozzle")


THERMAL_FIELDS = [("T", CMAPS["temperature"], r"$T$ (K)"), ("M", CMAPS["mach"], r"$M$ (–)")]


class ThermalLiveModel(ModelController):
    key = "ep.thermal"
    title = "Electrothermal thrusters"
    module = "Electric"
    mode = "live"
    Params = ThermalParams
    fields = [f[2] for f in THERMAL_FIELDS]

    def __init__(self) -> None:
        self._m: dict = {}
        self._built = None

    def evaluate(self, p: ThermalParams) -> dict:
        cp, g, _ = GASES[p.gas]
        md = p.mdot * 1e-6
        o = ep.electrothermal(p.power, p.eta_h, md, cp, 300.0, p.eta_n)
        R = cp * (g - 1) / g
        Gam = np.sqrt(g) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))
        cstar = np.sqrt(R * o["Tc"]) / Gam
        At = md * cstar / (p.Pc * 1e3)
        Rt = np.sqrt(At / np.pi)
        x, rr = contour(Rt, p.eps, Rc=2.5 * Rt, f=0.8, g=g)
        M = mach_along_contour(x, rr, Rt, g)
        sp = np.geomspace(2e6, 3e9, 120)                          # specific power P/mdot (J/kg)
        Tc_g = 300.0 + p.eta_h * sp / cp
        ve_g = np.sqrt(2 * p.eta_n * cp * Tc_g)
        T_lim_sp = (p.T_lim - 300.0) * cp / p.eta_h
        return dict(**o, cp=cp, g=g, cstar=cstar, Rt=Rt, x=x, r=rr, M=M, T=o["Tc"] * T_ratio(M, g),
                    p=p.Pc * 1e3 * p_ratio(M, g), sp=sp, Tc_g=Tc_g, isp_g=ve_g / G0, eta_g=ve_g**2 / (2 * sp),
                    sp_now=p.power / md, sp_lim=T_lim_sp, over=o["Tc"] > p.T_lim)

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        xs = r["sp"] / 1e6
        xl = r"$\frac{P}{\dot m}$ (MJ·kg⁻¹)"
        lim = r["sp_lim"] / 1e6
        tc = PlotSpec("Chamber temperature", xl, r"$T_c$ (K)",
                      [Curve(xs, r["Tc_g"], "hot"), Curve(xs, np.full_like(xs, p.T_lim), "text", "material limit", "dash")],
                      markers=[(r["sp_now"] / 1e6, r["Tc"], "hot")])
        isp = PlotSpec("Specific impulse", xl, r"$I_{sp}$ (s)", [Curve(xs, r["isp_g"], "thrust")],
                       markers=[(r["sp_now"] / 1e6, r["Isp"], "thrust"),
                                (lim, float(np.interp(lim, xs, r["isp_g"])), "text")])
        eta = PlotSpec("Thrust efficiency", xl, r"$\eta_T$ (–)", [Curve(xs, np.minimum(r["eta_g"], 1.0), "pressure")],
                       markers=[(r["sp_now"] / 1e6, r["eta_T"], "pressure")])
        all_g = []
        cols = ["thrust", "pressure", "fuel", "oxidizer"]
        for k, (gname, (cp, _, _)) in enumerate(GASES.items()):
            all_g.append(Curve(xs, np.sqrt(2 * p.eta_n * cp * (300.0 + p.eta_h * r["sp"] / cp)) / G0, cols[k],
                               GAS_LABELS[k].split(" (")[0].lower()))
        gas = PlotSpec("Specific impulse by propellant", xl, r"$I_{sp}$ (s)", all_g,
                       markers=[(r["sp_now"] / 1e6, r["Isp"], "text")])
        xmm = r["x"] * 1e3
        noz = PlotSpec("Nozzle temperature", r"$x$ (mm)", r"$T$ (K)", [Curve(xmm, r["T"], "hot")])
        return [TabSpec("Heating", [tc, isp, eta]), TabSpec("Propellants & nozzle", [gas, noz])]

    def readout(self, p, r, frame) -> list[str]:
        return [rf"$T_c$ = {r['Tc']:.0f} (K)", rf"$I_{{sp}}$ = {r['Isp']:.0f} (s)", rf"$F$ = {r['F'] * 1e3:.1f} (mN)",
                rf"$D_t$ = {2 * r['Rt'] * 1e3:.2f} (mm)"]

    def caption(self, p, r, frame) -> str:
        warn = (rf" · OVER the {p.T_lim:.0f} (K) material limit: lower the power or raise the flow" if r["over"] else "")
        return (rf"$\eta_T$ = {r['eta_T']:.2f} · $c^*$ = {r['cstar']:.0f} (m·s⁻¹) · frozen flow, no dissociation{warn}")

    def export_data(self, p, r):
        return {"P_over_mdot_Jkg": r["sp"], "Tc_K": r["Tc_g"], "Isp_s": r["isp_g"], "eta_T": r["eta_g"], "x_m": r["x"],
                "r_m": r["r"], "T_K": r["T"]}

    def _field(self, vp, r):
        key, cmap, title = THERMAL_FIELDS[vp.field_index]
        return r[key], cmap, title

    def build_scene(self, vp, p, r=None) -> None:
        r = r or self.evaluate(p)
        pl = vp.plotter
        pl.clear()
        th = thetas(vp.cutaway)
        v, cmap, _ = self._field(vp, r)
        xc = np.linspace(r["x"][0] - 4 * r["Rt"], r["x"][0], 8, endpoint=False)
        x = np.concatenate([xc, r["x"]])
        grid = rev_grid(x, np.concatenate([np.full(8, r["r"][0]), r["r"]]), th)
        grid["field"] = broadcast_scalar(np.concatenate([np.full(8, v[0]), v]), len(th))
        act = pl.add_mesh(grid, scalars="field", cmap=cmap, clim=(float(v.min()), float(v.max()) + 1e-9),
                          show_scalar_bar=False, name="thruster", smooth_shading=True)
        coil = rev_grid(xc, np.full(8, 0.7 * r["r"][0]), th)          # resistive heater element / arc column
        pl.add_mesh(coil, color="#ffb347", opacity=0.6, name="heater")
        self._m = dict(actor=act)
        self._built = (vp.cutaway, vp.field_index, round(float(r["Rt"]), 9), p.eps)
        vp.reset_view(zoom=1.1)

    def update_scene(self, vp, p, r, frame) -> None:
        if not self._m or self._built != (vp.cutaway, vp.field_index, round(float(r["Rt"]), 9), p.eps):
            self.build_scene(vp, p, r)
        v, cmap, title = self._field(vp, r)
        vp.set_colorbar(cmap, float(v.min()), float(v.max()), title)
        vp.render()


# ====================================================================================== gridded ion thruster (Live)
class IonLiveParams(ParamModel):
    prop: str = C("Xe", choices=PROP_CHOICES, labels=PROP_LABELS, symbol=r"\mathrm{propellant}",
                  desc="ion species (singly charged)", group="Propellant")
    Vb: float = P(1100.0, symbol=r"V_b", desc="net beam voltage", units="V", dims="M L² T⁻³ I⁻¹", lo=200.0, hi=5000.0,
                  step=10.0, group="Grid potentials")
    R: float = P(0.86, symbol=r"R=\frac{V_b}{V_T}", desc="net-to-total accelerating voltage ratio", units="", dims="1",
                 lo=0.5, hi=0.95, step=0.01, group="Grid potentials")
    l_g: float = P(0.66, symbol=r"l_g", desc="screen to accelerator gap", units="mm", dims="L", lo=0.3, hi=2.0,
                   step=0.01, group="Grid geometry")
    t_s: float = P(0.38, symbol=r"t_s", desc="screen-grid thickness", units="mm", dims="L", lo=0.2, hi=1.0, step=0.01,
                   group="Grid geometry")
    r_s: float = P(0.95, symbol=r"r_s", desc="screen aperture radius", units="mm", dims="L", lo=0.4, hi=1.5, step=0.01,
                   group="Grid geometry")
    pitch: float = P(2.5, symbol=r"l_{cc}", desc="hole pitch (centre to centre)", units="mm", dims="L", lo=1.0, hi=5.0,
                     step=0.05, group="Grid geometry")
    D: float = P(30.0, symbol=r"D_g", desc="active grid diameter", units="cm", dims="L", lo=5.0, hi=60.0, step=0.5,
                 group="Grid geometry")
    frac: float = P(0.5, symbol=r"\frac{I_b}{I_{max}}", desc="beamlet current as a fraction of the perveance limit",
                    units="", dims="1", lo=0.05, hi=1.0, step=0.01, group="Operating point")
    eta_m: float = P(0.9, symbol=r"\eta_m", desc="propellant utilization", units="", dims="1", lo=0.5, hi=0.98,
                     step=0.01, group="Operating point")
    eps_B: float = P(200.0, symbol=r"\varepsilon_B", desc="discharge loss (discharge power per beam ampere)",
                     units="W·A⁻¹", dims="M L² T⁻³ I⁻¹", lo=80.0, hi=600.0, step=5.0, group="Operating point")
    theta: float = P(10.0, symbol=r"\bar\theta", desc="thrust-weighted beam divergence half angle", units="deg",
                     dims="1", lo=0.0, hi=30.0, step=0.5, group="Operating point")
    f2: float = P(5.0, symbol=r"\frac{I^{++}}{I^+}", desc="doubly charged ion current ratio", units="%", dims="1",
                  lo=0.0, hi=40.0, step=0.5, group="Operating point")


SIGMA_CEX = 4.0e-19                       # charge-exchange cross-section (xenon, ~1 keV), m²


def ion_perf(p: IonLiveParams, Vb=None, eta_m=None) -> dict:
    Vb = p.Vb if Vb is None else Vb
    eta_m = p.eta_m if eta_m is None else eta_m
    M = ep.PROPELLANTS[p.prop] * ep.AMU
    Vt = Vb / p.R
    le = np.hypot((p.l_g + p.t_s) * 1e-3, p.r_s * 1e-3)
    j = ep.child_langmuir_j(Vt, le, M)
    I_let = p.frac * j * np.pi * (p.r_s * 1e-3) ** 2
    n_holes = (np.pi * (p.D * 1e-2) ** 2 / 4) / (np.sqrt(3) / 2 * (p.pitch * 1e-3) ** 2)
    Ib = I_let * n_holes
    alpha = ep.doubles_factor(p.f2 / 100)
    F = ep.ion_thrust(Ib, Vb, M) * np.cos(np.radians(p.theta)) * alpha
    mdot = Ib * M / (ep.E * eta_m)
    P_in = Ib * Vb + p.eps_B * Ib + 0.03 * Ib * Vb                 # beam + discharge + supplies / neutralizer
    return dict(j=j, I_let=I_let, n=n_holes, Ib=Ib, F=F, mdot=mdot, P=P_in, Isp=F / (mdot * G0),
                eta=F**2 / (2 * mdot * P_in), Vt=Vt, le=le, M=M)


class IonLiveModel(ModelController):
    key = "ep.ion"
    title = "Gridded ion thruster"
    module = "Electric"
    mode = "live"
    Params = IonLiveParams
    fields = [r"$j$ (A·m⁻²)"]

    def __init__(self) -> None:
        self._m: dict = {}

    def evaluate(self, p: IonLiveParams) -> dict:
        o = ion_perf(p)
        Vg = np.linspace(200.0, 5000.0, 60)
        sw = [ion_perf(p, Vb=v) for v in Vg]
        props = {k: [ep.isp_from_voltage(Vg, ep.PROPELLANTS[k] * ep.AMU) * np.sqrt(0.9) for _ in (0,)][0]
                 for k in PROP_CHOICES}
        # charge exchange (erosion driver): unionized neutrals leave through the accel open area
        etas = np.linspace(0.5, 0.98, 50)
        T_n, f_open = 500.0, 0.24
        vth = np.sqrt(8 * ep.KB * T_n / (np.pi * o["M"]))
        A_g = np.pi * (p.D * 1e-2) ** 2 / 4
        n0 = (o["Ib"] / ep.E) * (1 - etas) / etas / (0.25 * vth * f_open * A_g)
        L_cex = 4.0 * p.r_s * 1e-3
        cex = n0 * SIGMA_CEX * L_cex * 100.0                       # I_cex / I_b (%)
        n0_now = (o["Ib"] / ep.E) * (1 - p.eta_m) / p.eta_m / (0.25 * vth * f_open * A_g)
        Vt_g = np.linspace(200.0, 6000.0, 60)
        return dict(**o, Vg=Vg, F_g=np.array([s["F"] for s in sw]), isp_g=np.array([s["Isp"] for s in sw]),
                    eta_g=np.array([s["eta"] for s in sw]), P_g=np.array([s["P"] for s in sw]), props=props, etas=etas,
                    cex=cex, cex_now=n0_now * SIGMA_CEX * L_cex * 100.0, Vt_g=Vt_g,
                    j_g=ep.child_langmuir_j(Vt_g, o["le"], o["M"]))

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        cl = PlotSpec("Child–Langmuir current density", r"$V_T$ (V)", r"$j_{CL}$ (A·m⁻²)",
                      [Curve(r["Vt_g"], r["j_g"], "pressure"), Curve(r["Vt_g"], p.frac * r["j_g"], "dim", "operating", "dash")],
                      markers=[(r["Vt"], p.frac * r["j"], "pressure")])
        cex = PlotSpec("Charge-exchange current (erosion trend)", r"$\eta_m$ (–)", r"$\frac{I_{cex}}{I_b}$ (%)",
                       [Curve(r["etas"], r["cex"], "hot")], markers=[(p.eta_m, r["cex_now"], "hot")])
        th = PlotSpec("Thrust and power", r"$V_b$ (V)", r"$F$ (mN)", [Curve(r["Vg"], r["F_g"] * 1e3, "thrust", "thrust"),
                                                                       Curve(r["Vg"], r["P_g"] / 100.0, "dim", r"$P_{in}$ (hW)", "dash")],
                      markers=[(p.Vb, r["F"] * 1e3, "thrust")])
        cols = ["thrust", "pressure", "fuel"]
        isp = PlotSpec("Specific impulse", r"$V_b$ (V)", r"$I_{sp}$ (s)",
                       [Curve(r["Vg"], r["props"][k], cols[i], PROP_LABELS[i].lower()) for i, k in enumerate(PROP_CHOICES)],
                       markers=[(p.Vb, r["Isp"], "text")])
        eta = PlotSpec("Total efficiency", r"$V_b$ (V)", r"$\eta_T$ (–)", [Curve(r["Vg"], r["eta_g"], "oxidizer")],
                       markers=[(p.Vb, r["eta"], "oxidizer")])
        return [TabSpec("Grids", [cl, cex]), TabSpec("Performance", [th, isp, eta])]

    def readout(self, p, r, frame) -> list[str]:
        return [rf"$I_b$ = {r['Ib']:.2f} (A)", rf"$F$ = {r['F'] * 1e3:.1f} (mN)", rf"$I_{{sp}}$ = {r['Isp']:.0f} (s)",
                rf"$P_{{in}}$ = {r['P'] / 1e3:.2f} (kW)"]

    def caption(self, p, r, frame) -> str:
        return (rf"$j_{{CL}}$ = {r['j']:.1f} (A·m⁻²) over $l_e$ = {r['le'] * 1e3:.2f} (mm) · {r['n']:.0f} apertures · "
                rf"$\eta_T$ = {r['eta']:.2f} · beamlet focusing vs perveance: see the Ion Optics Run model")

    def export_data(self, p, r):
        return {"Vb_V": r["Vg"], "F_N": r["F_g"], "Isp_s": r["isp_g"], "eta_T": r["eta_g"], "VT_V": r["Vt_g"],
                "jCL_Am2": r["j_g"]}

    def build_scene(self, vp, p, r=None) -> None:
        vp.plotter.clear()
        self._m = {"built": True}
        self.update_scene(vp, p, r or self.evaluate(p), None)
        vp.reset_view(position="iso", zoom=1.0)

    def update_scene(self, vp, p, r, frame) -> None:
        if not self._m:
            return self.build_scene(vp, p, r)
        pl = vp.plotter
        Rg = 0.5 * p.D * 1e-2
        n_across = 16                                               # displayed hole count is reduced
        pitch = 2 * Rg / n_across
        ii = np.arange(-n_across, n_across + 1)
        X, Y = np.meshgrid(ii * pitch, ii * pitch * np.sqrt(3) / 2)
        X = X + (np.arange(len(ii))[:, None] % 2) * pitch / 2
        cx, cy = X.ravel(), Y.ravel()
        keep = np.hypot(cx, cy) < Rg - pitch * 0.6
        ang = np.arctan2(cy, cx) % (2 * np.pi)
        keep &= _cut_ok(ang, vp.cutaway)
        cx, cy = cx[keep], cy[keep]
        rh = pitch * p.r_s / p.pitch
        gap = max(3 * p.l_g * 1e-3 * Rg / 0.15, 0.01 * Rg)
        a = np.linspace(0, 2 * np.pi, 17)
        for zz, nm, col, rr in ((0.0, "screen", "#8a94a6", rh), (gap, "accel", "#5a6478", 0.6 * rh)):
            ph = np.linspace(0, np.radians(360 - vp.cutaway) if vp.cutaway else 2 * np.pi, 73)
            disc = pv.PolyData(np.c_[np.r_[0.0, Rg * np.cos(ph)], np.r_[0.0, Rg * np.sin(ph)], np.full(len(ph) + 1, zz)])
            disc.faces = np.hstack([[3, 0, k + 1, k + 2] for k in range(len(ph) - 1)])
            pl.add_mesh(disc, color=col, opacity=0.55, name=nm)
            rings = np.stack([np.c_[x0 + rr * np.cos(a), y0 + rr * np.sin(a), np.full_like(a, zz + 1e-4)]
                              for x0, y0 in zip(cx, cy)])
            pl.add_mesh(polylines(rings), color="#1b2230", line_width=1.5, name=nm + "_holes")
        L = 1.2 * Rg
        tn = np.tan(np.radians(max(p.theta, 0.5)))
        rad = np.hypot(cx, cy) / Rg
        beam = np.stack([np.c_[np.r_[x0, x0 * (1 + tn * L / Rg * (0.3 + rad_))], np.r_[y0, y0 * (1 + tn * L / Rg * (0.3 + rad_))],
                               np.r_[gap, gap + L]] for x0, y0, rad_ in zip(cx, cy, rad)])
        poly = polylines(beam)
        poly["j"] = np.full(poly.n_points, p.frac * r["j"])
        pl.add_mesh(poly, scalars="j", cmap="plasma", clim=(0.0, max(r["j"], 1e-9)), line_width=2, opacity=0.8,
                    show_scalar_bar=False, name="beamlets")
        vp.set_colorbar("plasma", 0.0, float(r["j"]), r"$j$ (A·m⁻²)")
        vp.render()


# ====================================================================================== Hall thruster (Live)
class HallLiveParams(ParamModel):
    Vd: float = P(300.0, symbol=r"V_d", desc="discharge voltage", units="V", dims="M L² T⁻³ I⁻¹", lo=100.0, hi=1000.0,
                  step=5.0, group="Operating point")
    mdot: float = P(5.0, symbol=r"\dot m_a", desc="anode xenon flow", units="mg·s⁻¹", dims="M T⁻¹", lo=0.5, hi=50.0,
                    step=0.1, group="Operating point")
    R_out: float = P(50.0, symbol=r"R_o", desc="channel outer radius", units="mm", dims="L", lo=10.0, hi=200.0, step=0.5,
                     group="Channel")
    R_in: float = P(34.0, symbol=r"R_i", desc="channel inner radius", units="mm", dims="L", lo=5.0, hi=190.0, step=0.5,
                    group="Channel")
    L: float = P(25.0, symbol=r"L_{ch}", desc="channel length", units="mm", dims="L", lo=8.0, hi=60.0, step=0.5,
                 group="Channel")
    B_max: float = P(20.0, symbol=r"B_{r,max}", desc="peak radial magnetic field", units="mT", dims="M T⁻² I⁻¹",
                     lo=5.0, hi=60.0, step=0.5, group="Magnetic field")
    Te_max: float = P(25.0, symbol=r"T_{e,max}", desc="peak electron temperature", units="eV", dims="M L² T⁻²", lo=5.0,
                      hi=60.0, step=0.5, group="Magnetic field")
    eta_b: float = P(0.75, symbol=r"\eta_b", desc="beam current fraction of the discharge current", units="", dims="1",
                     lo=0.4, hi=0.95, step=0.01, group="Efficiencies")
    eta_m: float = P(0.9, symbol=r"\eta_m", desc="propellant utilization", units="", dims="1", lo=0.5, hi=0.99,
                     step=0.01, group="Efficiencies")
    theta: float = P(20.0, symbol=r"\bar\theta", desc="plume divergence half angle", units="deg", dims="1", lo=0.0,
                     hi=45.0, step=0.5, group="Efficiencies")
    f2: float = P(10.0, symbol=r"\frac{I^{++}}{I^+}", desc="doubly charged ion current ratio", units="%", dims="1",
                  lo=0.0, hi=40.0, step=0.5, group="Efficiencies")


HALL_FIELDS = [("B", "plasma", r"$B_r$ (mT)"), ("Ek", "inferno", r"$E$ (kV·m⁻¹)"), ("Te", CMAPS["temperature"], r"$T_e$ (eV)")]


def _hall_params(p) -> hl.HallParams:
    mm = 1e-3
    Ro, Ri = p.R_out * mm, min(p.R_in, p.R_out - 2.0) * mm
    return hl.HallParams(L=p.L * mm, L_plume=0.4 * p.L * mm, A=np.pi * (Ro**2 - Ri**2), mdot=p.mdot * 1e-6, Vd=p.Vd,
                         Te_max=p.Te_max, B_peak_x=0.88 * p.L * mm, B_width=0.24 * p.L * mm, Te_peak_x=0.56 * p.L * mm,
                         Te_width=0.24 * p.L * mm, B_max=p.B_max * 1e-3)


class HallLiveModel(ModelController):
    key = "ep.hall"
    title = "Hall thruster"
    module = "Electric"
    mode = "live"
    Params = HallLiveParams
    fields = [f[2] for f in HALL_FIELDS]

    def __init__(self) -> None:
        self._m: dict = {}

    def _perf(self, p, Vd=None, mdot=None):
        Vd = p.Vd if Vd is None else Vd
        md = (p.mdot if mdot is None else mdot) * 1e-6
        M = 131.293 * ep.AMU
        Id = p.eta_m * ep.E * md / M / p.eta_b
        return hl.scaling(Vd, Id, md, eta_b=p.eta_b, eta_m=p.eta_m, theta_div=np.radians(p.theta), f2=p.f2 / 100), Id

    def evaluate(self, p: HallLiveParams) -> dict:
        hp = _hall_params(p)
        x, phi, vi, Te, kiz, vn, Ef = hl.profiles(hp)
        B = hp.B_max * hl.b_profile(x, hp)
        drift = Ef / np.maximum(B, 0.05 * hp.B_max)
        s, Id = self._perf(p)
        Vg = np.linspace(100.0, 1000.0, 50)
        sw = [self._perf(p, Vd=v)[0] for v in Vg]
        f_g = np.array([hl.fife_frequency(_hall_params(p.model_copy(update={"Vd": float(v)}))) for v in Vg[::5]])
        vth_e = np.sqrt(8 * ep.E * p.Te_max / (np.pi * ep.ME))
        rLe = ep.ME * vth_e / (ep.E * hp.B_max)
        vi_exit = float(vi[np.searchsorted(x, hp.L)])
        rLi = 131.293 * ep.AMU * vi_exit / (ep.E * hp.B_max)
        Ro, Ri = p.R_out * 1e-3, min(p.R_in, p.R_out - 2.0) * 1e-3
        return dict(**s, Id=Id, x=x, phi=phi, vi=vi, Te=Te, Ef=Ef, Ek=Ef / 1e3, B=B * 1e3, drift=drift, Vg=Vg,
                    F_g=np.array([w["F"] for w in sw]), isp_g=np.array([w["Isp"] for w in sw]),
                    eta_g=np.array([w["eta"] for w in sw]), Vf=Vg[::5], f_g=f_g, f_fife=hl.fife_frequency(hp),
                    rLe=rLe, rLi=rLi, w=Ro - Ri, Ro=Ro, Ri=Ri, Lch=hp.L, Lp=hp.L_plume, magnetized=rLe < 0.2 * (Ro - Ri) < rLi)

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        xmm = r["x"] * 1e3
        xl = r"$x$ (mm)"
        exit_ = np.full(2, p.L)
        b = PlotSpec("Radial magnetic field and electric field", xl, r"$B_r$ (mT)",
                     [Curve(xmm, r["B"], "pressure", r"$B_r$ (mT)"), Curve(xmm, r["Ek"] / max(r["Ek"].max(), 1e-9) * r["B"].max(),
                                                                         "hot", r"$E$ (scaled)", "dash"),
                      Curve(exit_, np.array([0.0, r["B"].max()]), "dim", "channel exit", "dash")])
        pot = PlotSpec("Potential and electron temperature", xl, r"$\phi$ (V)",
                       [Curve(xmm, r["phi"], "thrust", r"$\phi$ (V)"), Curve(xmm, r["Te"] * 10.0, "fuel", r"$10\,T_e$ (eV)", "dash")])
        dr = PlotSpec(r"$\vec E\times\vec B$ drift speed", xl, r"$v_{E\times B}$ (km·s⁻¹)", [Curve(xmm, r["drift"] / 1e3, "oxidizer")],
                      markers=[(xmm[int(np.argmax(r["B"]))], float(r["drift"][int(np.argmax(r["B"]))] / 1e3), "oxidizer")])
        th = PlotSpec("Thrust", r"$V_d$ (V)", r"$F$ (mN)", [Curve(r["Vg"], r["F_g"] * 1e3, "thrust")],
                      markers=[(p.Vd, r["F"] * 1e3, "thrust")])
        isp = PlotSpec("Specific impulse and efficiency", r"$V_d$ (V)", r"$I_{sp}$ (s)",
                       [Curve(r["Vg"], r["isp_g"], "pressure", r"$I_{sp}$ (s)"),
                        Curve(r["Vg"], r["eta_g"] * 1000.0, "hot", r"$1000\,\eta_T$", "dash")],
                       markers=[(p.Vd, r["Isp"], "pressure")])
        ff = PlotSpec("Breathing frequency estimate (Fife)", r"$V_d$ (V)", r"$f_{br}$ (kHz)", [Curve(r["Vf"], r["f_g"] / 1e3, "fuel")],
                      markers=[(p.Vd, r["f_fife"] / 1e3, "fuel")])
        return [TabSpec("Channel fields", [b, pot, dr]), TabSpec("Performance", [th, isp, ff])]

    def readout(self, p, r, frame) -> list[str]:
        return [rf"$I_d$ = {r['Id']:.2f} (A)", rf"$F$ = {r['F'] * 1e3:.1f} (mN)", rf"$I_{{sp}}$ = {r['Isp']:.0f} (s)",
                rf"$\eta_T$ = {r['eta']:.3f}"]

    def caption(self, p, r, frame) -> str:
        mag = ("electrons magnetized, ions not" if r["magnetized"] else
               "CHECK magnetization: need $r_{L,e}$ much smaller than the channel width, much smaller than $r_{L,i}$")
        return (rf"$P$ = {r['P']:.0f} (W) · $r_{{L,e}}$ = {r['rLe'] * 1e3:.2f} (mm), $r_{{L,i}}$ = {r['rLi']:.2f} (m) · "
                rf"{mag} · $f_{{br}}$ about {r['f_fife'] / 1e3:.0f} (kHz)")

    def export_data(self, p, r):
        return {"x_m": r["x"], "B_mT": r["B"], "E_Vm": r["Ef"], "phi_V": r["phi"], "Te_eV": r["Te"], "vi_ms": r["vi"],
                "Vd_V": r["Vg"], "F_N": r["F_g"], "Isp_s": r["isp_g"]}

    def build_scene(self, vp, p, r=None) -> None:
        vp.plotter.clear()
        self._m = {"built": True}
        self.update_scene(vp, p, r or self.evaluate(p), None)
        vp.reset_view(position="iso", zoom=1.0)

    def update_scene(self, vp, p, r, frame) -> None:
        if not self._m:
            return self.build_scene(vp, p, r)
        draw_hall(vp, r["x"], r[HALL_FIELDS[vp.field_index][0]], r["Ro"], r["Ri"], r["Lch"], np.radians(p.theta),
                  HALL_FIELDS[vp.field_index][1], HALL_FIELDS[vp.field_index][2], x_peak=float(r["x"][int(np.argmax(r["B"]))]))


def draw_hall(vp, x, v, Ro, Ri, Lch, theta, cmap, title, x_peak, clim=None):
    """Annular channel walls colored by a profile along x, anode, radial field lines, E x B drift ring and plume."""
    pl = vp.plotter
    th = thetas(vp.cutaway)
    inside = x <= Lch + 1e-12
    xi = x[inside]
    lo, hi = clim if clim is not None else (float(np.min(v)), float(np.max(v)) + 1e-12)
    for nm, R in (("outer", Ro), ("inner", Ri)):
        g = rev_grid(xi, np.full(len(xi), R), th)
        g["v"] = broadcast_scalar(v[inside], len(th))
        pl.add_mesh(g, scalars="v", cmap=cmap, clim=(lo, hi), show_scalar_bar=False, name=nm, smooth_shading=True,
                    opacity=0.9)
    rs = np.linspace(Ri, Ro, 5)
    anode = rev_grid(np.zeros(5), rs, th)
    pl.add_mesh(anode, color="#b8860b", name="anode")
    ph = th[::4]
    fl = np.stack([np.c_[np.full(6, x_peak), np.linspace(Ri, Ro, 6) * np.cos(a), np.linspace(Ri, Ro, 6) * np.sin(a)] for a in ph])
    pl.add_mesh(polylines(fl), color="#4ea1ff", line_width=2, name="field_lines")
    rm = 0.5 * (Ro + Ri)
    ring = np.c_[np.full(len(th), x_peak), rm * np.cos(th), rm * np.sin(th)][None]
    pl.add_mesh(polylines(ring), color="#ffd23f", line_width=4, name="exb")
    xp = np.linspace(Lch, Lch + 3.0 * Ro, 12)
    s = (xp - Lch) * np.tan(max(theta, 0.02))
    for nm, rr in (("plume_o", Ro + s), ("plume_i", np.maximum(Ri - s, 0.0))):
        pl.add_mesh(rev_grid(xp, rr, th), color="#b07cff", opacity=0.25, name=nm)
    vp.set_colorbar(cmap, lo, hi, title)
    vp.render()


# ====================================================================================== ion optics (Run)
class IonOpticsModel(ModelController):
    key = "ep.optics"
    title = "Ion optics"
    module = "Electric"
    mode = "run"
    Params = IonOpticsParams
    fields = [r"$\phi$ (V)"]

    def __init__(self) -> None:
        self.sim = IonOptics()
        self._m: dict = {}

    def _cell(self, p):
        g = p.geom(48)
        return g

    def build_scene(self, vp, p, r=None) -> None:
        pl = vp.plotter
        pl.clear()
        g = self._cell(p)
        th = thetas(vp.cutaway)
        mm = 1e3
        Rc = g.R_cell * mm
        for nm, z0, t, ra, col in (("screen", g.z_s0, g.t_s, g.r_s, "#8a94a6"), ("accel", g.z_a0, g.t_a, g.r_a, "#5a6478")):
            zz = np.array([z0, z0 + t, z0 + t, z0, z0]) * mm
            rr = np.array([ra, ra, Rc / mm, Rc / mm, ra]) * mm
            pl.add_mesh(rev_grid(zz, rr, th), color=col, name=nm, smooth_shading=False)
        self._m = dict(cut=vp.cutaway, built_r=r is not None, th=th)
        if r is not None:
            a = r.arrays
            Z, Rr = np.meshgrid(a["grid_z"] * mm, a["grid_r"] * mm, indexing="ij")
            plane = pv.StructuredGrid(np.r_[Z, Z[:, ::-1]], np.zeros_like(np.r_[Z, Z]), np.r_[Rr, -Rr[:, ::-1]])
            plane["phi"] = np.r_[a["phi"], a["phi"][:, ::-1]].ravel(order="F")
            lo, hi = float(a["phi"].min()), float(a["phi"].max())
            pl.add_mesh(plane, scalars="phi", cmap="coolwarm", clim=(lo, hi), opacity=0.55, show_scalar_bar=False, name="phi")
            ang = np.radians(np.linspace(0, 360 - max(vp.cutaway, 0), 6, endpoint=False))
            zt, rt = a["z_tr"] * mm, a["r_tr"] * mm
            tr3 = np.concatenate([np.stack([np.c_[zt[i], rt[i] * np.cos(q), rt[i] * np.sin(q)] for i in range(len(zt))])
                                  for q in ang])
            pl.add_mesh(polylines(tr3), color="#ff9d3c", line_width=1.2, opacity=0.6, name="tracks")
            self._m.update(ang=ang, clim=(lo, hi))
        vp.reset_view(zoom=1.2)

    def update_scene(self, vp, p, r, frame) -> None:
        if not self._m or self._m["cut"] != vp.cutaway or self._m["built_r"] != (r is not None):
            self.build_scene(vp, p, r)
        if r is None or frame is None:
            vp.set_colorbar("coolwarm", p.V_a, p.V_s, r"$\phi$ (V)")
            vp.render()
            return
        a = r.arrays
        zt, rt = a["z_tr"][:, frame] * 1e3, a["r_tr"][:, frame] * 1e3
        pts = np.concatenate([np.c_[zt, rt * np.cos(q), rt * np.sin(q)] for q in self._m["ang"]])
        vp.plotter.add_mesh(pv.PolyData(pts), color="#ffd23f", point_size=7, render_points_as_spheres=True, name="ions")
        vp.set_colorbar("coolwarm", *self._m["clim"], r"$\phi$ (V)")
        vp.render()

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        if r is None:
            return [TabSpec("Beamlet", [PlotSpec("Ion trajectories", r"$z$ (mm)", r"$r$ (mm)", []),
                                        PlotSpec("Axial potential", r"$z$ (mm)", r"$\phi$ (V)", []),
                                        PlotSpec("Ion energy gain", r"$t$ (ns)", r"$\bar E_k$ (eV)", [], cursor=True)]),
                    TabSpec("Focusing & erosion", [])]
        a, c, m = r.arrays, r.channels, r.meta
        mm = 1e3
        tracks = [Curve(a["z_tr"][i] * mm, a["r_tr"][i] * mm, "thrust" if a["z_tr"][i][-1] > m["z_exit"] else "hot")
                  for i in range(len(a["z_tr"]))]
        g = self._cell(p)
        for z0, t, ra in ((g.z_s0, g.t_s, g.r_s), (g.z_a0, g.t_a, g.r_a)):
            zz = np.array([z0, z0, z0 + t, z0 + t]) * mm
            tracks.append(Curve(zz, np.array([m["R_cell"], ra, ra, m["R_cell"]]) * mm, "dim"))
        traj = PlotSpec("Ion trajectories", r"$z$ (mm)", r"$r$ (mm)", tracks)
        ax = PlotSpec("Axial potential", r"$z$ (mm)", r"$\phi$ (V)",
                      [Curve(a["grid_z"] * mm, a["phi"][:, 0], "pressure", "axis"),
                       Curve(a["grid_z"] * mm, a["phi"][:, -1], "dim", "cell edge", "dash")])
        ke = PlotSpec("Ion energy gain", r"$t$ (ns)", r"$\bar E_k$ (eV)", [Curve(r.t, c["ke"], "hot")], cursor=True)
        out = [TabSpec("Beamlet", [traj, ax, ke])]
        foc = []
        if len(a["sweep_f"]):
            foc.append(PlotSpec("Beamlet divergence vs perveance", r"$\frac{I_b}{I_{max}}$ (–)", r"$\theta_{rms}$ (deg)",
                                [Curve(a["sweep_f"], a["sweep_th"], "pressure")], markers=[(p.frac, m["theta"], "hot")]))
            foc.append(PlotSpec("Accelerator-grid impingement", r"$\frac{I_b}{I_{max}}$ (–)", r"$f_{imp}$ (%)",
                                [Curve(a["sweep_f"], 100 * a["sweep_imp"], "hot")], markers=[(p.frac, 100 * m["impinge"], "hot")]))
        it = np.arange(1, len(a["theta_hist"]) + 1, dtype=float)
        foc.append(PlotSpec("Space-charge iteration", r"$k$ (–)", r"$\theta_{rms}$ (deg)", [Curve(it, a["theta_hist"], "fuel")]))
        out.append(TabSpec("Focusing & erosion", foc))
        return out

    def readout(self, p, r, frame) -> list[str]:
        if r is None or frame is None:
            return [r"$I_b$ = – (mA)", r"$\theta_{rms}$ = – (deg)", r"$\bar E_k$ = – (eV)", r"$I_{accel}$ = – (nA)"]
        m = r.meta
        return [rf"$I_b$ = {m['I_b'] * 1e3:.3f} (mA)", rf"$\theta_{{rms}}$ = {m['theta']:.2f} (deg)",
                rf"$\bar E_k$ = {r.channels['ke'][frame]:.0f} (eV)", rf"$I_{{accel}}$ = {m['I_accel'] * 1e9:.2f} (nA)"]

    def caption(self, p, r, frame) -> str:
        if r is None or frame is None:
            return r"press Run: self-consistent beamlet (Poisson + particles), charge exchange and a perveance sweep"
        m = r.meta
        conv = "" if m["converged"] else " · NOT converged (near the perveance limit)"
        return (rf"$t$ = {r.t[frame]:.0f} (ns) · {100 * r.channels['passed'][frame]:.0f} % of the tracked ions past the "
                rf"accelerator grid{conv}")

    def summary_line(self, p, r) -> str:
        m = r.meta
        return (rf"$\theta_{{rms}}$ = {m['theta']:.2f} (deg) · impingement {100 * m['impinge']:.1f} % · $F$ = "
                rf"{m['F_b'] * 1e6:.1f} ($\mathrm{{\mu N}}$) per beamlet · CEX back {100 * m['cex_back']:.0f} %")

    def export_data(self, p, r):
        a = r.arrays
        return {"t_ns": r.t, **r.channels, "sweep_frac": a["sweep_f"], "sweep_theta_deg": a["sweep_th"],
                "sweep_impinge": a["sweep_imp"], "axis_z_m": a["grid_z"], "axis_phi_V": a["phi"][:, 0]}


# ====================================================================================== Hall discharge (Run)
class HallDischargeModel(ModelController):
    key = "ep.hall_run"
    title = "Hall discharge"
    module = "Electric"
    mode = "run"
    Params = HallRunParams
    fields = [r"$n_i$ ($10^{17}$ m⁻³)", r"$n_n$ ($10^{17}$ m⁻³)"]

    def __init__(self) -> None:
        self.sim = HallDischarge()
        self._m: dict = {}

    def _geom(self, p):
        A = p.A * 1e-4
        Ro = np.sqrt(A / (np.pi * (1 - 0.68**2)))
        return Ro, 0.68 * Ro

    def build_scene(self, vp, p, r=None) -> None:
        vp.plotter.clear()
        self._m = {"built": True}
        vp.reset_view(position="iso", zoom=1.0)

    def update_scene(self, vp, p, r, frame) -> None:
        if not self._m:
            self.build_scene(vp, p, r)
        hp = p.hall()
        Ro, Ri = self._geom(p)
        key = ["ni", "nn"][vp.field_index]
        if r is None or frame is None:
            x = np.linspace(0, hp.L + hp.L_plume, 40)
            v, clim = np.zeros_like(x), (0.0, 1.0)
        else:
            x = r.arrays["x"]
            v = r.arrays[key][frame] / 1e17
            clim = (0.0, float(r.arrays[key].max() / 1e17) + 1e-12)
        b = hl.b_profile(x, hp)
        draw_hall(vp, x, v, Ro, Ri, hp.L, np.radians(20.0), "viridis", self.fields[vp.field_index],
                  x_peak=float(x[int(np.argmax(b))]), clim=clim)

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        TT = r"$t$ (ms)"
        if r is None:
            return [TabSpec("Discharge", [PlotSpec("Currents", TT, r"$I$ (A)", [], cursor=True),
                                          PlotSpec("Channel-mean densities", TT, r"$\bar n$ ($10^{17}$ m⁻³)", [], cursor=True)]),
                    TabSpec("Profiles", []), TabSpec("Breathing estimate", [])]
        c, a, t = r.channels, r.arrays, r.t
        cur = [PlotSpec("Currents", TT, r"$I$ (A)", [Curve(t, c["Iiz"], "hot", "ion production"),
                                                     Curve(t, c["Ib"], "thrust", "beam", "dash")], cursor=True),
               PlotSpec("Channel-mean densities", TT, r"$\bar n$ ($10^{17}$ m⁻³)",
                        [Curve(t, c["ni_avg"], "pressure", "ions"), Curve(t, c["nn_avg"], "fuel", "neutrals")], cursor=True)]
        xmm = a["x"] * 1e3
        prof = [PlotSpec("Final densities", r"$x$ (mm)", r"$n$ ($10^{17}$ m⁻³)",
                         [Curve(xmm, a["ni"][-1] / 1e17, "pressure", "ions"), Curve(xmm, a["nn"][-1] / 1e17, "fuel", "neutrals")]),
                PlotSpec("Potential and field shape", r"$x$ (mm)", r"$\phi$ (V)",
                         [Curve(xmm, a["phi"], "thrust", r"$\phi$ (V)"),
                          Curve(xmm, a["B"] * p.Vd, "dim", r"$B_r$ shape (scaled)", "dash")]),
                PlotSpec("Ion velocity and electron temperature", r"$x$ (mm)", r"$v_i$ (km·s⁻¹)",
                         [Curve(xmm, a["vi"] / 1e3, "oxidizer", r"$v_i$ (km·s⁻¹)"),
                          Curve(xmm, a["Te"], "hot", r"$T_e$ (eV)", "dash")])]
        Vg = np.linspace(150.0, 800.0, 14)
        fv = [hl.fife_frequency(p.model_copy(update={"Vd": float(v)}).hall()) / 1e3 for v in Vg]
        xg = np.linspace(4.0, 40.0, 14)
        fx = [hl.fife_frequency(p.model_copy(update={"L": float(max(L / 0.56, 10.0)), "Te_x": float(L)}).hall()) / 1e3
              for L in xg]
        br = [PlotSpec("Fife estimate vs discharge voltage", r"$V_d$ (V)", r"$f_{br}$ (kHz)", [Curve(Vg, fv, "fuel")],
                       markers=[(p.Vd, r.meta["f_fife"] / 1e3, "fuel")]),
              PlotSpec("Fife estimate vs ionization-zone location", r"$x_{iz}$ (mm)", r"$f_{br}$ (kHz)", [Curve(xg, fx, "pressure")])]
        return [TabSpec("Discharge", cur), TabSpec("Profiles", prof), TabSpec("Breathing estimate", br)]

    def readout(self, p, r, frame) -> list[str]:
        if r is None or frame is None:
            return [r"$I_{iz}$ = – (A)", r"$I_b$ = – (A)", r"$\bar n_i$ = – ($10^{17}$ m⁻³)", r"$f_{br}$ = – (kHz)"]
        c = r.channels
        return [rf"$I_{{iz}}$ = {c['Iiz'][frame]:.2f} (A)", rf"$I_b$ = {c['Ib'][frame]:.2f} (A)",
                rf"$\bar n_i$ = {c['ni_avg'][frame]:.2f} ($10^{{17}}$ m⁻³)", rf"$f_{{br}}$ = {r.meta['f_fife'] / 1e3:.1f} (kHz)"]

    def caption(self, p, r, frame) -> str:
        if r is None or frame is None:
            return r"press Run: 1-D neutral and ion transport with ionization in fixed field and $T_e$ profiles"
        return (rf"$t$ = {r.t[frame]:.3f} (ms) · reduced model: the discharge settles (no self-sustained breathing); "
                rf"Fife estimate {r.meta['f_fife'] / 1e3:.1f} (kHz)")

    def summary_line(self, p, r) -> str:
        m = r.meta
        return (rf"$I_b$ = {m['Ib_ss']:.2f} (A) · $F$ about {m['F'] * 1e3:.0f} (mN) · utilization {100 * min(m['eta_m'], 9.99):.0f} % · "
                rf"breathing not reproduced, estimate {m['f_fife'] / 1e3:.1f} (kHz)")

    def export_data(self, p, r):
        return {"t_ms": r.t, **r.channels, "x_m": r.arrays["x"], "ni_final": r.arrays["ni"][-1], "nn_final": r.arrays["nn"][-1]}


# ====================================================================================== low-thrust spiral (Run)
class SpiralModel(ModelController):
    key = "ep.spiral"
    title = "Low-thrust spiral"
    module = "Electric"
    mode = "run"
    Params = SpiralParams
    fields = [r"$h$ (km)"]
    MAX_REVS_SHOWN = 40

    def __init__(self) -> None:
        self.sim = LowThrustSpiral()
        self._m: dict = {}

    def _path(self, p, r):
        """Display path in Earth radii; revolutions compressed to at most ``MAX_REVS_SHOWN``."""
        c = r.channels
        n = 4000
        tt = np.linspace(r.t[0], r.t[-1], n)
        rev = np.interp(tt, r.t, c["rev"])
        k = min(1.0, self.MAX_REVS_SHOWN / max(rev[-1], 1e-9))
        ang = 2 * np.pi * rev * k
        rad = (np.interp(tt, r.t, c["h"]) * 1e3 + ep.R_EARTH) / ep.R_EARTH
        inc = np.radians(p.di - np.interp(tt, r.t, c["inc"]))               # inclination relative to the final plane
        P = np.c_[rad * np.cos(ang), rad * np.sin(ang) * np.cos(inc), rad * np.sin(ang) * np.sin(inc)]
        return tt, P, k

    def build_scene(self, vp, p, r=None) -> None:
        pl = vp.plotter
        pl.clear()
        pl.add_mesh(pv.Sphere(radius=1.0, theta_resolution=60, phi_resolution=30), color="#2f6db5", smooth_shading=True,
                    name="earth")
        self._m = dict(built_r=r is not None)
        if r is not None:
            tt, P, k = self._path(p, r)
            poly = pv.lines_from_points(P)
            poly["h"] = np.interp(tt, r.t, r.channels["h"])
            pl.add_mesh(poly, scalars="h", cmap="viridis", line_width=1.5, opacity=0.5, show_scalar_bar=False, name="path")
            self._m.update(tt=tt, P=P, k=k, hmax=float(r.channels["h"].max()))
        else:
            s = p.spiral()
            for nm, rad in (("orbit0", s.r0), ("orbit1", s.r1)):
                a = np.linspace(0, 2 * np.pi, 181)
                pl.add_mesh(pv.lines_from_points(np.c_[rad * np.cos(a), rad * np.sin(a), 0 * a] / ep.R_EARTH),
                            color="#8a94a6", line_width=1.5, name=nm)
        vp.reset_view(position="iso", zoom=1.0)

    def update_scene(self, vp, p, r, frame) -> None:
        if not self._m or self._m["built_r"] != (r is not None):
            self.build_scene(vp, p, r)
        if r is None or frame is None:
            vp.set_colorbar("viridis", p.h0, p.h0 + 1.0, r"$h$ (km)")
            vp.render()
            return
        m = self._m
        i = int(np.searchsorted(m["tt"], r.t[frame]))
        i = min(max(i, 1), len(m["tt"]) - 1)
        vp.plotter.add_mesh(pv.lines_from_points(m["P"][: i + 1]), color="#ffd23f", line_width=1.5, opacity=0.8, name="flown")
        vp.plotter.add_mesh(pv.Sphere(radius=0.12, center=m["P"][i]), color="#ff5a3c", name="craft")
        vp.set_colorbar("viridis", p.h0, m["hmax"], r"$h$ (km)")
        vp.render()

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        if r is None:
            e = lambda title, yl: PlotSpec(title, TT_DAYS, yl, [], cursor=True)
            return [TabSpec("Transfer", [e("Altitude", r"$h$ (km)"), e("Inclination change", r"$\Delta i$ (deg)"),
                                         e("Accumulated velocity increment", r"$\Delta v$ (km·s⁻¹)")]),
                    TabSpec("Spacecraft", []), TabSpec("Specific impulse trade", [])]
        c, t = r.channels, r.t
        cur = lambda y, col, label="", style="solid": Curve(t, y, col, label, style)
        tr = [PlotSpec("Altitude", TT_DAYS, r"$h$ (km)", [cur(c["h"], "pressure")], cursor=True),
              PlotSpec("Inclination change", TT_DAYS, r"$\Delta i$ (deg)", [cur(c["inc"], "fuel")], cursor=True),
              PlotSpec("Accumulated velocity increment", TT_DAYS, r"$\Delta v$ (km·s⁻¹)",
                       [cur(c["dv"], "thrust"), cur(np.full_like(t, r.meta["dv_tot"]), "dim", "Edelbaum total", "dash")], cursor=True)]
        sc = [PlotSpec("Spacecraft mass", TT_DAYS, r"$m$ (kg)", [cur(c["m"], "oxidizer")], cursor=True),
              PlotSpec("Power and thrust", TT_DAYS, r"$P$ (kW)", [cur(c["P"], "hot", r"$P$ (kW)"),
                                                                    cur(c["F"] / 100.0, "thrust", r"$F$ (hN, scaled)", "dash")], cursor=True),
              PlotSpec("Orbital speed", TT_DAYS, r"$v$ (km·s⁻¹)", [cur(c["v"], "pressure")], cursor=True)]
        # closed-form trade at constant power: m_p = m0 (1 - e^{-dv/ve}), t = m_p ve^2 / (2 eta P duty)
        s = p.spiral()
        isp = np.geomspace(300.0, 6000.0, 80)
        ve = isp * G0
        dv = r.meta["dv_tot"] * 1e3
        mp = s.m0 * (1 - np.exp(-dv / ve))
        days = mp * ve**2 / (2 * s.eta * s.P0 * s.duty) / DAY
        trade = [PlotSpec("Trip time vs specific impulse", r"$I_{sp}$ (s)", r"$t_{trip}$ (days)", [Curve(isp, days, "pressure")],
                          markers=[(p.isp, r.meta["days"], "hot")]),
                 PlotSpec("Propellant vs specific impulse", r"$I_{sp}$ (s)", r"$m_p$ (kg)", [Curve(isp, mp, "fuel")],
                          markers=[(p.isp, r.meta["m_prop"], "hot")])]
        return [TabSpec("Transfer", tr), TabSpec("Spacecraft", sc), TabSpec("Specific impulse trade", trade)]

    def readout(self, p, r, frame) -> list[str]:
        if r is None or frame is None:
            return [r"$h$ = – (km)", r"$\Delta i$ = – (deg)", r"$\Delta v$ = – (km·s⁻¹)", r"$m$ = – (kg)"]
        c = r.channels
        return [rf"$h$ = {c['h'][frame]:.0f} (km)", rf"$\Delta i$ = {c['inc'][frame]:.2f} (deg)",
                rf"$\Delta v$ = {c['dv'][frame]:.3f} (km·s⁻¹)", rf"$m$ = {c['m'][frame]:.0f} (kg)"]

    def caption(self, p, r, frame) -> str:
        if r is None or frame is None:
            return r"press Run: Edelbaum circle-to-circle spiral with plane change and solar-array degradation"
        k = self._m.get("k", 1.0) if self._m else 1.0
        shown = "" if k >= 1.0 else rf" · revolutions drawn compressed ({self.MAX_REVS_SHOWN} of {r.meta['revs']:.0f})"
        return rf"day {r.t[frame]:.1f} · $P$ = {r.channels['P'][frame]:.2f} (kW), $F$ = {r.channels['F'][frame]:.0f} (mN){shown}"

    def summary_line(self, p, r) -> str:
        m = r.meta
        return (rf"$\Delta v$ = {m['dv_tot']:.3f} (km·s⁻¹) · {m['days']:.0f} (days) · propellant {m['m_prop']:.0f} (kg) · "
                rf"{m['revs']:.0f} revolutions")

    def export_data(self, p, r):
        return {"t_days": r.t, **r.channels}
