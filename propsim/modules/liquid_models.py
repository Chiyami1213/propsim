"""Liquid module: injector & chamber and regenerative cooling (Live); feed transient, pump-fed cycles, chugging (Run)."""
from __future__ import annotations

import numpy as np
import pyvista as pv

from ..core import liquid as lq
from ..core.heat import COOLANT_LABELS, COOLANTS, RegenDesign, regen_march
from ..core.nozzle import T_ratio, contour, mach_along_contour, p_ratio
from ..core.result import Curve, PlotSpec, TabSpec
from ..core.schema import C, P
from ..core.stability import chugging as chg
from ..core.thermo.cea import PAIRS, Thermo
from ..sims.liquid_run import (CycleParams, Chugging, ChuggingParams, EngineGeometry, FeedParams, FeedTransient,
                               PumpFedCycle)
from ..ui.theme import CMAPS
from .base import RUN_COLORS, ModelController
from .geometry import broadcast_scalar, rev_grid, thetas
from .nozzle_live import NozzleLiveModel

TT = r"$t$ (s)"
LIQUID_PAIRS = ["LOX/RP-1", "LOX/LH2", "LOX/CH4", "LOX/Ethanol", "N2O4/MMH"]


def engine_profile(Dt: float, eps: float, L_star: float, contraction: float, g: float = 1.15):
    """Chamber cylinder + convergent + throat + bell: ``(x, r)`` in metres with x = 0 at the throat."""
    Rt = Dt / 2.0
    Rc = Rt * float(np.sqrt(contraction))
    x, r = contour(Rt, eps, Rc=Rc, f=0.8, g=g)
    geo = lq.chamber_geometry(L_star, Dt, contraction)
    n = 10
    xc = np.linspace(x[0] - geo["Lc"], x[0], n, endpoint=False)
    return np.concatenate([xc, x]), np.concatenate([np.full(n, r[0]), r]), geo


# ====================================================================================== injector & chamber (Live)
class InjectorLiveParams(EngineGeometry):
    pair: str = C("LOX/RP-1", choices=LIQUID_PAIRS, labels=["LOX / RP-1", "LOX / LH₂", "LOX / CH₄", "LOX / ethanol",
                                                              "N₂O₄ / MMH"], symbol=r"\mathrm{propellants}",
                  desc="oxidizer / fuel pair (CEA tables)", group="Engine")
    Pc: float = P(6.0, symbol=r"P_c", desc="chamber pressure", units="MPa", dims="M L⁻¹ T⁻²", lo=1.0, hi=20.0,
                  step=0.05, group="Operating point")
    MR: float = P(2.6, symbol=r"\frac{O}{F}", desc="mixture ratio (clamped to the table range)", units="", dims="1",
                  lo=0.8, hi=10.0, step=0.05, group="Operating point")
    dp_o: float = P(22.0, symbol=r"\frac{\Delta p_{ox}}{P_c}", desc="oxidizer injector drop", units="%", dims="1",
                    lo=5.0, hi=60.0, step=0.5, group="Injector")
    dp_f: float = P(22.0, symbol=r"\frac{\Delta p_f}{P_c}", desc="fuel injector drop", units="%", dims="1", lo=5.0,
                    hi=60.0, step=0.5, group="Injector")
    Cd: float = P(0.7, symbol=r"C_d", desc="orifice discharge coefficient", units="", dims="1", lo=0.3, hi=0.95,
                  step=0.01, group="Injector")
    d_o: float = P(1.2, symbol=r"d_{ox}", desc="oxidizer orifice diameter", units="mm", dims="L", lo=0.3, hi=4.0,
                   step=0.05, group="Injector")
    d_f: float = P(1.0, symbol=r"d_f", desc="fuel orifice diameter", units="mm", dims="L", lo=0.3, hi=4.0, step=0.05,
                   group="Injector")


def _liq_props(pair: str):
    ox, fu = pair.split("/")
    rho_o = lq.RHO_OX.get(ox, 1141.0)
    rho_f = lq.RHO_FUEL.get(fu, 810.0)
    return rho_o, rho_f


class InjectorLiveModel(NozzleLiveModel):
    key = "liquid.injector"
    title = "Injector & chamber"
    module = "Liquid"
    mode = "live"
    Params = InjectorLiveParams

    def evaluate(self, p: InjectorLiveParams) -> dict:
        pr = PAIRS[p.pair]
        mr = float(np.clip(p.MR, pr.mr_lo, pr.mr_hi))
        th = Thermo(p.pair, "tables")
        pt = th.point(mr, p.Pc * 1e6, p.eps)
        cs, g = pt["cstar"], pt["gamma"]
        Dt = p.D_t * 1e-3
        At = np.pi * Dt**2 / 4.0
        mdot = p.Pc * 1e6 * At / cs
        m_f, m_o = mdot / (1.0 + mr), mdot * mr / (1.0 + mr)
        rho_o, rho_f = _liq_props(p.pair)
        io = lq.injector_design(m_o, rho_o, p.dp_o / 100 * p.Pc * 1e6, p.Cd, p.d_o * 1e-3)
        if_ = lq.injector_design(m_f, rho_f, p.dp_f / 100 * p.Pc * 1e6, p.Cd, p.d_f * 1e-3)
        J = lq.momentum_ratio(m_f, rho_f, if_["A_tot"], m_o, rho_o, io["A_tot"])
        x, r, geo = engine_profile(Dt, p.eps, p.L_star, p.contraction, g)
        M = mach_along_contour(x, r, Dt / 2, g)
        RT = (cs * float(np.sqrt(g) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1))))) ** 2
        # curves
        dpg = np.linspace(0.05, 0.6, 50)
        Ls = np.linspace(0.4, 3.0, 40)
        cg = np.linspace(1.5, 8.0, 40)
        dg = np.linspace(0.3, 4.0, 40)
        ok_o, frac_o = lq.injector_stiffness_ok(p.dp_o / 100 * p.Pc * 1e6, p.Pc * 1e6)
        ok_f, frac_f = lq.injector_stiffness_ok(p.dp_f / 100 * p.Pc * 1e6, p.Pc * 1e6)
        return dict(cs=cs, g=g, Tc=pt["Tc"], mdot=mdot, m_o=m_o, m_f=m_f, mr=mr, io=io, if_=if_, J=J, geo=geo, x=x, r=r,
                    M=M, T=pt["Tc"] * T_ratio(M, g), p=p.Pc * 1e6 * p_ratio(M, g), Rt=Dt / 2, theta=lq.residence_time(
                        p.L_star, cs, RT), dpg=dpg, Ls=Ls, cg=cg, dg=dg, rho_o=rho_o, rho_f=rho_f, At=At,
                    ok=(ok_o and ok_f), frac_o=frac_o, frac_f=frac_f, clamped=(mr != p.MR), RT=RT, Dt=Dt)

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        Pc = p.Pc * 1e6
        mo_curve = lq.orifice_mdot(p.Cd, r["io"]["A_tot"], r["rho_o"], r["dpg"] * Pc)
        mf_curve = lq.orifice_mdot(p.Cd, r["if_"]["A_tot"], r["rho_f"], r["dpg"] * Pc)
        inj = [PlotSpec("Orifice flow vs pressure drop", r"$\frac{\Delta p}{P_c}$ (–)", r"$\dot m$ (kg·s⁻¹)",
                        [Curve(r["dpg"], mo_curve, "oxidizer", "oxidizer"), Curve(r["dpg"], mf_curve, "fuel", "fuel")],
                        markers=[(p.dp_o / 100, r["m_o"], "oxidizer"), (p.dp_f / 100, r["m_f"], "fuel")]),
               PlotSpec("Orifice count vs hole diameter", r"$d$ (mm)", r"$N_{holes}$ (–)",
                        [Curve(r["dg"], r["io"]["A_tot"] / (np.pi * (r["dg"] * 1e-3) ** 2 / 4), "oxidizer", "oxidizer"),
                         Curve(r["dg"], r["if_"]["A_tot"] / (np.pi * (r["dg"] * 1e-3) ** 2 / 4), "fuel", "fuel")],
                        markers=[(p.d_o, r["io"]["n_exact"], "oxidizer"), (p.d_f, r["if_"]["n_exact"], "fuel")]),
               PlotSpec("Jet velocity", r"$\frac{\Delta p}{P_c}$ (–)", r"$u$ (m·s⁻¹)",
                        [Curve(r["dpg"], p.Cd * np.sqrt(2 * r["dpg"] * Pc / r["rho_o"]), "oxidizer", "oxidizer"),
                         Curve(r["dpg"], p.Cd * np.sqrt(2 * r["dpg"] * Pc / r["rho_f"]), "fuel", "fuel")],
                        markers=[(p.dp_o / 100, r["io"]["jet_u_ideal"], "oxidizer"), (p.dp_f / 100, r["if_"]["jet_u_ideal"], "fuel")])]
        At = r["At"]
        cham = [PlotSpec("Chamber volume", r"$L^*$ (m)", r"$V_c$ (L)", [Curve(r["Ls"], r["Ls"] * At * 1e3, "thrust")],
                         markers=[(p.L_star, r["geo"]["Vc"] * 1e3, "hot")]),
                PlotSpec("Cylinder length vs contraction", r"$\frac{A_c}{A_t}$ (–)", r"$L_c$ (mm)",
                         [Curve(r["cg"], [lq.chamber_geometry(p.L_star, r["Dt"], c)["Lc"] * 1e3 for c in r["cg"]], "fuel")],
                         markers=[(p.contraction, r["geo"]["Lc"] * 1e3, "hot")]),
                PlotSpec("Gas residence time", r"$L^*$ (m)", r"$\theta_g$ (ms)",
                         [Curve(r["Ls"], r["Ls"] * r["cs"] / r["RT"] * 1e3, "pressure")],
                         markers=[(p.L_star, r["theta"] * 1e3, "hot")])]
        return [TabSpec("Injector", inj), TabSpec("Chamber", cham)]

    def readout(self, p, r, frame) -> list[str]:
        return [rf"$\dot m$ = {r['mdot']:.2f} (kg·s⁻¹)", rf"$N_{{ox}}$ / $N_f$ = {r['io']['n']} / {r['if_']['n']}",
                rf"$J$ = {r['J']:.2f}", rf"$\theta_g$ = {r['theta'] * 1e3:.2f} (ms)"]

    def caption(self, p, r, frame) -> str:
        warn = "" if r["ok"] else " · injector drop below 15 % of $P_c$"
        feas = "" if r["geo"]["feasible"] else " · chamber too small for the convergent: raise $L^*$"
        cl = r" · $\frac{O}{F}$ clamped" if r["clamped"] else ""
        return (rf"$c^*$ = {r['cs']:.0f} (m·s⁻¹) · $V_c$ = {r['geo']['Vc'] * 1e3:.2f} (L), $D_c$ = {r['geo']['Dc'] * 1e3:.0f}, "
                rf"$L_c$ = {r['geo']['Lc'] * 1e3:.0f} (mm) · jets {r['io']['u']:.0f} / {r['if_']['u']:.0f} (m·s⁻¹){warn}{feas}{cl}")

    def export_data(self, p, r) -> dict[str, np.ndarray]:
        return {"x_m": r["x"], "r_m": r["r"], "Mach": r["M"], "T_K": r["T"], "p_Pa": r["p"], "dp_over_Pc": r["dpg"],
                "mdot_ox_curve": lq.orifice_mdot(p.Cd, r["io"]["A_tot"], r["rho_o"], r["dpg"] * p.Pc * 1e6)}


# ====================================================================================== regenerative cooling (Live)
MATERIALS = {"cu": ("Copper alloy", 330.0, 800.0), "ni": ("Nickel alloy", 70.0, 1000.0),
             "inc": ("Inconel", 20.0, 1100.0), "ss": ("Stainless steel", 25.0, 900.0)}


class RegenLiveParams(EngineGeometry):
    pair: str = C("LOX/LH2", choices=LIQUID_PAIRS, labels=["LOX / RP-1", "LOX / LH₂", "LOX / CH₄", "LOX / ethanol",
                                                              "N₂O₄ / MMH"], symbol=r"\mathrm{propellants}",
                  desc="oxidizer / fuel pair (CEA tables)", group="Engine")
    Pc: float = P(7.0, symbol=r"P_c", desc="chamber pressure", units="MPa", dims="M L⁻¹ T⁻²", lo=1.0, hi=20.0,
                  step=0.05, group="Operating point")
    MR: float = P(6.0, symbol=r"\frac{O}{F}", desc="mixture ratio (clamped to the table range)", units="", dims="1",
                  lo=0.8, hi=10.0, step=0.05, group="Operating point")
    fluid: str = C("Hydrogen", choices=list(COOLANTS), labels=[COOLANT_LABELS[k] for k in COOLANTS],
                   symbol=r"\mathrm{coolant}", desc="regenerative coolant (CoolProp properties)", group="Coolant")
    cool_frac: float = P(1.0, symbol=r"f_c", desc="coolant flow as a fraction of the fuel flow", units="", dims="1",
                         lo=0.2, hi=1.0, step=0.01, group="Coolant")
    T_in: float = P(40.0, symbol=r"T_{c,in}", desc="coolant inlet temperature (at the nozzle exit)", units="K",
                    dims="Θ", lo=20.0, hi=500.0, step=1.0, group="Coolant")
    p_in: float = P(15.0, symbol=r"p_{c,in}", desc="coolant inlet pressure", units="MPa", dims="M L⁻¹ T⁻²", lo=1.0,
                    hi=30.0, step=0.25, group="Coolant")
    n_ch: float = P(120.0, symbol=r"N_{ch}", desc="number of cooling channels", units="", dims="1", lo=20.0, hi=300.0,
                    step=1.0, group="Channels")
    h_ch: float = P(1.2, symbol=r"h_{ch}", desc="channel height", units="mm", dims="L", lo=0.4, hi=6.0, step=0.05,
                    group="Channels")
    fill: float = P(0.5, symbol=r"\phi", desc="channel width as a fraction of the pitch", units="", dims="1", lo=0.3,
                    hi=0.8, step=0.01, group="Channels")
    fin: float = P(1.8, symbol=r"\eta_{fin}", desc="rib / fin enhancement of the coolant-side coefficient", units="",
                   dims="1", lo=1.0, hi=3.0, step=0.05, group="Channels")
    t_w: float = P(0.9, symbol=r"t_w", desc="hot-wall thickness", units="mm", dims="L", lo=0.3, hi=4.0, step=0.05,
                   group="Wall")
    mat: str = C("cu", choices=list(MATERIALS), labels=[v[0] for v in MATERIALS.values()],
                 symbol=r"\mathrm{wall\ material}", desc="wall material (conductivity and temperature limit)",
                 group="Wall")


FIELDS_REGEN = [("q", "inferno", r"$q$ (MW·m⁻²)"), ("Twg", CMAPS["temperature"], r"$T_{wg}$ (K)"),
                ("Tcool", "viridis", r"$T_{cool}$ (K)")]


class RegenLiveModel(NozzleLiveModel):
    key = "liquid.regen"
    title = "Regenerative cooling"
    module = "Liquid"
    mode = "live"
    Params = RegenLiveParams
    fields = [f[2] for f in FIELDS_REGEN]

    def evaluate(self, p: RegenLiveParams) -> dict:
        pr = PAIRS[p.pair]
        mr = float(np.clip(p.MR, pr.mr_lo, pr.mr_hi))
        pt = Thermo(p.pair, "tables").point(mr, p.Pc * 1e6, p.eps)
        cs, g = pt["cstar"], pt["gamma"]
        Dt = p.D_t * 1e-3
        mdot = p.Pc * 1e6 * np.pi * Dt**2 / 4.0 / cs
        m_f = mdot / (1.0 + mr)
        x, r, geo = engine_profile(Dt, p.eps, p.L_star, p.contraction, g)
        M = mach_along_contour(x, r, Dt / 2, g)
        name, kw, Tmax = MATERIALS[p.mat]
        lo, hi = COOLANTS[p.fluid]
        des = RegenDesign(p.fluid, mdot_c=p.cool_frac * m_f, T_in=float(np.clip(p.T_in, lo + 1, hi - 1)), p_in=p.p_in * 1e6,
                          n_ch=int(round(p.n_ch)), h_ch=p.h_ch * 1e-3, fill=p.fill, t_w=p.t_w * 1e-3, k_w=kw, T_w_max=Tmax,
                          fin=p.fin)
        gas = dict(Tc=pt["Tc"], p0=p.Pc * 1e6, cstar=cs, gamma=g, Mw=pt["Mw"])
        out = regen_march(x, r, Dt / 2, gas, M, des)
        i_t = int(np.argmin(np.abs(x)))
        return dict(x=x, r=r, M=M, T=pt["Tc"] * T_ratio(M, g), p=p.Pc * 1e6 * p_ratio(M, g), reg=out, mdot=mdot, m_f=m_f,
                    mdot_c=des.mdot_c, i_t=i_t, Tmax=Tmax, mat=name, cs=cs, g=g, Tc=pt["Tc"], Rt=Dt / 2, des=des, geo=geo,
                    clamped=(mr != p.MR), q=out["q"] / 1e6, Twg=out["T_wg"], Tcool=out["T_cool"])

    # ---------------------------------------------------------------- 3D
    def _field(self, vp, r):
        key, cmap, title = FIELDS_REGEN[vp.field_index]
        return r[key], cmap, title

    def update_scene(self, vp, p, r, frame) -> None:
        super().update_scene(vp, p, r, frame)
        pl = vp.plotter
        x, rr, des = r["x"], r["r"], r["des"]
        n_show = 16
        pts, conn, val = [], [], []
        for k in range(n_show):
            a = 2 * np.pi * k / n_show
            if vp.cutaway > 0 and a > np.radians(360 - vp.cutaway):
                continue
            R = rr + des.t_w + 0.5 * des.h_ch
            base = len(pts)
            pts.extend(np.c_[x, R * np.cos(a), R * np.sin(a)].tolist())
            conn.append([len(x)] + list(range(base, base + len(x))))
            val.extend(r["Tcool"].tolist())
        if pts:
            poly = pv.PolyData(np.array(pts))
            poly.lines = np.hstack(conn)
            poly["Tcool"] = np.array(val)
            pl.add_mesh(poly, scalars="Tcool", cmap="viridis", line_width=3, show_scalar_bar=False, name="channels")

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        x, o = r["x"] * 1e3, r["reg"]
        xl = r"$x$ (mm)"
        t_idx = r["i_t"]
        mk = lambda y: [(x[t_idx], float(y[t_idx]), "hot")]
        lim = np.full_like(x, r["Tmax"])
        flux = [PlotSpec("Heat flux", xl, r"$q$ (MW·m⁻²)", [Curve(x, o["q"] / 1e6, "hot")], markers=mk(o["q"] / 1e6)),
                PlotSpec("Wall and gas temperatures", xl, r"$T$ (K)",
                         [Curve(x, o["T_aw"], "dim", "adiabatic wall", "dash"), Curve(x, o["T_wg"], "hot", "gas-side wall"),
                          Curve(x, o["T_wc"], "pressure", "coolant-side wall"), Curve(x, lim, "text", "limit", "dash")]),
                PlotSpec("Coolant temperature", xl, r"$T_{cool}$ (K)", [Curve(x, o["T_cool"], "thrust")])]
        side = [PlotSpec("Film coefficients", xl, r"$h$ (kW·m⁻²·K⁻¹)",
                         [Curve(x, o["h_g"] / 1e3, "hot", "gas side"), Curve(x, o["h_c"] / 1e3, "thrust", "coolant side")]),
                PlotSpec("Coolant pressure", xl, r"$p_{cool}$ (MPa)", [Curve(x, o["p_cool"] / 1e6, "pressure")]),
                PlotSpec("Coolant velocity", xl, r"$v$ (m·s⁻¹)", [Curve(x, o["v"], "fuel")])]
        return [TabSpec("Heat flux & wall", flux), TabSpec("Coolant side", side)]

    def readout(self, p, r, frame) -> list[str]:
        o = r["reg"]
        return [rf"$q_{{max}}$ = {o['q_max'] / 1e6:.1f} (MW·m⁻²)", rf"$T_{{wg,max}}$ = {o['T_wg_max']:.0f} (K)",
                rf"$\Delta T_{{cool}}$ = {o['dT_cool']:.0f} (K)", rf"$\Delta p_{{cool}}$ = {o['dp'] / 1e6:.2f} (MPa)"]

    def caption(self, p, r, frame) -> str:
        o = r["reg"]
        ok = o["margin"] >= 0
        flag = ("within" if ok else "OVER") + rf" the {r['mat'].lower()} limit by {abs(o['margin']):.0f} (K)"
        return (rf"heat load {o['Q'] / 1e3:.0f} (kW) · coolant {r['mdot_c']:.2f} (kg·s⁻¹) · $T_{{wg,max}}$ {flag}"
                + (r" · $\frac{O}{F}$ clamped" if r["clamped"] else ""))

    def export_data(self, p, r) -> dict[str, np.ndarray]:
        o = r["reg"]
        return {"x_m": r["x"], "r_m": r["r"], "q_Wm2": o["q"], "Twg_K": o["T_wg"], "Twc_K": o["T_wc"],
                "Tcool_K": o["T_cool"], "pcool_Pa": o["p_cool"], "hg": o["h_g"], "hc": o["h_c"]}


# ====================================================================================== common scene for the Run models
class LiquidRunBase(ModelController):
    """Engine (chamber, throat, bell) with two tanks and feed lines, coloured by pressure / state."""

    module = "Liquid"
    mode = "run"
    cmap = "viridis"
    value_title = r"$P$ (MPa)"

    def __init__(self) -> None:
        self._m: dict = {}
        self._built = None

    def reset_scene(self) -> None:
        self._m, self._built = {}, None

    def _geometry(self, p):
        return engine_profile(p.D_t * 1e-3, p.eps, p.L_star, p.contraction)

    def _values(self, p, r, frame) -> dict:
        raise NotImplementedError

    def _clim(self, p, r) -> tuple[float, float]:
        return 0.0, 1.0

    def build_scene(self, vp, p, r=None) -> None:
        pl = vp.plotter
        pl.clear()
        th = thetas(vp.cutaway)
        x, rr, _ = self._geometry(p)
        eng = rev_grid(x, rr, th)
        eng["v"] = broadcast_scalar(np.zeros_like(x), len(th))
        lo, hi = self._clim(p, r)
        a_eng = pl.add_mesh(eng, scalars="v", cmap=self.cmap, clim=(lo, hi), show_scalar_bar=False, name="engine",
                            smooth_shading=True)
        Rc = rr[0]
        Lt = max(2.5 * Rc, 0.4)
        x0 = x[0] - 0.25 * Lt
        tanks = {}
        for name, y, col in (("tank_o", 2.2 * Rc, "#ff9d3c"), ("tank_f", -2.2 * Rc, "#5cc97a")):
            cyl = pv.Cylinder(center=(x0 - 0.5 * Lt, y, 0.0), direction=(1, 0, 0), radius=0.9 * Rc, height=Lt)
            cyl["v"] = np.zeros(cyl.n_points)
            tanks[name] = cyl
            pl.add_mesh(cyl, scalars="v", cmap=self.cmap, clim=(lo, hi), show_scalar_bar=False, name=name, opacity=0.85)
            line = pv.Line((x0, y, 0.0), (x[0] - 0.05, 0.4 * y, 0.0), resolution=6).tube(radius=0.07 * Rc)
            line["v"] = np.zeros(line.n_points)
            tanks[name.replace("tank", "line")] = line
            pl.add_mesh(line, scalars="v", cmap=self.cmap, clim=(lo, hi), show_scalar_bar=False, name=name.replace("tank", "line"))
        xe, re_ = x[-1], rr[-1]
        lines = [pv.Line((xe, 0.7 * re_ * np.cos(2 * np.pi * k / 8), 0.7 * re_ * np.sin(2 * np.pi * k / 8)),
                         (xe + 3.0 * re_, 2.0 * re_ * np.cos(2 * np.pi * k / 8), 2.0 * re_ * np.sin(2 * np.pi * k / 8)))
                 for k in range(8)]
        plume = lines[0].merge(lines[1:])
        act = pl.add_mesh(plume, color="#ff5a3c", opacity=0.45, line_width=2, name="plume")
        self._m = dict(eng=eng, th=th, tanks=tanks, plume=act, n=len(x), cut=vp.cutaway, clim=(lo, hi), actor=a_eng)
        vp.reset_view(zoom=1.1)

    def update_scene(self, vp, p, r, frame) -> None:
        if not self._m or self._m["cut"] != vp.cutaway:
            self.build_scene(vp, p, r)
        m = self._m
        v = self._values(p, r, frame)
        lo, hi = self._clim(p, r)
        m["eng"]["v"] = broadcast_scalar(np.full(m["n"], v.get("engine", 0.0)), len(m["th"]))
        for k in ("tank_o", "tank_f", "line_o", "line_f"):
            if k in m["tanks"]:
                m["tanks"][k]["v"] = np.full(m["tanks"][k].n_points, v.get(k, v.get("engine", 0.0)))
        m["actor"].mapper.scalar_range = (lo, hi)
        m["plume"].GetProperty().SetOpacity(0.5 if v.get("thrust", 0.0) > 0 else 0.0)
        vp.set_colorbar(self.cmap, lo, hi, self.value_title)
        vp.render()

    def _runs_overlay(self, compare, r, ch, title, yl):
        runs = compare or [("current", r)]
        cs = [Curve(rr.t, rr.channels[ch], RUN_COLORS[i % len(RUN_COLORS)], name) for i, (name, rr) in enumerate(runs)]
        return PlotSpec(title, TT, yl, cs, cursor=False)


# ====================================================================================== feed network transient (Run)
class FeedTransientModel(LiquidRunBase):
    key = "liquid.feed"
    title = "Feed network transient"
    Params = FeedParams
    fields = [r"$P$ (MPa)"]

    def __init__(self) -> None:
        super().__init__()
        self.sim = FeedTransient()

    def _clim(self, p, r):
        return 0.0, float(p.p_tank) * 1.1

    def _values(self, p, r, frame):
        if r is None or frame is None:
            return dict(engine=0.1, tank_o=p.p_tank, tank_f=p.p_tank, line_o=p.p_tank, line_f=p.p_tank)
        c = r.channels
        return dict(engine=c["Pc"][frame], tank_o=c["ptank_o"][frame], tank_f=c["ptank_f"][frame],
                    line_o=c["p1_o"][frame], line_f=c["p1_f"][frame], thrust=c["F"][frame])

    def readout(self, p, r, frame) -> list[str]:
        if r is None or frame is None:
            return [r"$P_c$ = – (MPa)", r"$F$ = – (kN)", r"$\frac{O}{F}$ = –", r"$p_{line}$ = – (MPa)"]
        c = r.channels
        of = c["OF"][frame]
        return [rf"$P_c$ = {c['Pc'][frame]:.2f} (MPa)", rf"$F$ = {c['F'][frame]:.2f} (kN)",
                rf"$\frac{{O}}{{F}}$ = {'–' if not np.isfinite(of) else f'{of:.2f}'}", rf"$p_{{line}}$ = {c['p1_o'][frame]:.2f} (MPa)"]

    def caption(self, p, r, frame) -> str:
        if r is None or frame is None:
            return r"press Run: start-up, steady burn and shutdown with valve timing and line dynamics"
        c = r.channels
        return (rf"$t$ = {r.t[frame]:.3f} (s) · valves {100 * c['valve_o'][frame]:.0f} % / {100 * c['valve_f'][frame]:.0f} % · "
                rf"tank {c['ptank_o'][frame]:.2f} (MPa)")

    def summary_line(self, p, r) -> str:
        s = r.meta
        return (rf"$P_{{c,ss}}$ = {s['Pc_ss_MPa']:.2f} (MPa) · overshoot {100 * s['overshoot']:.1f} % · surge {s['surge_o_MPa']:.2f} / "
                rf"{s['surge_f_MPa']:.2f} (MPa) · $I_t$ = {s['I_t_Ns'] / 1e3:.1f} (kN·s)")

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        if r is None:
            e = lambda title, yl: PlotSpec(title, TT, yl, [], cursor=True)
            return [TabSpec("Start-up & shutdown", [e("Chamber pressure", r"$P_c$ (MPa)"), e("Thrust", r"$F$ (kN)"),
                                                    e("Flows", r"$\dot m$ (kg·s⁻¹)")]), TabSpec("Lines", []), TabSpec("Compare", [])]
        c, t = r.channels, r.t
        cur = lambda y, col, label="", style="solid": Curve(t, y, col, label, style)
        ss = r.meta["Pc_ss_MPa"]
        main = [PlotSpec("Chamber pressure", TT, r"$P_c$ (MPa)", [cur(c["Pc"], "pressure"),
                                                                  cur(np.full_like(t, ss), "dim", "steady state", "dash")], cursor=True),
                PlotSpec("Thrust", TT, r"$F$ (kN)", [cur(c["F"], "thrust")], cursor=True),
                PlotSpec("Injector flows", TT, r"$\dot m$ (kg·s⁻¹)", [cur(c["mdot_o"], "oxidizer", "oxidizer"),
                                                                    cur(c["mdot_f"], "fuel", "fuel")], cursor=True)]
        lines = [PlotSpec("Valve-inlet pressure", TT, r"$p_1$ (MPa)", [cur(c["p1_o"], "oxidizer", "oxidizer"),
                                                                    cur(c["p1_f"], "fuel", "fuel")], cursor=True),
                 PlotSpec("Manifold pressure", TT, r"$p_2$ (MPa)", [cur(c["p2_o"], "oxidizer", "oxidizer"),
                                                                  cur(c["p2_f"], "fuel", "fuel")], cursor=True),
                 PlotSpec("Valve opening", TT, r"$f_v$ (–)", [cur(c["valve_o"], "oxidizer", "oxidizer"),
                                                              cur(c["valve_f"], "fuel", "fuel")], cursor=True)]
        cmp_ = [self._runs_overlay(compare, r, "Pc", "Chamber pressure", r"$P_c$ (MPa)"),
                self._runs_overlay(compare, r, "F", "Thrust", r"$F$ (kN)"),
                self._runs_overlay(compare, r, "p1_o", "Valve-inlet pressure", r"$p_1$ (MPa)")]
        return [TabSpec("Start-up & shutdown", main), TabSpec("Lines", lines), TabSpec("Compare", cmp_)]

    def export_data(self, p, r) -> dict[str, np.ndarray]:
        return {"t_s": r.t, **{k: v for k, v in r.channels.items()}}


# ====================================================================================== pump-fed cycles (Run)
class CycleModel(LiquidRunBase):
    key = "liquid.cycle"
    title = "Pump-fed cycles"
    Params = CycleParams
    fields = [r"$P_c$ (MPa)"]

    def __init__(self) -> None:
        super().__init__()
        self.sim = PumpFedCycle()

    def _clim(self, p, r):
        return 0.0, float(p.Pc) * 1.1

    def _values(self, p, r, frame):
        if r is None or frame is None:
            return dict(engine=0.1)
        c = r.channels
        return dict(engine=c["Pc"][frame], tank_o=c["p_o"][frame], tank_f=c["p_f"][frame], line_o=c["p_o"][frame],
                    line_f=c["p_f"][frame], thrust=c["F"][frame])

    def readout(self, p, r, frame) -> list[str]:
        if r is None or frame is None:
            return [r"$N$ = – (rpm)", r"$P_c$ = – (MPa)", r"$F$ = – (kN)", r"$P_t$ = – (kW)"]
        c = r.channels
        return [rf"$N$ = {c['N'][frame]:.0f} (rpm)", rf"$P_c$ = {c['Pc'][frame]:.2f} (MPa)", rf"$F$ = {c['F'][frame]:.2f} (kN)",
                rf"$P_t$ = {c['P_t'][frame]:.0f} (kW)"]

    def caption(self, p, r, frame) -> str:
        if r is None or frame is None:
            return r"press Run: starter spin-up, drive-valve opening and chamber build-up"
        c = r.channels
        return rf"$t$ = {r.t[frame]:.2f} (s) · {100 * c['N_ratio'][frame]:.0f} % of design speed · {p.cycle} cycle"

    def summary_line(self, p, r) -> str:
        s = r.meta
        return (rf"steady $N$ = {s['N_ss']:.0f} (rpm), $P_c$ = {s['Pc_ss']:.2f} (MPa), $F$ = {s['F_ss']:.1f} (kN) · "
                rf"$T_t$ {s['T_req']:.0f} (K) · bleed {100 * s['bleed_frac']:.1f} %")

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        if r is None:
            e = lambda title, yl: PlotSpec(title, TT, yl, [], cursor=True)
            return [TabSpec("Spool-up", [e("Shaft speed", r"$N$ (rpm)"), e("Chamber pressure", r"$P_c$ (MPa)"),
                                          e("Thrust", r"$F$ (kN)")]), TabSpec("Power balance", []), TabSpec("Compare", [])]
        c, t = r.channels, r.t
        cur = lambda y, col, label="", style="solid": Curve(t, y, col, label, style)
        s = r.meta
        spool = [PlotSpec("Shaft speed", TT, r"$N$ (rpm)", [cur(c["N"], "thrust"), cur(np.full_like(t, s["N_ss"]), "dim", "steady", "dash")],
                          cursor=True),
                 PlotSpec("Chamber pressure", TT, r"$P_c$ (MPa)", [cur(c["Pc"], "pressure"), cur(np.full_like(t, s["Pc_ss"]), "dim",
                                                                                                   "steady", "dash")], cursor=True),
                 PlotSpec("Thrust", TT, r"$F$ (kN)", [cur(c["F"], "hot")], cursor=True)]
        power = [PlotSpec("Turbine and pump power", TT, r"$P$ (kW)", [cur(c["P_t"], "hot", "turbine"), cur(c["P_pump"], "thrust", "pumps")],
                          cursor=True),
                 PlotSpec("Net shaft power", TT, r"$P_{net}$ (kW)", [cur(c["P_net"], "fuel")], cursor=True),
                 PlotSpec("Propellant flows", TT, r"$\dot m$ (kg·s⁻¹)", [cur(c["m_o"], "oxidizer", "oxidizer"), cur(c["m_f"], "fuel", "fuel"),
                                                                       cur(c["m_gg"], "dim", "turbine drive", "dash")], cursor=True)]
        cmp_ = [self._runs_overlay(compare, r, "N", "Shaft speed", r"$N$ (rpm)"),
                self._runs_overlay(compare, r, "Pc", "Chamber pressure", r"$P_c$ (MPa)"),
                self._runs_overlay(compare, r, "F", "Thrust", r"$F$ (kN)")]
        return [TabSpec("Spool-up", spool), TabSpec("Power balance", power), TabSpec("Compare", cmp_)]

    def export_data(self, p, r) -> dict[str, np.ndarray]:
        return {"t_s": r.t, **{k: v for k, v in r.channels.items()}}


# ====================================================================================== chugging (Run)
class ChuggingModel(LiquidRunBase):
    key = "liquid.chugging"
    title = "Chugging & L* stability"
    Params = ChuggingParams
    fields = [r"$\pi=\frac{\delta p_c}{p_c}$ (%)"]
    cmap = "coolwarm"
    value_title = r"$\pi$ (%)"

    def __init__(self) -> None:
        super().__init__()
        self.sim = Chugging()

    def _clim(self, p, r):
        a = 1.0 if r is None else max(float(np.max(np.abs(r.channels["pi"]))), 1e-3)
        return -a, a

    def _values(self, p, r, frame):
        if r is None or frame is None:
            return dict(engine=0.0)
        v = float(r.channels["pi"][frame])
        return dict(engine=v, tank_o=0.0, tank_f=0.0, line_o=float(r.channels["m"][frame]), line_f=float(r.channels["m"][frame]),
                    thrust=1.0)

    def readout(self, p, r, frame) -> list[str]:
        if r is None or frame is None:
            return [r"$\pi$ = – (%)", r"gain margin = –", r"$f_{pc}$ = – (Hz)", r"$\Delta p_{min}$ = – (%)"]
        m = r.meta
        return [rf"$\pi$ = {r.channels['pi'][frame]:.3f} (%)", rf"gain margin = {m['gain_margin']:.2f}",
                rf"$f_{{pc}}$ = {m['f_pc']:.0f} (Hz)", rf"$\Delta p_{{min}}$ = {100 * m['dp_min_frac']:.1f} (%)"]

    def caption(self, p, r, frame) -> str:
        if r is None or frame is None:
            return r"press Run: Nyquist analysis and delay-differential time trace"
        m = r.meta
        verdict = "STABLE" if m["stable"] else "UNSTABLE"
        return (rf"$t$ = {r.t[frame] * 1e3:.1f} (ms) · {verdict} · $\kappa$ = {m['kappa']:.2f}, $\theta_g$ = {m['theta_g_ms']:.2f}, "
                rf"$\tau_c$ = {m['tau_c_ms']:.2f} (ms)")

    def summary_line(self, p, r) -> str:
        m = r.meta
        return (rf"{'stable' if m['stable'] else 'UNSTABLE'} · gain margin {m['gain_margin']:.2f} · growth {m['growth']:.0f} (1/s) · "
                rf"min injector drop {100 * m['dp_min_frac']:.1f} (%) of $P_c$")

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        if r is None:
            return [TabSpec("Stability", [PlotSpec("Nyquist plot", r"$\mathrm{Re}\,G$ (–)", r"$\mathrm{Im}\,G$ (–)", []),
                                          PlotSpec("Chamber pressure", r"$t$ (ms)", r"$\pi$ (%)", [], cursor=True),
                                          PlotSpec("Injector flow", r"$t$ (ms)", r"$m$ (%)", [], cursor=True)]),
                    TabSpec("Margins", [])]
        a, c, tm = r.arrays, r.channels, r.t * 1e3
        w = a["nyq_w"]
        theta = np.linspace(0, 2 * np.pi, 120)
        nyq = PlotSpec("Nyquist plot", r"$\mathrm{Re}\,G$ (–)", r"$\mathrm{Im}\,G$ (–)",
                       [Curve(a["nyq_re"], a["nyq_im"], "pressure", "G(iω)"),
                        Curve(np.cos(theta) - 1.0 + 1.0, np.sin(theta), "dim", "unit circle", "dash")],
                       markers=[(-1.0, 0.0, "hot")])
        pi = PlotSpec("Chamber pressure", r"$t$ (ms)", r"$\pi$ (%)", [Curve(tm, c["pi"], "pressure")], cursor=False)
        mm = PlotSpec("Injector flow", r"$t$ (ms)", r"$m$ (%)", [Curve(tm, c["m"], "oxidizer")], cursor=False)
        # margins versus L* and tau_c for the current feed system
        tmv = p.terms()
        Ls = np.linspace(0.3, 3.0, 28)
        dpmin = []
        for L in Ls:
            theta_g = L * tmv["cs"] / tmv["RT"]
            mg = chg.stability_margins(tmv["kappa"], tmv["tau_c"], theta_g, tmv["tau_f"])
            dpmin.append(100.0 * p.Pc * 1e6 / (2.0 * mg["kappa_crit"]) / (p.Pc * 1e6) if np.isfinite(mg["kappa_crit"]) else 0.0)
        taus = np.linspace(0.2e-3, 10e-3, 28)
        gm_tau = [chg.stability_margins(tmv["kappa"], tc, tmv["theta"], tmv["tau_f"])["gain_margin"] for tc in taus]
        marg = [PlotSpec("Minimum stable injector drop", r"$L^*$ (m)", r"$\frac{\Delta p_{min}}{P_c}$ (%)",
                         [Curve(Ls, dpmin, "hot"), Curve(Ls, np.full_like(Ls, p.dp_inj), "dim", "design", "dash")],
                         markers=[(p.L_star, 100 * r.meta["dp_min_frac"], "text")]),
                PlotSpec("Gain margin vs combustion delay", r"$\tau_c$ (ms)", r"gain margin (–)",
                         [Curve(taus * 1e3, np.minimum(gm_tau, 20), "pressure"), Curve(taus * 1e3, np.ones_like(taus), "dim", "limit", "dash")],
                         markers=[(p.tau_c, min(r.meta["gain_margin"], 20), "hot")]),
                PlotSpec("Open-loop magnitude", r"$f$ (Hz)", r"$|G|$ (–)", [Curve(w / (2 * np.pi), np.hypot(a["nyq_re"], a["nyq_im"]), "thrust"),
                                                                      Curve(w / (2 * np.pi), np.ones_like(w), "dim", "unity", "dash")])]
        return [TabSpec("Stability", [nyq, pi, mm]), TabSpec("Margins", marg)]

    def export_data(self, p, r) -> dict[str, np.ndarray]:
        return {"t_s": r.t, "pi_pct": r.channels["pi"], "m_pct": r.channels["m"], "nyq_w": r.arrays["nyq_w"],
                "nyq_re": r.arrays["nyq_re"], "nyq_im": r.arrays["nyq_im"]}

