"""Solid module: grain designer and burn-rate tool (Live); 3-D level-set burnback and internal ballistics (Run)."""
from __future__ import annotations

import numpy as np
import pyvista as pv

from ..core.ballistics import solid as sol
from ..core.result import Curve, PlotSpec, TabSpec
from ..core.schema import P, ParamModel
from ..sims.solid_burn import (Solid3DParams, SolidBallistics, SolidBallisticsParams, SolidBurnback3D,
                               SolidGrainFields, SolidPropFields)
from ..ui.theme import CMAPS, QUANTITY
from .base import RUN_COLORS, ModelController
from .geometry import rev_grid
from .hybrid_burn import HybridBurnModel
from .hybrid_live import TubeLive, draw_section_slab

TT = r"$t$ (s)"
WEB = r"$\frac{w}{w_{max}}$ (%)"
FRAC_COLORS = ["#4ea1ff", "#5cc97a", "#e6c34a", "#ff9d3c", "#ff5a3c"]


# ====================================================================================== 2-D grain designer (Live)
class SolidGrainLiveParams(SolidGrainFields, SolidPropFields):
    D_t: float = P(36.0, symbol=r"D_t", desc="throat diameter", units="mm", dims="L", lo=10.0, hi=120.0, step=0.5,
                   group="Nozzle")
    burn: float = P(35.0, symbol=r"\frac{w}{w_{max}}", desc="web burned (percent of the web)", units="%", dims="1",
                    lo=0.0, hi=100.0, step=0.5, group="Burnback")


class SolidGrainLiveModel(ModelController):
    key = "solid.grain"
    title = "Grain designer"
    module = "Solid"
    mode = "live"
    Params = SolidGrainLiveParams
    fields = [r"burn distance $\frac{d}{w_{max}}$ (–)"]

    def __init__(self) -> None:
        self._built = None

    def evaluate(self, p: SolidGrainLiveParams) -> dict:
        b = p.section()
        prop = p.prop(alpha=0.0)
        At = np.pi * (p.D_t * 1e-3) ** 2 / 4.0
        tab = p.section_table()
        nseg, k = int(round(p.n_seg)), p.faces
        web = sol.segment_web(tab, p.L_g, k)                       # m: radial web, or the axial one if the ends win
        d = np.linspace(0.0, web, 61)
        Ab = sol.segment_burn_area(tab, p.L_g, d, nseg, k)
        Ab[-1] = 0.0
        Kn = Ab / At
        pc = np.array([sol.p_equilibrium(prop, kn) if kn > 0 else 0.0 for kn in Kn]) / 1e6
        rate = prop.rate(pc * 1e6) * 1e3
        rate[Kn <= 0] = 0.0
        perim = tab.perim(d) * 1e3
        area = tab.area(d) * 1e6
        Lr = np.maximum(p.L_g - k * d, 0.0)
        V0 = nseg * (tab.A_case - float(tab.A_p[0])) * p.L_g
        remain = nseg * np.maximum(tab.A_case - tab.area(d), 0.0) * Lr / max(V0, 1e-30)
        remain[-1] = 0.0
        dmm = d * 1e3
        d_now = p.burn / 100.0 * dmm[-1]
        pct = 100.0 * d / web
        ok = rate[:-1] > 0
        tb = float(np.trapezoid(1.0 / np.maximum(rate[:-1][ok], 1e-9), dmm[:-1][ok])) if ok.sum() > 1 else 0.0
        i = lambda y: float(np.interp(d_now, dmm, y))
        Kn0 = max(float(Kn[0]), 1e-12)
        A_free = float(tab.A_p[0]) * 1e6
        return dict(b=b, d=dmm, pct=pct, Kn=Kn, pc=pc, rate=rate, d_now=d_now, Pi=i(perim), perim=perim, area=area,
                    remain=remain, Ap=i(area), rem=i(remain), Kn_now=i(Kn), pc_now=i(pc), Ab=Ab,
                    load=1.0 - A_free / (tab.A_case * 1e6), prog=float(Kn.max() / Kn0), t_burn=tb, web=dmm[-1],
                    m_prop=prop.rho_p * V0, J0=(A_free * 1e-6 / At) if A_free > 0 else float("nan"),
                    note=sol.stability_note(prop.n_max))

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        x = r["pct"]
        mk = lambda y, c="hot": [(p.burn, float(np.interp(p.burn, x, y)), c)]
        sect = [PlotSpec("Burning area", WEB, r"$A_b$ (cm²)", [Curve(x, r["Ab"] * 1e4, "fuel")],
                         markers=mk(r["Ab"] * 1e4)),
                PlotSpec("Port area", WEB, r"$A_p$ (cm²)", [Curve(x, r["area"] / 100.0, "oxidizer")],
                         markers=mk(r["area"] / 100.0)),
                PlotSpec("Propellant remaining", WEB, r"$\frac{m_p}{m_{p,0}}$ (–)", [Curve(x, r["remain"], "thrust")],
                         markers=mk(r["remain"]))]
        bal = [PlotSpec("Burning-area ratio", WEB, r"$K_n=\frac{A_b}{A_t}$ (–)", [Curve(x, r["Kn"], "hot")],
                        markers=mk(r["Kn"])),
               PlotSpec("Equilibrium pressure", WEB, r"$P_c$ (MPa)", [Curve(x, r["pc"], "pressure")],
                        markers=mk(r["pc"])),
               PlotSpec("Burn rate at equilibrium", WEB, r"$r$ (mm·s⁻¹)", [Curve(x, r["rate"], "fuel")],
                        markers=mk(r["rate"]))]
        return [TabSpec("Section", sect), TabSpec("Ballistics", bal)]

    def readout(self, p, r, frame) -> list[str]:
        return [rf"$\frac{{w}}{{w_{{max}}}}$ = {p.burn:.0f} (%)", rf"$A_b$ = {np.interp(r['d_now'], r['d'], r['Ab']) * 1e4:.0f} (cm²)",
                rf"$K_n$ = {r['Kn_now']:.0f}", rf"$P_c$ = {r['pc_now']:.2f} (MPa)"]

    def caption(self, p, r, frame) -> str:
        j = "" if not np.isfinite(r["J0"]) else rf" · $\frac{{A_p}}{{A_t}}$ = {r['J0']:.2f}"
        return (rf"web {r['web']:.1f} (mm) · loading {100 * r['load']:.0f} % · {r['m_prop']:.2f} (kg) · "
                rf"$\frac{{K_{{n,max}}}}{{K_{{n,0}}}}$ = {r['prog']:.2f} · burn time ≈ {r['t_burn']:.2f} (s){j} · "
                rf"{r['note']}")

    def export_data(self, p, r) -> dict[str, np.ndarray]:
        return {"burn_distance_mm": r["d"], "port_area_mm2": r["area"], "perimeter_mm": r["perim"],
                "burning_area_mm2": r["Ab"] * 1e6, "remaining": r["remain"], "Kn": r["Kn"], "Pc_eq_MPa": r["pc"],
                "rate_mmps": r["rate"]}

    legend_labels = {"grain": "Propellant (burn distance)", "case": "Motor case"}

    def build_scene(self, vp, p, r=None) -> None:
        vp.plotter.clear()
        vp._slab_view_set = False
        self._built = True

    def update_scene(self, vp, p, r, frame) -> None:
        b = r["b"]
        draw_section_slab(vp, np.maximum(b.sdf, 0.0), b.inside, b.h, b.web, min(r["d_now"], b.web), p.R_out, L=p.L_g)


# ====================================================================================== burn rate & equilibrium (Live)
class BurnRateLiveParams(SolidPropFields):
    Kn: float = P(200.0, symbol=r"K_n=\frac{A_b}{A_t}", desc="burning-area to throat-area ratio", units="", dims="1",
                  lo=40.0, hi=600.0, step=1.0, group="Motor")


class BurnRateLiveModel(TubeLive):
    key = "solid.burnrate"
    title = "Burn rate & K_n"
    module = "Solid"
    Params = BurnRateLiveParams
    fields = [r"$P_c$ (MPa)"]
    cmap = "viridis"

    def evaluate(self, p: BurnRateLiveParams) -> dict:
        prop = p.prop(alpha=0.0)
        pg = np.linspace(0.5e6, 15e6, 80)
        dT = (-30.0, 0.0, 30.0)
        curves = {d: sol.SolidProp(**{**prop.__dict__, "Ti": prop.Ti + d}).rate(pg) * 1e3 for d in dT}
        Kg = np.linspace(40.0, 600.0, 60)
        Tg = np.linspace(233.0, 343.0, 40)
        ng = np.linspace(0.1, 0.95, 50)
        peq = lambda pr, k: sol.p_equilibrium(pr, k) / 1e6
        pc = peq(prop, p.Kn)
        return dict(pg=pg, curves=curves, Kg=Kg, pc_K=np.array([peq(prop, k) for k in Kg]),
                    Tg=Tg, pc_T=np.array([peq(sol.SolidProp(**{**prop.__dict__, "Ti": t}), p.Kn) for t in Tg]),
                    a_T=np.array([sol.SolidProp(**{**prop.__dict__, "Ti": t}).a_eff * 1e6**prop.n * 1e3 for t in Tg]),
                    ng=ng, pc_n=np.array([peq(sol.SolidProp(**{**prop.__dict__, "n": n_}), p.Kn) for n_ in ng]),
                    pc=pc, r_eq=float(prop.rate(pc * 1e6)) * 1e3, pi_K=prop.sigma_p / (1.0 - prop.n),
                    note=sol.stability_note(p.n), mass_balance=float(
                        (prop.rho_p * prop.a_eff * (pc * 1e6) ** prop.n * p.Kn - pc * 1e6 / prop.cstar)
                        / (pc * 1e6 / prop.cstar)))

    def _profile(self, p, r):
        x = np.linspace(0.0, self.L_tube, 40)
        return x, np.full_like(x, self.R_in), np.full_like(x, r["pc"]), r"$P_c$ (MPa)"

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        lab = {-30.0: "T_i − 30 K", 0.0: "T_i", 30.0: "T_i + 30 K"}
        col = {-30.0: "thrust", 0.0: "fuel", 30.0: "hot"}
        law = [PlotSpec("Burn rate vs pressure", r"$P_c$ (MPa)", r"$r$ (mm·s⁻¹)",
                        [Curve(r["pg"] / 1e6, r["curves"][d], col[d], lab[d]) for d in r["curves"]],
                        markers=[(r["pc"], r["r_eq"], "text")]),
               PlotSpec("Equilibrium pressure vs K_n", r"$K_n$ (–)", r"$P_c$ (MPa)", [Curve(r["Kg"], r["pc_K"], "pressure")],
                        markers=[(p.Kn, r["pc"], "hot")]),
               PlotSpec("Equilibrium vs exponent", r"$n$ (–)", r"$P_c$ (MPa)", [Curve(r["ng"], r["pc_n"], "oxidizer")],
                        markers=[(p.n, r["pc"], "hot")])]
        temp = [PlotSpec("Burn-rate coefficient vs temperature", r"$T_i$ (K)", r"$r_{1\,MPa}$ (mm·s⁻¹)",
                         [Curve(r["Tg"], r["a_T"], "fuel")], markers=[(p.T_i, float(np.interp(p.T_i, r["Tg"], r["a_T"])), "hot")]),
                PlotSpec("Pressure vs grain temperature", r"$T_i$ (K)", r"$P_c$ (MPa)", [Curve(r["Tg"], r["pc_T"], "hot")],
                         markers=[(p.T_i, r["pc"], "text")]),
                PlotSpec("Equilibrium pressure vs K_n (zoom)", r"$K_n$ (–)", r"$P_c$ (MPa)",
                         [Curve(r["Kg"], r["pc_K"], "pressure")], markers=[(p.Kn, r["pc"], "hot")])]
        return [TabSpec("Burn rate", law), TabSpec("Temperature", temp)]

    def readout(self, p, r, frame) -> list[str]:
        return [rf"$P_c$ = {r['pc']:.2f} (MPa)", rf"$r$ = {r['r_eq']:.2f} (mm·s⁻¹)", rf"$K_n$ = {p.Kn:.0f}",
                rf"$\pi_K$ = {100 * r['pi_K']:.2f} (%·K⁻¹)"]

    def caption(self, p, r, frame) -> str:
        return (rf"$P_c=(a\rho_pc^*K_n)^{{\frac{{1}}{{1-n}}}}$ · mass-balance residual {r['mass_balance']:.1e} · {r['note']}")

    def export_data(self, p, r) -> dict[str, np.ndarray]:
        return {"p_MPa": r["pg"] / 1e6, "r_mmps": r["curves"][0.0], "Kn": r["Kg"], "Pc_eq_MPa": r["pc_K"]}


# ====================================================================================== internal ballistics (Run)
FIELDS_SOLID = [("Pc", "viridis", r"$P_c$ (MPa)", (0.0, 8.0)), ("rdot", "cividis", r"$r$ (mm·s⁻¹)", (0.0, 15.0)),
                ("erosive", "inferno", r"$\frac{r}{r_0}$ (–)", (1.0, 1.5))]


class SolidBallisticsModel(HybridBurnModel):
    key = "solid.ballistics"
    title = "Internal ballistics"
    module = "Solid"
    mode = "run"
    Params = SolidBallisticsParams
    FIELDS = FIELDS_SOLID
    fields = [f[2] for f in FIELDS_SOLID]

    def __init__(self) -> None:
        super().__init__()
        self.sim = SolidBallistics()

    # ---------------------------------------------------------------- 3D hooks
    def _stations(self, r):
        xc = np.asarray(r.meta["x_cells"], float)
        return xc, len(xc)

    def _phase(self, p, r) -> np.ndarray:
        if self._cum_for != id(r):
            dt = np.diff(r.t, prepend=r.t[0])
            self._cum = np.cumsum(r.channels["mdot"] * dt) / max(float(np.nanmax(r.channels["mdot"]) * r.t[-1]), 1e-9)
            self._cum_for = id(r)
        return self._cum

    def _radii(self, si, r, frame) -> np.ndarray:
        """Equivalent circular port radius (the 3-D scene is an axisymmetric stand-in for the real section)."""
        if r is None or frame is None:
            return np.full_like(self._xg, si.R0)
        xc, n = self._stations(r)
        w = np.array([r.channels[f"w_{i:02d}"][frame] for i in range(n)]) * 1e-3
        Req = np.sqrt(si.table.area(w) / np.pi)
        return np.interp(self._xg, xc, np.minimum(Req, si.R_out * 0.999))

    def _field_x(self, si, r, frame, ch: str) -> np.ndarray:
        if r is None or frame is None:
            return np.zeros_like(self._xg)
        xc, n = self._stations(r)
        if ch == "Pc":
            return np.full_like(self._xg, float(r.channels["Pc"][frame]))
        rd = np.array([r.channels[f"rd_{i:02d}"][frame] for i in range(n)])
        if ch == "rdot":
            return np.interp(self._xg, xc, rd)
        r0 = si.prop.rate(max(float(r.channels["Pc"][frame]) * 1e6, 1.0)) * 1e3
        return np.interp(self._xg, xc, np.where(rd > 0, rd / max(r0, 1e-9), 1.0))

    def _range(self, r, ch, default):
        if r is None:
            return default
        n = len(r.meta["x_cells"])
        if ch == "Pc":
            return 0.0, float(np.nanmax(r.channels["Pc"]))
        if ch == "rdot":
            return 0.0, float(max(np.max([r.channels[f"rd_{i:02d}"].max() for i in range(n)]), 1e-6))
        return 1.0, float(max(np.nanmax(r.channels["erosive_aft"]), 1.01))

    # ---------------------------------------------------------------- text
    def readout(self, p, r, frame) -> list[str]:
        if r is None or frame is None:
            return [r"$P_c$ = – (MPa)", r"$F$ = – (kN)", r"$K_n$ = –", r"$r$ = – (mm·s⁻¹)"]
        c = r.channels
        return [rf"$P_c$ = {c['Pc'][frame]:.2f} (MPa)", rf"$F$ = {c['F'][frame]:.2f} (kN)",
                rf"$K_n$ = {c['Kn'][frame]:.0f}", rf"$r$ = {c['rdot'][frame]:.2f} (mm·s⁻¹)"]

    def caption(self, p, r, frame) -> str:
        if r is None or frame is None:
            return r"press Run to simulate · section shown as its equivalent circular port"
        c = r.channels
        return (rf"$t$ = {r.t[frame]:.2f} (s) · cells lit {100 * c['lit_frac'][frame]:.0f} % · aft erosion "
                rf"$\frac{{r}}{{r_0}}$ = {c['erosive_aft'][frame]:.2f} · section → equivalent circular port")

    def summary_line(self, p, r) -> str:
        s = r.meta
        return (rf"$I_t$ = {s['I_t_Ns'] / 1e3:.2f} (kN·s) ({s['impulse_class']}),  $\bar{{I}}_{{sp}}$ = "
                rf"{s['Isp_avg']:.0f} (s),  $P_{{c,max}}$ = {s['Pc_max_MPa']:.2f} (MPa)")

    # ---------------------------------------------------------------- plots
    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        if r is None:
            e = lambda title, yl: PlotSpec(title, TT, yl, [], cursor=True)
            return [TabSpec("Time histories", [e("Chamber pressure", r"$P_c$ (MPa)"), e("Thrust", r"$F$ (kN)"),
                                               e("Burning-area ratio", r"$K_n$ (–)")]),
                    TabSpec("Burn rate", []), TabSpec("Nozzle & Isp", []), TabSpec("Port evolution", []),
                    TabSpec("Ignition & tail-off", []), TabSpec("Compare", [])]
        c, t = r.channels, r.t
        n = len(r.meta["x_cells"])
        cur = lambda y, col, label="", style="solid": Curve(t, y, col, label, style)
        hist = [PlotSpec("Chamber pressure", TT, r"$P_c$ (MPa)",
                         [cur(c["Pc"], "pressure"), Curve(t, np.full_like(t, r.meta["Pc_eq_MPa"]), "dim", "equilibrium",
                                                          "dash")], cursor=True),
                PlotSpec("Thrust", TT, r"$F$ (kN)", [cur(c["F"], "thrust")], cursor=True),
                PlotSpec("Burning-area ratio", TT, r"$K_n$ (–)", [cur(c["Kn"], "hot")], cursor=True)]
        rate = [PlotSpec("Burn rate", TT, r"$r$ (mm·s⁻¹)",
                         [cur(c["rdot_head"], "oxidizer", "head"), cur(c["rdot_aft"], "fuel", "aft"),
                          cur(c["rdot"], "text", "mean", "dash")], cursor=True),
                PlotSpec("Erosive enhancement (aft)", TT, r"$\frac{r}{r_0}$ (–)", [cur(c["erosive_aft"], "hot")],
                         cursor=True),
                PlotSpec("Mass flow", TT, r"$\dot m$ (kg·s⁻¹)",
                         [cur(c["mdot"], "thrust", "nozzle"), cur(c["mdot_gen"], "fuel", "generated", "dash")],
                         cursor=True)]
        noz = [PlotSpec("Specific impulse", TT, r"$I_{sp}$ (s)", [cur(c["Isp"], "thrust")], cursor=True),
               PlotSpec("Thrust coefficient", TT, r"$C_F$ (–)", [cur(c["CF"], "oxidizer")], cursor=True),
               PlotSpec("Throat and port", TT, r"$D_t$ (mm), $\frac{A_{p,aft}}{A_t}$ (–)",
                        [cur(c["Dt"], "hot", "throat diameter"), cur(c["J_aft"], "fuel", "aft port / throat", "dash")],
                        cursor=True)]
        x = np.array(r.meta["x_cells"]) * 1e3
        idx = [int(round(f * (r.n - 1))) for f in (0.0, 0.2, 0.4, 0.6, 0.8)]
        prof = lambda pre, fmin=None: [Curve(x, np.array([c[f"{pre}_{k:02d}"][i] for k in range(n)]), col, f"t = {t[i]:.1f} s")
                                       for i, col in zip(idx, FRAC_COLORS)]
        xl = r"$x$ (mm)"
        port = [PlotSpec("Burn distance along the grain", xl, r"$w$ (mm)", prof("w")),
                PlotSpec("Local burn rate", xl, r"$r$ (mm·s⁻¹)", prof("rd")[1:]),
                PlotSpec("Core mass flux", xl, r"$G$ (kg·m⁻²·s⁻¹)", prof("G")[1:])]
        t_end = float(t[-1])
        early = t <= min(0.6, t_end)
        ign = [PlotSpec("Ignition transient", TT, r"$P_c$ (MPa)", [Curve(t[early], c["Pc"][early], "pressure")],
                        cursor=True),
               PlotSpec("Cells burning", TT, r"$f_{lit}$ (–)", [cur(c["lit_frac"], "fuel")], cursor=True),
               PlotSpec("Tail-off", TT, r"$P_c$ (MPa)", [cur(c["Pc"], "pressure")], cursor=True)]
        runs = compare or [("current", r)]

        def overlay(ch, title, yl):
            cs = [Curve(rr.t, rr.channels[ch], RUN_COLORS[i % len(RUN_COLORS)], name) for i, (name, rr) in enumerate(runs)]
            return PlotSpec(title, TT, yl, cs, cursor=False)
        cmp_ = [overlay("Pc", "Chamber pressure", r"$P_c$ (MPa)"), overlay("F", "Thrust", r"$F$ (kN)"),
                overlay("Kn", "Burning-area ratio", r"$K_n$ (–)")]
        return [TabSpec("Time histories", hist), TabSpec("Burn rate", rate), TabSpec("Nozzle & Isp", noz),
                TabSpec("Port evolution", port),
                TabSpec("Ignition & tail-off", ign), TabSpec("Compare", cmp_)]

    def export_eng(self, p, r) -> str:
        """The predicted thrust curve as a RASP ``.eng`` file (motor diameter = grain diameter)."""
        L = p.L_g * max(int(round(p.n_seg)), 1) if not p.stack_json.strip() else float(r.meta.get("L_total", p.L_g))
        return sol.eng_text(r.t, r.channels["F"] * 1e3, "PropSim-" + str(r.meta["impulse_class"]).split()[0],
                            2.0 * p.R_out, L * 1e3, r.meta["m_prop"])

    def export_data(self, p, r) -> dict[str, np.ndarray]:
        names = {"Pc": "Pc_MPa", "F": "F_kN", "Kn": "Kn", "mdot": "mdot_kgps", "mdot_gen": "mdot_gen_kgps",
                 "rdot": "rdot_mean_mmps", "rdot_head": "rdot_head_mmps", "rdot_aft": "rdot_aft_mmps",
                 "erosive_aft": "erosive_ratio_aft", "lit_frac": "lit_fraction", "Ab": "Ab_m2", "Vc": "Vc_m3",
                 "Isp": "Isp_s", "CF": "CF", "Dt": "Dt_mm", "G_max": "G_max_kgpm2s", "F_vac": "F_vac_kN"}
        return {"t_s": r.t, **{v: r.channels[k] for k, v in names.items()}}


# ====================================================================================== 3-D level-set burnback (Run)
class Burnback3DModel(ModelController):
    key = "solid.burnback3d"
    title = "3D burnback"
    module = "Solid"
    mode = "run"
    Params = Solid3DParams
    fields = [r"distance from the axis $\rho$ (mm)"]

    def __init__(self) -> None:
        self.sim = SolidBurnback3D()
        self._m: dict = {}

    def reset_scene(self) -> None:
        self._m = {}

    def build_scene(self, vp, p, r=None) -> None:
        pl = vp.plotter
        pl.clear()
        L, Ro = p.L_g, p.R_out * 1e-3
        th = np.radians(np.linspace(0.0, 360.0 - (vp.cutaway or 0), 72))
        pl.add_mesh(rev_grid(np.array([0.0, L]), np.full(2, Ro + 0.004), th), color="#6b768a", opacity=0.22,
                    name="case", smooth_shading=True)
        self._m = {"cut": vp.cutaway}
        vp.reset_view(zoom=1.1)

    def _surface(self, r, frame) -> pv.PolyData | None:
        from skimage import measure
        phi = r.arrays["phi_snaps"][frame].astype(np.float32)
        if not (phi.min() < 0.0 < phi.max()):
            return None
        hs = float(r.arrays["snap_h"][0])
        v, f, _, _ = measure.marching_cubes(phi, 0.0, spacing=(hs, hs, hs))
        ny, nz = phi.shape[1], phi.shape[2]
        # to metres, axis along x, centred like the grid (cell-centred, padded)
        v = v * 1e-3
        v[:, 1] -= (ny - 1) / 2 * hs * 1e-3
        v[:, 2] -= (nz - 1) / 2 * hs * 1e-3
        v[:, 0] -= 2 * hs * 1e-3 * 0                       # head end near x = 0 (padding is a few cells, < 1 %)
        mesh = pv.PolyData(v, np.hstack([np.full((len(f), 1), 3), f]).ravel())
        mesh["rho"] = np.hypot(v[:, 1], v[:, 2]) * 1e3
        return mesh

    def update_scene(self, vp, p, r, frame) -> None:
        if not self._m or self._m.get("cut") != vp.cutaway:
            self.build_scene(vp, p, r)
        pl = vp.plotter
        if r is None or frame is None or "phi_snaps" not in r.arrays:
            pl.remove_actor("surface") if "surface" in pl.renderer.actors else None
            vp.set_colorbar("viridis", 0.0, p.R_out, r"$\rho$ (mm)")
            vp.render()
            return
        mesh = self._surface(r, frame)
        if mesh is not None:
            if vp.cutaway > 0:
                mesh = mesh.clip(normal=(0.0, 0.0, 1.0), origin=(0.0, 0.0, 0.0), invert=False)
            pl.add_mesh(mesh, scalars="rho", cmap="viridis", clim=(0.0, p.R_out), show_scalar_bar=False,
                        name="surface", smooth_shading=True)
        elif "surface" in pl.renderer.actors:
            pl.remove_actor("surface")
        vp.set_colorbar("viridis", 0.0, p.R_out, r"$\rho$ (mm)")
        vp.render()

    def readout(self, p, r, frame) -> list[str]:
        if r is None or frame is None:
            return [r"$w$ = – (mm)", r"$A_b$ = – (cm²)", r"$V_{gas}$ = – (cm³)", r"left = –"]
        c = r.channels
        return [rf"$w$ = {c['w'][frame]:.1f} (mm)", rf"$A_b$ = {c['A_b'][frame]:.0f} (cm²)",
                rf"$V_{{gas}}$ = {c['V_gas'][frame]:.0f} (cm³)", rf"left = {100 * c['propellant_left'][frame]:.0f} (%)"]

    def caption(self, p, r, frame) -> str:
        if r is None or frame is None:
            return r"press Run: voxel level set, Godunov upwind, marching-cubes surface"
        m = r.meta
        return (rf"$t=\frac{{w}}{{r_{{nom}}}}$ = {r.t[frame]:.2f} (s) · grid {m['cells'][0]}×{m['cells'][1]}×{m['cells'][2]}, "
                rf"$h$ = {m['h_mm']:.2f} (mm)")

    def summary_line(self, p, r) -> str:
        m = r.meta
        return (rf"web {m['web_mm']:.1f} (mm) · $A_{{b,0}}$ = {m['A_b0_cm2']:.0f}, peak {m['A_b_max_cm2']:.0f} (cm²) · "
                rf"sliver {100 * m['sliver_frac']:.1f} (%)")

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        xl = r"$t=\frac{w}{r_{nom}}$ (s)"
        if r is None:
            return [TabSpec("Burn tables", [PlotSpec("Burning area", xl, r"$A_b$ (cm²)", [], cursor=True),
                                            PlotSpec("Gas volume", xl, r"$V_{gas}$ (cm³)", [], cursor=True),
                                            PlotSpec("Propellant left", xl, r"$\frac{m_p}{m_{p,0}}$ (–)", [], cursor=True)])]
        c, t = r.channels, r.t
        tabs = [PlotSpec("Burning area", xl, r"$A_b$ (cm²)",
                         [Curve(t, c["A_b"], "hot", "3-D level set"), Curve(t, c["A_b_section"], "dim", "L × section", "dash")],
                         cursor=True),
                PlotSpec("Gas volume", xl, r"$V_{gas}$ (cm³)",
                         [Curve(t, c["V_gas"], "oxidizer", "3-D level set"),
                          Curve(t, c["V_gas_section"], "dim", "L × section", "dash")], cursor=True),
                PlotSpec("Propellant left", xl, r"$\frac{m_p}{m_{p,0}}$ (–)", [Curve(t, c["propellant_left"], "thrust")],
                         cursor=True)]
        port = [PlotSpec("Port area (mid-length)", xl, r"$A_{port}$ (cm²)", [Curve(t, c["A_port_mid"], "fuel")],
                         cursor=True),
                PlotSpec("Burn distance", xl, r"$w$ (mm)", [Curve(t, c["w"], "text")], cursor=True),
                PlotSpec("Area / initial area", xl, r"$\frac{A_b}{A_{b,0}}$ (–)", [Curve(t, c["A_b"] / max(c["A_b"][0], 1e-9), "hot")],
                         cursor=True)]
        return [TabSpec("Burn tables", tabs), TabSpec("Port", port)]

    def export_data(self, p, r) -> dict[str, np.ndarray]:
        return {"t_s": r.t, **{k: v for k, v in r.channels.items()}}


_ = (CMAPS, QUANTITY, ParamModel)
