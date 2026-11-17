"""Common module, Live model: nozzle area ratio -> C_F plot, Mach/pressure distribution, 3D bell nozzle.

Every value is one evaluation of the same functions the Run models use (``core.nozzle``). Target < 50 ms.
Combustion gas is described by (gamma, T_c, molar mass) until the Phase 2 CEA wrapper lands.
"""
from __future__ import annotations

import numpy as np

from ..core.constants import G0, R_U
from ..core.nozzle import (CF, T_ratio, cf_curve, contour, cstar, eps_from_pratio, exit_state,
                           mach_along_contour, p_ratio)
from ..core.result import Curve, PlotSpec, TabSpec
from ..core.schema import P, ParamModel
from ..ui.theme import CMAPS
from .base import ModelController
from .geometry import broadcast_scalar, rev_grid, rev_points, thetas


class NozzleLiveParams(ParamModel):
    gamma: float = P(1.20, symbol=r"\gamma", desc="ratio of specific heats (frozen)", units="", dims="1",
                     lo=1.1, hi=1.67, step=0.01, group="Combustion gas")
    T_c: float = P(3200.0, symbol=r"T_c", desc="chamber (stagnation) temperature", units="K", dims="Θ",
                   lo=1500.0, hi=4000.0, step=10.0, group="Combustion gas")
    M_bar: float = P(24.0, symbol=r"\bar{M}", desc="mean molar mass of exhaust", units="g·mol⁻¹", dims="M N⁻¹",
                     lo=2.0, hi=40.0, step=0.1, group="Combustion gas")
    p_c: float = P(3.5, symbol=r"P_c", desc="chamber pressure", units="MPa", dims="M L⁻¹ T⁻²", lo=0.5, hi=20.0,
                   step=0.05, group="Chamber")
    D_t: float = P(32.0, symbol=r"D_t", desc="throat diameter", units="mm", dims="L", lo=10.0, hi=100.0, step=0.5,
                   group="Nozzle")
    eps: float = P(6.0, symbol=r"\varepsilon=\frac{A_e}{A_t}", desc="nozzle expansion ratio", units="", dims="1",
                   lo=1.5, hi=100.0, step=0.1, group="Nozzle")
    p_a: float = P(101.325, symbol=r"P_a", desc="ambient pressure (0 = vacuum)", units="kPa", dims="M L⁻¹ T⁻²",
                   lo=0.0, hi=101.325, step=0.5, group="Ambient")


# (key, cmap, colorbar title)
FIELDS = [("M", CMAPS["mach"], r"$M$ (–)"), ("T", CMAPS["temperature"], r"$T$ (K)"), ("p", "plasma", r"$p$ (MPa)")]


class NozzleLiveModel(ModelController):
    key = "common.nozzle"
    title = "Nozzle thrust coefficient"
    module = "Hybrid"
    mode = "live"
    Params = NozzleLiveParams
    fields = [r"$M$ (–)", r"$T$ (K)", r"$p$ (MPa)"]

    def __init__(self) -> None:
        self._m: dict = {}
        self._built = None

    # ------------------------------------------------------------------ physics (pure, Live)
    def evaluate(self, p: NozzleLiveParams) -> dict:
        g, Tc, Pc, pa = p.gamma, p.T_c, p.p_c * 1e6, p.p_a * 1e3
        Rt = p.D_t * 1e-3 / 2
        At = np.pi * Rt**2
        RT = R_U / (p.M_bar * 1e-3) * Tc
        cs = float(cstar(g, RT))
        Me, pe_pc = exit_state(p.eps, g)
        pa_pc = pa / Pc
        cf = float(CF(g, pe_pc, pa_pc, p.eps))
        cf_vac = float(CF(g, pe_pc, 0.0, p.eps))
        eps_g, cf_a = cf_curve(g, pa_pc)
        _, cf_v = cf_curve(g, 0.0)
        keep = eps_g <= 120.0
        # optimum expansion (p_e = p_a): closed form
        if pa_pc > 0:
            eps_opt = float(eps_from_pratio(pa_pc, g)[0])
            cf_opt = float(CF(g, pa_pc, pa_pc, eps_opt))
        else:
            eps_opt, cf_opt = np.nan, np.nan
        x, r = contour(Rt, p.eps)
        M = mach_along_contour(x, r, Rt, g)
        mdot = Pc * At / cs
        return dict(cs=cs, Me=float(Me), pe_pc=float(pe_pc), cf=cf, cf_vac=cf_vac, F=cf * Pc * At, mdot=mdot,
                    Isp=cf * cs / G0, Isp_vac=cf_vac * cs / G0, eps_g=eps_g[keep], cf_a=cf_a[keep], cf_v=cf_v[keep],
                    eps_opt=eps_opt, cf_opt=cf_opt, x=x, r=r, M=M, T=Tc * T_ratio(M, g), p=Pc * p_ratio(M, g),
                    Pc=Pc, pa=pa, Rt=Rt, Tc=Tc, g=g)

    # ------------------------------------------------------------------ plots
    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        xmm = r["x"] * 1e3
        pa_line = np.full_like(xmm, r["pa"] / r["Pc"])
        cf_plot = PlotSpec("Thrust coefficient", r"$\varepsilon=\frac{A_e}{A_t}$ (–)", r"$C_F$ (–)",
                           [Curve(r["eps_g"], r["cf_a"], "thrust", "at P_a"),
                            Curve(r["eps_g"], r["cf_v"], "dim", "vacuum", "dash")],
                           markers=[(p.eps, r["cf"], "thrust")] + (
                               [(r["eps_opt"], r["cf_opt"], "hot")] if np.isfinite(r["eps_opt"]) and
                               r["eps_opt"] < 120 else []))
        mach = PlotSpec("Mach number", r"$x$ (mm)", r"$M$ (–)", [Curve(xmm, r["M"], "#21918c")],
                        markers=[(xmm[-1], r["Me"], "#21918c")])
        pres = PlotSpec("Static pressure", r"$x$ (mm)", r"$\frac{p}{P_c}$ (–)",
                        [Curve(xmm, r["p"] / r["Pc"], "pressure"), Curve(xmm, pa_line, "dim", "ambient", "dash")],
                        markers=[(xmm[-1], r["pe_pc"], "pressure")])
        return [TabSpec("Performance", [cf_plot, mach, pres])]

    def readout(self, p, r, frame) -> list[str]:
        return [rf"$P_c$ = {p.p_c:.2f} (MPa)", rf"$F$ = {r['F'] / 1e3:.2f} (kN)", rf"$C_F$ = {r['cf']:.3f}",
                rf"$I_{{sp}}$ = {r['Isp']:.1f} (s)"]

    def caption(self, p, r, frame) -> str:
        state = ("over-expanded" if r["pe_pc"] * r["Pc"] < r["pa"] * 0.98 else
                 "under-expanded" if r["pe_pc"] * r["Pc"] > r["pa"] * 1.02 else "optimally expanded")
        return (rf"$M_e$ = {r['Me']:.2f} · $c^*$ = {r['cs']:.0f} (m·s⁻¹) · $I_{{sp,vac}}$ = {r['Isp_vac']:.1f} (s)"
                f" · {state}")

    def export_data(self, p, r) -> dict[str, np.ndarray]:
        return {"eps": r["eps_g"], "CF_ambient": r["cf_a"], "CF_vac": r["cf_v"], "x_m": r["x"], "r_m": r["r"],
                "Mach": r["M"], "T_K": r["T"], "p_Pa": r["p"]}

    # ------------------------------------------------------------------ 3D
    def _field(self, vp, r):
        key, cmap, title = FIELDS[vp.field_index]
        v = {"M": r["M"], "T": r["T"], "p": r["p"] / 1e6}[key]
        return v, cmap, title

    def build_scene(self, vp, p, r=None) -> None:
        r = r or self.evaluate(p)
        pl = vp.plotter
        pl.clear()
        th = thetas(vp.cutaway)
        v, cmap, _ = self._field(vp, r)
        grid = rev_grid(r["x"], r["r"], th)
        grid["field"] = broadcast_scalar(v, len(th))
        self._m = dict(grid=grid, th=th, actor=pl.add_mesh(grid, scalars="field", cmap=cmap, clim=(v.min(), v.max()),
                                                         show_scalar_bar=False, name="nozzle", smooth_shading=True))
        self._built = (vp.cutaway, vp.field_index)
        vp.reset_view(zoom=1.1)

    def update_scene(self, vp, p, r, frame) -> None:
        if not self._m or self._built != (vp.cutaway, vp.field_index):
            self.build_scene(vp, p, r)
        v, cmap, title = self._field(vp, r)
        g, th = self._m["grid"], self._m["th"]
        g.points = rev_points(r["x"], r["r"], th)            # fixed topology -> in-place update
        g["field"] = broadcast_scalar(v, len(th))
        self._m["actor"].mapper.scalar_range = (float(v.min()), float(v.max()))
        vp.set_colorbar(cmap, float(v.min()), float(v.max()), title)
        vp.render()
