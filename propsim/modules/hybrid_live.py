"""Hybrid module, Live tools (shared 3D grain-tube scene): regression-law explorer and port / multiport grain."""
from __future__ import annotations

from functools import lru_cache

import contourpy
import numpy as np
import pyvista as pv

from ..core.ballistics import hybrid1d as h1
from ..core.regression import (POWER_LAWS, arrhenius_rdot, marxman_rdot, power_law_rdot, rdot_radiation,
                               thermal_wave)
from ..core.result import Curve, PlotSpec, TabSpec
from ..core.schema import C, P, ParamModel
from ..ui.theme import CMAPS, QUANTITY
from .base import ModelController
from .geometry import broadcast_scalar, cut_face, cut_face_points, rev_grid, rev_points, thetas

PRESETS = ["htpb_gox", "paraffin_gox", "custom"]
PRESET_LABELS = ["HTPB / GOX  (0.0541, 0.647)", "Paraffin / GOX  (0.1171, 0.62)", "Custom a, n"]
KEYS = {"htpb_gox": "HTPB / GOX", "paraffin_gox": "Paraffin / GOX"}


class TubeLive(ModelController):
    legend_labels = {"cut": "Fuel grain (cut face)"}
    """Base for Live tools whose 3D view is a cut-away grain tube colored by a per-station field."""

    module = "Hybrid"
    mode = "live"
    cmap = "viridis"
    L_tube, R_in, R_out = 0.6, 0.025, 0.06
    port_opacity = 1.0

    def __init__(self) -> None:
        self._m: dict = {}
        self._built = None

    # subclasses: (x[40], R_in[40], field[40], color-bar title)
    def _profile(self, p, r) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        raise NotImplementedError

    def _extra(self, vp, p, r) -> None:
        """Optional extra meshes (streamlines, ...); called on every scene update."""

    def build_scene(self, vp, p, r=None) -> None:
        r = r if r is not None else self.evaluate(p)
        pl = vp.plotter
        pl.clear()
        th = thetas(vp.cutaway)
        x, Rin, fld, _ = self._profile(p, r)
        case = rev_grid(np.array([0.0, self.L_tube]), np.full(2, self.R_out + 0.006), th)
        pl.add_mesh(case, color="#6b768a", opacity=0.30, name="case", smooth_shading=True)
        port = rev_grid(x, Rin, th)
        port["field"] = broadcast_scalar(fld, len(th))
        actor = pl.add_mesh(port, scalars="field", cmap=self.cmap, show_scalar_bar=False, name="port",
                            smooth_shading=True, opacity=self.port_opacity)
        cuts = []
        if vp.cutaway > 0:
            for k, ang in enumerate((0.0, th[-1])):
                f = cut_face(x, Rin, np.full_like(x, self.R_out), ang)
                pl.add_mesh(f, color=QUANTITY["fuel"], opacity=0.85, name=f"cut{k}")
                cuts.append((f, ang))
        self._m = dict(port=port, actor=actor, th=th, x=x, cuts=cuts)
        self._built = vp.cutaway
        vp.reset_view(zoom=1.2)

    def update_scene(self, vp, p, r, frame) -> None:
        if not self._m or self._built != vp.cutaway:
            self.build_scene(vp, p, r)
        m = self._m
        x, Rin, fld, title = self._profile(p, r)
        lo, hi = float(np.min(fld)), float(np.max(fld))
        if hi - lo < 1e-12:
            hi = lo + 1.0
        m["port"].points = rev_points(x, Rin, m["th"])
        m["port"]["field"] = broadcast_scalar(fld, len(m["th"]))
        m["actor"].mapper.scalar_range = (lo, hi)
        for f, ang in m["cuts"]:
            f.points = cut_face_points(x, Rin, np.full_like(x, self.R_out), ang)
        self._extra(vp, p, r)
        vp.set_colorbar(self.cmap, lo, hi, title)
        vp.render()


# ====================================================================================== regression explorer
class RegressionLiveParams(ParamModel):
    preset: str = C("htpb_gox", choices=PRESETS, labels=PRESET_LABELS, symbol=r"\mathrm{power\ law}",
                    desc="space-averaged power law rdot = a G^n (Chiaverini Ch. 10 style; MATHKIT 3.1 values)",
                    group="Power law")
    a: float = P(0.0541, symbol=r"a", desc="custom power-law coefficient (rdot in mm/s, G in kg/m²/s)",
                 units="mm·s⁻¹", dims="L T⁻¹ (M L⁻² T⁻¹)⁻ⁿ", lo=0.01, hi=0.5, step=0.0001, group="Power law")
    n: float = P(0.647, symbol=r"n", desc="custom power-law exponent", units="", dims="1", lo=0.3, hi=1.0,
                 step=0.001, group="Power law")
    G: float = P(200.0, symbol=r"G", desc="mass flux at the operating point", units="kg·m⁻²·s⁻¹",
                 dims="M L⁻² T⁻¹", lo=20.0, hi=800.0, step=1.0, group="Operating point")
    x: float = P(0.5, symbol=r"x", desc="axial station from the port head end", units="m", dims="L", lo=0.05,
                 hi=1.5, step=0.01, group="Operating point")
    B: float = P(6.0, symbol=r"B", desc="blowing number", units="", dims="1", lo=1.0, hi=20.0, step=0.1,
                 group="Marxman")
    mu: float = P(8.0, symbol=r"\mu", desc="gas dynamic viscosity (x 1e-5)", units="10⁻⁵ Pa·s", dims="M L⁻¹ T⁻¹",
                  lo=4.0, hi=15.0, step=0.1, group="Marxman")
    rho_f: float = P(920.0, symbol=r"\rho_f", desc="fuel density", units="kg·m⁻³", dims="M L⁻³", lo=700.0,
                     hi=1200.0, step=5.0, group="Marxman")
    Qc: float = P(1.0, symbol=r"Q_c", desc="convective heat flux (unblocked)", units="MW·m⁻²", dims="M T⁻³",
                  lo=0.1, hi=5.0, step=0.05, group="Radiation blocking")
    Qr: float = P(0.3, symbol=r"Q_{rad}", desc="radiative heat flux", units="MW·m⁻²", dims="M T⁻³", lo=0.0, hi=2.0,
                  step=0.05, group="Radiation blocking")
    dHv: float = P(1.81, symbol=r"\Delta H_{v,eff}", desc="effective heat of gasification", units="MJ·kg⁻¹",
                   dims="L² T⁻²", lo=0.5, hi=4.0, step=0.01, group="Radiation blocking")
    Ts: float = P(800.0, symbol=r"T_s", desc="regressing surface temperature", units="K", dims="Θ", lo=600.0,
                  hi=1000.0, step=1.0, group="Surface pyrolysis")
    log10A: float = P(6.8, symbol=r"\log_{10}A", desc="Arrhenius pre-exponential factor (A in mm/s)", units="",
                      dims="1", lo=3.0, hi=11.0, step=0.05, group="Surface pyrolysis")
    Ea: float = P(100.0, symbol=r"E_a", desc="pyrolysis activation energy", units="kJ·mol⁻¹", dims="M L² T⁻² N⁻¹",
                  lo=40.0, hi=200.0, step=1.0, group="Surface pyrolysis")
    alpha: float = P(1.0, symbol=r"\alpha", desc="solid thermal diffusivity (x 1e-7)", units="10⁻⁷ m²·s⁻¹",
                     dims="L² T⁻¹", lo=0.3, hi=2.0, step=0.01, group="Surface pyrolysis")


class RegressionLiveModel(TubeLive):
    key = "hybrid.regression"
    title = "Regression laws"
    Params = RegressionLiveParams
    fields = [r"$\dot r$ (mm·s⁻¹)"]
    cmap = "cividis"

    @staticmethod
    def _law(p) -> tuple[float, float]:
        if p.preset == "custom":
            return p.a * 1e-3, p.n
        return POWER_LAWS[KEYS[p.preset]]

    def evaluate(self, p: RegressionLiveParams) -> dict:
        a, n = self._law(p)
        mu = p.mu * 1e-5
        Gg = np.linspace(20.0, 800.0, 80)
        xg = np.linspace(0.05, 1.5, 60)
        Qg = np.linspace(0.1, 5.0, 60)
        r_pow = float(power_law_rdot(p.G, a, n)) * 1e3
        r_mar = float(marxman_rdot(p.G, p.x, p.B, mu, p.rho_f)) * 1e3
        r_rad = float(rdot_radiation(p.Qc * 1e6, p.Qr * 1e6, p.rho_f, p.dHv * 1e6)) * 1e3
        r_ar = float(arrhenius_rdot(p.Ts, 10.0**p.log10A, p.Ea * 1e3)) * 1e-0           # mm/s (A in mm/s)
        delta = p.alpha * 1e-7 / max(r_ar * 1e-3, 1e-12)                                 # thermal thickness (m)
        zdepth = np.linspace(0.0, 5.0 * delta, 80)
        Tprof = 293.0 + (p.Ts - 293.0) * thermal_wave(zdepth, r_ar * 1e-3, p.alpha * 1e-7)
        xs = np.linspace(0.0, self.L_tube, 40)
        r_x = marxman_rdot(p.G, np.maximum(xs, 0.02), p.B, mu, p.rho_f) * 1e3
        return dict(a=a, n=n, Gg=Gg, xg=xg, Qg=Qg, r_pow=r_pow, r_mar=r_mar, r_rad=r_rad, r_ar=r_ar, delta=delta,
                    pow_sel=power_law_rdot(Gg, a, n) * 1e3,
                    pow_htpb=power_law_rdot(Gg, *POWER_LAWS["HTPB / GOX"]) * 1e3,
                    pow_par=power_law_rdot(Gg, *POWER_LAWS["Paraffin / GOX"]) * 1e3,
                    mar_G=marxman_rdot(Gg, p.x, p.B, mu, p.rho_f) * 1e3,
                    mar_x=marxman_rdot(p.G, xg, p.B, mu, p.rho_f) * 1e3,
                    rad0=rdot_radiation(Qg * 1e6, 0.0, p.rho_f, p.dHv * 1e6) * 1e3,
                    rad1=rdot_radiation(Qg * 1e6, p.Qr * 1e6, p.rho_f, p.dHv * 1e6) * 1e3,
                    Tg=np.linspace(600.0, 1000.0, 60),
                    ar_T=arrhenius_rdot(np.linspace(600.0, 1000.0, 60), 10.0**p.log10A, p.Ea * 1e3),
                    zdepth=zdepth * 1e3, Tprof=Tprof, xs=xs, r_x=r_x)

    def _profile(self, p, r):
        return r["xs"], np.full_like(r["xs"], self.R_in), r["r_x"], r"$\dot r$ (mm·s⁻¹)"

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        xg_l, G_l = r"$x$ (m)", r"$G$ (kg·m⁻²·s⁻¹)"
        laws = [PlotSpec("Regression rate vs mass flux", G_l, r"$\dot r$ (mm·s⁻¹)",
                         [Curve(r["Gg"], r["pow_sel"], "thrust", "power law"),
                          Curve(r["Gg"], r["pow_htpb"], "fuel", "HTPB/GOX", "dash"),
                          Curve(r["Gg"], r["pow_par"], "oxidizer", "paraffin", "dash"),
                          Curve(r["Gg"], r["mar_G"], "hot", "Marxman")],
                         markers=[(p.G, r["r_pow"], "thrust"), (p.G, r["r_mar"], "hot")]),
                PlotSpec("Marxman: axial dependence", xg_l, r"$\dot r$ (mm·s⁻¹)", [Curve(r["xg"], r["mar_x"], "hot")],
                         markers=[(p.x, r["r_mar"], "hot")]),
                PlotSpec("Radiation blocking", r"$Q_c$ (MW·m⁻²)", r"$\dot r$ (mm·s⁻¹)",
                         [Curve(r["Qg"], r["rad0"], "dim", "no radiation", "dash"),
                          Curve(r["Qg"], r["rad1"], "pressure", "with radiation")],
                         markers=[(p.Qc, r["r_rad"], "pressure")])]
        pyro = [PlotSpec("Surface pyrolysis (Arrhenius)", r"$T_s$ (K)", r"$\dot r$ (mm·s⁻¹)",
                         [Curve(r["Tg"], r["ar_T"], "fuel")], markers=[(p.Ts, r["r_ar"], "fuel")]),
                PlotSpec("Thermal wave in the solid", r"$z_d$ (mm)", r"$T$ (K)", [Curve(r["zdepth"], r["Tprof"], "hot")]),
                PlotSpec("Blowing effect (Marxman)", r"$B$ (–)", r"$\dot r$ (mm·s⁻¹)",
                         [Curve(np.linspace(1, 20, 40),
                                marxman_rdot(p.G, p.x, np.linspace(1, 20, 40), p.mu * 1e-5, p.rho_f) * 1e3, "hot")],
                         markers=[(p.B, r["r_mar"], "hot")])]
        return [TabSpec("Laws", laws), TabSpec("Pyrolysis", pyro)]

    def readout(self, p, r, frame) -> list[str]:
        return [rf"$G$ = {p.G:.0f} (kg·m⁻²·s⁻¹)", rf"$\dot r_{{pow}}$ = {r['r_pow']:.3f} (mm·s⁻¹)",
                rf"$\dot r_{{Marxman}}$ = {r['r_mar']:.3f} (mm·s⁻¹)", rf"$\dot r_{{rad}}$ = {r['r_rad']:.3f} (mm·s⁻¹)"]

    def caption(self, p, r, frame) -> str:
        return (rf"power law $a$ = {r['a'] * 1e3:.4f}, $n$ = {r['n']:.3f} · thermal thickness "
                rf"$\delta=\frac{{\alpha}}{{\dot r}}$ = {r['delta'] * 1e3:.3f} (mm) · Arrhenius constants are illustrative")

    def export_data(self, p, r) -> dict[str, np.ndarray]:
        return {"G": r["Gg"], "rdot_power_mmps": r["pow_sel"], "rdot_marxman_mmps": r["mar_G"], "x_m": r["xg"],
                "rdot_marxman_x_mmps": r["mar_x"], "Qc_MWm2": r["Qg"], "rdot_rad_mmps": r["rad1"], "Ts_K": r["Tg"],
                "rdot_arrhenius_mmps": r["ar_T"]}


# ====================================================================================== port & multiport grain
LAYOUTS = ["single", "wagon", "center4", "center6"]
LAYOUT_LABELS = ["Single circular port", "Wagon wheel (centre + slots)", "Centre port + 4 ports",
                 "Centre port + 6 ports"]
NGRID = 160


class PortLiveParams(ParamModel):
    layout: str = C("wagon", choices=LAYOUTS, labels=LAYOUT_LABELS, symbol=r"\mathrm{cross\ section}",
                    desc="grain port layout", group="Grain cross-section")
    R_out: float = P(60.0, symbol=r"R_{out}", desc="grain outer radius", units="mm", dims="L", lo=30.0, hi=90.0,
                     step=0.5, group="Grain cross-section")
    R_p: float = P(14.0, symbol=r"R_p", desc="centre-port radius (and satellite-port radius)", units="mm", dims="L",
                   lo=6.0, hi=30.0, step=0.5, group="Grain cross-section")
    n_slot: float = P(6.0, symbol=r"N_s", desc="number of slots (wagon wheel)", units="", dims="1", lo=3.0, hi=12.0,
                      step=1.0, group="Grain cross-section")
    w_slot: float = P(8.0, symbol=r"w_s", desc="slot width (wagon wheel)", units="mm", dims="L", lo=3.0, hi=20.0,
                      step=0.5, group="Grain cross-section")
    r_slot: float = P(0.55, symbol=r"\frac{r_s}{R_{out}}", desc="slot depth as a fraction of the outer radius",
                      units="", dims="1", lo=0.3, hi=0.9, step=0.01, group="Grain cross-section")
    burn: float = P(35.0, symbol=r"\frac{w}{w_{max}}", desc="web burned (percent of the maximum burn distance)",
                    units="%", dims="1", lo=0.0, hi=100.0, step=0.5, group="Burnback")
    L_g: float = P(0.6, symbol=r"L_g", desc="grain length", units="m", dims="L", lo=0.2, hi=2.0, step=0.01,
                   group="Port flow")
    mdot_ox: float = P(1.8, symbol=r"\dot{m}_{ox}", desc="oxidizer mass flow", units="kg·s⁻¹", dims="M T⁻¹", lo=0.2,
                       hi=5.0, step=0.01, group="Port flow")
    law: str = C("power", choices=["power", "marxman"], labels=[r"Power law $\dot r=aG_{ox}^{n}$", "Marxman"],
                 symbol=r"\mathrm{regression\ law}", desc="local regression law", group="Port flow")


def _sdf_outside(layout: str, R_out: float, R_p: float, n_slot: int, w: float, r_slot: float):
    """Exact distance (mm) from every raster point to the union of the initial ports (0 inside the ports).

    Ports are disks and flat-ended slots, so the distance is the minimum of the analytic distances to each primitive
    (disk: |p-c|-R; slot: box signed distance). Burnback by distance d is then the level set {dist <= d}."""
    xs = np.linspace(-R_out, R_out, NGRID)
    X, Y = np.meshgrid(xs, xs)
    h = xs[1] - xs[0]
    sd = np.hypot(X, Y) - R_p
    if layout == "wagon":
        hl = 0.5 * r_slot * R_out
        for k in range(n_slot):
            a = 2 * np.pi * k / n_slot
            u, v = X * np.cos(a) + Y * np.sin(a), -X * np.sin(a) + Y * np.cos(a)
            qx, qy = np.abs(u - hl) - hl, np.abs(v) - w / 2
            box = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) + np.minimum(np.maximum(qx, qy), 0.0)
            sd = np.minimum(sd, box)
    elif layout in ("center4", "center6"):
        ns = 4 if layout == "center4" else 6
        for k in range(ns):
            a = 2 * np.pi * k / ns
            sd = np.minimum(sd, np.hypot(X - 0.5 * R_out * np.cos(a), Y - 0.5 * R_out * np.sin(a)) - R_p)
    return X, Y, sd, h                     # signed: negative inside the initial ports


@lru_cache(maxsize=16)
def _burnback(layout, R_out, R_p, n_slot, w, r_slot) -> dict:
    """Distance field from the initial port and area / perimeter / remaining-fuel curves vs burn distance."""
    X, Y, sdf, h = _sdf_outside(layout, R_out, R_p, n_slot, w, r_slot)
    dist = np.maximum(sdf, 0.0)
    inside = X**2 + Y**2 <= R_out**2
    web_max = float(dist[inside].max())
    # anti-aliased coverage (boundary pixels weighted by the signed distance) instead of counting pixels
    cov_in = np.clip((R_out - np.hypot(X, Y)) / h + 0.5, 0.0, 1.0)
    A_out = float(cov_in.sum() * h * h)
    d = np.linspace(0.0, web_max, 26)
    area = np.array([float((np.clip((di - sdf) / h + 0.5, 0.0, 1.0) * cov_in).sum() * h * h) for di in d])
    gen = contourpy.contour_generator(X, Y, np.ma.masked_array(sdf, mask=~inside))
    length = lambda lvl: sum(float(np.sum(np.hypot(*np.diff(ln, axis=0).T))) for ln in gen.lines(lvl))
    perim = np.array([length(di) for di in d])
    area[-1], perim[-1] = A_out, 0.0                      # at d = w_max the web is consumed by definition
    area = np.minimum(np.maximum.accumulate(area), A_out)
    remain = (A_out - area) / max(A_out - area[0], 1e-9)
    return dict(dist=dist, inside=inside, h=h, web_max=web_max, d=d, area=area, perim=perim, remain=remain,
                A_out=A_out)


class PortLiveModel(ModelController):
    key = "hybrid.port"
    title = "Port & grain"
    module = "Hybrid"
    mode = "live"
    Params = PortLiveParams
    fields = [r"burn distance $\frac{w}{w_{max}}$ (–)"]

    def __init__(self) -> None:
        self._built = None

    def evaluate(self, p: PortLiveParams) -> dict:
        b = _burnback(p.layout, p.R_out, p.R_p, int(round(p.n_slot)), p.w_slot, p.r_slot)
        d_now = p.burn / 100.0 * b["web_max"]
        A_now = float(np.interp(d_now, b["d"], b["area"]))
        P_now = float(np.interp(d_now, b["d"], b["perim"]))
        rem = float(np.interp(d_now, b["d"], b["remain"]))
        # axial march of a single equivalent circular port of the same cross-section area
        Rh = np.sqrt(A_now * 1e-6 / np.pi)
        a, n = POWER_LAWS["HTPB / GOX"]
        mot = h1.Motor1D(pair="GOX/HTPB", L=p.L_g, R0=Rh, R_out=p.R_out * 1e-3, rho_f=920.0, law=p.law, a=a, n=n,
                         B=6.0, mu=8e-5, Dt=0.03, eps=6.0, eta_cstar=0.95, N=30)
        rdot, mf, G = h1._march(mot, np.full(mot.N, Rh), p.mdot_ox)
        xs = (np.arange(mot.N) + 0.5) * mot.dx
        return dict(b=b, d_now=d_now, A_now=A_now, P_now=P_now, rem=rem, Rh=Rh, xs=xs, rdot=rdot * 1e3, G=G,
                    OF=p.mdot_ox / max(mf.sum(), 1e-12), mf=float(mf.sum()))

    # ------------------------------------------------------------------ plots
    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        b, W = r["b"], r"$\frac{w}{w_{max}}$ (%)"
        pct = 100.0 * b["d"] / b["web_max"]
        sect = [PlotSpec("Port area", W, r"$A_p$ (cm²)", [Curve(pct, b["area"] / 100.0, "oxidizer")],
                         markers=[(p.burn, r["A_now"] / 100.0, "hot")]),
                PlotSpec("Burning perimeter", W, r"$\Pi$ (mm)", [Curve(pct, b["perim"], "fuel")],
                         markers=[(p.burn, r["P_now"], "hot")]),
                PlotSpec("Fuel remaining", W, r"$\frac{m_f}{m_{f,0}}$ (–)", [Curve(pct, b["remain"], "thrust")],
                         markers=[(p.burn, r["rem"], "hot")])]
        flow = [PlotSpec("Local regression rate", r"$x$ (m)", r"$\dot r$ (mm·s⁻¹)", [Curve(r["xs"], r["rdot"], "fuel")]),
                PlotSpec("Local mass flux", r"$x$ (m)", r"$G$ (kg·m⁻²·s⁻¹)", [Curve(r["xs"], r["G"], "oxidizer")]),
                PlotSpec("Local equivalent radius", r"$x$ (m)", r"$R_h$ (mm)",
                         [Curve(r["xs"], np.full_like(r["xs"], r["Rh"] * 1e3), "dim")])]
        return [TabSpec("Cross-section", sect), TabSpec("Port flow", flow)]

    def readout(self, p, r, frame) -> list[str]:
        return [rf"$\frac{{w}}{{w_{{max}}}}$ = {p.burn:.0f} (%)", rf"$A_p$ = {r['A_now'] / 100.0:.1f} (cm²)",
                rf"$\Pi$ = {r['P_now']:.0f} (mm)", rf"$\frac{{O}}{{F}}$ = {r['OF']:.2f}"]

    def caption(self, p, r, frame) -> str:
        return (rf"{LAYOUT_LABELS[LAYOUTS.index(p.layout)]} · web {r['b']['web_max']:.1f} (mm) · "
                rf"burn distance {r['d_now']:.1f} (mm) · perimeter excludes the insulated case wall")

    def export_data(self, p, r) -> dict[str, np.ndarray]:
        b = r["b"]
        return {"burn_distance_mm": b["d"], "port_area_mm2": b["area"], "perimeter_mm": b["perim"],
                "fuel_remaining": b["remain"], "x_m": r["xs"], "rdot_mmps": r["rdot"], "G": r["G"]}

    # ------------------------------------------------------------------ 3D: extruded cross-section
    def build_scene(self, vp, p, r=None) -> None:
        vp.plotter.clear()
        self._built = None
        vp.reset_view(zoom=1.0)

    def update_scene(self, vp, p, r, frame) -> None:
        b = r["b"]
        first = self._built is None
        draw_section_slab(vp, b["dist"], b["inside"], b["h"], b["web_max"], r["d_now"], p.R_out)
        if first:
            self._built = True


def draw_section_slab(vp, dist: np.ndarray, inside: np.ndarray, h: float, web_max: float, d_now: float,
                      R_out: float, cmap: str | None = None) -> None:
    """Extruded grain slice (thin slab) with the burned-out port removed, colored by burn distance / web."""
    cmap = cmap or CMAPS.get("temperature", "plasma")
    n = dist.shape[0]
    hm = h * 1e-3
    thick = 0.5 * R_out * 1e-3                                  # display thickness: a slice, not the full grain length
    g = pv.ImageData(dimensions=(n + 1, n + 1, 2), spacing=(hm, hm, thick), origin=(-R_out * 1e-3,) * 2 + (0.0,))
    keep = inside & (dist > d_now)
    field = np.full((n, n), -1.0)
    field[keep] = dist[keep] / max(web_max, 1e-9)
    g.cell_data["w"] = field.ravel()                            # cell order: x fastest, then y
    vp.plotter.add_mesh(g.threshold(0.0, scalars="w"), scalars="w", cmap=cmap, clim=(0.0, 1.0),
                        show_scalar_bar=False, name="grain")
    vp.reset_view(position="iso", zoom=1.0) if not getattr(vp, "_slab_view_set", False) else None
    vp._slab_view_set = True
    vp.set_colorbar(cmap, 0.0, 1.0, r"$\frac{d}{w_{max}}$ (–)")
    vp.render()
