"""Hybrid module, Run model: 0-D burn (HTPB/GOX power law) with 3D grain cutaway playback."""
from __future__ import annotations

import numpy as np
import pyvista as pv

from ..core.nozzle import contour, mach_along_contour
from ..core.result import Curve, PlotSpec, Result, TabSpec
from ..core.ballistics.hybrid0d import GAMMA_DEFAULT
from ..sims.hybrid_burn import HybridBurn0D, HybridBurnParams
from ..ui.theme import CMAPS, QUANTITY
from .base import RUN_COLORS, ModelController
from .geometry import broadcast_scalar, cut_face, cut_face_points, rev_grid, rev_points, thetas

T_LBL, P_LBL, R_LBL = r"$T$ (K)", r"$P_c$ (MPa)", r"$\dot r$ (mm·s⁻¹)"
# (channel, cmap, colorbar title, default range)
FIELDS = [("Tc", CMAPS["temperature"], T_LBL, (700.0, 3600.0)),
          ("Pc", "viridis", P_LBL, (0.0, 5.0)),
          ("rdot", "cividis", R_LBL, (0.0, 6.0))]
N_TRACERS = 160


class HybridBurnModel(ModelController):
    key = "hybrid.burn0d"
    title = "0-D burn"
    module = "Hybrid"
    mode = "run"
    Params = HybridBurnParams
    FIELDS = FIELDS
    fields = [f[2] for f in FIELDS]
    legend_labels = {"cut": "Fuel grain (cut face)", "tracers": "Oxidizer flow"}

    def __init__(self) -> None:
        self.sim = HybridBurn0D()
        self._m: dict = {}
        self._cut_built = None
        self._field_built = None
        rng = np.random.default_rng(7)                    # deterministic tracer cloud
        self._tr = rng.uniform(size=(N_TRACERS, 3))
        self._cum_for: int | None = None
        self._cum = np.zeros(1)

    # ---------------------------------------------------------------- scene
    def build_scene(self, vp, p, r=None) -> None:
        si = p.to_si()
        pl = vp.plotter
        pl.clear()
        th = thetas(vp.cutaway)
        self._th, self._cut_built, self._field_built = th, vp.cutaway, vp.field_index
        xg = np.linspace(0.0, si.L, 40)
        self._xg = xg
        m: dict = {}
        case = rev_grid(np.array([0.0, si.L]), np.full(2, si.R_out + 0.006), th)
        pl.add_mesh(case, color="#6b768a", opacity=0.30, name="case", smooth_shading=True)
        port = rev_grid(xg, self._radii(si, r, None), th)
        port["field"] = broadcast_scalar(np.full_like(xg, 1.0), len(th))
        ch, cmap, _, rng_ = self.FIELDS[vp.field_index]
        lo, hi = self._range(r, ch, rng_)
        pl.add_mesh(port, scalars="field", cmap=cmap, clim=(lo, hi), show_scalar_bar=False, name="port",
                    smooth_shading=True)
        m["port"] = port
        m["cuts"] = []
        if vp.cutaway > 0:
            for k, ang in enumerate((0.0, th[-1])):
                f = cut_face(xg, self._radii(si, r, None), np.full_like(xg, si.R_out), ang)
                pl.add_mesh(f, color=QUANTITY["fuel"], opacity=0.85, name=f"cut{k}")
                m["cuts"].append((f, ang))
        # nozzle coloured by Mach (viridis), joined at x = L
        Rt = si.Dt / 2
        xc, rc = contour(Rt, si.eps, Rc=si.R_out)
        Mn = mach_along_contour(xc, rc, Rt, r.meta.get("gamma_mean", GAMMA_DEFAULT) if r is not None else GAMMA_DEFAULT)
        xn = xc - xc[0] + si.L
        noz = rev_grid(xn, rc, th)
        noz["Mach"] = broadcast_scalar(Mn, len(th))
        pl.add_mesh(noz, scalars="Mach", cmap=CMAPS["mach"], clim=(0, float(Mn.max())), show_scalar_bar=False,
                    name="nozzle", smooth_shading=True)
        xe, re_ = xn[-1], rc[-1]
        lines = []
        for k in range(8):
            ph = 2 * np.pi * k / 8
            lines.append(pv.Line((xe, 0.7 * re_ * np.cos(ph), 0.7 * re_ * np.sin(ph)),
                                 (xe + 3.5 * re_, 2.2 * re_ * np.cos(ph), 2.2 * re_ * np.sin(ph))))
        plume = lines[0].merge(lines[1:]) if len(lines) > 1 else lines[0]
        m["plume_actor"] = pl.add_mesh(plume, color=QUANTITY["hot"], opacity=0.45, line_width=2, name="plume")
        tr = pv.PolyData(np.zeros((N_TRACERS, 3)))
        m["tr"] = tr
        pl.add_mesh(tr, color=QUANTITY["oxidizer"], point_size=4, render_points_as_spheres=True, name="tracers")
        self._m = m
        self._extra_build(pl, si, p, th, m)
        vp.reset_view()

    # hooks for the quasi-1D and blowdown models -------------------------------------------------------
    def _radii(self, si, r, frame) -> np.ndarray:
        """Port radius (m) at the mesh stations ``self._xg`` (uniform for the 0-D model)."""
        R = si.R0 if (r is None or frame is None) else r.channels["R"][frame] * 1e-3
        return np.full_like(self._xg, R)

    def _field_x(self, si, r, frame, ch: str) -> np.ndarray:
        v = 0.0 if (r is None or frame is None) else float(np.nan_to_num(r.channels[ch][frame]))
        return np.full_like(self._xg, v)

    def _extra_build(self, pl, si, p, th, m) -> None:
        """Add model-specific meshes (e.g. the oxidizer tank)."""

    def _extra_update(self, si, p, r, frame, m, th) -> None:
        """Per-frame update of model-specific meshes."""

    def _range(self, r, ch, default):
        if r is not None and ch in r.channels:
            v = r.channels[ch][np.isfinite(r.channels[ch])]
            if v.size and v.max() > v.min():
                return float(v.min()), float(v.max())
        return default

    def _phase(self, p, r) -> np.ndarray:
        """Cumulative oxidizer throughput (normalised) so tracer motion tracks the scrubber."""
        if self._cum_for != id(r):
            dt = np.diff(r.t, prepend=r.t[0])
            self._cum = np.cumsum(r.channels["mdot_ox"] * dt) / max(float(np.nanmax(r.channels["mdot_ox"])), 1e-9)
            self._cum_for = id(r)
        return self._cum

    def update_scene(self, vp, p, r, frame) -> None:
        if not self._m or vp.cutaway != self._cut_built or vp.field_index != self._field_built:
            self.build_scene(vp, p, r)
        si = p.to_si()
        m, th, xg = self._m, self._th, self._xg
        ch, cmap, title, rng_ = self.FIELDS[vp.field_index]
        if r is None or frame is None:
            F, phase = 0.0, 0.0
            lo, hi = rng_
        else:
            F = r.channels["F"][frame]
            phase = float(self._phase(si, r)[frame])
            lo, hi = self._range(r, ch, rng_)
        Rx = self._radii(si, r, frame)
        m["port"].points = rev_points(xg, Rx, th)
        m["port"]["field"] = broadcast_scalar(self._field_x(si, r, frame, ch), len(th))
        for f, ang in m["cuts"]:
            f.points = cut_face_points(xg, Rx, np.full_like(xg, si.R_out), ang)
        u = self._tr
        x = ((u[:, 0] + 0.25 * phase) % 1.0) * si.L
        rr = np.sqrt(u[:, 1]) * 0.9 * np.interp(x, xg, Rx)
        ph = u[:, 2] * 2 * np.pi
        m["tr"].points = np.c_[x, rr * np.cos(ph), rr * np.sin(ph)]
        Fmax = np.nanmax(r.channels["F"]) if r is not None else 1.0
        m["plume_actor"].GetProperty().SetOpacity(0.5 if F > 0.05 * Fmax else 0.0)
        self._extra_update(si, p, r, frame, m, th)
        vp.set_colorbar(cmap, lo, hi, title)
        vp.render()

    # ---------------------------------------------------------------- text
    def readout(self, p, r, frame) -> list[str]:
        if r is None or frame is None:
            return [r"$P_c$ = – (MPa)", r"$F$ = – (kN)", r"$\frac{O}{F}$ = –", r"$I_{sp}$ = – (s)"]
        c = r.channels
        fmt = lambda k, f: "–" if not np.isfinite(c[k][frame]) else f % c[k][frame]
        return [rf"$P_c$ = {fmt('Pc', '%.2f')} (MPa)", rf"$F$ = {fmt('F', '%.2f')} (kN)",
                rf"$\frac{{O}}{{F}}$ = {fmt('OF', '%.2f')}", rf"$I_{{sp}}$ = {fmt('Isp', '%.1f')} (s)"]

    def caption(self, p, r, frame) -> str:
        if r is None or frame is None:
            return r"press Run to simulate · port radius grows with $\dot r(t)$"
        web = (r.channels["R"][frame] * 1e-3 - p.R_p0 * 1e-3) / (p.web * 1e-3)
        return (rf"$t$ = {r.t[frame]:.2f} (s) · web burned {100 * web:.0f} % · "
                r"port radius grows with $\dot r(t)$")

    def summary_line(self, p, r) -> str:
        s = r.meta
        return rf"$I_t$ = {s['I_t_kNs']:.1f} (kN·s),  $\bar{{I}}_{{sp}}$ = {s['Isp_avg']:.1f} (s)"

    # ---------------------------------------------------------------- plots
    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        TT = r"$t$ (s)"
        if r is None:
            empty = lambda title, yl, col: PlotSpec(title, TT, yl, [], cursor=True)
            th = [empty("Chamber pressure", r"$P_c$ (MPa)", "pressure"), empty("Thrust", r"$F$ (kN)", "thrust"),
                  empty("Mixture-ratio shift", r"$\frac{O}{F}$ (–)", "oxidizer")]
            return [TabSpec("Time histories", th), TabSpec(r"Regression $\dot r(t)$", []), TabSpec("Tank", []),
                    TabSpec("Compare", [])]
        c, t = r.channels, r.t
        Gox = c["mdot_ox"] / (np.pi * (c["R"] * 1e-3) ** 2)
        cur = lambda y, col, **k: Curve(t, y, col, **k)
        hist = [PlotSpec("Chamber pressure", TT, r"$P_c$ (MPa)", [cur(c["Pc"], "pressure")], cursor=True),
                PlotSpec("Thrust", TT, r"$F$ (kN)", [cur(c["F"], "thrust")], cursor=True),
                PlotSpec("Mixture-ratio shift", TT, r"$\frac{O}{F}$ (–)", [cur(c["OF"], "oxidizer")], cursor=True)]
        reg = [PlotSpec("Regression rate", TT, r"$\dot r$ (mm·s⁻¹)", [cur(c["rdot"], "fuel")], cursor=True),
               PlotSpec("Port radius", TT, r"$R_p$ (mm)", [cur(c["R"], "fuel")], cursor=True),
               PlotSpec("Oxidizer mass flux", TT, r"$G_{ox}$ (kg·m⁻²·s⁻¹)", [cur(Gox, "oxidizer")], cursor=True)]
        tank = [PlotSpec("Oxidizer flow", TT, r"$\dot{m}_{ox}$ (kg·s⁻¹)", [cur(c["mdot_ox"], "oxidizer")],
                         cursor=True),
                PlotSpec("Feed margin", TT, r"$P$ (MPa)",
                         [cur(c["Pc"], "pressure", label="Pc"),
                          Curve(t, np.full_like(t, p.p_tank), "oxidizer", "tank", "dash")], cursor=True),
                PlotSpec("Fuel flow", TT, r"$\dot{m}_f$ (kg·s⁻¹)", [cur(c["mdot_f"], "fuel")], cursor=True)]
        runs = compare or [("current", r)]
        def overlay(ch, title, yl):
            cs = [Curve(rr.t, rr.channels[ch], RUN_COLORS[i % len(RUN_COLORS)], name) for i, (name, rr) in
                  enumerate(runs)]
            return PlotSpec(title, TT, yl, cs, cursor=False)
        cmp_ = [overlay("Pc", "Chamber pressure", r"$P_c$ (MPa)"), overlay("F", "Thrust", r"$F$ (kN)"),
                overlay("OF", "Mixture-ratio shift", r"$\frac{O}{F}$ (–)")]
        return [TabSpec("Time histories", hist), TabSpec(r"Regression $\dot r(t)$", reg), TabSpec("Tank", tank),
                TabSpec("Compare", cmp_)]

    def export_data(self, p, r: Result) -> dict[str, np.ndarray]:
        names = {"Pc": "Pc_MPa", "F": "F_kN", "OF": "O_over_F", "mdot_ox": "mdot_ox_kgps", "mdot_f": "mdot_f_kgps",
                 "rdot": "rdot_mmps", "R": "Rp_mm", "Isp": "Isp_s", "Tc": "Tc_K"}
        return {"t_s": r.t, **{v: r.channels[k] for k, v in names.items()}}
