"""Toy Run model: damped spring-mass (pipeline proof; not a propulsion model)."""
from __future__ import annotations

import numpy as np
import pyvista as pv

from ..core.result import Curve, PlotSpec, TabSpec
from ..sims.toy import SpringMass, SpringMassParams
from .base import ModelController


class SpringMassModel(ModelController):
    key = "toy.springmass"
    title = "Spring-mass"
    module = "Toy"
    mode = "run"
    Params = SpringMassParams
    fields: list[str] = []

    def __init__(self) -> None:
        self.sim = SpringMass()
        self._m: dict = {}

    def build_scene(self, vp, p, r=None) -> None:
        pl = vp.plotter
        pl.clear()
        ball = pv.Sphere(radius=0.09)
        self._m = dict(ball=ball, base=np.array(ball.points), line=pv.Line((-1.2, 0, 0), (0, 0, 0)))
        pl.add_mesh(pv.Box(bounds=(-1.3, -1.2, -0.3, 0.3, -0.3, 0.3)), color="#6b768a", name="wall")
        pl.add_mesh(self._m["line"], color="#ff9d3c", line_width=4, name="spring")
        pl.add_mesh(ball, color="#4ea1ff", name="ball", smooth_shading=True)
        vp.reset_view()

    def update_scene(self, vp, p, r, frame) -> None:
        if not self._m:
            self.build_scene(vp, p, r)
        x = p.x0 if r is None or frame is None else r.channels["x"][frame]
        self._m["ball"].points = self._m["base"] + np.array([x, 0, 0])
        self._m["line"].points = np.array([[-1.2, 0, 0], [x - 0.09, 0, 0]])
        vp.render()

    def tabs(self, p, r, compare=None) -> list[TabSpec]:
        TT = r"$t$ (s)"
        if r is None:
            return [TabSpec("Time histories", [PlotSpec("Displacement", TT, r"$x$ (m)", [], cursor=True),
                                               PlotSpec("Velocity", TT, r"$v$ (m·s⁻¹)", [], cursor=True),
                                               PlotSpec("Energy", TT, r"$E$ (J)", [], cursor=True)])]
        c, t = r.channels, r.t
        return [TabSpec("Time histories", [
            PlotSpec("Displacement", TT, r"$x$ (m)", [Curve(t, c["x"], "thrust")], cursor=True),
            PlotSpec("Velocity", TT, r"$v$ (m·s⁻¹)", [Curve(t, c["v"], "oxidizer")], cursor=True),
            PlotSpec("Energy", TT, r"$E$ (J)", [Curve(t, c["E"], "pressure")], cursor=True)])]

    def readout(self, p, r, frame) -> list[str]:
        if r is None or frame is None:
            return [r"$x$ = – (m)", r"$v$ = – (m·s⁻¹)", r"$E$ = – (J)", ""]
        c = r.channels
        return [rf"$x$ = {c['x'][frame]:.3f} (m)", rf"$v$ = {c['v'][frame]:.3f} (m·s⁻¹)",
                rf"$E$ = {c['E'][frame]:.3f} (J)", ""]

    def caption(self, p, r, frame) -> str:
        return "" if r is None or frame is None else rf"$t$ = {r.t[frame]:.2f} (s)"

    def summary_line(self, p, r) -> str:
        return rf"$E_0$ = {r.meta['E0']:.2f} (J)"

    def export_data(self, p, r):
        return {"t_s": r.t, "x_m": r.channels["x"], "v_mps": r.channels["v"], "E_J": r.channels["E"]}
