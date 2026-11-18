"""Toy Run model: damped spring-mass (pipeline proof; not a propulsion model)."""
from __future__ import annotations

import numpy as np
import pyvista as pv

from ..core.result import Curve, PlotSpec, TabSpec
from ..sims.toy import SpringMass, SpringMassParams
from .base import ModelController


WALL_X, BALL_R = -1.2, 0.09
COILS, COIL_R, WIRE_R = 10, 0.075, 0.013            # turns, coil radius and wire radius (m)
N_ALONG, N_AROUND = 220, 10


def coil_surface(x_end: float) -> np.ndarray:
    """Surface points (3, N_ALONG, N_AROUND) of a helical spring (a tube of radius ``WIRE_R`` swept along a helix) from
    the wall to ``x_end``; the topology is fixed, so the mesh is updated by moving its points."""
    s = np.linspace(0.0, 1.0, N_ALONG)
    phi = 2.0 * np.pi * COILS * s
    L = x_end - WALL_X
    cx = WALL_X + L * s
    c = np.stack([cx, COIL_R * np.cos(phi), COIL_R * np.sin(phi)])                         # centre line
    t = np.stack([np.full_like(s, L), -COIL_R * 2.0 * np.pi * COILS * np.sin(phi),
                  COIL_R * 2.0 * np.pi * COILS * np.cos(phi)])                           # d(centre)/ds
    t /= np.linalg.norm(t, axis=0)
    n = np.stack([np.zeros_like(s), np.cos(phi), np.sin(phi)])                           # radial direction
    n -= t * np.sum(n * t, axis=0)
    n /= np.linalg.norm(n, axis=0)
    b = np.cross(t, n, axis=0)
    a = np.linspace(0.0, 2.0 * np.pi, N_AROUND)
    return c[:, :, None] + WIRE_R * (n[:, :, None] * np.cos(a) + b[:, :, None] * np.sin(a))


def _flat(surf: np.ndarray) -> np.ndarray:
    return np.c_[surf[0].ravel(order="F"), surf[1].ravel(order="F"), surf[2].ravel(order="F")]


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
        ball = pv.Sphere(radius=BALL_R)
        x0 = p.x0 if r is None else r.channels["x"][0]
        surf = coil_surface(x0 - BALL_R)
        spring = pv.StructuredGrid(surf[0], surf[1], surf[2])
        self._m = dict(ball=ball, base=np.array(ball.points), spring=spring)
        pl.add_mesh(pv.Box(bounds=(WALL_X - 0.1, WALL_X, -0.3, 0.3, -0.3, 0.3)), color="#6b768a", name="wall")
        pl.add_mesh(spring, color="#ff9d3c", name="spring", smooth_shading=True)
        pl.add_mesh(ball, color="#4ea1ff", name="ball", smooth_shading=True)
        vp.reset_view()

    def update_scene(self, vp, p, r, frame) -> None:
        if not self._m:
            self.build_scene(vp, p, r)
        x = p.x0 if r is None or frame is None else r.channels["x"][frame]
        self._m["ball"].points = self._m["base"] + np.array([x, 0, 0])
        self._m["spring"].points = _flat(coil_surface(x - BALL_R))
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
