r"""Parametric 3-D models of the engine-graph components (the cycle sandbox's *3D Model* panel).

Every component of the library gets a model built from its own parameters, so a part drawn in the sandbox has the
size the solver uses: a line's length and bore, a valve's bore, a pump's impeller tip diameter, stage count and
impeller kind, a turbine's mean blade diameter and arrangement, a tank's volume and fill, a chamber's :math:`L^*`,
throat area and contraction ratio, a nozzle's expansion ratio and bell fraction (Rao contour, ``core.nozzle``), a
jacket's channel count and size.  The solved operating point sets what moves: valve positions, and the wall and
coolant temperatures and heat flux along the jacket march colour the chamber, nozzle and jacket.

Proportions the library does not hold (casing walls, flanges, stems, actuators, blade counts, injector element
pattern, gas-generator and shaft sizes) are drawing choices, not data: each model says which of its sizes are
illustrative in ``Model.notes``.  All lengths are metres; the flow axis is ``+x``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyvista as pv

from .geometry import cut_face, rev_grid, thetas

STEEL, DARK, ROTOR, STATOR, COPPER, SEAL = "#a7aeb7", "#4b525c", "#d2a83f", "#7e8fa3", "#c47a45", "#2f3640"
FLUID_COLORS = {"LOX": "#7cc4ff", "LH2": "#d6efff", "RP1": "#d8a441", "CH4": "#a6e3a0", "N2O4": "#d2694a",
                "MMH": "#e8d48a", "UDMH": "#e8d48a", "N2H4": "#e8d48a", "H2O2": "#cfe8f0", "ETHANOL": "#e9e2c0"}
GAS = "#ff8a3d"


@dataclass
class Part:
    name: str
    mesh: pv.DataSet
    color: str = STEEL
    opacity: float = 1.0
    scalars: str | None = None          # point-data array to colour by (a field of the model)


@dataclass
class Model:
    parts: list[Part]
    title: str
    size: str                           # one line: the main dimensions
    notes: list[str] = field(default_factory=list)
    fields: dict[str, tuple[str, str]] = field(default_factory=dict)   # array -> (title, cmap)
    labels: dict[str, str] = field(default_factory=dict)               # part name -> legend label
    vertical: bool = False              # stood on end for display (tanks)


# ----------------------------------------------------------------------------------------------- primitives
def shell(x, r_in, r_out, cut: float, n: int = 64, ends: bool = True) -> pv.PolyData:
    """Closed wall of revolution between radius profiles ``r_in(x)`` and ``r_out(x)`` with a ``cut`` degree wedge
    removed (inner and outer skins, the two cut faces and the end rings)."""
    x, r_in, r_out = (np.asarray(a, float) for a in (x, r_in, r_out))
    th = thetas(cut, n)
    blocks = [rev_grid(x, r_out, th), rev_grid(x, r_in, th)]
    if cut > 0:
        blocks += [cut_face(x, r_in, r_out, th[0]), cut_face(x, r_in, r_out, th[-1])]
    if ends:
        for i in (0, -1):
            blocks.append(_ring(x[i], r_in[i], r_out[i], th))
    return _merge(blocks)


def solid(x, r, cut: float, n: int = 64) -> pv.PolyData:
    """Solid of revolution (a fluid volume or a hub): the skin, the cut faces and the end discs."""
    x, r = np.asarray(x, float), np.asarray(r, float)
    th = thetas(cut, n)
    blocks = [rev_grid(x, r, th)]
    z = np.zeros_like(r)
    if cut > 0:
        blocks += [cut_face(x, z, r, th[0]), cut_face(x, z, r, th[-1])]
    for i in (0, -1):
        if r[i] > 0:
            blocks.append(_ring(x[i], 0.0, r[i], th))
    return _merge(blocks)


def _ring(x0, r0, r1, th, nr: int = 4) -> pv.StructuredGrid:
    s = np.linspace(r0, r1, nr)
    R, T = np.meshgrid(s, th, indexing="ij")
    g = pv.StructuredGrid(np.full_like(R, x0), R * np.cos(T), R * np.sin(T))
    return g


def _merge(blocks) -> pv.PolyData:
    out = None
    for b in blocks:
        p = b.extract_surface(algorithm='dataset_surface') if not isinstance(b, pv.PolyData) else b
        out = p if out is None else out.merge(p)
    return out.clean() if out is not None else pv.PolyData()


def tube_along(points, radius: float, n_sides: int = 24) -> pv.PolyData:
    line = pv.lines_from_points(np.asarray(points, float))
    return line.tube(radius=radius, n_sides=n_sides, capping=True)


def cyl(center, direction, radius, height, res: int = 40) -> pv.PolyData:
    return pv.Cylinder(center=center, direction=direction, radius=radius, height=height, resolution=res)


def blade(r0: float, r1: float, x0: float, x1: float, a0: float, twist: float, thick: float, nr: int = 6,
          nx: int = 8) -> pv.PolyData:
    """Thin blade between radii ``r0..r1`` and axial ``x0..x1``; its angle runs from ``a0`` to ``a0+twist`` along x
    (a helix for an inducer or axial stage, a straight vane for twist 0); ``thick`` (m) is offset both ways."""
    r, s = np.linspace(r0, r1, nr), np.linspace(0.0, 1.0, nx)
    R, S = np.meshgrid(r, s, indexing="ij")
    X = x0 + (x1 - x0) * S
    out = None
    for d in (-0.5, 0.5):
        A = a0 + twist * S + d * thick / R
        g = pv.StructuredGrid(X, R * np.cos(A), R * np.sin(A)).extract_surface(algorithm='dataset_surface')
        out = g if out is None else out.merge(g)
    return out


def radial_vane(r0: float, r1: float, x0: float, x1: float, a0: float, sweep: float, thick: float) -> pv.PolyData:
    """Backward-swept centrifugal vane in the plane normal to x: angle ``a0 → a0+sweep`` from hub to tip."""
    r, s = np.linspace(r0, r1, 12), np.linspace(0.0, 1.0, 2)
    R, S = np.meshgrid(r, s, indexing="ij")
    A = a0 + sweep * (R - r0) / (r1 - r0)
    X = x0 + (x1 - x0) * S
    out = None
    for d in (-0.5, 0.5):
        g = pv.StructuredGrid(X, R * np.cos(A + d * thick / R), R * np.sin(A + d * thick / R)).extract_surface(algorithm='dataset_surface')
        out = g if out is None else out.merge(g)
    return out


def flange(x0: float, r_pipe: float, cut: float, side: float = 1.0) -> pv.PolyData:
    w = max(0.15 * r_pipe, 0.004)
    return shell([x0 - side * w, x0], [r_pipe] * 2, [1.6 * r_pipe] * 2, cut)


def pipe(x0: float, x1: float, r: float, cut: float, t: float | None = None, flanges: bool = True) -> list[Part]:
    t = t or max(0.08 * r, 0.0015)
    parts = [Part("pipe", shell([x0, x1], [r, r], [r + t, r + t], cut))]
    if flanges:
        parts += [Part("flange", flange(x0 + 0.15 * r, r + t, cut)), Part("flange", flange(x1, r + t, cut))]
    return parts


def fluid_color(fluid: str | None) -> str:
    return GAS if fluid in (None, "combustion_gas") else FLUID_COLORS.get(fluid, "#9ad0e0")


# ----------------------------------------------------------------------------------------------- context
class Ctx:
    """What a builder needs: the node, its graph, the solution (optional) and the cutaway angle."""

    def __init__(self, graph: dict, node: dict, sol=None, cut: float = 90.0) -> None:
        self.graph, self.node, self.sol, self.cut = graph, node, sol, float(cut)
        self.id = node["id"]
        self.nodes = {n["id"]: n for n in graph["nodes"]}

    def par(self, name: str, default=None):
        from ..core.engine_graph.edit import param_value
        v = param_value(self.node, name)[0]
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    def opt(self, name: str, default=None):
        return self.node.get("options", {}).get(name, default)

    def out(self, name: str, default=None, nid: str | None = None):
        if self.sol is None:
            return default
        return self.sol.components.get(nid or self.id, {}).get(name, default)

    def neighbours(self, kind: str | None = None) -> list[dict]:
        ids = []
        for e in self.graph["edges"]:
            if kind and e.get("kind") != kind:
                continue
            a, b = e["from"]["node"], e["to"]["node"]
            if a == self.id:
                ids.append(b)
            elif b == self.id:
                ids.append(a)
        return [self.nodes[i] for i in ids if i in self.nodes]

    def medium(self) -> str | None:
        for e in self.graph["edges"]:
            if e.get("kind") == "fluid" and self.id in (e["from"]["node"], e["to"]["node"]):
                return e.get("medium")
        return self.opt("fluid")

    def node_par(self, node: dict, name: str, default=None):
        from ..core.engine_graph.edit import param_value
        v = param_value(node, name)[0]
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default


def _mm(v: float) -> str:
    if v < 0.01:
        return f"{v * 1e3:.1f} mm"
    return f"{v * 1e3:.0f} mm" if v < 1.0 else f"{v:.2f} m"


# ----------------------------------------------------------------------------------------------- builders
def _line(c: Ctx) -> Model:
    d, L = c.par("diameter", 0.05), c.par("length", 1.0)
    r = d / 2
    shown = min(L, 40 * d)
    parts = pipe(0.0, shown, r, c.cut)
    parts.append(Part("fluid", solid([0.0, shown], [r * 0.999] * 2, c.cut), fluid_color(c.medium()), 0.55))
    notes = [] if shown == L else [f"drawn {shown:.2f} m of the {L:.2f} m length"]
    return Model(parts, "Feed line", f"bore {_mm(d)}, length {L:.2f} m", notes + ["wall and flanges illustrative"])


def _valve_body(c: Ctx, d: float, bulge: float) -> list[Part]:
    r = d / 2
    t = max(0.1 * r, 0.002)
    x = np.linspace(-1.6 * d, 1.6 * d, 40)
    ro = r + t + (bulge * r) * np.exp(-(x / (0.75 * d)) ** 2)
    ri = r + (bulge * r - t) * np.clip(np.exp(-(x / (0.75 * d)) ** 2), 0, 1)
    ri = np.maximum(ri, r)
    return [Part("valve_body", shell(x, ri, ro, c.cut)), Part("flange", flange(x[0] + 0.15 * r, r + t, c.cut)),
            Part("flange", flange(x[-1], r + t, c.cut))]


def _position(c: Ctx) -> float:
    p = c.out("position")
    if p is None:
        p = c.opt("position", 1.0)
    try:
        return float(np.clip(float(p), 0.0, 1.0))
    except (TypeError, ValueError):
        return 1.0


def _actuator(d: float, h: float) -> list[Part]:
    return [Part("stem", cyl((0, h / 2, 0), (0, 1, 0), 0.08 * d, h)),
            Part("actuator", pv.Box((-0.45 * d, 0.45 * d, h, h + 0.7 * d, -0.4 * d, 0.4 * d)), DARK)]


def _valve_ball(c: Ctx) -> Model:
    d = c.par("diameter", 0.05)
    pos = _position(c)
    ang = (1.0 - pos) * 90.0
    ball = pv.Sphere(radius=0.8 * d, theta_resolution=48, phi_resolution=48)
    bore = cyl((0, 0, 0), (1, 0, 0), 0.5 * d, 1.62 * d)
    ball = ball.rotate_y(ang, inplace=False)
    bore = bore.rotate_y(ang, inplace=False)
    parts = _valve_body(c, d, 1.0) + [Part("ball", ball, ROTOR), Part("bore", bore, SEAL)] + _actuator(d, 1.5 * d)
    return Model(parts, "Ball valve", f"bore {_mm(d)}, {pos * 100:.0f} % open",
                 ["body, ball, stem and actuator proportions illustrative"], labels={"ball": "Ball", "bore": "Ball bore"})


def _valve_butterfly(c: Ctx) -> Model:
    d = c.par("diameter", 0.1)
    pos = _position(c)
    # open (position 1): the disc lies along the flow; closed (0): its normal is the flow axis
    disc = cyl((0, 0, 0), (np.cos(np.radians(90 * pos)), 0, np.sin(np.radians(90 * pos))), 0.495 * d, 0.06 * d)
    parts = pipe(-0.6 * d, 0.6 * d, d / 2, c.cut) + [Part("disc", disc, ROTOR),
                                                       Part("stem", cyl((0, 0, 0), (0, 1, 0), 0.06 * d, 1.1 * d))]
    parts += [Part("actuator", pv.Box((-0.35 * d, 0.35 * d, 0.55 * d, 1.05 * d, -0.35 * d, 0.35 * d)), DARK)]
    return Model(parts, "Butterfly valve", f"bore {_mm(d)}, {pos * 100:.0f} % open",
                 ["spool, disc thickness and actuator illustrative"])


def _valve_poppet(c: Ctx) -> Model:
    d = c.par("diameter", 0.03)
    pos = _position(c)
    lift = 0.25 * d * pos                                   # full lift d/4 gives the bore's curtain area
    parts = _valve_body(c, d, 1.2)
    seat_x = 0.2 * d
    parts.append(Part("seat", shell([seat_x - 0.08 * d, seat_x], [0.5 * d] * 2, [1.2 * d] * 2, c.cut), DARK))
    xh = seat_x - lift
    head = pv.Cone(center=(xh - 0.25 * d, 0, 0), direction=(1, 0, 0), height=0.5 * d, radius=0.62 * d,
                   resolution=48)
    parts += [Part("poppet", head, ROTOR), Part("stem", cyl((xh - 0.9 * d, 0, 0), (1, 0, 0), 0.08 * d, 1.0 * d)),
              Part("spring", _helix(-1.3 * d, -0.5 * d, 0.25 * d, 6).tube(radius=0.03 * d), DARK)]
    return Model(parts, "Poppet valve", f"bore {_mm(d)}, lift {_mm(lift)} ({pos * 100:.0f} % open)",
                 ["body, spring and stem illustrative; full lift taken as d/4"])


def _helix(x0: float, x1: float, r: float, turns: float, n: int = 240) -> pv.PolyData:
    t = np.linspace(0, 2 * np.pi * turns, n)
    pts = np.c_[np.linspace(x0, x1, n), r * np.cos(t), r * np.sin(t)]
    return pv.lines_from_points(pts)


def _check_valve(c: Ctx) -> Model:
    d = c.par("diameter", 0.03)
    parts = _valve_body(c, d, 0.9)
    flap = cyl((0.05 * d, 0, 0), (1, 0, 0), 0.6 * d, 0.05 * d).rotate_y(-35.0, point=(0, 0, 0.6 * d), inplace=False)
    parts += [Part("seat", shell([-0.05 * d, 0.05 * d], [0.5 * d] * 2, [1.1 * d] * 2, c.cut), DARK),
              Part("flapper", flap, ROTOR)]
    return Model(parts, "Check valve", f"bore {_mm(d)}", ["swing-disc layout illustrative"])


def _orifice(c: Ctx) -> Model:
    d = c.par("diameter", 0.02)
    D = max(2.0 * d, d + 0.01)
    parts = pipe(-1.5 * D, 1.5 * D, D / 2, c.cut, flanges=False)
    parts.append(Part("plate", shell([-0.03 * D, 0.03 * D], [d / 2] * 2, [0.8 * D] * 2, c.cut), ROTOR))
    return Model(parts, "Orifice", f"orifice {_mm(d)} in a {_mm(D)} pipe", ["pipe bore and plate thickness illustrative"])


def _venturi(c: Ctx) -> Model:
    dt = c.par("throat_diameter", 0.01)
    D = 2.5 * dt
    x = np.r_[np.linspace(-2.0 * D, 0.0, 20), np.linspace(0.0, 4.0 * D, 30)[1:]]
    r = np.where(x < 0, dt / 2 + (D - dt) / 2 * np.clip(-x / (1.2 * D), 0, 1),
                 dt / 2 + (D - dt) / 2 * np.clip(x / (3.5 * D), 0, 1))
    parts = [Part("venturi", shell(x, r, np.full_like(x, D / 2 + 0.12 * D), c.cut)),
             Part("fluid", solid(x, r * 0.999, c.cut), fluid_color(c.medium()), 0.55)]
    return Model(parts, "Cavitating venturi", f"throat {_mm(dt)}",
                 ["inlet bore 2.5 throat diameters; 20° inlet and 7° diffuser cones illustrative"])


def _manifold(c: Ctx) -> Model:
    V = c.par("volume", 0.01)
    # torus with minor radius a quarter of the major: V = 2 pi^2 R a^2 = pi^2 R^3 / 8
    R = (8.0 * V / np.pi**2) ** (1 / 3)
    a = R / 4
    t = np.linspace(0, 2 * np.pi, 48)
    xo, ro = a * np.cos(t), R + a * np.sin(t)
    xi, ri = 0.9 * a * np.cos(t), R + 0.9 * a * np.sin(t)
    th = thetas(c.cut, 72)
    torus = _merge([rev_grid(xo, ro, th)])
    parts = [Part("manifold", torus, STEEL, 0.6 if c.cut else 1.0),
             Part("fluid", _merge([rev_grid(xi, ri, th)]), fluid_color(c.medium()), 0.7),
             Part("inlet", cyl((0, R + a + 0.6 * a, 0), (0, 1, 0), 0.5 * a, 1.2 * a))]
    return Model(parts, "Manifold", f"volume {V * 1e3:.1f} L (torus, major radius {_mm(R)})",
                 ["toroidal shape and its 4:1 radius ratio illustrative"])


def _tank(c: Ctx) -> Model:
    V, ff = c.par("V_tank", 1.0), c.par("fill_fraction", 0.9)
    # cylinder of length D with hemispherical domes: V = 5 pi D^3 / 12
    D = (12.0 * V / (5.0 * np.pi)) ** (1 / 3)
    R = D / 2
    t = max(0.004 * D, 0.002)
    s = np.linspace(0, np.pi / 2, 24)
    x = np.r_[-R * np.cos(s), np.linspace(0, D, 10)[1:-1], D + R * np.sin(s)]
    r = np.r_[R * np.sin(s), np.full(8, R), R * np.cos(s)]
    r[0], r[-1] = 1e-4, 1e-4
    wall = shell(x, r, r + t, c.cut, ends=False)
    # liquid up to the fill level (the tank stands on its x axis: x = -R is the bottom)
    Vl, lo, hi = ff * V, -R, D + R
    for _ in range(60):
        h = 0.5 * (lo + hi)
        m = x <= h
        xs = np.r_[x[m], h]
        rs = np.r_[r[m], np.interp(h, x, r)]
        v = np.trapezoid(np.pi * rs**2, xs)
        lo, hi = (h, hi) if v < Vl else (lo, h)
    liq = solid(xs, rs * 0.995, c.cut)
    parts = [Part("tank_wall", wall, STEEL, 0.55 if c.cut == 0 else 1.0),
             Part("propellant", liq, fluid_color(c.opt("fluid") or c.medium()), 0.85),
             Part("outlet", cyl((-R - 0.1 * R, 0, 0), (1, 0, 0), 0.08 * D, 0.25 * R))]
    if c.node["type"] == "tank_regulated":
        parts.append(Part("pressurant_inlet", cyl((D + R + 0.1 * R, 0, 0), (1, 0, 0), 0.04 * D, 0.25 * R), DARK))
    fluid = c.opt("fluid") or c.medium() or ""
    return Model(parts, "Regulated tank" if c.node["type"] == "tank_regulated" else "Blowdown tank",
                 f"{V:.3g} m³, diameter {D:.2f} m, {ff * 100:.0f} % full of {fluid}",
                 ["cylinder of length D with hemispherical domes (shape illustrative, volume from the part)"],
                 labels={"propellant": f"Liquid {fluid}" if fluid else "Propellant"}, vertical=True)


def _shaft_d(c: Ctx) -> float:
    ds = [c.node_par(n, "D_tip") or c.node_par(n, "D_mean") for n in c.neighbours("shaft")]
    ds = [d for d in ds if d]
    return 0.18 * max(ds) if ds else 0.06


def _pump(c: Ctx) -> Model:
    D = c.par("D_tip", 0.2)
    R = D / 2
    stages = int(float(c.opt("stages", 1) or 1))
    kind = c.opt("impeller", "centrifugal")
    rs = max(0.09 * D, 0.006)
    parts: list[Part] = []
    notes = ["casing, vane counts and hub sizes illustrative"]
    if kind == "axial":
        Lb = 0.6 * D
        hub = solid([-0.2 * D, 0.0, Lb, Lb + 0.15 * D], [0.0, 0.3 * R, 0.45 * R, 0.0], 0.0)
        parts.append(Part("hub", hub, ROTOR))
        for k in range(4):
            parts.append(Part("blade", blade(0.4 * R, 0.98 * R, 0.05 * D, Lb - 0.05 * D, 2 * np.pi * k / 4,
                                             0.9 * np.pi, 0.012 * D), ROTOR))
        x = np.array([-0.4 * D, Lb + 0.4 * D])
        parts.append(Part("casing", shell(x, [R * 1.01] * 2, [R * 1.12] * 2, c.cut)))
        parts.append(Part("shaft", cyl((Lb + 0.6 * D, 0, 0), (1, 0, 0), rs, 0.6 * D), DARK))
        title = "Axial pump"
    else:
        w = 0.12 * D
        L = stages * 0.45 * D
        for s in range(stages):
            x0 = s * 0.45 * D
            parts.append(Part("hub", solid([x0, x0 + 0.02 * D], [R, R], 0.0), ROTOR))
            parts.append(Part("hub", solid([x0 - 0.1 * D, x0], [0.25 * R, 0.32 * R], 0.0), ROTOR))
            for k in range(7):
                parts.append(Part("vane", radial_vane(0.32 * R, R, x0 - w, x0, 2 * np.pi * k / 7, -0.9,
                                                      0.01 * D), ROTOR))
        # shroud: from the eye, around the impeller tips to the back plate
        xs = np.r_[np.linspace(-0.35 * D, -w - 0.03 * D, 8), np.linspace(-w, L + 0.03 * D, 6)]
        ri = np.r_[np.linspace(0.38 * R, 1.06 * R, 8) ** 1.0, np.full(6, 1.06 * R)]
        ri[:8] = 0.38 * R + (1.06 * R - 0.38 * R) * np.sin(np.linspace(0, np.pi / 2, 8))
        parts.append(Part("casing", shell(xs, ri, ri + 0.06 * R, c.cut)))
        parts.append(Part("back_plate", shell([L + 0.03 * D, L + 0.07 * D], [rs] * 2, [1.12 * R] * 2, c.cut)))
        # volute: a tube spiralling around the last stage, growing toward the discharge
        a = np.linspace(0, 2 * np.pi, 80)
        rv = 0.06 * D + 0.1 * D * a / (2 * np.pi)
        xv = (stages - 1) * 0.45 * D - 0.05 * D
        pts = np.c_[np.full_like(a, xv), (1.14 * R + rv) * np.cos(a), (1.14 * R + rv) * np.sin(a)]
        pts = np.r_[pts, [[xv, (1.14 * R + rv[-1]) + 0.0, -0.6 * D]]]
        spl = pv.Spline(pts, 160)
        spl["r"] = np.r_[np.linspace(0.06 * D, 0.16 * D, 140), np.full(20, 0.16 * D)]
        volute = spl.tube(scalars="r", absolute=True, n_sides=24, capping=True)
        volute.point_data.clear()
        parts.append(Part("volute", volute, STEEL, 0.55 if c.cut else 1.0))
        parts.append(Part("eye", shell([-0.9 * D, -0.35 * D], [0.32 * R] * 2, [0.38 * R] * 2, c.cut)))
        parts.append(Part("shaft", cyl((L + 0.35 * D, 0, 0), (1, 0, 0), rs, 0.7 * D), DARK))
        title = f"Centrifugal pump ({stages} stage{'s' if stages > 1 else ''})"
    N = c.out("speed") or c.par("N_design")
    rpm = f", {N * 60 / (2 * np.pi):,.0f} rpm" if N else ""
    return Model(parts, title, f"impeller tip {_mm(D)}{rpm}", notes)


def _inducer(c: Ctx) -> Model:
    pumps = [c.node_par(n, "D_tip") for n in c.neighbours("shaft") if n["type"] == "pump_centrifugal"]
    D = 0.6 * pumps[0] if pumps and pumps[0] else 0.1
    R = D / 2
    L = 0.5 * D
    parts = [Part("hub", solid([-0.1 * D, 0.0, L, L + 0.1 * D], [0.0, 0.25 * R, 0.4 * R, 0.4 * R], 0.0), ROTOR)]
    for k in range(3):
        parts.append(Part("blade", blade(0.3 * R, 0.98 * R, 0.0, L, 2 * np.pi * k / 3, 1.6 * np.pi, 0.01 * D), ROTOR))
    parts.append(Part("casing", shell([-0.3 * D, L + 0.3 * D], [1.01 * R] * 2, [1.1 * R] * 2, c.cut)))
    return Model(parts, "Inducer", f"tip {_mm(D)}",
                 ["tip diameter taken as 0.6 of the pump impeller on its shaft (the library has no inducer size)",
                  "three blades, hub and lead illustrative"])


def _turbine(c: Ctx) -> Model:
    Dm = c.par("D_mean", 0.3)
    Rm = Dm / 2
    arr = str(c.opt("arrangement", "") or "")
    rows = 6 if "6stage" in arr else (2 if ("2" in arr) else 1)
    h = 0.12 * Dm if rows < 6 else 0.08 * Dm
    pitch = 0.16 * Dm
    parts: list[Part] = []
    for s in range(rows):
        x0 = s * pitch
        parts.append(Part("disk", solid([x0 - 0.03 * Dm, x0 + 0.03 * Dm], [Rm - h / 2] * 2, 0.0), ROTOR))
        nb = 48
        for k in range(nb):
            parts.append(Part("blade", blade(Rm - h / 2, Rm + h / 2, x0 - 0.035 * Dm, x0 + 0.035 * Dm,
                                             2 * np.pi * k / nb, 0.35, 0.004 * Dm, nr=3, nx=4), ROTOR))
        xs = x0 - 0.5 * pitch
        for k in range(36 if s > 0 else 0):
            parts.append(Part("stator", blade(Rm - h / 2, Rm + h / 2, xs - 0.03 * Dm, xs + 0.03 * Dm,
                                              2 * np.pi * k / 36, -0.35, 0.004 * Dm, nr=3, nx=4), STATOR))
    L = (rows - 1) * pitch
    x = np.array([-0.35 * Dm, L + 0.3 * Dm])
    parts.append(Part("casing", shell(x, [Rm + 0.6 * h] * 2, [Rm + 0.75 * h] * 2, c.cut)))
    t = np.linspace(0, 2 * np.pi, 40)
    a = 0.12 * Dm
    torus = rev_grid(-0.25 * Dm + a * np.cos(t), Rm + 0.4 * h + a + a * np.sin(t), thetas(c.cut, 72))
    parts.append(Part("inlet_manifold", _merge([torus]), STEEL, 0.6 if c.cut else 1.0))
    parts.append(Part("shaft", cyl((-0.5 * Dm, 0, 0), (1, 0, 0), 0.06 * Dm, 0.8 * Dm), DARK))
    names = {1: "single-row impulse", 2: "two-row", 6: "six-stage hydraulic"}
    T = c.out("T_inlet") or c.par("T_inlet")
    extra = f", inlet {T:.0f} K" if T else ""
    return Model(parts, f"Turbine ({names.get(rows, arr)})", f"mean blade diameter {_mm(Dm)}{extra}",
                 ["blade counts, heights and casing illustrative"],
                 labels={"blade": "Rotor blades", "stator": "Stator vanes"})


def _shaft(c: Ctx) -> Model:
    d = _shaft_d(c)
    L = 12 * d
    parts = [Part("shaft", cyl((L / 2, 0, 0), (1, 0, 0), d / 2, L), DARK)]
    for xb in (0.15 * L, 0.85 * L):
        parts.append(Part("bearing", shell([xb - 0.4 * d, xb + 0.4 * d], [0.5 * d] * 2, [0.9 * d] * 2, c.cut), STEEL))
    rpm = c.out("speed")
    return Model(parts, "Shaft", f"diameter {_mm(d)}" + (f", {rpm * 60 / (2 * np.pi):,.0f} rpm" if rpm else ""),
                 ["diameter 0.18 of the largest wheel on the shaft; length and bearings illustrative"])


# ---- combustion devices ------------------------------------------------------------------------------------
def _chamber_node(c: Ctx) -> dict | None:
    if c.node["type"] == "chamber":
        return c.node
    for e in c.graph["edges"]:
        if e.get("kind") == "thermal" and c.id in (e["from"]["node"], e["to"]["node"]):
            other = e["from"]["node"] if e["to"]["node"] == c.id else e["to"]["node"]
            n = c.nodes.get(other)
            if n and n["type"] == "chamber":
                return n
    if c.node["type"] == "nozzle_bell":
        for e in c.graph["edges"]:
            if e["to"]["node"] == c.id and c.nodes.get(e["from"]["node"], {}).get("type") == "chamber":
                return c.nodes[e["from"]["node"]]
    return next((n for n in c.graph["nodes"] if n["type"] == "chamber"), None)


def _nozzle_node(c: Ctx, ch: dict | None) -> dict | None:
    if ch is None:
        return None
    for e in c.graph["edges"]:
        if e["from"]["node"] == ch["id"] and c.nodes.get(e["to"]["node"], {}).get("type") == "nozzle_bell":
            return c.nodes[e["to"]["node"]]
    return None


def engine_contour(c: Ctx):
    """``(x, r, Rt, eps, Lc)`` of the chamber and nozzle the part belongs to (injector face at ``x[0]``)."""
    from ..core.engine_graph.jacket import chamber_contour
    from ..core.liquid import chamber_geometry
    ch = _chamber_node(c)
    noz = _nozzle_node(c, ch)
    At = c.node_par(ch, "A_t", 0.01) if ch else 0.01
    Ls = c.node_par(ch, "Lstar", 1.0) if ch else 1.0
    cr = c.node_par(ch, "contraction_ratio", 3.0) if ch else 3.0
    eps = c.node_par(noz, "expansion_ratio", 4.0) if noz else 4.0
    bell = c.node_par(noz, "bell_fraction", 0.8) if noz else 0.8
    g = (c.out("gamma", None, ch["id"]) if ch else None) or 1.2
    Dt = float(np.sqrt(4 * At / np.pi))
    x, r = chamber_contour(Dt, eps, Ls, cr, g, bell)
    return x, r, Dt / 2, eps, chamber_geometry(Ls, Dt, cr)["Lc"]


def _wall_fields(c: Ctx, x: np.ndarray) -> dict[str, np.ndarray]:
    """Jacket march results (gas-side wall T, heat flux, coolant T) on the stations ``x`` (NaN where uncooled)."""
    out: dict[str, np.ndarray] = {}
    if c.sol is None:
        return out
    ch = _chamber_node(c)
    jk = [n["id"] for n in c.graph["nodes"] if n["type"] == "regen_jacket"]
    if c.node["type"] == "regen_jacket":
        jk = [c.id]
    elif ch is not None:
        jk = [j for j in jk if _chamber_node(Ctx(c.graph, c.nodes[j], c.sol)) is ch]
    for key in ("T_wg", "q", "T_cool"):
        vals = np.full(len(x), np.nan)
        for j in jk:
            p = c.sol.profiles.get(j)
            if not p:
                continue
            px, pv_ = np.asarray(p["x"], float), np.asarray(p[key], float)
            o = np.argsort(px)
            inside = (x >= px.min() - 1e-9) & (x <= px.max() + 1e-9)
            vals[inside] = np.interp(x[inside], px[o], pv_[o])
        if np.isfinite(vals).any():
            out[key] = vals
    return out


FIELD_TITLES = {"T_wg": (r"Wall $T_{wg}$ (K)", "inferno"), "q": (r"Heat flux $q$ (MW/m$^2$)", "plasma"),
                "T_cool": (r"Coolant $T$ (K)", "viridis"), "p": (r"Pressure $p$ (MPa)", "turbo"),
                "T_f": (r"Fluid $T$ (K)", "inferno")}


def _revolved_with_fields(c: Ctx, x, ri, ro, name: str, color: str) -> tuple[Part, dict]:
    th = thetas(c.cut, 72)
    m = shell(x, ri, ro, c.cut, n=72)
    f = _wall_fields(c, np.asarray(x, float))
    fields = {}
    if f:
        # rebuild the skin as a grid so the per-station values map onto it
        g = _merge([rev_grid(x, ri, th), rev_grid(x, ro, th)])
        for k, v in f.items():
            vv = v * (1e-6 if k == "q" else 1.0)
            g.point_data[k] = np.interp(g.points[:, 0], x, np.where(np.isfinite(vv), vv, np.nan))
            fields[k] = FIELD_TITLES[k]
        return Part(name, g, color, 1.0, next(iter(fields))), fields
    return Part(name, m, color), fields


def _chamber(c: Ctx) -> Model:
    x, r, Rt, eps, Lc = engine_contour(c)
    keep = x <= 0.0
    xc, rc = x[keep], r[keep]
    t = max(0.03 * Rt, 0.002)
    liner, fields = _revolved_with_fields(c, xc, rc, rc + t, "liner", COPPER)
    Rc = rc[0]
    x0 = xc[0]
    dx = np.linspace(x0 - 0.45 * Rc, x0, 12)
    dr = Rc * np.sqrt(np.clip(1 - ((dx - x0) / (0.45 * Rc)) ** 2, 0, 1)) + t
    dome = shell(dx, dr - t, dr, c.cut, ends=False)
    parts = [liner, Part("injector_dome", dome, STEEL),
             Part("injector_face", shell([x0 - 0.02 * Rc, x0], [0.0, 0.0], [Rc, Rc], c.cut), DARK),
             Part("combustion_gas", solid(xc, rc * 0.995, c.cut), GAS, 0.35)]
    pc = c.out("p_c")
    st = f"throat {_mm(2 * Rt)}, chamber {_mm(2 * Rc)} × {Lc:.2f} m (L* and contraction ratio)"
    return Model(parts, "Thrust chamber", st + (f", pc {pc / 1e6:.2f} MPa" if pc else ""),
                 ["liner thickness and injector dome shape illustrative"], fields,
                 labels={"liner": "Chamber liner", "combustion_gas": "Combustion gas"})


def _nozzle(c: Ctx) -> Model:
    x, r, Rt, eps, _ = engine_contour(c)
    keep = x >= 0.0
    xn, rn = x[keep], r[keep]
    t = max(0.01 * Rt, 0.0015)
    cool = c.opt("cooling", "regenerative")
    color = {"radiative": "#8a6d5a", "ablative": "#3a3533", "uncooled": STEEL}.get(cool, COPPER)
    wall, fields = _revolved_with_fields(c, xn, rn, rn + t, "nozzle_wall", color)
    parts = [wall]
    F = c.out("thrust")
    return Model(parts, f"Bell nozzle ({cool})", f"area ratio {eps:g}, exit {_mm(2 * rn[-1])}, length {xn[-1]:.2f} m"
                 + (f", thrust {F / 1e3:,.0f} kN" if F else ""), ["wall thickness illustrative"], fields,
                 labels={"nozzle_wall": "Nozzle wall"})


def _jacket(c: Ctx) -> Model:
    x, r, Rt, eps, _ = engine_contour(c)
    section = c.opt("section", "full")
    split = float(c.opt("split_area_ratio", 5.0) or 5.0)
    A = (r / Rt) ** 2
    keep = {"chamber": (x <= 0) | (A <= split), "nozzle": (x > 0) & (A >= split)}.get(section, np.ones_like(x, bool))
    xs, rs = x[keep], r[keep]
    tw, w, hgt = c.par("wall_thickness", 0.001), c.par("channel_width", 0.002), c.par("channel_height", 0.003)
    n = int(round(c.par("n_channels", 100)))
    liner, fields = _revolved_with_fields(c, xs, rs, rs + tw, "liner", COPPER)
    parts = [liner, Part("closeout", shell(xs, rs + tw + hgt, rs + 2 * tw + hgt, c.cut), STEEL, 0.35)]
    # ribs between channels: drawn up to 180 of them (every k-th) on the uncut arc
    th_max = np.radians(360.0 - c.cut)
    k = max(1, int(np.ceil(n / 180)))
    pitch = 2 * np.pi / n
    for i in range(0, n, k):
        a = i * pitch + pitch / 2
        if a > th_max:
            break
        rib_w = max(pitch * (rs.mean() + tw) - w, 0.3 * w)
        X, Rr = np.meshgrid(xs, [0.0, hgt], indexing="ij")
        R0 = rs[:, None] + tw + Rr
        blocks = []
        for d in (-0.5, 0.5):
            A_ = a + d * rib_w / R0
            blocks.append(pv.StructuredGrid(X, R0 * np.cos(A_), R0 * np.sin(A_)))
        parts.append(Part("rib", _merge(blocks), "#9a6a45"))
    f = _wall_fields(c, xs)
    if "T_cool" in f:
        th = thetas(c.cut, 72)
        g = rev_grid(xs, rs + tw + 0.5 * hgt, th).extract_surface(algorithm='dataset_surface')
        g.point_data["T_cool"] = np.interp(g.points[:, 0], xs, f["T_cool"])
        parts.append(Part("coolant", g, "#5fb0ff", 0.85, "T_cool"))
    Q = c.out("Q")
    note = [] if k == 1 else [f"one rib in {k} drawn ({n} channels)"]
    return Model(parts, f"Regenerative jacket ({section})",
                 f"{n} channels {_mm(w)} × {_mm(hgt)}, wall {_mm(tw)}" + (f", Q {Q / 1e6:.1f} MW" if Q else ""),
                 note + ["closeout thickness illustrative"], fields, labels={"liner": "Hot-gas wall", "rib": "Lands"})


def _injector(c: Ctx) -> Model:
    At = c.par("A_total", 0.001)
    ch = _chamber_node(c)
    downstream = [c.nodes[e["to"]["node"]] for e in c.graph["edges"] if e["from"]["node"] == c.id
                  and e["to"]["node"] in c.nodes]
    into_chamber = any(n["type"] == "chamber" for n in downstream)
    if into_chamber and ch is not None:
        Dc = float(np.sqrt(4 * c.node_par(ch, "A_t", 0.01) * c.node_par(ch, "contraction_ratio", 3.0) / np.pi))
        note = "face diameter from the chamber"
    else:
        Dc = float(np.sqrt(4 * At * 12 / np.pi))
        note = "face diameter taken as 12 × the orifice area (preburner or GG face)"
    R = Dc / 2
    t = 0.04 * R
    xd = np.linspace(-0.45 * R, -t, 10)
    rd = R * np.sqrt(np.clip(1 - (xd / (0.45 * R)) ** 2, 0.02, 1))
    parts = [Part("injector_face", shell([-t, 0.0], [0.0, 0.0], [R, R], c.cut), STEEL),
             Part("injector_dome", shell(xd, 0.95 * rd, rd, c.cut, ends=False), STEEL, 0.6 if c.cut else 1.0)]
    # elements on concentric rings, total orifice area = A_total
    n_el = int(np.clip(round(At / (np.pi / 4 * (0.06 * R) ** 2)), 6, 400))
    d_el = float(np.sqrt(4 * At / (np.pi * n_el)))
    rings = max(1, int(round(np.sqrt(n_el / 3.0))))
    th_max = np.radians(360.0 - c.cut)
    for j in range(rings):
        rr = R * (j + 0.6) / (rings + 0.3)
        m = max(1, int(round(n_el * rr / sum(R * (i + 0.6) / (rings + 0.3) for i in range(rings)))))
        for k in range(m):
            a = 2 * np.pi * (k + 0.5 * (j % 2)) / m
            if a > th_max:
                continue
            ctr = (0.001, rr * np.cos(a), rr * np.sin(a))
            parts.append(Part("element", cyl(ctr, (1, 0, 0), d_el / 2, 0.003, res=12), SEAL))
            if (c.opt("type") or "") == "coaxial":
                parts.append(Part("element_post", shell([0.0, 0.002], [d_el / 2] * 2, [0.75 * d_el] * 2, 0.0,
                                                        n=12).translate(ctr, inplace=False), DARK))
    return Model(parts, f"Injector ({c.opt('type') or 'impinging'})",
                 f"face {_mm(Dc)}, {n_el} elements of {d_el * 1e3:.1f} mm, total {At * 1e4:.2f} cm²",
                 [note, "element count and pattern illustrative (total orifice area from the part)"],
                 labels={"element": "Injector orifices"})


def _gas_generator(c: Ctx) -> Model:
    m = c.out("mdot") or 10.0
    D = 0.12 * np.sqrt(max(m, 0.1) / 10.0) + 0.05
    R = D / 2
    L = 2.5 * D
    t = 0.04 * D
    x = np.linspace(0, L, 20)
    parts = [Part("gg_wall", shell(x, np.full_like(x, R), np.full_like(x, R + t), c.cut)),
             Part("injector_face", shell([-t, 0.0], [0.0, 0.0], [R, R], c.cut), DARK),
             Part("combustion_gas", solid(x, np.full_like(x, R * 0.99), c.cut), GAS, 0.4),
             Part("outlet_duct", shell([L, L + 1.2 * D], [0.45 * R] * 2, [0.45 * R + t] * 2, c.cut))]
    T = c.out("T_out")
    return Model(parts, "Gas generator / preburner", f"diameter {_mm(D)}" + (f", {T:.0f} K out" if T else ""),
                 ["size scaled with the solved flow for the picture only (the library has no geometry for it)"],
                 labels={"gg_wall": "Combustor wall", "combustion_gas": "Drive gas"})


BUILDERS = {"line": _line, "valve_ball": _valve_ball, "valve_butterfly": _valve_butterfly,
            "valve_poppet": _valve_poppet, "check_valve": _check_valve, "orifice": _orifice,
            "cavitating_venturi": _venturi, "manifold": _manifold, "tank_regulated": _tank, "tank_blowdown": _tank,
            "pump_centrifugal": _pump, "inducer": _inducer, "turbine_impulse": _turbine, "shaft": _shaft,
            "chamber": _chamber, "nozzle_bell": _nozzle, "regen_jacket": _jacket, "injector": _injector,
            "gas_generator": _gas_generator}


def build(graph: dict, node_id: str, sol=None, cut: float = 90.0) -> Model:
    """3-D model of component ``node_id`` of ``graph`` (with the solved state of ``sol`` when given)."""
    node = next(n for n in graph["nodes"] if n["id"] == node_id)
    c = Ctx(graph, node, sol, cut)
    m = BUILDERS[node["type"]](c)
    # turn the removed wedge toward the default (isometric) camera, and stand tanks on end
    for p in m.parts:
        p.mesh = p.mesh.rotate_x((-45.0 if m.vertical else 45.0) + 0.5 * c.cut, inplace=False)
        if m.vertical:
            p.mesh = p.mesh.rotate_y(-90.0, inplace=False)
    return m


def merged(m: Model) -> list[Part]:
    """Parts with the same name, colour and field joined into one mesh (one actor each, for a fast scene)."""
    groups: dict[tuple, Part] = {}
    for p in m.parts:
        k = (p.name, p.color, p.opacity, p.scalars)
        if k in groups:
            groups[k].mesh = groups[k].mesh.merge(p.mesh)
        else:
            groups[k] = Part(p.name, p.mesh.copy(), p.color, p.opacity, p.scalars)
    return list(groups.values())
