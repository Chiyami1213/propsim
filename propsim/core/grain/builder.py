r"""User-built grain cross-sections: a port drawn as a list of shapes, with optional inhibited surfaces.

A section is the outer radius, a list of :class:`Shape` voids (circle, rectangle, radial slot, polygon, star) and the
flag ``outer_burn``.  Each shape can be repeated around the axis (``repeat``) and can be marked ``inhibited``: its
surface is coated and does not regress, so it keeps its initial outline while the burning surfaces move around it.

With the burning shapes' signed distance :math:`F=\min_i\phi_i`, the inhibited shapes' :math:`S=\min_j\phi_j` and,
when the outside also burns, :math:`O=R_{out}-r`, the propellant left after a burn distance :math:`d` is

.. math::  \{x:\ \min(F,O)>d\}\ \cap\ \{S>0\}\ \cap\ \{r<R_{out}\},

and the burning perimeter is the length of the level :math:`\min(F,O)=d` in :math:`\{S>0\}` (coated surfaces do not
count).  The result is the same :class:`~propsim.core.grain.section2d.Burnback` table the fixed section kinds give, so
the ballistics code does not care how a section was made.  Lengths are in mm, angles in degrees.

JSON form (what the editor stores)::

    {"R_out": 60, "outer_burn": false,
     "shapes": [{"kind": "circle", "r": 14},
                {"kind": "slot", "angle": 0, "r0": 0, "r1": 45, "w": 8, "repeat": 6},
                {"kind": "circle", "r": 5, "cx": 30, "inhibited": true}]}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache

import contourpy
import numpy as np

from .section2d import NGRID, Burnback, grid, rot, sd_circle, sd_polygon, sd_star

# kind -> (required parameters, optional parameters with defaults)
SHAPES: dict[str, tuple[tuple[str, ...], dict[str, float]]] = {
    "circle": (("r",), {"cx": 0.0, "cy": 0.0}),
    "box": (("w", "h"), {"cx": 0.0, "cy": 0.0, "angle": 0.0}),
    "slot": (("r1", "w"), {"r0": 0.0, "angle": 0.0}),
    "polygon": (("pts",), {}),
    "star": (("n", "r_tip", "r_valley"), {}),
}
SHAPE_LABELS = {"circle": "Circle", "box": "Rectangle", "slot": "Radial slot", "polygon": "Polygon", "star": "Star"}


@dataclass(frozen=True)
class Shape:
    kind: str
    params: tuple[tuple[str, object], ...] = ()
    inhibited: bool = False
    repeat: int = 1

    def p(self, name: str):
        return dict(self.params)[name]

    def to_dict(self) -> dict:
        d = {"kind": self.kind, **{k: (list(map(list, v)) if k == "pts" else v) for k, v in self.params}}
        if self.inhibited:
            d["inhibited"] = True
        if self.repeat != 1:
            d["repeat"] = self.repeat
        return d


@dataclass(frozen=True)
class SectionSpec:
    R_out: float
    shapes: tuple[Shape, ...] = field(default_factory=tuple)
    outer_burn: bool = False

    def to_dict(self) -> dict:
        return {"R_out": self.R_out, "outer_burn": self.outer_burn, "shapes": [s.to_dict() for s in self.shapes]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def make_shape(kind: str, inhibited: bool = False, repeat: int = 1, **params) -> Shape:
    """Validate a shape and fill in its default parameters."""
    if kind not in SHAPES:
        raise ValueError(f"unknown shape kind {kind!r}; use one of {sorted(SHAPES)}")
    req, opt = SHAPES[kind]
    missing = [k for k in req if k not in params]
    if missing:
        raise ValueError(f"{kind}: missing parameter(s) {missing}")
    extra = [k for k in params if k not in req and k not in opt]
    if extra:
        raise ValueError(f"{kind}: unknown parameter(s) {extra}")
    full = {**opt, **params}
    if kind == "polygon":
        pts = np.asarray(full["pts"], float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise ValueError("polygon: pts must be a list of at least three [x, y] pairs")
        full["pts"] = tuple(map(tuple, pts.tolist()))
    else:
        for k, v in full.items():
            if not np.isfinite(float(v)):
                raise ValueError(f"{kind}: {k} must be finite")
            full[k] = float(v)
    if kind in ("circle", "slot") and full.get("r", full.get("w", 1.0)) <= 0:
        raise ValueError(f"{kind}: size must be positive")
    if kind == "box" and (full["w"] <= 0 or full["h"] <= 0):
        raise ValueError("box: w and h must be positive")
    if kind == "slot" and (full["w"] <= 0 or full["r1"] <= full["r0"]):
        raise ValueError("slot: need w > 0 and r1 > r0")
    if kind == "star":
        full["n"] = int(round(full["n"]))
        if full["n"] < 3 or not 0 < full["r_valley"] < full["r_tip"]:
            raise ValueError("star: need n >= 3 and 0 < r_valley < r_tip")
    repeat = int(repeat)
    if not 1 <= repeat <= 48:
        raise ValueError("repeat must be between 1 and 48")
    return Shape(kind, tuple(sorted(full.items())), bool(inhibited), repeat)


def spec_from_dict(d: dict) -> SectionSpec:
    if not isinstance(d, dict) or "R_out" not in d or "shapes" not in d:
        raise ValueError("a section needs 'R_out' and 'shapes'")
    R = float(d["R_out"])
    if not R > 0:
        raise ValueError("R_out must be positive")
    shapes = []
    for s in d["shapes"]:
        s = dict(s)
        kind = s.pop("kind", None)
        shapes.append(make_shape(kind, bool(s.pop("inhibited", False)), s.pop("repeat", 1), **s))
    if not shapes:
        raise ValueError("a section needs at least one shape")
    return SectionSpec(R, tuple(shapes), bool(d.get("outer_burn", False)))


def spec_from_json(text: str) -> SectionSpec:
    try:
        return spec_from_dict(json.loads(text))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e


def _rot_by(X, Y, ang):
    return rot(X, Y, ang)


def _sd_obox(X, Y, cx, cy, hw, hh, ang):
    """Exact signed distance to a box rotated by ``ang`` (rad) about its centre ``(cx, cy)``."""
    u, v = _rot_by(X - cx, Y - cy, ang)
    qx, qy = np.abs(u) - hw, np.abs(v) - hh
    return np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) + np.minimum(np.maximum(qx, qy), 0.0)


def shape_sdf(s: Shape, X, Y) -> np.ndarray:
    """Signed distance (negative inside the void) of one shape, including its polar repeats."""
    out = None
    for k in range(s.repeat):
        u, v = _rot_by(X, Y, 2.0 * np.pi * k / s.repeat)         # evaluating at the rotated point = rotating the shape
        q = dict(s.params)
        if s.kind == "circle":
            f = sd_circle(u, v, q["r"], q["cx"], q["cy"])
        elif s.kind == "box":
            f = _sd_obox(u, v, q["cx"], q["cy"], q["w"] / 2, q["h"] / 2, np.radians(q["angle"]))
        elif s.kind == "slot":
            hl = (q["r1"] - q["r0"]) / 2
            ang = np.radians(q["angle"])
            f = _sd_obox(u, v, (q["r0"] + hl) * np.cos(ang), (q["r0"] + hl) * np.sin(ang), hl, q["w"] / 2, ang)
        elif s.kind == "polygon":
            f = sd_polygon(u, v, np.asarray(q["pts"], float))
        else:
            f = sd_star(u, v, q["n"], q["r_tip"], q["r_valley"])
        out = f if out is None else np.minimum(out, f)
    return out


def _fields(spec: SectionSpec, X, Y):
    burn = [shape_sdf(s, X, Y) for s in spec.shapes if not s.inhibited]
    coat = [shape_sdf(s, X, Y) for s in spec.shapes if s.inhibited]
    F = np.minimum.reduce(burn) if burn else None
    S = np.minimum.reduce(coat) if coat else None
    return F, S


def front_field(spec: SectionSpec, X, Y):
    """``(H, S)``: the burn-front distance field (``None`` when nothing burns) and the coated-void field (or ``None``)."""
    F, S = _fields(spec, X, Y)
    H = F
    if spec.outer_burn:
        Ro = spec.R_out - np.hypot(X, Y)
        H = Ro if H is None else np.minimum(H, Ro)
    return H, S


def build_section(spec: SectionSpec, n_grid: int = NGRID, n: int = 31) -> Burnback:
    """Burnback tables (``d``, port area, burning perimeter, ...) of a user-built section; lengths in mm."""
    R = spec.R_out
    X, Y, h = grid(R * (1.0 + 4.0 / n_grid) if spec.outer_burn else R, n_grid)   # pad: the outer front touches the edge
    H, S = front_field(spec, X, Y)
    if H is None:
        raise ValueError("nothing burns: mark at least one shape as burning or set outer_burn")
    r = np.hypot(X, Y)
    inside = r <= R
    cov_in = np.clip((R - r) / h + 0.5, 0.0, 1.0)
    covS = np.ones_like(H) if S is None else np.clip(S / h + 0.5, 0.0, 1.0)
    prop_ok = inside & (covS > 0.5)
    if not prop_ok.any() or float(H[prop_ok].max()) <= 0.0:
        raise ValueError("the shapes leave no propellant to burn")
    web = float(H[prop_ok].max())
    A_case = float(cov_in.sum() * h * h)
    d = np.linspace(0.0, web, n)
    area = np.array([float(((1.0 - np.clip((H - di) / h + 0.5, 0.0, 1.0) * covS) * cov_in).sum() * h * h) for di in d])
    mask = ~(prop_ok if not spec.outer_burn else ((r <= R + 2.0 * h) & (covS > 0.5)))
    gen = contourpy.contour_generator(X, Y, np.ma.masked_array(H, mask=mask))
    perim = np.array([sum(float(np.sum(np.hypot(*np.diff(ln, axis=0).T))) for ln in gen.lines(di)) for di in d])
    area[-1], perim[-1] = A_case, 0.0
    area = np.minimum(np.maximum.accumulate(area), A_case)
    remain = (A_case - area) / max(A_case - area[0], 1e-9)
    port = H if S is None else np.minimum(H, S)
    return Burnback(d, area, perim, remain, web, A_case, port, inside, h)


@lru_cache(maxsize=32)
def cached_section(text: str, n_grid: int = NGRID) -> Burnback:
    """Cache keyed by the canonical JSON: the UI calls ``to_si`` many times per session."""
    return build_section(spec_from_json(text), n_grid)


def default_spec(R_out: float = 60.0) -> SectionSpec:
    """A starting point for the editor: centre bore plus six radial slots."""
    return spec_from_dict({"R_out": R_out, "shapes": [{"kind": "circle", "r": 14.0},
                                                      {"kind": "slot", "r0": 0.0, "r1": 0.5 * R_out, "w": 8.0,
                                                       "repeat": 6}]})


def phi_3d(spec: SectionSpec, L: float, h: float, thick: float | None = None):
    """Level-set inputs for the 3-D burnback of an extruded user section: ``(phi, active, h)`` (lengths in mm).

    A coated surface is modelled as an inert shell of cells ``thick`` (default 2 h) around the coated voids: the front
    stops at inert cells, exactly like at the case wall.  The end faces do not burn (no artificial end caps)."""
    from .levelset import box_grid

    X, Y, Z, active = box_grid(spec.R_out, L, h)
    H, S = front_field(spec, Y[0], Z[0])
    sd = S if H is None else H if S is None else np.minimum(H, S)
    if S is not None:
        lining = (S > 0.0) & (S <= (2.0 * h if thick is None else thick))
        active = active & ~lining[None]
    return np.broadcast_to(sd[None], X.shape).copy(), active, h
