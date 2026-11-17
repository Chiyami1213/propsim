r"""2-D grain cross-sections: exact signed distance to the initial port and burnback tables.

The port is a union of primitives (disks, flat/rounded slots, star polygon), so the distance from any point to the
port is the minimum of analytic distances.  Burnback by distance :math:`d` is the level set :math:`\{\phi\le d\}`
(:math:`\phi<0` inside the port, :math:`\phi>0` in the propellant), clipped to the case: area by anti-aliased
coverage and burning perimeter by contour length of the signed field (the insulated case wall does not burn).
"""
from __future__ import annotations

from dataclasses import dataclass

import contourpy
import numpy as np
from matplotlib.path import Path

NGRID = 192


def grid(R_out: float, n: int = NGRID) -> tuple[np.ndarray, np.ndarray, float]:
    xs = np.linspace(-R_out, R_out, n)
    X, Y = np.meshgrid(xs, xs)
    return X, Y, float(xs[1] - xs[0])


def sd_circle(X, Y, R: float, cx: float = 0.0, cy: float = 0.0):
    return np.hypot(X - cx, Y - cy) - R


def sd_box(u, v, hl: float, hw: float, uc: float):
    """Signed distance to the box centred at ``(uc, 0)`` with half-length ``hl`` (u) and half-width ``hw`` (v)."""
    qx, qy = np.abs(u - uc) - hl, np.abs(v) - hw
    return np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) + np.minimum(np.maximum(qx, qy), 0.0)


def sd_segment(u, v, a: float, b: float):
    """Unsigned distance to the segment from ``(a, 0)`` to ``(b, 0)``."""
    return np.hypot(u - np.clip(u, a, b), v)


def sd_polygon(X, Y, pts: np.ndarray):
    """Exact signed distance to a closed polygon (negative inside)."""
    P = np.c_[X.ravel(), Y.ravel()]
    d2 = np.full(len(P), np.inf)
    for k in range(len(pts)):
        a, b = pts[k], pts[(k + 1) % len(pts)]
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0.0, 1.0)
        d2 = np.minimum(d2, np.sum((P - (a + t[:, None] * ab)) ** 2, axis=1))
    inside = Path(pts).contains_points(P)
    return (np.sqrt(d2) * np.where(inside, -1.0, 1.0)).reshape(X.shape)


def rot(X, Y, ang: float):
    return X * np.cos(ang) + Y * np.sin(ang), -X * np.sin(ang) + Y * np.cos(ang)


KINDS = ["circle", "star", "wagon", "finocyl", "center4", "center6"]


def sdf_field(kind: str, X, Y, R_out: float, R_p: float = 15.0, n: int = 6, w: float = 8.0, r_tip: float = 0.6,
              r_valley: float = 0.3, depth: float = 0.55):
    """Signed distance of the initial port at points ``(X, Y)``; lengths in the same unit as ``R_out`` (mm).

    * ``circle``: bore of radius ``R_p``.
    * ``star``: ``n``-point star polygon, tips at ``r_tip * R_out``, valleys at ``r_valley * R_out``.
    * ``wagon``: bore ``R_p`` plus ``n`` flat-ended slots of width ``w`` reaching ``depth * R_out``.
    * ``finocyl``: bore ``R_p`` plus ``n`` fins (slots with rounded ends) of width ``w`` reaching ``depth * R_out``.
    * ``center4`` / ``center6``: bore plus 4 / 6 satellite bores of radius ``R_p`` on a ring at ``R_out / 2``.
    """
    if kind == "star":
        k = np.arange(2 * n)
        ang = np.pi * k / n
        rad = np.where(k % 2 == 0, r_tip, r_valley) * R_out
        return sd_polygon(X, Y, np.c_[rad * np.cos(ang), rad * np.sin(ang)])
    sd = sd_circle(X, Y, R_p)
    if kind == "wagon":
        hl = 0.5 * depth * R_out
        for j in range(n):
            u, v = rot(X, Y, 2 * np.pi * j / n)
            sd = np.minimum(sd, sd_box(u, v, hl, w / 2, hl))
    elif kind == "finocyl":
        L = depth * R_out - w / 2
        for j in range(n):
            u, v = rot(X, Y, 2 * np.pi * j / n)
            sd = np.minimum(sd, sd_segment(u, v, 0.0, max(L, 0.0)) - w / 2)
    elif kind in ("center4", "center6"):
        m = 4 if kind == "center4" else 6
        for j in range(m):
            a = 2 * np.pi * j / m
            sd = np.minimum(sd, sd_circle(X, Y, R_p, 0.5 * R_out * np.cos(a), 0.5 * R_out * np.sin(a)))
    return sd


def make_sdf(kind: str, R_out: float, n_grid: int = NGRID, **kw):
    """Signed distance on a square raster spanning the case: returns ``(X, Y, sdf, h)``."""
    X, Y, h = grid(R_out, n_grid)
    return X, Y, sdf_field(kind, X, Y, R_out, **kw), h


@dataclass
class Burnback:
    d: np.ndarray            # burn distances (same unit as the grid)
    area: np.ndarray         # port area A_p(d)
    perim: np.ndarray        # burning perimeter Pi(d)
    remain: np.ndarray       # propellant fraction remaining
    web: float               # maximum burn distance (web)
    A_case: float            # grain outer area
    sdf: np.ndarray
    inside: np.ndarray
    h: float


def burnback_tables(X, Y, sdf, h, R_out: float, n: int = 31) -> Burnback:
    dist = np.maximum(sdf, 0.0)
    cov_in = np.clip((R_out - np.hypot(X, Y)) / h + 0.5, 0.0, 1.0)
    inside = np.hypot(X, Y) <= R_out
    web = float(dist[inside].max())
    A_case = float(cov_in.sum() * h * h)
    d = np.linspace(0.0, web, n)
    area = np.array([float((np.clip((di - sdf) / h + 0.5, 0.0, 1.0) * cov_in).sum() * h * h) for di in d])
    gen = contourpy.contour_generator(X, Y, np.ma.masked_array(sdf, mask=~inside))
    length = lambda lvl: sum(float(np.sum(np.hypot(*np.diff(ln, axis=0).T))) for ln in gen.lines(lvl))
    perim = np.array([length(di) for di in d])
    area[-1], perim[-1] = A_case, 0.0                       # at d = web the propellant is consumed by definition
    area = np.minimum(np.maximum.accumulate(area), A_case)
    remain = (A_case - area) / max(A_case - area[0], 1e-9)
    return Burnback(d, area, perim, remain, web, A_case, sdf, inside, h)


def section(kind: str, R_out: float, **kw) -> Burnback:
    X, Y, sd, h = make_sdf(kind, R_out, **kw)
    return burnback_tables(X, Y, sd, h, R_out)
