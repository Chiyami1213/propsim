"""PyVista geometry helpers: surfaces of revolution as StructuredGrids whose points can be updated in place."""
from __future__ import annotations

import numpy as np
import pyvista as pv


def thetas(cut_deg: float, n: int = 72) -> np.ndarray:
    """Azimuth samples for a cutaway that removes a ``cut_deg`` wedge (0 -> closed body)."""
    return np.radians(np.linspace(0.0, 360.0 - cut_deg, n))


def rev_points(x: np.ndarray, r: np.ndarray, th: np.ndarray) -> np.ndarray:
    """Points of a revolved profile in StructuredGrid order (axial index fastest), shape (nx*nth, 3)."""
    X = np.repeat(x[:, None], len(th), axis=1)
    R = np.repeat(r[:, None], len(th), axis=1)
    Y = R * np.cos(th)[None, :]
    Z = R * np.sin(th)[None, :]
    return np.c_[X.ravel(order="F"), Y.ravel(order="F"), Z.ravel(order="F")]


def rev_grid(x: np.ndarray, r: np.ndarray, th: np.ndarray) -> pv.StructuredGrid:
    g = pv.StructuredGrid()
    g.points = rev_points(x, r, th)
    g.dimensions = (len(x), len(th), 1)
    return g


def broadcast_scalar(profile: np.ndarray, nth: int) -> np.ndarray:
    """Per-axial-station scalar -> point scalars in the same order as ``rev_points``."""
    return np.repeat(np.asarray(profile, float)[:, None], nth, axis=1).ravel(order="F")


def cut_face(x: np.ndarray, r0: np.ndarray, r1: np.ndarray, theta: float, nr: int = 6) -> pv.StructuredGrid:
    """Planar face at azimuth ``theta`` between radius profiles ``r0(x)`` and ``r1(x)`` (grain cut face)."""
    g = pv.StructuredGrid()
    g.points = cut_face_points(x, r0, r1, theta, nr)
    g.dimensions = (len(x), nr, 1)
    return g


def cut_face_points(x, r0, r1, theta, nr: int = 6) -> np.ndarray:
    s = np.linspace(0, 1, nr)
    R = r0[:, None] + (r1 - r0)[:, None] * s[None, :]
    X = np.repeat(x[:, None], nr, axis=1)
    return np.c_[X.ravel(order="F"), (R * np.cos(theta)).ravel(order="F"), (R * np.sin(theta)).ravel(order="F")]
