"""Axisymmetric nozzle contour (MATHKIT.md Section 1.4, Rao parabolic approximation).

1. Convergent: cylinder of radius ``Rc`` + 45 deg cone, upstream throat arc 1.5 Rt (-135 -> -90 deg).
2. Downstream throat arc 0.382 Rt from -90 deg to (theta_n - 90 deg).
3. Bell: quadratic Bezier from N to E=(Ln, Re); control point Q at the intersection of the
   tangents at N (angle theta_n) and E (angle theta_e).  Ln = f (sqrt(eps)-1) Rt / tan(15 deg).

theta_n and theta_e are *placeholder* trends standing in for the Rao chart (whose values are not reproduced here);
theta_e scales with the bell length fraction.  A method-of-characteristics design is not implemented.
The planar nu(M_e)/2 estimate was tried for theta_n and rejected (it exceeds 40 deg for eps >= 20).
The point count is fixed so that the 3D mesh can be updated in place.
"""
from __future__ import annotations

import numpy as np


def rao_angles(eps: float, f: float = 0.8, g: float = 1.2) -> tuple[float, float]:
    """Bell (theta_n, theta_e) in radians, PLACEHOLDER trends: theta_n 26 -> 36 deg and theta_e 14 -> 7 deg over
    eps = 4..100 for an 80 % bell; theta_e is scaled by (1-f)/0.2 so that a 100 % bell exits axially."""
    s = float(np.clip((np.log10(eps) - np.log10(4.0)) / (np.log10(100.0) - np.log10(4.0)), 0.0, 1.0))
    thn = np.radians(26.0 + 10.0 * s)
    the = np.radians(14.0 - 7.0 * s) * float(np.clip((1.0 - f) / 0.2, 0.0, 2.0))
    return thn, min(float(the), thn - np.radians(1.0))


def contour(Rt: float, eps: float, Rc: float | None = None, f: float = 0.8, n_conv: int = 12, n_arc: int = 10,
            n_bell: int = 60, g: float = 1.2, kind: str = "bell",
            cone_deg: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x, r)`` with x = 0 at the throat, x < 0 convergent. Fixed length for fixed counts.

    ``kind='cone'``: the divergent section is a straight cone of half-angle ``cone_deg`` after the 0.382 Rt arc."""
    Rc = max(Rc if Rc is not None else 2.0 * Rt, 1.6 * Rt)
    thn, the = rao_angles(eps, f, g)
    if kind == "cone":
        thn = np.radians(cone_deg)
    # upstream arc, centre (0, 2.5 Rt), angle -135 -> -90 deg
    a1 = np.linspace(np.radians(-135), np.radians(-90), n_arc)
    xu = 1.5 * Rt * np.cos(a1)
    ru = 2.5 * Rt + 1.5 * Rt * np.sin(a1)
    # 45 deg cone from arc start back up to radius Rc
    x0, r0 = xu[0], ru[0]
    xc = np.linspace(x0 - (Rc - r0), x0, n_conv, endpoint=False)
    rc = r0 + (x0 - xc)
    # downstream arc, centre (0, 1.382 Rt), -90 -> theta_n - 90 deg
    a2 = np.linspace(np.radians(-90), thn - np.radians(90), n_arc)
    xd = 0.382 * Rt * np.cos(a2)
    rd = 1.382 * Rt + 0.382 * Rt * np.sin(a2)
    N = np.array([xd[-1], rd[-1]])
    if kind == "cone":                                   # straight cone N -> E
        Re = np.sqrt(eps) * Rt
        Ee = N + np.array([(Re - N[1]) / np.tan(thn), Re - N[1]])
        s = np.linspace(0, 1, n_bell + 1)[1:, None]
        B = N + s * (Ee - N)
        return (np.concatenate([xc, xu, xd[1:], B[:, 0]]), np.concatenate([rc, ru, rd[1:], B[:, 1]]))
    # bell Bezier N -> E
    Ln = f * (np.sqrt(eps) - 1.0) * Rt / np.tan(np.radians(15.0))
    E = np.array([Ln, np.sqrt(eps) * Rt])
    tn, te = np.tan(thn), np.tan(the)
    xq = (E[1] - N[1] + tn * N[0] - te * E[0]) / (tn - te)
    Q = np.array([xq, N[1] + tn * (xq - N[0])])
    s = np.linspace(0, 1, n_bell + 1)[1:, None]
    B = (1 - s) ** 2 * N + 2 * (1 - s) * s * Q + s**2 * E
    x = np.concatenate([xc, xu, xd[1:], B[:, 0]])
    r = np.concatenate([rc, ru, rd[1:], B[:, 1]])
    return x, r


def mach_along_contour(x: np.ndarray, r: np.ndarray, Rt: float, g: float) -> np.ndarray:
    """Isentropic Mach number along the contour (subsonic for x < 0, supersonic for x > 0)."""
    from .relations import mach_from_area_array
    A = np.maximum((r / Rt) ** 2, 1.0)
    sup = mach_from_area_array(A, g, True)
    sub = mach_from_area_array(A, g, False)
    return np.where(x < 0, sub, sup)
