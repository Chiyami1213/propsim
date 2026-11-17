r"""Analytic core flows for hybrid ports (MATHKIT.md Section 3.4), nondimensional: radius ``a``, injection speed ``U_w``.

Taylor-Culick (Majdalani): :math:`u_r=-\frac{\sin(\frac{\pi}{2}r^2)}{r}`, :math:`u_z=\pi z\cos(\frac{\pi}{2}r^2)`,
stream function :math:`\psi=z\sin(\frac{\pi}{2}r^2)`; with headwall injection :math:`\psi=(z+u_h)\sin(\frac{\pi}{2}r^2)`.
Bidirectional vortex mantle with fuel blowing (Chiaverini Ch. 6, Eq. 78) -> :math:`\beta^*`.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def taylor_culick(r, z, u_h: float = 0.0):
    """Velocity ``(u_r, u_z)``; ``u_h`` = headwall injection speed (nondimensional), 0 for plain Taylor-Culick."""
    r = np.asarray(r, float)
    s, c = np.sin(0.5 * np.pi * r**2), np.cos(0.5 * np.pi * r**2)
    ur = -np.where(r == 0, 0.0, s / np.where(r == 0, 1.0, r))              # u_r = -psi_z / r  (axis: 0)
    uz = np.pi * (np.asarray(z, float) + u_h) * c                           # u_z = psi_r / r
    return ur, uz


def stream_function(r, z, u_h: float = 0.0):
    return (np.asarray(z, float) + u_h) * np.sin(0.5 * np.pi * np.asarray(r, float) ** 2)


def streamline(r0: float, z0: float, z_end: float, u_h: float = 0.0, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Streamline through (r0, z0): ``psi`` is constant, so r(z) follows from ``(z+u_h) sin(pi r^2/2) = psi``."""
    z = np.linspace(z0, z_end, n)
    psi = (z0 + u_h) * np.sin(0.5 * np.pi * r0**2)
    arg = np.clip(psi / (z + u_h), 0.0, 1.0)
    return z, np.sqrt(2.0 / np.pi * np.arcsin(arg))


def bidirectional_mantle(kappa: float, eps: float) -> float:
    """Root of Chiaverini Ch. 6 Eq. 78 in [0.6, 0.97] (``brentq``, raises if the bracket fails)."""
    def f(b):
        s, c = np.sin(np.pi * b * b), np.cos(np.pi * b * b)
        return c * ((np.pi * kappa + np.pi * eps * (1 + b * b * c)) / s - eps) + np.pi * eps * b * b * s
    return brentq(f, 0.6, 0.97, xtol=1e-14)


def mantle_asymptotic(kappa: float, eps: float) -> float:
    """Eqs. 79-80: beta* = (A - sqrt C)/B."""
    p = np.pi
    A = p**2 * eps / np.sqrt(2)
    B = 4 * eps - 2 * p * eps + p**2 * eps - 2 * p * kappa
    C = (4 * eps**2 - 6 * p * eps**2 + p**2 * eps**2 - 6 * p * eps * kappa + 4 * p**2 * eps * kappa
         + 2 * p**2 * kappa**2)
    return (A - np.sqrt(C)) / B
