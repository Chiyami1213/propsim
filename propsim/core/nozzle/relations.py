"""Isentropic nozzle relations and performance (MATHKIT.md Sections 1.1-1.2).

The scalar functions are ported unchanged from ``mathkit_check.py``; the ``*_array``
helpers are vectorised versions for Live-mode plots (no per-point root finding).
Calorically perfect gas with frozen gamma. SI units.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def area_ratio(M, g):
    r"""$\frac{A}{A^*} = \frac{1}{M}\left[\frac{2}{\gamma+1}\left(1+\frac{\gamma-1}{2}M^2\right)\right]^{\frac{\gamma+1}{2(\gamma-1)}}$
    (Sutton Eq. 3-14; Anderson Eq. 5.20)."""
    return (1 / M) * ((2 / (g + 1)) * (1 + 0.5 * (g - 1) * M**2)) ** ((g + 1) / (2 * (g - 1)))


def mach_from_area(eps, g, supersonic=True):
    """Mach number from area ratio. Bracketed ``brentq`` (never Newton: minimum at M = 1)."""
    f = lambda M: area_ratio(M, g) - eps
    return brentq(f, 1.0 + 1e-12, 100.0) if supersonic else brentq(f, 1e-9, 1.0 - 1e-12)


def p_ratio(M, g):
    r"""$\frac{p}{p_0} = (1+\frac{\gamma-1}{2}M^2)^{-\frac{\gamma}{\gamma-1}}$."""
    return (1 + 0.5 * (g - 1) * M**2) ** (-g / (g - 1))


def T_ratio(M, g):
    r"""$\frac{T}{T_0} = (1+\frac{\gamma-1}{2}M^2)^{-1}$."""
    return 1.0 / (1 + 0.5 * (g - 1) * M**2)


def eps_from_pratio(pe_pc, g):
    r"""Expansion ratio and exit Mach from $\frac{p_e}{p_c}$ (closed form)."""
    Me = np.sqrt(2 / (g - 1) * (pe_pc ** (-(g - 1) / g) - 1))
    return area_ratio(Me, g), Me


def Gamma(g):
    r"""Vandenkerckhove function $\Gamma(\gamma)=\sqrt{\gamma}\left(\frac{2}{\gamma+1}\right)^{\frac{\gamma+1}{2(\gamma-1)}}$."""
    return np.sqrt(g) * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))


def CF(g, pe_pc, pa_pc, eps):
    r"""Thrust coefficient (Sutton Eq. 3-30):
    $C_F=\sqrt{\frac{2\gamma^2}{\gamma-1}(\frac{2}{\gamma+1})^{\frac{\gamma+1}{\gamma-1}}[1-(\frac{p_e}{p_c})^{\frac{\gamma-1}{\gamma}}]}+(\frac{p_e}{p_c}-\frac{p_a}{p_c})\frac{A_e}{A_t}$."""
    mom = np.sqrt(2 * g**2 / (g - 1) * (2 / (g + 1)) ** ((g + 1) / (g - 1)) * (1 - pe_pc ** ((g - 1) / g)))
    return mom + (pe_pc - pa_pc) * eps


def cstar(g, RT):
    r"""$c^*=\frac{\sqrt{R T_c}}{\Gamma}$; ``RT`` in (m²·s⁻²)."""
    return np.sqrt(RT) / Gamma(g)


# ----------------------------------------------------------------------------------------
# Vectorised helpers (Live mode). Monotone-branch table inversion; validated against brentq.
# ----------------------------------------------------------------------------------------
_M_SUP = np.concatenate([np.linspace(1.0, 6.0, 12001), np.linspace(6.0, 40.0, 8001)[1:]])
_M_SUB = np.linspace(1e-3, 1.0, 12001)


def mach_from_area_array(eps, g, supersonic=True):
    """Vectorised Mach from area ratio (``eps >= 1``) by inverting the monotone branch."""
    eps = np.asarray(eps, float)
    if supersonic:
        a = area_ratio(_M_SUP, g)
        return np.interp(eps, a, _M_SUP)
    a = area_ratio(_M_SUB, g)  # decreasing in M
    return np.interp(eps, a[::-1], _M_SUB[::-1])


def cf_curve(g, pa_pc, eps_grid=None, n=300):
    r"""$C_F$ against $\varepsilon$ with ambient ratio ``pa_pc`` (set 0 for vacuum).

    Built on a Mach grid, so no root finding. Returns ``(eps, CF)``."""
    Me = np.linspace(1.02, 12.0, n)
    eps = area_ratio(Me, g)
    pe = p_ratio(Me, g)
    return eps, CF(g, pe, pa_pc, eps)


def exit_state(eps, g):
    r"""Exit Mach and $\frac{p_e}{p_c}$ for expansion ratio ``eps`` (scalar, bracketed root)."""
    Me = mach_from_area(eps, g, True)
    return Me, p_ratio(Me, g)


def prandtl_meyer(M, g):
    r"""Prandtl–Meyer angle (rad): :math:`\nu(M)=\sqrt{\frac{\gamma+1}{\gamma-1}}\arctan\sqrt{\frac{\gamma-1}{\gamma+1}(M^2-1)}-\arctan\sqrt{M^2-1}`
    (Anderson, *Modern Compressible Flow*, Eq. 4.44)."""
    M = np.maximum(np.asarray(M, float), 1.0)
    k = np.sqrt((g + 1.0) / (g - 1.0))
    return k * np.arctan(np.sqrt((M**2 - 1.0) / k**2)) - np.arctan(np.sqrt(M**2 - 1.0))


def divergence_factor(alpha):
    r"""Conical-nozzle divergence correction :math:`\lambda=\frac{1+\cos\alpha}{2}` (Sutton Eq. 3-34); ``alpha`` in rad."""
    return 0.5 * (1.0 + np.cos(alpha))
