r"""Low-thrust circle-to-circle transfer with inclination change (Edelbaum) and solar-array power degradation.

Edelbaum's optimal constant-yaw-steering solution relates the accumulated :math:`\Delta v` to the orbital speed and the
inclination change:

.. math::
    \tan\beta_0=\frac{\sin(\frac\pi2\Delta i)}{\frac{v_0}{v_1}-\cos(\frac\pi2\Delta i)},\quad
    v(\Delta v)=\sqrt{v_0^2-2v_0\Delta v\cos\beta_0+\Delta v^2},\quad
    i(\Delta v)=\frac2\pi\Big[\tan^{-1}\frac{\Delta v-v_0\cos\beta_0}{v_0\sin\beta_0}+\frac\pi2-\beta_0\Big]

(the total :math:`\Delta v` equals the MathKit Edelbaum formula).  The thrust is power-limited,
:math:`F=\frac{2\eta P(t)}{g_0I_{sp}}`, with :math:`P(t)=P_0(1-\delta)^{t/\mathrm{yr}}`, and :math:`\dot m=-F/(g_0I_{sp})`;
the spacecraft accumulates :math:`\Delta v` at :math:`F/m`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from ..constants import G0
from . import MU_EARTH, edelbaum_dv

YEAR = 365.25 * 86400.0


@dataclass(frozen=True)
class Spiral:
    r0: float          # initial circular radius (m)
    r1: float          # final circular radius (m)
    di: float          # inclination change (rad)
    m0: float          # initial mass (kg)
    P0: float          # beginning-of-life electric power (W)
    eta: float         # thruster total efficiency
    isp: float         # (s)
    degr: float        # fractional power loss per year
    duty: float = 1.0  # thrusting fraction (eclipses, operations)


def geometry(s: Spiral):
    v0, v1 = np.sqrt(MU_EARTH / s.r0), np.sqrt(MU_EARTH / s.r1)
    a = 0.5 * np.pi * s.di
    beta0 = np.arctan2(np.sin(a), v0 / v1 - np.cos(a))
    return v0, v1, beta0, float(edelbaum_dv(v0, v1, s.di))


def state_along(s: Spiral, dv):
    """Orbital speed, radius and inclination change after an accumulated ``dv``."""
    v0, v1, b0, _ = geometry(s)
    dv = np.asarray(dv, float)
    v = np.sqrt(np.maximum(v0**2 - 2 * v0 * dv * np.cos(b0) + dv**2, 1e-9))
    inc = 2 / np.pi * (np.arctan((dv - v0 * np.cos(b0)) / (v0 * np.sin(b0) + 1e-300)) + np.pi / 2 - b0) if s.di > 0 else 0 * dv
    return v, MU_EARTH / v**2, inc


def simulate(s: Spiral, n_out: int = 601) -> dict[str, np.ndarray]:
    """Integrate ``[dv, m]`` in time until the Edelbaum total is reached (event)."""
    _, _, _, dv_tot = geometry(s)
    c = G0 * s.isp

    def power(t):
        return s.P0 * (1.0 - s.degr) ** (t / YEAR)

    def rhs(t, y):
        F = s.duty * 2.0 * s.eta * power(t) / c
        return [F / y[1], -F / c]

    done = lambda t, y: y[0] - dv_tot
    done.terminal, done.direction = True, 1
    t_guess = dv_tot * s.m0 / (2.0 * s.eta * s.P0 / c) * 5.0 + YEAR
    sol = solve_ivp(rhs, (0.0, t_guess), [0.0, s.m0], events=done, rtol=1e-9, atol=[1e-6, 1e-9], dense_output=True)
    t_end = float(sol.t_events[0][0]) if sol.t_events[0].size else float(sol.t[-1])
    t = np.linspace(0.0, t_end, n_out)
    dv, m = sol.sol(t)
    v, r, inc = state_along(s, np.minimum(dv, dv_tot))
    P = power(t)
    F = s.duty * 2.0 * s.eta * P / c
    return dict(t=t, dv=dv, m=m, v=v, r=r, inc=inc, P=P, F=F, dv_tot=dv_tot, t_end=t_end,
                m_prop=float(s.m0 - m[-1]), reached=bool(sol.t_events[0].size))
