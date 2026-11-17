"""Hybrid regression laws (MATHKIT.md Section 3). Ported unchanged from ``mathkit_check.py``."""
from __future__ import annotations

import numpy as np


def marxman_rdot(G, x, B, mu, rho_f):
    r"""$\rho_f \dot r = 0.036\,G B^{0.23} Re_x^{-0.2}$, $Re_x=\frac{G x}{\mu}$ (Chiaverini Ch. 2). Returns (m·s⁻¹)."""
    Re = G * x / mu
    return 0.036 * G * B**0.23 * Re**-0.2 / rho_f


def rdot_radiation(Qc, Qr, rho_f, dHv):
    r"""$\rho_f \dot r \Delta H_{v,eff} = Q_c e^{-Q_{rad}/Q_c} + Q_{rad}$ (radiation with blocking)."""
    return (Qc * np.exp(-Qr / Qc) + Qr) / (rho_f * dHv)


def power_law_rdot(G_ox, a, n):
    r"""Space-averaged law $\dot r = a G_{ox}^n$; ``a`` must be consistent with units of $\dot r$ and $G_{ox}$."""
    return a * np.power(G_ox, n)


def port_radius_analytic(t, R0, a, n, mdot_ox):
    r"""Single circular port, constant oxidizer flow:
    $R^{2n+1} = R_0^{2n+1} + (2n+1)\,a\,(\frac{\dot m_{ox}}{\pi})^n t$."""
    return (R0 ** (2 * n + 1) + (2 * n + 1) * a * (mdot_ox / np.pi) ** n * t) ** (1 / (2 * n + 1))
