"""Hybrid regression laws (MATHKIT.md Section 3).  Marxman, radiation and power-law forms are ported unchanged from
``mathkit_check.py``; the Arrhenius pyrolysis form is the generic textbook surface-kinetics law."""
from __future__ import annotations

import numpy as np

R_GAS = 8.314462618

# Space-averaged power laws rdot = a G^n with a in (m/s) when G in (kg m^-2 s^-1). MATHKIT.md Section 3.1:
# HTPB/GOX 0.0541 (mm/s); paraffin 0.488 in (g cm^-2 s^-1) units -> SI 0.1171 (mm/s).
POWER_LAWS = {
    "HTPB / GOX": (0.0541e-3, 0.647),
    "Paraffin / GOX": (0.1171e-3, 0.62),
}


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


def arrhenius_rdot(Ts, A, Ea):
    r"""Surface pyrolysis (kinetics-limited) law $\dot r = A e^{-E_a/(R_u T_s)}$, ``Ea`` in (J·mol⁻¹)."""
    return A * np.exp(-Ea / (R_GAS * np.asarray(Ts, float)))


def thermal_wave(x, rdot, alpha):
    r"""Steady condensed-phase temperature profile below the surface, $\frac{T-T_0}{T_s-T_0}=e^{-\dot r x/\alpha}$
    (x measured into the solid); returns the normalized profile. Thermal thickness is $\delta=\frac{\alpha}{\dot r}$."""
    return np.exp(-rdot * np.asarray(x, float) / alpha)


def port_radius_analytic(t, R0, a, n, mdot_ox):
    r"""Single circular port, constant oxidizer flow:
    $R^{2n+1} = R_0^{2n+1} + (2n+1)\,a\,(\frac{\dot m_{ox}}{\pi})^n t$."""
    return (R0 ** (2 * n + 1) + (2 * n + 1) * a * (mdot_ox / np.pi) ** n * t) ** (1 / (2 * n + 1))
