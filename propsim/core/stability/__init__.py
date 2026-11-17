r"""Hybrid combustion-instability estimates (MATHKIT.md Section 3.5; Chiaverini Ch. 9, Karabeyoglu).

ILFI boundary-layer delay: :math:`\tau_{bl2}=\frac{c\,L\,P_c}{(2+\frac{\dot m_f}{\dot m_{ox}})\,G_o\,RT_{av}}`, ``c`` = 2.05,
and the low-frequency instability at :math:`f=\frac{0.48}{\tau_{bl2}}`.  The brief's quick estimate is
:math:`f=0.234\,(2+\frac{\dot m_f}{\dot m_{ox}})\frac{G_o RT_{av}}{L P_c}`.
"""
from __future__ import annotations

import numpy as np


def ilfi_tau_bl2(L, Pc, Go, OF, RTav, c: float = 2.05):
    """Boundary-layer delay (s). ``Go`` oxidizer mass flux (kg m^-2 s^-1), ``RTav`` (m^2 s^-2), ``Pc`` (Pa)."""
    return c * L * Pc / ((2.0 + 1.0 / OF) * Go * RTav)


def ilfi_freq(tau):
    """Low-frequency instability frequency (Hz) = 0.48 / tau_bl2."""
    return 0.48 / tau


def freq_estimate(L, Pc, Go, OF, RTav):
    """Quick estimate f = 0.234 (2 + mf/mox) Go RTav / (L Pc) (Hz)."""
    return 0.234 * (2.0 + 1.0 / OF) * Go * RTav / (L * Pc)


def similarity_scale(base: dict, s: float, n: float = 0.5) -> dict:
    r"""Geometric similarity scaling of a hybrid motor by length factor ``s`` at fixed :math:`G_{ox}`, :math:`P_c`,
    O/F and propellant: mass flows and thrust scale as :math:`s^2`, web and burn time as :math:`s` (since
    :math:`\dot r` depends only on :math:`G`), propellant mass as :math:`s^3`, and the ILFI frequency as :math:`s^{-1}`
    (because :math:`\tau_{bl2}\propto L`).  ``base`` keys: mdot_ox, thrust, t_burn, L, freq (any subset)."""
    out = {}
    for k, e in (("mdot_ox", 2), ("thrust", 2), ("t_burn", 1), ("L", 1), ("D", 1), ("m_prop", 3), ("freq", -1)):
        if k in base:
            out[k] = base[k] * s**e
    return out
