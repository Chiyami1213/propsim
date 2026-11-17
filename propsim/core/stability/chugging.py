r"""Chugging / low-frequency chamber-feed stability, time-lag model (MATHKIT.md Section 4.4).

Normalized perturbations :math:`\pi=\delta p_c/p_c` and :math:`m=\delta\dot m_i/\dot m` with a feed line (inertance),
an injector, a combustion delay :math:`\tau_c` and a chamber of gas residence time :math:`\theta_g=L^*c^*/RT_c`:

.. math::
    \tau_f\dot m+m=-\kappa\,\pi,\quad \theta_g\dot\pi+\pi=m(t-\tau_c),\qquad
    \kappa=\frac{p_c}{2\Delta p_{inj}},\ \tau_f=\frac{\mathcal L\dot m_0}{2\Delta p_{inj}}

Open loop :math:`G(i\omega)=\frac{\kappa\,e^{-i\omega\tau_c}}{(1+i\omega\theta_g)(1+i\omega\tau_f)}`; the system is stable when
the Nyquist curve does not encircle :math:`-1`.  The delay equation is integrated with a fixed-step RK4 and a history
buffer (``solve_ivp`` does not handle delays), step :math:`\le\tau/50`.
"""
from __future__ import annotations

import numpy as np


def open_loop(omega, kappa: float, tau_c: float, theta_g: float, tau_f: float = 0.0):
    w = np.asarray(omega, float)
    return kappa * np.exp(-1j * w * tau_c) / ((1.0 + 1j * w * theta_g) * (1.0 + 1j * w * tau_f))


def stability_margins(kappa: float, tau_c: float, theta_g: float, tau_f: float = 0.0) -> dict[str, float]:
    """Gain margin at the phase crossover (-180°) and the critical gain; ``gain_margin`` > 1 means stable."""
    w = np.geomspace(1e-2 / max(tau_c, theta_g, tau_f, 1e-6), 1e3 / max(min(tau_c, theta_g, 1e9), 1e-9), 40000)
    L = open_loop(w, kappa, tau_c, theta_g, tau_f)
    ph = np.unwrap(np.angle(L))
    below = np.where((ph[:-1] > -np.pi) & (ph[1:] <= -np.pi))[0]
    if below.size == 0:
        return dict(gain_margin=np.inf, omega_pc=np.nan, f_pc=np.nan, kappa_crit=np.inf, stable=True)
    i = below[0]
    t = (-np.pi - ph[i]) / (ph[i + 1] - ph[i])
    wpc = w[i] + t * (w[i + 1] - w[i])
    mag = abs(open_loop(wpc, kappa, tau_c, theta_g, tau_f))
    return dict(gain_margin=float(1.0 / mag), omega_pc=float(wpc), f_pc=float(wpc / (2 * np.pi)),
                kappa_crit=float(kappa / mag), stable=bool(mag < 1.0))


def simulate(kappa: float, tau_c: float, theta_g: float, tau_f: float, t_end: float, pi0: float = 0.01,
             n_per_tau: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed-step RK4 with a delay buffer: returns ``t, pi(t), m(t)`` for an initial pressure perturbation ``pi0``."""
    tau = max(min(tau_c, theta_g, tau_f if tau_f > 0 else np.inf), 1e-7)
    dt = tau / n_per_tau
    dt = min(dt, t_end / 2000.0) if t_end / dt > 2e6 else dt
    n = int(np.ceil(t_end / dt))
    nd = max(int(round(tau_c / dt)), 1)
    pi, m = np.zeros(n + 1), np.zeros(n + 1)
    pi[0] = pi0

    def delayed(k: int, sub: float = 0.0) -> float:
        """m at time (k + sub) dt - tau_c by linear interpolation of the stored history (0 before t = 0)."""
        x = k + sub - nd
        if x <= 0:
            return 0.0
        i = int(np.floor(x))
        f = x - i
        return m[i] * (1.0 - f) + (m[min(i + 1, k)] * f if i + 1 <= k else m[i] * f)

    def f(pi_, m_, md):
        dpi = (md - pi_) / theta_g
        dm = (-kappa * pi_ - m_) / tau_f if tau_f > 0 else 0.0
        return dpi, dm

    for k in range(n):
        if tau_f <= 0:
            m[k] = -kappa * pi[k]
            k1, _ = f(pi[k], m[k], delayed(k))
            m_mid = -kappa * (pi[k] + 0.5 * dt * k1)
            k2, _ = f(pi[k] + 0.5 * dt * k1, m_mid, 0.5 * (delayed(k) + delayed(k, 1.0)))
            m_m2 = -kappa * (pi[k] + 0.5 * dt * k2)
            k3, _ = f(pi[k] + 0.5 * dt * k2, m_m2, 0.5 * (delayed(k) + delayed(k, 1.0)))
            m_e = -kappa * (pi[k] + dt * k3)
            k4, _ = f(pi[k] + dt * k3, m_e, delayed(k, 1.0))
            pi[k + 1] = pi[k] + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            d0, d1 = delayed(k), delayed(k, 1.0)
            dm0 = 0.5 * (d0 + d1)
            a1, b1 = f(pi[k], m[k], d0)
            a2, b2 = f(pi[k] + 0.5 * dt * a1, m[k] + 0.5 * dt * b1, dm0)
            a3, b3 = f(pi[k] + 0.5 * dt * a2, m[k] + 0.5 * dt * b2, dm0)
            a4, b4 = f(pi[k] + dt * a3, m[k] + dt * b3, d1)
            pi[k + 1] = pi[k] + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
            m[k + 1] = m[k] + dt / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4)
    if tau_f <= 0:
        m[n] = -kappa * pi[n]
    return np.arange(n + 1) * dt, pi, m


def growth_rate(t: np.ndarray, pi: np.ndarray) -> float:
    """Exponential envelope rate (1/s) from the peaks of ``|pi|`` (positive = unstable, negative = decaying)."""
    a = np.abs(pi)
    pk = np.where((a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:]))[0] + 1
    if len(pk) < 4:
        return float("nan")
    k = slice(len(pk) // 4, None)
    return float(np.polyfit(t[pk][k], np.log(np.maximum(a[pk][k], 1e-300)), 1)[0])
