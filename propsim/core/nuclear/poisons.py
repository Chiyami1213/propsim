r"""Fission-product poisons over a burn / shutdown / restart history (Emrich Ch. 13).

Iodine-xenon chain (Emrich Eqs. 13.16-13.17, Te-135 treated as decaying instantly to I-135):

.. math::
    \dot N_I=\gamma_{Te}\Sigma_f\phi-\lambda_IN_I,\qquad
    \dot N_{Xe}=\gamma_{Xe}\Sigma_f\phi+\lambda_IN_I-(\sigma_a^{Xe}\phi+\lambda_{Xe})N_{Xe}

and the promethium-samarium chain (Eqs. 13.27-13.28): :math:`\dot N_{Pm}=\gamma_{Nd}\Sigma_f\phi-\lambda_{Pm}N_{Pm}`,
:math:`\dot N_{Sm}=\lambda_{Pm}N_{Pm}-\sigma_a^{Sm}\phi N_{Sm}` (Sm-149 is stable). Everything is linear in
:math:`\Sigma_f`, so the results are the poison absorption over the fission cross section,
:math:`\frac{\Sigma_a^{P}}{\Sigma_f}`. The reactivity worth uses the usual estimate
:math:`\Delta\rho\approx-\frac{\Sigma_a^P}{\nu\Sigma_f}` for a core that is critical without the poison.

Data (Emrich 2e Ch. 13, pp. 256-262): sigma_a(Xe-135) = 2.7e6 b, gamma(Te-135) = 0.064, gamma(Xe-135) = 0.003,
half-lives I-135 6.57 h and Xe-135 9.14 h; sigma_a(Sm-149) = 40 800 b, gamma(Nd-149) = 0.0113, half-life Pm-149 53.1 h.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

HOUR = 3600.0
SIG_XE = 2.7e6 * 1e-24        # (cm²)
SIG_SM = 40800.0 * 1e-24      # (cm²)
G_TE, G_XE, G_ND = 0.064, 0.003, 0.0113
LAM_I = np.log(2) / (6.57 * HOUR)
LAM_XE = np.log(2) / (9.14 * HOUR)
LAM_PM = np.log(2) / (53.1 * HOUR)


def xenon_equilibrium(phi: float) -> float:
    r""":math:`\frac{\Sigma_a^{Xe}(\infty)}{\Sigma_f}=\frac{\gamma_{Te}+\gamma_{Xe}}{1+\lambda_{Xe}/(\sigma_a^{Xe}\phi)}`
    (Emrich Eq. 13.22); ``phi`` in n·cm⁻²·s⁻¹."""
    return (G_TE + G_XE) / (1.0 + LAM_XE / (SIG_XE * phi))


def _matrix(phi: float) -> np.ndarray:
    """Augmented 5x5 system for y = [N_I, N_Xe, N_Pm, N_Sm, 1] with Sigma_f = 1 (cm⁻¹)."""
    A = np.zeros((5, 5))
    A[0, 0] = -LAM_I
    A[0, 4] = G_TE * phi
    A[1, 0] = LAM_I
    A[1, 1] = -(SIG_XE * phi + LAM_XE)
    A[1, 4] = G_XE * phi
    A[2, 2] = -LAM_PM
    A[2, 4] = G_ND * phi
    A[3, 2] = LAM_PM
    A[3, 3] = -SIG_SM * phi
    return A


def history(phases: list[tuple[float, float]], n: int = 600, equilibrium_start: bool = False) -> dict:
    """March the chains through ``phases`` = [(duration s, flux n·cm⁻²·s⁻¹), ...] on ``n`` uniform time steps.

    ``equilibrium_start`` starts from the equilibrium of the first phase's flux (otherwise a clean core). Returns
    ``t`` (s), the flux, ``xe`` and ``sm`` as Sigma_a / Sigma_f, and the iodine and promethium inventories."""
    T = sum(d for d, _ in phases)
    t = np.linspace(0.0, T, n)
    dt = t[1] - t[0]
    y = np.zeros(5)
    y[4] = 1.0
    if equilibrium_start and phases[0][1] > 0:
        phi = phases[0][1]
        y[0] = G_TE * phi / LAM_I
        y[1] = (G_TE + G_XE) * phi / (SIG_XE * phi + LAM_XE)
        y[2] = G_ND * phi / LAM_PM
        y[3] = G_ND / SIG_SM
    ends = np.cumsum([d for d, _ in phases])
    Y = np.empty((n, 5))
    F = np.empty(n)
    Y[0] = y
    F[0] = phases[0][1]
    cache: dict[float, np.ndarray] = {}
    for k in range(1, n):
        ta, tb = t[k - 1], t[k]
        # split the step at phase boundaries so a short burn is never smeared across a step
        cuts = [ta] + [e for e in ends if ta < e < tb] + [tb]
        for a, b in zip(cuts[:-1], cuts[1:]):
            ip = int(min(np.searchsorted(ends, 0.5 * (a + b)), len(phases) - 1))
            phi = phases[ip][1]
            h = b - a
            if abs(h - dt) < 1e-9 * dt:
                key = phi
                if key not in cache:
                    cache[key] = expm(_matrix(phi) * dt)
                y = cache[key] @ y
            else:
                y = expm(_matrix(phi) * h) @ y
        Y[k] = y
        F[k] = phases[int(min(np.searchsorted(ends, tb - 1e-9), len(phases) - 1))][1]
    return dict(t=t, phi=F, I=Y[:, 0], xe=SIG_XE * Y[:, 1], pm=Y[:, 2], sm=SIG_SM * Y[:, 3])


def post_shutdown_peak(phi: float, horizon_h: float = 72.0, n: int = 2000) -> tuple[float, float]:
    """Time (h) and size (relative to equilibrium) of the xenon peak after shutdown from equilibrium at ``phi``."""
    h = history([(1.0, phi), (horizon_h * HOUR, 0.0)], n=n, equilibrium_start=True)
    i = int(np.argmax(h["xe"]))
    return float((h["t"][i] - 1.0) / HOUR), float(h["xe"][i] / xenon_equilibrium(phi))
