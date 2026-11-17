r"""Hall thruster: scaling, :math:`\vec E\times\vec B` drift, efficiency decomposition, and a 1-D breathing-mode model
(MATHKIT.md Section 5).

1-D two-fluid (neutrals + ions, quasi-neutral electrons) along the channel axis ``x``:

.. math::
    \partial_t n_n+v_n\partial_x n_n=-k_{iz}n_nn_i,\qquad \partial_t n_i+\partial_x(n_iv_i)=k_{iz}n_nn_i

Neutrals enter at the anode with the thermal drift speed :math:`v_n`.  The ion velocity follows the potential drop
:math:`v_i(x)=\sqrt{v_{i,0}^2+\frac{2e}{M}(V_d-\phi(x))}`, with the electric field concentrated where the radial
magnetic field (and the anomalous, Bohm-like electron resistivity) peaks, :math:`E\propto B_r^2`.  Ionization from a
Maxwellian rate fit with :math:`T_e` peaking just upstream of the field peak (the ionization zone).  In this reduced model
(fixed field and electron-temperature profiles) the start-up transient is overdamped and the discharge settles; a
self-sustained breathing mode is NOT reproduced. :func:`fife_frequency` gives the predator-prey estimate of the
breathing frequency (typically 10-30 kHz).  Explicit first-order upwind under the ion CFL limit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import AMU, E, KB


@dataclass(frozen=True)
class HallParams:
    L: float = 0.025            # channel length (m)
    L_plume: float = 0.010      # near plume included (m)
    A: float = 4.0e-3           # channel annulus area (m²)
    mdot: float = 5.0e-6        # anode flow (kg·s⁻¹)
    Vd: float = 300.0           # discharge voltage (V)
    T_n: float = 600.0          # neutral temperature at the anode (K)
    Te_max: float = 25.0        # peak electron temperature (eV)
    Te_min: float = 3.0         # anode electron temperature (eV)
    B_peak_x: float = 0.022     # location of the radial-field maximum (m)
    B_width: float = 0.006      # Gaussian width of B_r (m)
    Te_peak_x: float = 0.014    # electron-temperature peak (ionization zone, upstream of the B_r peak) (m)
    Te_width: float = 0.006     # (m)
    sigma0: float = 5.0e-20     # ionization cross-section scale (m²)
    eps_i: float = 12.13        # xenon first ionization energy (eV)
    M_amu: float = 131.293
    N: int = 140
    B_max: float = 0.02         # peak radial magnetic field (T), for the E x B drift display


def k_iz(Te, sigma0: float = 5.0e-20, eps_i: float = 12.13):
    r"""Maxwellian-averaged ionization rate fit (m³·s⁻¹), Te in eV:
    :math:`k=\sigma_0\bar v_e\big(1+\frac{T_e\epsilon_i}{(T_e+\epsilon_i)^2}\big)e^{-\epsilon_i/T_e}`."""
    Te = np.asarray(Te, float)
    vbar = np.sqrt(8.0 * E * Te / (np.pi * 9.1093837015e-31))
    return sigma0 * vbar * (1.0 + Te * eps_i / (Te + eps_i) ** 2) * np.exp(-eps_i / Te)


def b_profile(x: np.ndarray, p: HallParams) -> np.ndarray:
    """Normalized radial magnetic field shape along the channel (peak 1 near the exit)."""
    return np.exp(-0.5 * ((x - p.B_peak_x) / p.B_width) ** 2)


def profiles(p: HallParams):
    """Grid, potential, ion velocity, electron temperature and ionization rate profiles."""
    x = np.linspace(0.0, p.L + p.L_plume, p.N)
    b2 = b_profile(x, p) ** 2
    cum = np.cumsum(b2)
    phi = p.Vd * (1.0 - cum / cum[-1])                         # E ~ B^2: the drop concentrates at the field peak
    M = p.M_amu * AMU
    vi0 = 300.0
    vi = np.sqrt(vi0**2 + 2.0 * E * np.maximum(p.Vd - phi, 0.0) / M)
    Te = p.Te_min + (p.Te_max - p.Te_min) * np.exp(-0.5 * ((x - p.Te_peak_x) / p.Te_width) ** 2)
    kiz = k_iz(Te, p.sigma0, p.eps_i)
    vn = np.sqrt(8 * KB * p.T_n / (np.pi * M)) / 2.0           # one-sided thermal drift
    E_field = -np.gradient(phi, x)
    return x, phi, vi, Te, kiz, vn, E_field


def simulate(p: HallParams, t_end: float = 1.5e-3, n_out: int = 601, seed_frac: float = 1e-3,
             progress=None, cancel=None) -> dict[str, np.ndarray]:
    x, phi, vi, Te, kiz, vn, Ef = profiles(p)
    M = p.M_amu * AMU
    dx = x[1] - x[0]
    nn_in = p.mdot / (M * vn * p.A)
    nn = np.full(p.N, nn_in)
    ni = np.full(p.N, seed_frac * nn_in)
    dt = 0.4 * dx / vi.max()
    n_steps = int(np.ceil(t_end / dt))
    t_out = np.linspace(0.0, t_end, n_out)
    k_out = (t_out / dt).astype(int)
    Id = np.zeros(n_out)
    Ib = np.zeros(n_out)
    nn_t = np.zeros((n_out, p.N))
    ni_t = np.zeros((n_out, p.N))
    j = 0
    flux_in = flux_out_n = flux_out_i = 0.0
    inv0 = float(np.trapezoid(nn + ni, x))
    for k in range(n_steps + 1):
        S = kiz * nn * ni
        while j < n_out and k_out[j] == k:
            ion_rate = float(np.trapezoid(S, x)) * p.A
            Ib[j] = E * ni[-1] * vi[-1] * p.A                      # beam current leaving the domain
            Id[j] = E * ion_rate                                   # ionization (discharge) current
            nn_t[j], ni_t[j] = nn, ni
            j += 1
            if progress and j % 60 == 0:
                progress(j / n_out)
        if cancel is not None and k % 2000 == 0 and cancel.is_set():
            raise InterruptedError("cancelled")
        if k == n_steps:
            break
        # upwind advection (both species move toward +x); inflow boundaries at the anode
        dnn = -vn * (nn - np.concatenate([[nn_in], nn[:-1]])) / dx - S
        flux = ni * vi
        dni = -(flux - np.concatenate([[0.0], flux[:-1]])) / dx + S
        nn = np.maximum(nn + dt * dnn, 0.0)
        ni = np.maximum(ni + dt * dni, 1e-6 * nn_in)
        flux_in += nn_in * vn * dt
        flux_out_n += nn[-1] * vn * dt
        flux_out_i += ni[-1] * vi[-1] * dt
    return dict(t=t_out, Id=Id, Ib=Ib, nn=nn_t, ni=ni_t, x=x, phi=phi, vi=vi, Te=Te, Ef=Ef,
                B=b_profile(x, p), mass_in=flux_in * M * p.A, mass_out=(flux_out_n + flux_out_i) * M * p.A,
                mass_stored=(float(np.trapezoid(nn + ni, x)) - inv0) * M * p.A)


def fife_frequency(p: HallParams) -> float:
    r"""Predator-prey (Fife) breathing-frequency estimate :math:`f=\frac{\sqrt{v_iv_n}}{2\pi L_{iz}}` (Hz), with the
    mean ion speed over the ionization zone of length :math:`L_{iz}`."""
    x, _, vi, _, _, vn, _ = profiles(p)
    z = np.abs(x - p.Te_peak_x) < p.Te_width
    return float(np.sqrt(vi[z].mean() * vn) / (2.0 * np.pi * 2.0 * p.Te_width))


def spectrum(t: np.ndarray, y: np.ndarray, t_skip: float) -> tuple[np.ndarray, np.ndarray, float]:
    """One-sided amplitude spectrum of ``y`` after ``t_skip`` (transient removed) and its peak frequency (Hz)."""
    m = t >= t_skip
    yy = y[m] - y[m].mean()
    dt = t[1] - t[0]
    w = np.hanning(len(yy))
    F = np.abs(np.fft.rfft(yy * w)) * 2 / max(w.sum(), 1e-30)
    f = np.fft.rfftfreq(len(yy), dt)
    i = int(np.argmax(F[1:]) + 1) if len(F) > 1 else 0
    return f, F, float(f[i])


def scaling(Vd: float, Id: float, mdot: float, M_amu: float = 131.293, eta_b: float = 0.8, eta_m: float = 0.9,
            theta_div: float = np.radians(20.0), f2: float = 0.1) -> dict[str, float]:
    """Performance from the efficiency decomposition: beam current fraction, utilization, divergence and charge."""
    from . import divergence_eff, doubles_factor, ion_thrust
    M = M_amu * AMU
    Ib = eta_b * Id
    eta_div = divergence_eff(theta_div)
    q = doubles_factor(f2) ** 2
    F = float(ion_thrust(Ib, 0.9 * Vd, M)) * np.sqrt(eta_div) * doubles_factor(f2) * min(eta_m / max(Ib * M / (E * mdot), 1e-9), 1.0)
    P = Vd * Id
    isp = F / (mdot * 9.80665)
    eta = F**2 / (2 * mdot * P)
    return dict(F=F, P=P, Isp=isp, eta=eta, eta_div=eta_div, eta_q=q, Ib=Ib)
