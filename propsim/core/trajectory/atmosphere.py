r"""U.S. Standard Atmosphere 1976 (MATHKIT.md Section 6: lapse-rate layers, never a single exponential above 11 km).

Below 86 km geometric (84.852 km geopotential) the seven layers of the standard are evaluated in closed form:

.. math::
    T=T_b+L_b(H-H_b),\qquad p=p_b\Big(\frac{T_b}{T}\Big)^{\frac{g_0M_0}{R^*L_b}}\ (L_b\neq0),\qquad
    p=p_b\,e^{-\frac{g_0M_0(H-H_b)}{R^*T_b}}\ (L_b=0),\qquad \rho=\frac{pM_0}{R^*T},

with geopotential height :math:`H=\frac{r_0Z}{r_0+Z}`.  From 86 to 1000 km the standard's tabulated pressure and density
ratios (as listed in the public-domain PDAS ``atmos76`` code) are interpolated in log space and the kinetic temperature
is the standard's analytic profile (the speed of sound there ignores the change in mean molecular weight; at those
altitudes the dynamic pressure is negligible).  Above 1000 km the 1000 km values are held.
Viscosity is Sutherland's law.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

R_STAR = 8314.32            # N·m·kmol⁻¹·K⁻¹
M0 = 28.9644                # kg·kmol⁻¹
G0 = 9.80665                # m·s⁻²
R0 = 6356.766e3             # m (radius used for geopotential height)
GAMMA = 1.4
P0, RHO0, T0 = 101325.0, 1.225, 288.15
R_AIR = R_STAR / M0         # J·kg⁻¹·K⁻¹
GMR = 1000.0 * G0 * M0 / R_STAR            # K·km⁻¹, 34.163195

# geopotential base heights (km), base temperatures (K), lapse rates (K·km⁻¹)
_HB = np.array([0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 84.852])
_TB = np.array([288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946])
_LB = np.array([-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0, 0.0])


def _base_pressures() -> np.ndarray:
    p = [P0]
    for k in range(len(_HB) - 1):
        dH = _HB[k + 1] - _HB[k]
        if _LB[k] == 0.0:
            p.append(p[-1] * np.exp(-GMR * dH / _TB[k]))
        else:
            p.append(p[-1] * (_TB[k] / (_TB[k] + _LB[k] * dH)) ** (GMR / _LB[k]))
    return np.array(p)


_PB = _base_pressures()

# 86 - 1000 km: geometric altitude (km), pressure ratio p/p0, density ratio rho/rho0 (PDAS atmos76, U.S. Std. Atm. 1976)
_Z = np.array([86, 93, 100, 107, 114, 121, 128, 135, 142, 150, 160, 170, 180, 190, 200, 220, 260, 300, 400, 500, 600,
               700, 800, 900, 1000], float)
_DELTA = np.array([3.6850e-6, 1.0660e-6, 3.1593e-7, 1.0611e-7, 4.3892e-8, 2.3095e-8, 1.3997e-8, 9.2345e-9, 6.4440e-9,
                   4.4828e-9, 2.9997e-9, 2.0933e-9, 1.5072e-9, 1.1118e-9, 8.3628e-10, 4.9494e-10, 1.9634e-10,
                   8.6557e-11, 1.4328e-11, 2.9840e-12, 8.1056e-13, 3.1491e-13, 1.6813e-13, 1.0731e-13, 7.4155e-14])
_SIGMA = np.array([5.680e-6, 1.632e-6, 4.575e-7, 1.341e-7, 4.061e-8, 1.614e-8, 7.932e-9, 4.461e-9, 2.741e-9, 1.694e-9,
                   1.007e-9, 6.380e-10, 4.240e-10, 2.923e-10, 2.074e-10, 1.116e-10, 3.871e-11, 1.564e-11, 2.288e-12,
                   4.257e-13, 9.279e-14, 2.506e-14, 9.272e-15, 4.701e-15, 2.907e-15])
_lnp = PchipInterpolator(_Z, np.log(_DELTA))
_lnr = PchipInterpolator(_Z, np.log(_SIGMA))


def _kinetic_T_high(z_km: np.ndarray) -> np.ndarray:
    """Standard's kinetic temperature profile above 86 km (z geometric, km)."""
    z = np.asarray(z_km, float)
    T = np.full_like(z, 186.8673)
    m = (z > 91.0) & (z < 110.0)
    T[m] = 263.1905 - 76.3232 * np.sqrt(np.maximum(1.0 - ((z[m] - 91.0) / 19.9429) ** 2, 0.0))
    m = (z >= 110.0) & (z <= 120.0)
    T[m] = 240.0 + 12.0 * (z[m] - 110.0)
    m = z > 120.0
    zz = np.minimum(z[m], 1000.0)
    T[m] = 1000.0 - 640.0 * np.exp(-0.01875 * (zz - 120.0) * (R0 / 1e3 + 120.0) / (R0 / 1e3 + zz))
    return T


def atmosphere(h) -> dict[str, np.ndarray]:
    """Standard atmosphere at geometric altitude ``h`` (m; scalar or array): T (K), p (Pa), rho (kg·m⁻³),
    a (m·s⁻¹), mu (Pa·s)."""
    h = np.atleast_1d(np.asarray(h, float))
    z_km = np.maximum(h, 0.0) / 1e3
    H = R0 / 1e3 * z_km / (R0 / 1e3 + z_km)                              # geopotential km
    k = np.clip(np.searchsorted(_HB, H, side="right") - 1, 0, len(_HB) - 1)
    Tb, Lb, Hb, Pb = _TB[k], _LB[k], _HB[k], _PB[k]
    dH = H - Hb
    T = Tb + Lb * dH
    with np.errstate(divide="ignore", invalid="ignore"):
        p_l = Pb * (Tb / T) ** (GMR / np.where(Lb == 0, 1.0, Lb))
    p_i = Pb * np.exp(-GMR * dH / Tb)
    p = np.where(Lb == 0.0, p_i, p_l)
    rho = p / (R_AIR * T)
    hi = z_km > 86.0
    if np.any(hi):
        zc = np.minimum(z_km[hi], 1000.0)
        p[hi] = np.exp(_lnp(zc)) * P0
        rho[hi] = np.exp(_lnr(zc)) * RHO0
        T[hi] = _kinetic_T_high(z_km[hi])
    a = np.sqrt(GAMMA * R_AIR * T)
    mu = 1.458e-6 * T ** 1.5 / (T + 110.4)
    return dict(T=T, p=p, rho=rho, a=a, mu=mu)
