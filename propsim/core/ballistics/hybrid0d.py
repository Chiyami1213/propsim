"""0-D hybrid motor burn: single circular port, power-law regression, filling-ODE chamber.

State ``y = [R, Pc/1e6]`` (port radius (m), chamber pressure (MPa)).

.. math::
    \\dot R = a\\,G_{ox}^{n},\\quad G_{ox}=\\frac{\\dot m_{ox}}{\\pi R^2},\\quad
    \\dot m_f=\\rho_f\\,2\\pi R L\\,\\dot R,\\quad
    \\frac{dP_c}{dt}=\\frac{R T_c}{V_c}\\left(\\dot m_{ox}+\\dot m_f-\\frac{P_c A_t}{\\eta c^*}\\right)

(MATHKIT.md Sections 3.1-3.3; the filling ODE ignores the small $dV_c/dt$ term for hybrids.)
Oxidizer valve profile: first-order opening (``tau_open``) and closing (``tau_close``) at ``t_burn``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from ..constants import G0, P_SEA_LEVEL
from ..nozzle import CF, Gamma, exit_state
from ..thermo.surrogate import GAMMA, cstar_gox_htpb, tc_gox_htpb


@dataclass(frozen=True)
class HybridBurnSI:
    mdot_ox: float      # (kg·s⁻¹)
    p_tank: float       # (Pa)
    R0: float           # (m)
    L: float            # (m)
    R_out: float        # (m)   case/grain outer radius (web = R_out - R0)
    rho_f: float        # (kg·m⁻³)
    a: float            # (m·s⁻¹ with G in kg·m⁻²·s⁻¹)
    n: float
    Dt: float           # (m)
    eps: float
    t_burn: float       # (s)
    eta_cstar: float
    V_extra: float = 2.5e-4   # (m³) pre- plus post-chamber free volume
    tau_open: float = 0.25    # (s)
    tau_close: float = 0.15   # (s)

    @property
    def At(self) -> float:
        return np.pi * self.Dt**2 / 4.0


def mdot_ox_of_t(p: HybridBurnSI, t):
    t = np.asarray(t, float)
    return p.mdot_ox * (1 - np.exp(-t / p.tau_open)) * (1 - np.exp(-np.maximum(p.t_burn - t, 0.0) / p.tau_close))


def _rates(p: HybridBurnSI, t, R, Pc):
    """Return (mdot_ox, rdot, mdot_f, OF, c*_eff)."""
    mox = float(mdot_ox_of_t(p, t))
    R = max(R, 1e-6)
    G = mox / (np.pi * R**2)
    rdot = p.a * G**p.n if R < p.R_out else 0.0       # burnout: web consumed
    mf = p.rho_f * 2 * np.pi * R * p.L * rdot
    OF = mox / mf if mf > 1e-12 else np.nan
    cs = p.eta_cstar * float(cstar_gox_htpb(OF if np.isfinite(OF) else 2.1))
    return mox, rdot, mf, OF, cs


def rhs(t: float, y, p: HybridBurnSI):
    R, Pc = y[0], y[1] * 1e6
    mox, rdot, mf, OF, cs = _rates(p, t, R, Pc)
    RT = (cs * Gamma(GAMMA)) ** 2
    Vc = np.pi * R**2 * p.L + p.V_extra
    dPc = RT / Vc * (mox + mf - Pc * p.At / cs)
    return [rdot, dPc / 1e6]


def integrate(p: HybridBurnSI, n_frames: int = 601, progress: Callable[[float], None] | None = None,
              cancel=None, n_chunks: int = 50) -> dict[str, np.ndarray]:
    """Integrate in chunks (LSODA, stiff chamber filling) so progress/cancel are serviced between chunks."""
    t = np.linspace(0.0, p.t_burn, n_frames)
    y0 = [p.R0, P_SEA_LEVEL / 1e6]
    Y = np.empty((2, n_frames))
    Y[:, 0] = y0
    edges = np.linspace(0, n_frames - 1, n_chunks + 1).astype(int)
    y = np.array(y0)
    for k in range(n_chunks):
        i0, i1 = edges[k], edges[k + 1]
        if i1 <= i0:
            continue
        if cancel is not None and cancel.is_set():
            raise InterruptedError("cancelled")
        sol = solve_ivp(rhs, (t[i0], t[i1]), y, args=(p,), method="LSODA", t_eval=t[i0:i1 + 1],
                        rtol=1e-8, atol=[1e-9, 1e-9])
        if not sol.success:
            raise RuntimeError(f"integrator failed: {sol.message}")
        Y[:, i0:i1 + 1] = sol.y
        y = sol.y[:, -1]
        if progress:
            progress((k + 1) / n_chunks)
    return derived_channels(p, t, Y[0], Y[1] * 1e6)


def derived_channels(p: HybridBurnSI, t, R, Pc) -> dict[str, np.ndarray]:
    """Thrust and performance from the integrated state (one evaluation of the same physics)."""
    n = len(t)
    mox = np.empty(n); rdot = np.empty(n); mf = np.empty(n); OF = np.empty(n); cs = np.empty(n)
    for i in range(n):
        mox[i], rdot[i], mf[i], OF[i], cs[i] = _rates(p, t[i], R[i], Pc[i])
    Me, pe_pc = exit_state(p.eps, GAMMA)
    cf = np.array([CF(GAMMA, pe_pc, P_SEA_LEVEL / max(pc, 1.0), p.eps) for pc in Pc])
    F = np.maximum(cf * Pc * p.At, 0.0)       # clamp: no negative thrust below ambient (flow-separation regime)
    mdot = mox + mf
    Isp = np.where(mdot > 1e-3 * p.mdot_ox, F / np.maximum(mdot, 1e-12) / G0, np.nan)
    return {
        "R": R * 1e3, "Pc": Pc / 1e6, "F": F / 1e3, "OF": np.where(mox > 1e-3 * p.mdot_ox, OF, np.nan),
        "mdot_ox": mox, "mdot_f": mf, "rdot": rdot * 1e3, "Isp": Isp, "Tc": tc_gox_htpb(np.nan_to_num(OF, nan=2.1)),
        "CF": cf, "cstar": cs,
    }


def summary(p: HybridBurnSI, t, ch: dict[str, np.ndarray]) -> dict[str, float]:
    It = float(np.trapezoid(ch["F"] * 1e3, t))              # (N·s)
    mprop = float(np.trapezoid(ch["mdot_ox"] + ch["mdot_f"], t))
    return {"I_t_kNs": It / 1e3, "Isp_avg": It / (mprop * G0) if mprop > 0 else float("nan"),
            "m_prop": mprop, "Pc_max_MPa": float(np.nanmax(ch["Pc"])), "R_end_mm": float(ch["R"][-1]),
            "web_burned": float((ch["R"][-1] * 1e-3 - p.R0) / (p.R_out - p.R0)),
            "dp_inj_margin": float((p.p_tank - np.nanmax(ch["Pc"]) * 1e6) / (np.nanmax(ch["Pc"]) * 1e6))}
