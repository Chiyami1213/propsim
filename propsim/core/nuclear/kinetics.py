r"""Reactor kinetics, thermal feedback, decay heat and stability (Emrich Ch. 11-12).

Point kinetics (Emrich Eqs. 11.14-11.15):

.. math::
    \frac{dn}{dt}=\frac{\rho-\beta}{\Lambda}n+\sum_i\lambda_iC_i,\qquad
    \frac{dC_i}{dt}=\frac{\beta_i}{\Lambda}n-\lambda_iC_i

with ``n`` the power relative to full power and the precursors scaled so that :math:`C_i=\frac{\beta_i}{\lambda_i\Lambda}n`
at steady state. The core is three lumped regions per unit volume (Emrich Eqs. 12.19-12.23; :math:`T_p` is the mean of
the inlet and outlet propellant temperatures):

.. math::
    \rho_fc_{pf}\dot T_f=q-U_{fm}(T_f-T_m),\quad
    \rho_mc_{pm}\dot T_m=U_{fm}(T_f-T_m)-U_{mp}(T_m-T_p),\quad
    \rho_pc_{pp}\dot T_p=U_{mp}(T_m-T_p)-2\dot mc_{pp}(T_p-T_{in})

and the reactivity is the control input plus fuel, moderator and propellant-density feedback,
:math:`\rho=\rho_{ext}-\alpha_f\delta T_f-\alpha_m\delta T_m+\beta_p\delta\rho_p` with :math:`\rho_p=\frac{P}{R_pT_p}`
(Emrich Eqs. 12.39-12.46; :math:`\alpha_f`, :math:`\alpha_m` are the magnitudes of negative coefficients and
:math:`\beta_p>0` means more hydrogen adds reactivity). After shutdown the Todreas-Kazimi decay heat (Emrich Eq. 11.35)
is added to the fuel heat source.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from . import DELAYED_U235

LAM = np.array(DELAYED_U235["lambda"])
BETA_I = np.array(DELAYED_U235["beta"])
BETA = float(BETA_I.sum())
LAM_1G = float(BETA / np.sum(BETA_I / LAM))           # composite one-group decay constant, 0.0764 s⁻¹


# ---------------------------------------------------------------------------------------------- reactivity relations
def inhour_rho(omega, Lam: float, beta_i=BETA_I, lam=LAM):
    r"""Inhour equation :math:`\rho=\omega\Lambda+\sum_i\frac{\omega\beta_i}{\omega+\lambda_i}` (Emrich Eq. 11.22)."""
    w = np.asarray(omega, float)
    return w * Lam + np.sum(w[..., None] * beta_i / (w[..., None] + lam), axis=-1)


def inhour_root(rho: float, Lam: float, beta_i=BETA_I, lam=LAM) -> float:
    """Largest (asymptotic) root omega_0 of the inhour equation; positive for rho > 0, in (-lambda_1, 0) for rho < 0."""
    if abs(rho) < 1e-14:
        return 0.0
    f = lambda w: float(inhour_rho(w, Lam, beta_i, lam)) - rho
    if rho > 0:
        hi = 1.0
        while f(hi) < 0:
            hi *= 10.0
        return float(brentq(f, 1e-14, hi))
    return float(brentq(f, -float(np.min(lam)) * (1 - 1e-12), -1e-14))


def one_group_step(t, rho: float, Lam: float, beta: float = BETA, lam: float = LAM_1G):
    r"""One-group step response (Emrich Eq. 11.33):
    :math:`\frac{n}{n_0}=\frac{\rho}{\rho-\beta}e^{\frac{\rho-\beta}{\Lambda}t}+\frac{\beta}{\beta-\rho}e^{\frac{\rho\lambda}{\beta-\rho}t}`."""
    t = np.asarray(t, float)
    return rho / (rho - beta) * np.exp((rho - beta) / Lam * t) + beta / (beta - rho) * np.exp(rho * lam / (beta - rho) * t)


# ---------------------------------------------------------------------------------------------- decay heat
def decay_heat_fraction(t, t_fp: float):
    r"""Todreas-Kazimi beta + gamma decay power (Emrich Eq. 11.35): :math:`0.066\left[t^{-0.2}-(t_{fp}+t)^{-0.2}\right]`
    with ``t`` seconds after shutdown and ``t_fp`` seconds at full power before it. Clipped at ``t >= 0.1 s``."""
    t = np.maximum(np.asarray(t, float), 0.1)
    return 0.066 * (t**-0.2 - (t_fp + t) ** -0.2)


def shutdown_energy_ratio(t_fp: float, rho_sd: float, Lam: float, beta: float = BETA, lam: float = LAM_1G) -> float:
    r"""Shutdown energy over full power, :math:`\frac{Q_{sd}}{P_{fp}}=0.0825t_{fp}^{0.8}-\frac{\beta}{\rho\lambda}-
    \frac{\Lambda\rho}{(\beta-\rho)^2}` (s), for a shutdown reactivity ``rho_sd < 0`` (Emrich Eq. 11.38)."""
    return 0.0825 * t_fp**0.8 - beta / (rho_sd * lam) - Lam * rho_sd / (beta - rho_sd) ** 2


def cooldown_fraction(t_fp: float, rho_sd: float, Lam: float) -> float:
    r"""Fraction of all propellant spent on decay-heat removal, :math:`f_{sd}=\frac{m_{dh}}{m_{dh}+m_{fp}}` with both
    masses heated over the same temperature span (Emrich Eq. 11.41)."""
    q = shutdown_energy_ratio(t_fp, rho_sd, Lam)
    return q / (t_fp + q)


# ---------------------------------------------------------------------------------------------- lumped core model
@dataclass
class Core:
    """Per-unit-volume core data, SI. Defaults are the NERVA-type example of Emrich 2e Ch. 12 (pp. 236-238):
    q0 = 3750 W·cm⁻³, Lambda = 1e-4 s, fuel rho c = 13.5 x 0.15, graphite rho c = 1.7 x 1.9 J·cm⁻³·K⁻¹,
    U_fm = 32.3 and U_mp = 3.14 W·cm⁻³·K⁻¹, H2 at 7 MPa with R = 4.2 and c_p = 16.8 J·g⁻¹·K⁻¹,
    mdot = 0.083 g·s⁻¹·cm⁻³, alpha_f = 2e-5, alpha_m = 4.5e-5 K⁻¹, beta_p = 100 cm³·g⁻¹."""

    q0: float = 3.75e9               # (W·m⁻³) full-power density
    Lam: float = 1e-4                # (s) prompt generation time
    rc_f: float = 13.5 * 0.15 * 1e6  # (J·m⁻³·K⁻¹) fuel rho c_p
    rc_m: float = 1.7 * 1.9 * 1e6    # (J·m⁻³·K⁻¹) moderator rho c_p
    U_fm: float = 32.3e6             # (W·m⁻³·K⁻¹)
    U_mp: float = 3.14e6             # (W·m⁻³·K⁻¹)
    P: float = 7e6                   # (Pa) core propellant pressure
    R_p: float = 4200.0              # (J·kg⁻¹·K⁻¹) propellant gas constant (Emrich's rounded value)
    cp_p: float = 16800.0            # (J·kg⁻¹·K⁻¹)
    mdot: float = 83.0               # (kg·s⁻¹·m⁻³) full-power propellant flow per unit core volume
    T_in: float = 155.0              # (K) core inlet temperature (gives a 1500 K mean propellant temperature)
    alpha_f: float = 2e-5            # (K⁻¹) magnitude of the (negative) fuel coefficient
    alpha_m: float = 4.5e-5          # (K⁻¹) magnitude of the (negative) moderator coefficient
    beta_p: float = 0.1              # (m³·kg⁻¹) propellant density coefficient (100 cm³·g⁻¹)
    beta_i: np.ndarray = field(default_factory=lambda: BETA_I.copy())
    lam: np.ndarray = field(default_factory=lambda: LAM.copy())

    @property
    def beta(self) -> float:
        return float(np.sum(self.beta_i))

    def steady(self, n: float, mfrac: float) -> tuple[float, float, float]:
        """Steady ``(T_f, T_m, T_p)`` at relative power ``n`` and relative flow ``mfrac``."""
        q = self.q0 * n
        Tp = self.T_in + q / (2.0 * self.mdot * mfrac * self.cp_p)
        Tm = Tp + q / self.U_mp
        return Tm + q / self.U_fm, Tm, Tp

    def constants(self, Tp: float | None = None) -> dict:
        r"""Time constants and resistances of Emrich Eq. 12.30-12.35 and the propellant term :math:`B_p` (Eq. 12.46) at
        the full-power mean propellant temperature (or ``Tp``)."""
        Tp = self.steady(1.0, 1.0)[2] if Tp is None else Tp
        rho_p = self.P / (self.R_p * Tp)
        m2c = 2.0 * self.mdot * self.cp_p
        Bp = self.U_mp * self.P / (self.R_p * Tp**2 * (self.U_fm + self.U_mp) * (self.U_mp + m2c))
        Af = 1.0 / self.U_fm
        Am = 1.0 / (self.U_fm + self.U_mp)
        return dict(tau_f=self.rc_f / self.U_fm, tau_m=self.rc_m / (self.U_fm + self.U_mp),
                    tau_p=rho_p * self.cp_p / (self.U_mp + m2c), A_f=Af, A_m=Am, B_p=Bp, rho_p=rho_p, Tp=Tp,
                    net=self.alpha_f * Af + self.alpha_m * Am - self.beta_p * Bp,
                    beta_p_crit=(self.alpha_f * Af + self.alpha_m * Am) / Bp)

    def transfer(self, w: np.ndarray) -> np.ndarray:
        r"""Closed-loop power-to-reactivity transfer function :math:`K_{RT}G_{RT}(i\omega)` (W·m⁻³ per unit
        reactivity), Emrich Eq. 12.47."""
        c = self.constants()
        s = 1j * np.asarray(w, float)
        kin = (s * self.Lam + np.sum(s[:, None] * self.beta_i / (s[:, None] + self.lam), axis=1)) / self.q0
        lf = 1.0 + s * c["tau_f"]
        lm = 1.0 + s * c["tau_m"]
        lp = 1.0 + s * c["tau_p"]
        fb = self.alpha_f * c["A_f"] / lf + self.alpha_m * c["A_m"] / (lf * lm) - self.beta_p * c["B_p"] / (lf * lm * lp)
        return 1.0 / (kin + fb)


@dataclass
class Program:
    """Control and flow program for :func:`simulate`. Reactivities are absolute (multiply dollars by beta).

    ``control="open"``: the drums add ``rho1`` at t = 0 (step, or a linear ramp over ``t_ramp``).
    ``control="temperature"``: the drums move at up to ``rate_max`` (reactivity per second) with a rate proportional to
    the outlet-temperature error, ``gain`` (reactivity per second per K), following a set point that ramps linearly
    from the initial outlet temperature to ``T_set`` over ``t_ramp`` (an integral controller with a rate limit).
    The propellant flow ramps linearly from ``m0`` to ``m1`` over ``t_flow``. At ``t_sd`` the drums scram (``rho_sd``)
    and the flow then follows the shutdown power with a lag ``tau_flow``, never below ``m_cool``."""

    n0: float = 1.0                   # initial relative power (steady)
    m0: float = 1.0                   # initial relative flow
    m1: float = 1.0                   # relative flow at the end of the flow ramp
    t_flow: float = 0.0               # flow ramp time (s), 0 = step to m1
    control: str = "open"             # "open" | "temperature"
    rho1: float = 0.0                 # open loop: control insertion
    t_ramp: float = 0.0               # open loop: insertion ramp time; temperature control: set-point ramp time (s)
    T_set: float = 2845.0             # (K) temperature control: outlet set point
    gain: float = 2e-6                # temperature control: drum reactivity rate per K of error (s⁻¹·K⁻¹)
    rate_max: float = 0.0013          # temperature control: maximum drum reactivity rate (s⁻¹), 0.2 $/s
    t_sd: float = float("inf")        # shutdown (scram) time (s)
    rho_sd: float = 0.0               # reactivity added at shutdown (negative)
    t_fp: float = 600.0               # full-power time before the scram, for the decay heat (s)
    m_cool: float = 0.03              # minimum relative flow after the scram (cooldown flow)
    tau_flow: float = 1.0             # (s) flow lag after the scram
    decay_heat: bool = True


def simulate(core: Core, prog: Program, t_end: float, n_frames: int = 401, progress=None, cancel=None) -> dict:
    """Integrate point kinetics + lumped thermal feedback. Returns uniform-time channels."""
    beta_i, lam, Lam = core.beta_i, core.lam, core.Lam
    BETA_SUM = float(np.sum(beta_i))
    Tf0, Tm0, Tp0 = core.steady(prog.n0, prog.m0)
    Tout0 = 2.0 * Tp0 - core.T_in
    Tout_d = 2.0 * core.steady(1.0, 1.0)[2] - core.T_in

    def pressure(mf, Tp):
        """Choked nozzle: chamber pressure scales with flow and the square root of the outlet temperature."""
        return core.P * mf * np.sqrt(max(2.0 * Tp - core.T_in, 1.0) / Tout_d)

    rho_p0 = pressure(prog.m0, Tp0) / (core.R_p * Tp0)
    C0 = beta_i / (lam * Lam) * prog.n0
    # state: n, C1..C6, Tf, Tm, Tp, flow (used after the scram), drum reactivity
    y0 = np.concatenate([[prog.n0], C0, [Tf0, Tm0, Tp0, prog.m0, 0.0]])

    def decay(t):
        if not prog.decay_heat or t < prog.t_sd:
            return 0.0
        return float(decay_heat_fraction(t - prog.t_sd, prog.t_fp))

    def flow(t, y):
        if t >= prog.t_sd:
            return y[10]
        if prog.t_flow <= 0:
            return prog.m1
        return prog.m0 + (prog.m1 - prog.m0) * min(t / prog.t_flow, 1.0)

    def setpoint(t):
        if prog.t_ramp <= 0:
            return prog.T_set
        return Tout0 + (prog.T_set - Tout0) * min(t / prog.t_ramp, 1.0)

    def parts(t, y):
        Tf, Tm, Tp = y[7], y[8], y[9]
        if prog.control == "temperature":
            re = y[11]
        else:
            re = prog.rho1 * (min(t / prog.t_ramp, 1.0) if prog.t_ramp > 0 else 1.0)
        if t >= prog.t_sd:
            re = re + prog.rho_sd
        rho_p = pressure(flow(t, y), Tp) / (core.R_p * max(Tp, 1.0))
        return re, -core.alpha_f * (Tf - Tf0), -core.alpha_m * (Tm - Tm0), core.beta_p * (rho_p - rho_p0)

    t_eval = np.linspace(0.0, t_end, n_frames)
    last = [0.0]

    def rhs(t, y):
        if cancel is not None and cancel.is_set():
            raise InterruptedError("cancelled")
        if progress is not None and t > last[0] + 0.02 * t_end:
            last[0] = t
            progress(min(0.95, t / t_end))
        n = y[0]
        C = y[1:7]
        Tf, Tm, Tp = y[7:10]
        rho = sum(parts(t, y))
        dn = (rho - BETA_SUM) / Lam * n + float(np.dot(lam, C))
        dC = beta_i / Lam * n - lam * C
        dh = decay(t)
        q = core.q0 * (max(n, 0.0) + dh)
        mf = flow(t, y)
        dTf = (q - core.U_fm * (Tf - Tm)) / core.rc_f
        dTm = (core.U_fm * (Tf - Tm) - core.U_mp * (Tm - Tp)) / core.rc_m
        rho_p = pressure(mf, Tp) / (core.R_p * max(Tp, 1.0))
        dTp = (core.U_mp * (Tm - Tp) - 2.0 * core.mdot * mf * core.cp_p * (Tp - core.T_in)) / (rho_p * core.cp_p)
        dm = (max(max(n, 0.0) + dh, prog.m_cool) - y[10]) / prog.tau_flow if t >= prog.t_sd else 0.0
        dd = 0.0
        if prog.control == "temperature" and t < prog.t_sd:
            dd = float(np.clip(prog.gain * (setpoint(t) - (2.0 * Tp - core.T_in)), -prog.rate_max, prog.rate_max))
        return np.concatenate([[dn], dC, [dTf, dTm, dTp, dm, dd]])

    breaks = sorted({0.0, t_end, *(x for x in (prog.t_ramp, prog.t_flow, prog.t_sd) if 0 < x < t_end)})
    ts, ys = [], []
    y = y0
    for a, b in zip(breaks[:-1], breaks[1:]):
        if a >= prog.t_sd and ts and abs(a - prog.t_sd) < 1e-12:
            y = y.copy()
            y[10] = flow(a - 1e-9, y)                          # flow continues smoothly into the cooldown phase
        te = t_eval[(t_eval >= a) & (t_eval <= b)]
        sol = solve_ivp(rhs, (a, b), y, method="Radau", t_eval=te, rtol=1e-6, atol=1e-10, max_step=(b - a) / 20)
        if not sol.success:
            raise RuntimeError(sol.message)
        ts.append(sol.t)
        ys.append(sol.y)
        y = sol.y[:, -1]
    t = np.concatenate(ts)
    Y = np.concatenate(ys, axis=1)
    t, idx = np.unique(t, return_index=True)
    Y = Y[:, idx]
    n = Y[0]
    rp = np.array([parts(tt, Y[:, k]) for k, tt in enumerate(t)])
    dec = np.array([decay(tt) for tt in t])
    mf = np.array([flow(tt, Y[:, k]) for k, tt in enumerate(t)])
    Tp = Y[9]
    pc = np.array([pressure(m, T) for m, T in zip(mf, Tp)])
    return dict(t=t, n=n, P_tot=n + dec, pc=pc, decay=dec, rho_ext=rp[:, 0], rho_f=rp[:, 1], rho_m=rp[:, 2], rho_pr=rp[:, 3],
                rho=rp.sum(axis=1), Tf=Y[7], Tm=Y[8], Tp=Tp, Tout=2.0 * Tp - core.T_in, mflow=mf,
                T_set=np.array([setpoint(tt) for tt in t]), steady0=(Tf0, Tm0, Tp0))
