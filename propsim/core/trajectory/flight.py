r"""Point-mass ascent in a vertical plane over a spherical, non-rotating Earth (MATHKIT.md Section 6).

MathKit form, flight-path angle :math:`\gamma` from the local horizontal:

.. math::
    \dot V=\frac{F-D}{m}-g\sin\gamma,\quad \dot\gamma=-\Big(\frac gV-\frac V{R_E+h}\Big)\cos\gamma,\quad
    \dot h=V\sin\gamma,\quad \dot x=\frac{R_E}{R_E+h}V\cos\gamma,\quad \dot m=-\frac F{g_0I_{sp}},

with :math:`g=g_0\big(\frac{R_E}{R_E+h}\big)^2`, :math:`D=\tfrac12\rho V^2C_D(M)A` and :math:`F=F_{vac}-p_aA_e`
(:func:`mathkit_rhs`).  On the launch rail :math:`\gamma` is held fixed and that form is integrated directly.  In free
flight the same equations are integrated in polar components :math:`(r,\theta,v_r,v_t)`, which are identical but have no
singularity at :math:`V=0` (apogee of a vertical flight):

.. math::
    \dot v_r=\frac{v_t^2}r-\frac\mu{r^2}+\frac{a_r}{},\quad \dot v_t=-\frac{v_rv_t}r+a_t,\qquad
    \gamma=\operatorname{atan2}(v_r,v_t).

Added to the MathKit model (all reduce to it when set to zero):

* a thrust angle :math:`\alpha` above the velocity vector (``level`` guidance holds :math:`\gamma\to0` for orbit insertion),
* a horizontal wind profile (drag acts along the air-relative velocity),
* staging (dry mass jettisoned at burnout, a coast, then the next stage lights), a fairing, an instantaneous pitch kick,
* thrust profiles imported from other modules' Run results (used as given; no back-pressure correction).

There is no lift and no Earth rotation; the 2-D plane is the launch azimuth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp

from ..constants import G0
from .atmosphere import atmosphere

R_E = 6378.137e3                      # m
MU = G0 * R_E**2                      # m³·s⁻² (so that g(0) = g0)

# generic drag shapes: Mach -> C_D (typical slender-body trends; replace with your own data for a real vehicle)
DRAG_TABLES = {
    "slender": ([0.0, 0.6, 0.9, 1.05, 1.2, 1.6, 2.2, 3.5, 6.0, 10.0],
                [0.28, 0.28, 0.34, 0.52, 0.50, 0.42, 0.34, 0.27, 0.22, 0.20]),
    "launcher": ([0.0, 0.6, 0.9, 1.05, 1.2, 1.6, 2.2, 3.5, 6.0, 10.0],
                 [0.40, 0.40, 0.52, 0.78, 0.74, 0.60, 0.48, 0.38, 0.30, 0.26]),
    "blunt": ([0.0, 0.6, 0.9, 1.05, 1.2, 1.6, 2.2, 3.5, 6.0, 10.0],
              [0.55, 0.55, 0.68, 1.00, 0.96, 0.82, 0.70, 0.60, 0.52, 0.48]),
}

# ------------------------------------------------------------------------------------------- fast atmosphere table
_H_GRID = np.arange(0.0, 1.2e6 + 1.0, 50.0)
_ATM = atmosphere(_H_GRID)
_LNRHO, _LNP, _A = np.log(_ATM["rho"]), np.log(_ATM["p"]), _ATM["a"]


def atm(h: float) -> tuple[float, float, float]:
    """``(rho, p, a)`` at altitude ``h`` (m) from a 50 m table of :func:`atmosphere` (negligible interpolation error)."""
    h = min(max(h, 0.0), 1.2e6)
    return (float(np.exp(np.interp(h, _H_GRID, _LNRHO))), float(np.exp(np.interp(h, _H_GRID, _LNP))),
            float(np.interp(h, _H_GRID, _A)))


# ------------------------------------------------------------------------------------------- vehicle description
@dataclass(frozen=True)
class StageSpec:
    m_dry: float                      # kg
    m_prop: float                     # kg
    F_vac: float                      # N (constant-thrust source)
    isp: float                        # s (vacuum; sets the mass flow)
    A_e: float = 0.0                  # nozzle exit area (m²), back-pressure term
    d: float = 1.0                    # outer diameter (m): drag reference area is the diameter of the current lowest stage
    coast: float = 0.0                # s from this stage's burnout to the next ignition
    profile: tuple | None = None      # (t, F_N, mdot or None) imported thrust profile


@dataclass(frozen=True)
class Vehicle:
    stages: tuple[StageSpec, ...]
    m_payload: float
    m_fairing: float = 0.0
    h_fairing: float = 110e3          # m


@dataclass(frozen=True)
class Aero:
    mach: tuple = tuple(DRAG_TABLES["slender"][0])
    cd: tuple = tuple(DRAG_TABLES["slender"][1])
    scale: float = 1.0

    def cd_of(self, M: float) -> float:
        return self.scale * float(np.interp(M, self.mach, self.cd))


@dataclass(frozen=True)
class Wind:
    w_surf: float = 0.0               # m/s along the downrange direction (tailwind positive)
    w_jet: float = 0.0
    h_jet: float = 11e3
    width: float = 5e3

    def at(self, h: float) -> float:
        return self.w_surf + self.w_jet * float(np.exp(-(((h - self.h_jet) / max(self.width, 1.0)) ** 2)))


@dataclass(frozen=True)
class Guidance:
    elevation: float = np.pi / 2      # rad, launch elevation (rail angle)
    rail: float = 20.0                # m
    h_kick: float = 0.0               # m (0 = no kick)
    kick: float = 0.0                 # rad, instantaneous flight-path change at the kick altitude
    mode: str = "gt"                  # "gt" (thrust along velocity) | "level" (hold gamma -> 0 above h_level)
    h_level: float = 80e3             # m
    alpha_max: float = np.radians(25.0)
    kp: float = 0.25                  # 1/s, level-off gain
    perigee_cut: float = 0.0          # m, last-stage cutoff when the orbit's perigee altitude reaches this (0 = off)


def _lowest_d(veh: Vehicle, k: int) -> float:
    return veh.stages[min(k, len(veh.stages) - 1)].d


def stack_mass(veh: Vehicle, k: int, fairing: bool = True) -> float:
    """Mass of everything still attached when stage ``k`` ignites (all later stages full)."""
    return (veh.m_payload + (veh.m_fairing if fairing else 0.0)
            + sum(s.m_dry + s.m_prop for s in veh.stages[k:]))


def ideal_dv(veh: Vehicle) -> list[float]:
    """Tsiolkovsky :math:`\\Delta v` of each stage (fairing already jettisoned for stages 2+; vacuum Isp)."""
    out = []
    for k, s in enumerate(veh.stages):
        m0 = stack_mass(veh, k, fairing=(k == 0))
        mf = m0 - s.m_prop
        out.append(s.isp * G0 * float(np.log(m0 / mf)) if s.m_prop > 0 and mf > 0 else 0.0)
    return out


# ------------------------------------------------------------------------------------------- forces
def thrust(stage: StageSpec, tau: float, p_a: float) -> tuple[float, float]:
    """``(F, mdot)`` of a burning stage ``tau`` seconds after ignition."""
    if stage.profile is not None:
        t, F, md = stage.profile
        if tau < 0.0 or tau > t[-1]:
            return 0.0, 0.0
        f = float(np.interp(tau, t, F))
        mdot = float(np.interp(tau, t, md)) if md is not None else f / (G0 * stage.isp)
        return f, mdot
    return max(stage.F_vac - p_a * stage.A_e, 0.0), stage.F_vac / (G0 * stage.isp)


def mathkit_rhs(t, y, F_vac: float, isp: float, A_e: float, cd: float, area: float):
    """The MathKit Section 6 equations for ``y = [V, gamma, h, x, m]`` (reference form, constant Cd, no wind)."""
    V, gam, h, x, m = y
    g = G0 * (R_E / (R_E + h)) ** 2
    rho, p_a, _ = atm(h)
    D = 0.5 * rho * V * V * cd * area
    F = max(F_vac - p_a * A_e, 0.0)
    return [(F - D) / m - g * np.sin(gam), -(g / V - V / (R_E + h)) * np.cos(gam), V * np.sin(gam),
            R_E / (R_E + h) * V * np.cos(gam), -F_vac / (G0 * isp)]


def orbit(r: float, vr: float, vt: float) -> tuple[float, float, float]:
    """``(perigee_alt, apogee_alt, eccentricity)`` (m) of the Keplerian orbit through the state; apogee is ``inf`` for
    an unbound state."""
    V2 = vr * vr + vt * vt
    E = 0.5 * V2 - MU / r
    hang = r * vt
    e = float(np.sqrt(max(1.0 + 2.0 * E * hang * hang / MU**2, 0.0)))
    if E >= 0.0:
        p = hang * hang / MU
        return p / (1.0 + e) - R_E, float("inf"), e
    a = -MU / (2.0 * E)
    return a * (1.0 - e) - R_E, a * (1.0 + e) - R_E, e


@dataclass
class Flight:
    t: np.ndarray
    ch: dict[str, np.ndarray]
    events: list[tuple[str, float]]       # (name, time)
    meta: dict
    log: list[str] = field(default_factory=list)


def simulate(veh: Vehicle, guid: Guidance, aero: Aero, wind: Wind, t_max: float = 1200.0, n_out: int = 601,
             initial: dict | None = None, atmosphere_on: bool = True, progress=None, cancel=None) -> Flight:
    """Integrate the ascent. ``initial`` (tests, air launch) = ``dict(h=, V=, gamma=)`` starts in free flight instead
    of on the rail. Returns uniform-time channels and the event list."""
    stages = veh.stages
    n_st = len(stages)
    events: list[tuple[str, float]] = []
    log: list[str] = []
    segs: list[dict] = []
    area_of = lambda k: np.pi * _lowest_d(veh, k) ** 2 / 4.0
    fairing_on = [veh.m_fairing > 0.0]

    def air(h):
        return atm(h) if atmosphere_on else (0.0, 0.0, 340.0)

    def stage_thrust(k, tau, h):
        _, p_a, _ = air(h)
        return thrust(stages[k], tau, p_a)

    # ---------------------------------------------------------------- free-flight right-hand side (polar)
    def accel(t, y, k, burn, t_ign, level, kd=None):
        kd = k if kd is None else kd
        r, th, vr, vt, m, mp = y
        h = r - R_E
        rho, p_a, a_s = air(h)
        V = float(np.hypot(vr, vt))
        w = wind.at(h)
        ur, ut = vr, vt - w
        U = float(np.hypot(ur, ut))
        D = 0.5 * rho * U * U * aero.cd_of(U / a_s) * area_of(kd) if U > 0 else 0.0
        dr = -D * ur / U if U > 0 else 0.0
        dt_ = -D * ut / U if U > 0 else 0.0
        F, mdot = (stage_thrust(k, t - t_ign, h) if burn else (0.0, 0.0))
        ev = np.array([vr, vt]) / V if V > 1e-6 else np.array([np.sin(guid.elevation), np.cos(guid.elevation)])
        en = np.array([ev[1], -ev[0]])
        base_r = vt * vt / r - MU / (r * r) + dr / m + F * ev[0] / m
        base_t = -vr * vt / r + dt_ / m + F * ev[1] / m
        al = 0.0
        if burn and F > 0.0 and level > 0.0 and V > 1e-6:
            gam = np.arctan2(vr, vt)
            need = V * guid.kp * (0.0 - gam) - (base_r * en[0] + base_t * en[1])
            al = level * float(np.arcsin(np.clip(m * need / F, -np.sin(guid.alpha_max), np.sin(guid.alpha_max))))
        dvr = base_r + F * np.sin(al) * en[0] / m + F * (np.cos(al) - 1.0) * ev[0] / m
        dvt = base_t + F * np.sin(al) * en[1] / m + F * (np.cos(al) - 1.0) * ev[1] / m
        return dict(dvr=dvr, dvt=dvt, F=F, mdot=mdot, D=D, U=U, M=U / a_s, rho=rho, alpha=al, V=V, h=h, p_a=p_a)

    def level_blend(h):
        if guid.mode != "level":
            return 0.0
        s = float(np.clip((h - guid.h_level) / 3000.0, 0.0, 1.0))
        return s * s * (3.0 - 2.0 * s)

    def rhs_free(t, y, k, burn, t_ign, kd):
        a = accel(t, y, k, burn, t_ign, level_blend(y[0] - R_E), kd)
        return [y[2], y[3] / y[0], a["dvr"], a["dvt"], -a["mdot"], -a["mdot"] if burn else 0.0]

    # ---------------------------------------------------------------- launch
    k, t, phase, t_ign = 0, 0.0, "burn", 0.0
    pend_kick = initial is None and guid.kick != 0.0 and guid.h_kick > 0.0 and guid.elevation > np.radians(80.0)
    if initial is None:
        gam0 = guid.elevation
        m0 = stack_mass(veh, 0)
        F0, _ = thrust(stages[0], 0.0, atm(0.0)[1] if atmosphere_on else 0.0)
        if F0 / (m0 * G0) <= 1.0:
            raise ValueError(f"thrust-to-weight at lift-off is {F0 / (m0 * G0):.2f}: the vehicle cannot leave the pad")
        events.append(("Liftoff", 0.0))
        if guid.rail > 0.0:
            def rhs_rail(tt, y):
                s, V, m, mp = y
                h = s * np.sin(gam0)
                rho, p_a, a_s = air(h)
                w = wind.at(h)
                ux, uz = V * np.cos(gam0) - w, V * np.sin(gam0)
                U = float(np.hypot(ux, uz))
                D = 0.5 * rho * U * U * aero.cd_of(U / a_s) * area_of(0) if U > 0 else 0.0
                Dt = -D * (ux * np.cos(gam0) + uz * np.sin(gam0)) / U if U > 0 else 0.0
                F, mdot = stage_thrust(0, tt, h)
                g = MU / (R_E + h) ** 2
                return [V, (F + Dt) / m - g * np.sin(gam0), -mdot, -mdot]
            end_rail = lambda tt, y: y[0] - guid.rail
            end_rail.terminal, end_rail.direction = True, 1
            sol = solve_ivp(rhs_rail, (0.0, t_max), [0.0, 0.0, m0, stages[0].m_prop], events=end_rail, rtol=1e-9,
                            atol=[1e-6, 1e-6, 1e-6, 1e-6], dense_output=True)
            segs.append(dict(kind="rail", t0=0.0, t1=float(sol.t[-1]), sol=sol, k=0, burn=True, t_ign=0.0, kd=0))
            t = float(sol.t[-1])
            if not sol.t_events[0].size:
                raise ValueError("the vehicle does not clear the launch rail")
            s_, V_, m_, mp_ = sol.y[:, -1]
            y = np.array([R_E + s_ * np.sin(gam0), s_ * np.cos(gam0) / R_E, V_ * np.sin(gam0), V_ * np.cos(gam0), m_, mp_])
            events.append(("Rail exit", t))
        else:
            y = np.array([R_E, 0.0, 0.0, 1e-3 * np.cos(gam0), m0, stages[0].m_prop])
            y[2] = 1e-3 * np.sin(gam0)
    else:
        gam = np.radians(initial.get("gamma_deg", 90.0))
        V = float(initial.get("V", 0.0))
        m0 = stack_mass(veh, 0)
        y = np.array([R_E + initial.get("h", 0.0), 0.0, V * np.sin(gam), V * np.cos(gam), m0, stages[0].m_prop])
        events.append(("Start", 0.0))
    if not (stages[0].m_prop > 0 and (stages[0].F_vac > 0 or stages[0].profile is not None)):
        phase = "none"

    # ---------------------------------------------------------------- stage loop
    guard = 0
    while t < t_max - 1e-9 and guard < 40:
        guard += 1
        if cancel is not None and cancel.is_set():
            raise InterruptedError("cancelled")
        burn = phase == "burn"
        last = k == n_st - 1
        if phase == "none":                       # no usable stage: coast to the end (ballistic tests)
            t_stop, burn = t_max, False
        elif phase == "burn":
            t_stop = t_max
        elif phase == "coast":
            t_stop = min(t_max, t_next)
        else:
            t_stop = t_max
        evs, names = [], []
        if burn:
            f = lambda tt, yy: yy[5] - 1e-9
            f.terminal, f.direction = True, -1
            evs.append(f), names.append("burnout")
            if stages[k].profile is not None:
                tend = float(stages[k].profile[0][-1])
                fp = lambda tt, yy, te=tend, ti=t_ign: tt - ti - te
                fp.terminal, fp.direction = True, 1
                evs.append(fp), names.append("burnout")
            if last and guid.perigee_cut > 0.0:
                def fc(tt, yy):
                    pa, _, _ = orbit(yy[0], yy[2], yy[3])
                    return min(pa - guid.perigee_cut, 1e9)
                fc.terminal, fc.direction = True, 1
                evs.append(fc), names.append("cutoff")
        if pend_kick:
            fk = lambda tt, yy: yy[0] - R_E - guid.h_kick
            fk.terminal, fk.direction = True, 1
            evs.append(fk), names.append("kick")
        if fairing_on[0]:
            ff = lambda tt, yy: yy[0] - R_E - veh.h_fairing
            ff.terminal, ff.direction = True, 1
            evs.append(ff), names.append("fairing")
        fi = lambda tt, yy: yy[0] - R_E
        fi.terminal, fi.direction = True, -1
        evs.append(fi), names.append("impact")
        fa = lambda tt, yy: yy[2]
        fa.terminal, fa.direction = False, -1
        evs.append(fa), names.append("apogee")
        kd = min(k + 1, n_st - 1) if phase == "coast" else k
        sol = solve_ivp(lambda tt, yy, k=k, burn=burn, ti=t_ign, kd=kd: rhs_free(tt, yy, k, burn, ti, kd), (t, t_stop), y,
                        events=evs, method="DOP853", rtol=1e-9,
                        atol=[0.05, 1e-10, 1e-5, 1e-5, 1e-6, 1e-6], dense_output=True)
        segs.append(dict(kind="free", t0=t, t1=float(sol.t[-1]), sol=sol, k=k, burn=burn, t_ign=t_ign, kd=kd))
        if progress:
            progress(min(float(sol.t[-1]) / t_max, 0.99))
        for nm, te in zip(names, sol.t_events):
            if nm == "apogee" and te.size:
                for tt in te:
                    events.append(("Apogee", float(tt)))
        t, y = float(sol.t[-1]), sol.y[:, -1].copy()
        fired = [nm for nm, te in zip(names, sol.t_events) if te.size and nm != "apogee"
                 and abs(float(te[-1]) - t) < 1e-6 and sol.status == 1]
        if "impact" in fired:
            events.append(("Impact", t))
            break
        if "kick" in fired:
            events.append(("Pitch kick", t))
            gam_now = np.arctan2(y[2], y[3])
            V = float(np.hypot(y[2], y[3]))
            gam_new = gam_now - guid.kick
            y[2], y[3] = V * np.sin(gam_new), V * np.cos(gam_new)
            pend_kick = False
            continue
        if "fairing" in fired:
            events.append(("Fairing jettison", t))
            y[4] -= veh.m_fairing
            fairing_on[0] = False
            continue
        if "burnout" in fired or "cutoff" in fired:
            if "cutoff" in fired:
                events.append(("Orbit-insertion cutoff", t))
                phase = "final"
                continue
            events.append((f"Stage {k + 1} burnout", t))
            y[4] -= stages[k].m_dry + max(y[5], 0.0)           # dry mass and any residual propellant leave
            y[5] = 0.0
            if last:
                phase = "final"
            else:
                events.append((f"Stage {k + 1} separation", t))
                phase, t_next = "coast", t + stages[k].coast
            continue
        if sol.status == 0 and phase == "coast" and t < t_max - 1e-9:     # coast over: next stage lights
            k += 1
            t_ign, phase = t, "burn"
            y[5] = stages[k].m_prop
            events.append((f"Stage {k + 1} ignition", t))
            if not (stages[k].m_prop > 0 and (stages[k].F_vac > 0 or stages[k].profile is not None)):
                phase = "none"
            continue
        break
    t_end = t
    # ------------------------------------------------------------------------------- sampling
    tg = np.linspace(0.0, t_end, n_out)
    starts = np.array([s["t0"] for s in segs])
    ch = {key: np.zeros(n_out) for key in ("h", "x", "V", "gamma", "m", "M", "q", "F", "D", "g_load", "alpha", "perigee",
                                           "apogee_alt", "mass_flow", "rho")}
    stage_idx = np.zeros(n_out, int)
    for i, ti in enumerate(tg):
        j = int(np.clip(np.searchsorted(starts, ti, side="right") - 1, 0, len(segs) - 1))
        s = segs[j]
        yy = s["sol"].sol(min(ti, s["t1"]))
        if s["kind"] == "rail":
            ss, V, m, mp = yy
            gam0 = guid.elevation
            yy = np.array([R_E + ss * np.sin(gam0), ss * np.cos(gam0) / R_E, V * np.sin(gam0), V * np.cos(gam0), m, mp])
        r, th, vr, vt, m, mp = yy
        a = accel(ti, yy, s["k"], s["burn"], s["t_ign"], level_blend(r - R_E), s["kd"])
        pa_, ap_, _ = orbit(r, vr, vt)
        ch["h"][i], ch["x"][i] = (r - R_E) / 1e3, th * R_E / 1e3
        ch["V"][i], ch["gamma"][i] = float(np.hypot(vr, vt)), float(np.degrees(np.arctan2(vr, vt)))
        ch["m"][i], ch["M"][i] = m, a["M"]
        ch["q"][i], ch["F"][i], ch["D"][i] = 0.5 * a["rho"] * a["U"] ** 2 / 1e3, a["F"] / 1e3, a["D"] / 1e3
        ch["g_load"][i] = (a["F"] * np.cos(a["alpha"]) - a["D"]) / (m * G0)
        ch["alpha"][i], ch["rho"][i] = float(np.degrees(a["alpha"])), a["rho"]
        ch["perigee"][i], ch["apogee_alt"][i] = pa_ / 1e3, min(ap_ / 1e3, 1e5)
        ch["mass_flow"][i] = a["mdot"]
        stage_idx[i] = s["kd"]
    ch["stage"] = stage_idx.astype(float) + 1.0
    # ------------------------------------------------------------------------------- derived events and summary
    i_q = int(np.argmax(ch["q"]))
    if 0 < i_q < n_out - 1:
        events.append(("Max-Q", float(tg[i_q])))
    mach_up = np.where((ch["M"][:-1] < 1.0) & (ch["M"][1:] >= 1.0))[0]
    if mach_up.size:
        events.append(("Mach 1", float(tg[mach_up[0]])))
    events.sort(key=lambda e: e[1])
    i_ap = int(np.argmax(ch["h"]))
    last_burn = max([te for nm, te in events if nm.endswith("burnout") or nm == "Orbit-insertion cutoff"], default=0.0)
    i_bo = int(min(np.searchsorted(tg, last_burn + 1e-9), n_out - 1))      # first sample at or after the last cutoff
    meta = dict(apogee_km=float(ch["h"][i_ap]), t_apogee=float(tg[i_ap]), range_km=float(ch["x"][-1]),
                maxq_kpa=float(ch["q"][i_q]), t_maxq=float(tg[i_q]), h_maxq_km=float(ch["h"][i_q]),
                max_mach=float(ch["M"].max()), max_g=float(ch["g_load"].max()), v_burnout=float(ch["V"][i_bo]),
                t_burnout=float(last_burn), perigee_km=float(ch["perigee"][i_bo]),
                apogee_orbit_km=float(ch["apogee_alt"][i_bo]), t_end=float(t_end),
                impact=bool(any(nm == "Impact" for nm, _ in events)), m_burnout=float(ch["m"][i_bo]),
                orbit=bool(ch["perigee"][i_bo] > 100.0), dv_ideal=float(sum(ideal_dv(veh))))
    return Flight(tg, ch, events, meta, log)
