r"""Regenerative-jacket heat load and friction for the engine-graph solver.

The chamber contour (cylinder sized by :math:`L^*` and the contraction ratio, Rao bell of the joined nozzle) is split
into the jacket's ``section``: ``full`` (injector face to nozzle exit), ``chamber`` (injector face to the split area
ratio in the divergent part) or ``nozzle`` (split area ratio to the exit).  The coolant enters at the aft end of its
section and flows toward the injector.  Per station the Bartz gas film (``core.heat``), the wall conduction and the
Dittus-Boelter coolant film act in series,

.. math:: q=\frac{T_{aw}-T_{cool}}{1/h_g+t_w/k_w+1/h_c},

the coolant picks up :math:`q\,dA` in enthalpy and loses :math:`f\frac{ds}{D_h}\frac{\rho v^2}2` by Darcy friction
(Blasius-type :math:`f=0.184Re^{-0.2}`).  Coolant properties come from the actual coolant state (CoolProp).
"""
from __future__ import annotations

import numpy as np

from ..heat import adiabatic_wall_T, bartz_hg, gas_transport, sigma
from ..liquid import chamber_geometry
from ..nozzle import contour, mach_along_contour
from .props import props


def chamber_contour(Dt: float, eps: float, Lstar: float, contraction: float, g: float, bell: float = 0.8):
    Rt = Dt / 2.0
    Rc = Rt * float(np.sqrt(contraction))
    x, r = contour(Rt, eps, Rc=Rc, f=bell, g=g)
    geo = chamber_geometry(Lstar, Dt, contraction)
    n = 10
    xc = np.linspace(x[0] - geo["Lc"], x[0], n, endpoint=False)
    return np.concatenate([xc, x]), np.concatenate([np.full(n, r[0]), r])


def march(jk, ev, gas: dict, mdot: float) -> dict:
    """Heat load ``Q`` (W), friction coefficient ``K`` (dp = K m|m|/rho) and wall temperatures of jacket ``jk``."""
    net = jk.net
    ch_id = net.wall_source(jk.id)
    ch = net.models[ch_id]
    noz_id = net.downstream_nozzle(ch_id)
    eps = net.param(noz_id, "expansion_ratio") if noz_id else 4.0
    bell = (net.param(noz_id, "bell_fraction", 0.8) or 0.8) if noz_id else 0.8
    At = gas["At"]
    Dt = float(np.sqrt(4.0 * At / np.pi))
    Lstar = ch.par("Lstar", None) or 1.0
    cr = ch.par("contraction_ratio", None) or 3.0
    x, r = chamber_contour(Dt, eps, Lstar, cr, gas["gamma"], bell)
    Rt = Dt / 2.0
    section = jk.opt.get("section", "full")
    split = float(jk.opt.get("split_area_ratio", 5.0))
    A = (r / Rt) ** 2
    if section == "chamber":
        keep = (x <= 0) | (A <= split)
    elif section == "nozzle":
        keep = (x > 0) & (A >= split)
    else:
        keep = np.ones_like(x, bool)
    x, r = x[keep], r[keep]
    if len(x) < 2 or mdot <= 0:
        return dict(Q=0.0, K=0.0, section=section)
    M = mach_along_contour(x, r, Rt, gas["gamma"])
    Tc, p0, cs, g, Mw = gas["Tc"], gas["p0"], gas["cstar"], gas["gamma"], gas["Mw"]
    cpg, mug, Pr = gas_transport(Tc, Mw, g)
    Taw = adiabatic_wall_T(Tc, g, M, Pr)
    n_ch = max(int(round(jk.par("n_channels"))), 1)
    w, hc_ = jk.par("channel_width"), jk.par("channel_height")
    t_w, k_w = jk.par("wall_thickness"), jk.par("k_wall", None) or 20.0
    Af, Dh = w * hc_, 2.0 * w * hc_ / (w + hc_)
    s_in = ev.inlet(jk.id, jk.pin)
    p = ev.p(jk.id, jk.pin)
    pp = None if s_in.kind == "gas" else props(s_in.fluid)
    h, T = s_in.h, s_in.T
    ds = np.hypot(np.diff(x), np.diff(r))
    Q, dp_tot, Twg_max, Twc_max, q_max = 0.0, 0.0, 0.0, 0.0, 0.0
    Twg = np.zeros(len(x))
    qs = np.zeros(len(x))
    Tcool = np.zeros(len(x))
    for i in range(len(x) - 1, -1, -1):
        if pp is None:
            rho, mu, cp, k = p / (s_in.R * T), 3e-5, s_in.cp, 0.1
        else:
            a = pp.at(p, h, T)
            rho, mu, cp, k, T = a["rho"], a["mu"], a["cp"], a["k"], a["T"]
        v = mdot / (n_ch * rho * Af)
        Re = max(rho * v * Dh / mu, 1e3)
        Prc = mu * cp / k
        hcool = 0.023 * Re**0.8 * Prc**0.4 * k / Dh
        Tw = 0.5 * (T + Taw[i])
        for _ in range(6):
            sig = float(sigma(Tw / Tc, g, M[i]))
            hg = float(bartz_hg(Dt, mug, cpg, Pr, p0, cs, 0.382 * Rt * 2.0, (r[i] / Rt) ** 2, sig))
            qi = (Taw[i] - T) / (1.0 / hg + t_w / k_w + 1.0 / hcool)
            Tw = float(np.clip(Taw[i] - qi / hg, T, Taw[i]))
        Twg[i], qs[i], Tcool[i] = Tw, qi, T
        Twg_max = max(Twg_max, Tw)
        Twc_max = max(Twc_max, Tw - qi * t_w / k_w)
        q_max = max(q_max, qi)
        if i > 0:
            dA = 2.0 * np.pi * 0.5 * (r[i] + r[i - 1]) * ds[i - 1]
            Q += qi * dA
            if pp is None:
                T += qi * dA / (mdot * cp)
            else:
                h += qi * dA / mdot
                T = pp.T_ph(p, h, T)
            dp = 0.184 * Re**-0.2 * ds[i - 1] / Dh * rho * v * v / 2.0
            dp_tot += dp
            p -= dp
    rho_in = (ev.p(jk.id, jk.pin) / (s_in.R * s_in.T)) if pp is None else pp.at(ev.p(jk.id, jk.pin), s_in.h, s_in.T)["rho"]
    return dict(Q=float(Q), K=float(dp_tot * rho_in / mdot**2), T_wg_max=float(Twg_max), T_wc_max=float(Twc_max),
                q_max=float(q_max), section=section, profile=dict(x=x, r=r, T_wg=Twg, q=qs, T_cool=Tcool))
