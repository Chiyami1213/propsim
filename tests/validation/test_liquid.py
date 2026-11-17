"""Liquid-engine validation: MathKit Bartz values, regen energy balance, injector and chamber sizing, feed-network
steady state and water hammer, cycle balances and spool-up, chugging stability."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from propsim.core import heat as H
from propsim.core import liquid as lq
from propsim.core.feed import cycles as cy
from propsim.core.feed import network as nw
from propsim.core.nozzle import contour, mach_along_contour
from propsim.core.stability import chugging as chg
from propsim.sims.liquid_run import Chugging, ChuggingParams, CycleParams, FeedParams, FeedTransient, PumpFedCycle

from .figs import new_fig, save


# ------------------------------------------------------------------------------------------- Bartz (MATHKIT 4.2)
def test_bartz_mathkit_values():
    sig = float(H.sigma(0.5, 1.2, 1.0))
    assert sig == pytest.approx(1.1757, abs=1e-4)
    hg = float(H.bartz_hg(0.1, 8e-5, 2000.0, 0.6, 5e6, 1700.0, 0.05, 1.0, sig))
    assert hg == pytest.approx(12734.0, abs=1.0)


def test_gas_transport_estimates():
    cp, mu, Pr = H.gas_transport(3600.0, 0.0225, 1.15)
    assert 2.0e-5 < mu < 1.5e-4 and 0.6 < Pr < 0.9 and cp == pytest.approx(1.15 / 0.15 * 8.314462618 / 0.0225)


@pytest.fixture(scope="module")
def regen():
    Rt, g = 0.03, 1.15
    x, r = contour(Rt, 12.0, Rc=2.5 * Rt, f=0.8, g=g)
    M = mach_along_contour(x, r, Rt, g)
    gas = dict(Tc=3600.0, p0=7e6, cstar=1780.0, gamma=g, Mw=0.0225)
    des = H.RegenDesign("n-Dodecane", mdot_c=3.4, T_in=300.0, p_in=10e6, n_ch=100, h_ch=0.001, fill=0.5, t_w=0.001,
                        k_w=330.0, T_w_max=800.0, fin=1.6)
    return x, r, H.regen_march(x, r, Rt, gas, M, des), des


def test_regen_energy_balance_and_shape(regen):
    x, r, o, des = regen
    cool = H.coolant(des.fluid, des.p_in)
    T = np.linspace(des.T_in, o["T_cool"][0], 40)
    cp_mean = np.trapezoid([cool.at(t)[1] for t in T], T) / (T[-1] - T[0])
    assert o["Q"] == pytest.approx(des.mdot_c * cp_mean * o["dT_cool"], rel=5e-3)        # heat in = coolant enthalpy rise
    i_peak = int(np.argmax(o["q"]))
    assert abs(x[i_peak]) < 0.25 * 0.03 * 2                                           # peak heat flux at the throat
    assert np.all(np.diff(o["T_cool"][::-1]) >= -1e-9)                               # coolant heats along its path
    assert np.all(o["T_wg"] > o["T_wc"]) and np.all(o["T_wc"] > o["T_cool"])          # gradient through the wall
    assert np.all(o["T_wg"] < o["T_aw"])
    fig, (ax1, ax2) = new_fig(2)
    ax1.plot(x * 1e3, o["q"] / 1e6, color="#ff5a3c")
    ax1.set_xlabel(r"$x$ (mm)")
    ax1.set_ylabel(r"$q$ ($\mathrm{MW{\cdot}m^{-2}}$)")
    ax2.plot(x * 1e3, o["T_wg"], color="#ff5a3c", label=r"$T_{wg}$")
    ax2.plot(x * 1e3, o["T_cool"], color="#4ea1ff", label=r"$T_{cool}$")
    ax2.set_xlabel(r"$x$ (mm)")
    ax2.set_ylabel(r"$T$ (K)")
    ax2.legend()
    save(fig, "liquid_regen_march")


def test_regen_more_coolant_runs_cooler(regen):
    x, r, o, des = regen
    Rt, g = 0.03, 1.15
    M = mach_along_contour(x, r, Rt, g)
    gas = dict(Tc=3600.0, p0=7e6, cstar=1780.0, gamma=g, Mw=0.0225)
    o2 = H.regen_march(x, r, Rt, gas, M, H.RegenDesign(**{**des.__dict__, "fin": 2.5}))
    assert o2["T_wg_max"] < o["T_wg_max"]


# ------------------------------------------------------------------------------------------- injector and chamber
def test_injector_and_chamber_sizing():
    d = lq.injector_design(2.0, 810.0, 1.75e6, 0.7, 1.0e-3)
    assert lq.orifice_mdot(0.7, d["A_tot"], 810.0, 1.75e6) == pytest.approx(2.0, rel=1e-12)
    assert d["u"] == pytest.approx(0.7 * np.sqrt(2 * 1.75e6 / 810.0), rel=1e-12)       # u = Cd sqrt(2 dp / rho)
    J = lq.momentum_ratio(2.0, 810.0, 1e-4, 5.0, 1141.0, 2e-4)
    assert J == pytest.approx((2.0**2 / (810.0 * 1e-8)) / (5.0**2 / (1141.0 * 4e-8)), rel=1e-12)
    g = lq.chamber_geometry(1.0, 0.06, 3.0)
    Vcyl = np.pi * (g["Dc"] / 2) ** 2 * g["Lc"]
    assert Vcyl + g["Vconv"] == pytest.approx(1.0 * np.pi * 0.06**2 / 4.0, rel=1e-12)  # V_c = L* A_t
    assert lq.residence_time(1.0, 1700.0, 3.0e6) == pytest.approx(1700.0 / 3.0e6)


# ------------------------------------------------------------------------------------------- feed network
def _chain(rho, a, mdot, pc, dpf=0.25, **kw):
    A1 = 1.2e-3
    dp = dpf * pc
    d = dict(rho=rho, a_snd=a, p_tank=7.5e6, V_ullage=0.02, gamma_g=1.3, regulated=True, l1=3.0, A1=A1, K1=2.0,
             Kv0=3.0, l2=0.3, A2=A1, V_m=2e-4, Cd=0.7, A_inj=mdot / (0.7 * np.sqrt(2 * rho * dp)), t_open=0.02,
             ramp_open=0.05, t_close=0.5, ramp_close=0.05)
    d.update(kw)
    return nw.Chain(**d)


def _system(**kw):
    At = np.pi * 0.06**2 / 4
    return nw.FeedSystem("LOX/RP-1", _chain(1141.0, 900.0, 6.0, 5e6, **kw), _chain(810.0, 1300.0, 2.5, 5e6, **kw),
                         Vc=0.004, At=At, eps=10.0, tau_c=0.002)


def test_feed_transient_reaches_steady_state_and_shuts_down():
    s = _system()
    ss = nw.steady_state(s)
    ch = nw.integrate(s, 0.7, 701)
    i = int(0.45 / 0.7 * 700)
    assert ch["Pc"][i] * 1e6 == pytest.approx(ss["pc"], rel=1e-4)
    assert ch["mdot_o"][i] == pytest.approx(ss["mdot_o"], rel=1e-4) and ch["mdot_f"][i] == pytest.approx(ss["mdot_f"], rel=1e-4)
    assert ch["Pc"][-1] < 0.12 and ch["mdot_o"][-1] < 0.01 * ss["mdot_o"]                # shut down, valves sealed


@pytest.mark.parametrize("ramp,lo,hi", [(2e-4, 0.97, 1.03), (1e-3, 0.95, 1.05), (0.1, 0.0, 0.5)])
def test_water_hammer_joukowsky_limit(ramp, lo, hi):
    """Instantaneous closure: surge = rho a v (Joukowsky); slow closure (>> 4L/a) stays well below it."""
    s = _system(ramp_close=ramp)
    ss = nw.steady_state(s)
    ch = nw.integrate(s, 0.7, 701)
    v = ss["mdot_o"] / (1141.0 * 1.2e-3)
    jk = 1141.0 * 900.0 * v
    surge = (ch["p1_o"].max() - 7.5) * 1e6
    assert lo < surge / jk < hi


def test_blowdown_tank_follows_polytropic_law():
    s = _system(regulated=False, t_close=2.0)
    ch = nw.integrate(s, 1.5, 301)
    t = np.linspace(0, 1.5, 301)
    expelled = np.cumsum(ch["m1_o"] / 1141.0 * np.gradient(t))
    i = 280
    expect = 7.5 * (0.02 / (0.02 + expelled[i])) ** 1.3
    assert ch["ptank_o"][i] == pytest.approx(expect, rel=0.01)
    assert ch["Pc"][250] < ch["Pc"][100]                                              # pressure decays in blowdown


# ------------------------------------------------------------------------------------------- pump-fed cycles
BASE = dict(pair="LOX/RP-1", Pc=6e6, Dt=0.05, eps=12.0, MR=2.6, eta_cstar=0.97, dp_inj_o=0.2, dp_inj_f=0.2,
            dp_jac=0.1, eta_po=0.7, eta_pf=0.65, eta_t=0.6, N0=20000.0, T_t=900.0, cp_t=2700.0, gamma_t=1.15,
            MR_gg=0.35, PR_t=1.4)
CYC = {"gg": {}, "expander": dict(cp_t=3500.0, gamma_t=1.2), "staged": dict(PR_t=1.6, k_gg=1.25, T_t=800.0)}


@pytest.mark.parametrize("cycle", list(CYC))
def test_cycle_balances_exactly_at_design(cycle):
    e, info = cy.design_engine(cy.CycleDesign(cycle=cycle, **{**BASE, **CYC[cycle]}))
    s = cy.steady(e)
    assert s["N"] == pytest.approx(20000.0, rel=1e-5) and s["pc"] == pytest.approx(6e6, rel=1e-5)
    assert e.eta_mech * s["P_t"] == pytest.approx(s["P_pump"], rel=1e-6)
    assert s["m_o"] / s["m_f"] == pytest.approx(2.6, rel=1e-4)
    if cycle == "gg":
        assert 0.005 < info["bleed"] / info["mdot"] < 0.05                           # a few percent bleed overboard


@pytest.mark.parametrize("cycle", ["gg", "staged"])
def test_cycle_spool_up_converges_to_balance(cycle):
    e, _ = cy.design_engine(cy.CycleDesign(cycle=cycle, **{**BASE, **CYC[cycle]}))
    s = cy.steady(e)
    tr = cy.transient(e, 12.0, gg_ramp=0.8, n_frames=61)
    assert tr["N"][-1] / s["N"] > 0.98 and tr["pc"][-1] / s["pc"] > 0.97
    assert np.all(np.diff(tr["N"]) > -1.0)                                            # monotone spool-up


# ------------------------------------------------------------------------------------------- chugging (MATHKIT 4.4)
def test_chugging_margin_agrees_with_time_domain():
    theta, tau = 0.002, 0.001
    m = chg.stability_margins(1.0, tau, theta)
    kc = m["kappa_crit"]
    t, pi, _ = chg.simulate(0.8 * kc, tau, theta, 0.0, 0.3)
    assert chg.growth_rate(t, pi) < 0                                                 # below critical gain: decays
    t, pi, _ = chg.simulate(1.2 * kc, tau, theta, 0.0, 0.3)
    assert chg.growth_rate(t, pi) > 0                                                 # above: grows
    t, pi, _ = chg.simulate(1.0, 1e-6, 0.01, 0.0, 0.05)
    rate = np.polyfit(t[10:2000], np.log(np.abs(pi[10:2000])), 1)[0]
    assert rate == pytest.approx(-(1 + 1.0) / 0.01, rel=1e-3)                         # no-delay analytic limit


# ------------------------------------------------------------------------------------------- Run drivers
def test_run_drivers_produce_consistent_results():
    ev = threading.Event()
    f = FeedTransient().run(FeedParams(), lambda x: None, ev)
    assert f.meta["Pc_max_MPa"] == pytest.approx(f.meta["Pc_ss_MPa"], rel=0.01)
    c = PumpFedCycle().run(CycleParams(), lambda x: None, ev)
    assert c.meta["N_ss"] == pytest.approx(CycleParams().N0, rel=1e-5) and c.meta["settle"] > 0.95
    stable = Chugging().run(ChuggingParams(dp_inj=40.0), lambda x: None, ev)
    unstable = Chugging().run(ChuggingParams(dp_inj=4.0), lambda x: None, ev)
    assert stable.meta["stable"] and not unstable.meta["stable"]
    assert stable.meta["growth"] < 0 < unstable.meta["growth"]
