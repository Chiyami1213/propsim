"""Solid-motor validation: MathKit Section 2 values, level-set convergence and conservation, ballistics checks."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from propsim.core.ballistics import solid as S
from propsim.core.grain import levelset as LS
from propsim.core.grain import section2d as S2
from propsim.sims.solid_burn import (Solid3DParams, SolidBallistics, SolidBallisticsParams, SolidBurnback3D)

from .figs import new_fig, save


# ------------------------------------------------------------------------------------------- burn law (Section 2.1)
def test_equilibrium_pressure_mathkit():
    prop = S.SolidProp()
    pc = S.p_equilibrium(prop, 250.0)
    assert pc / 1e6 == pytest.approx(6.222, abs=5e-4)
    resid = (prop.rho_p * prop.a_eff * pc**prop.n * 0.5 - pc * 2e-3 / prop.cstar) / (pc * 2e-3 / prop.cstar)
    assert abs(resid) < 1e-12


def test_unit_trap_and_temperature_sensitivity():
    assert S.a_from_mm_MPa(5.0, 0.35) * 1e6**0.35 == pytest.approx(5e-3, rel=1e-12)       # 5 mm/s at 1 MPa
    cold, hot = S.SolidProp(Ti=273.15), S.SolidProp(Ti=313.15)
    ratio = S.p_equilibrium(hot, 250.0) / S.p_equilibrium(cold, 250.0)
    assert ratio == pytest.approx(np.exp(hot.sigma_p * 40.0 / (1 - hot.n)), rel=1e-12)      # pi_K = sigma_p / (1 - n)


def test_stability_flags():
    assert "stable" in S.stability_note(0.35)
    assert "WARNING" in S.stability_note(0.7) and "UNSTABLE" in S.stability_note(1.0)
    with pytest.raises(ValueError):
        S.p_equilibrium(S.SolidProp(n=1.0), 100.0)


# ------------------------------------------------------------------------------------------- 0-D transient (2.2)
def _const_ab_motor(**kw):
    """MATHKIT 2.2 setup: constant A_b = 0.5 m^2, A_t = 2e-3 m^2, V_c0 = 0.02 m^3, igniter and thresholds off."""
    tab = S.SectionTable(np.array([0.0, 10.0]), np.array([0.01, 0.01]), np.array([1.0, 1.0]))
    d = dict(prop=S.SolidProp(alpha=0.0), table=tab, L=0.5, At=2e-3, eps=6.0, N=1, V_extra=0.02 - 0.01 * 0.5,
             m_ig=0.0, t_ig=1e-3, v_fs=1e9, p_ign=0.0, p0=2e5)
    d.update(kw)
    return S.SolidMotor(**d)


def test_zero_d_transient_approaches_equilibrium_from_below():
    m = _const_ab_motor()
    ch = S.integrate(m, 0.5, 201)
    peq = S.p_equilibrium(m.prop, 250.0) / 1e6
    assert ch["Pc"][-1] / peq == pytest.approx(0.9943, abs=5e-4)                 # MATHKIT value
    assert np.all(np.diff(ch["Pc"]) > 0) and ch["Pc"].max() < peq               # monotone, from below


def test_spatial_equals_zero_d_without_erosion():
    tab = S.SectionTable(np.array([0.0, 0.03]), np.array([8e-4, 8e-3]), np.array([0.35, 0.35]))
    base = dict(prop=S.SolidProp(alpha=0.0), table=tab, L=0.6, At=1.0e-3, eps=8.0, m_ig=0.0, t_ig=1e-3, v_fs=1e9,
                p_ign=0.0, p0=2e5, R_case=0.06)
    a = S.integrate(S.SolidMotor(N=1, **base), 3.0, 301)
    b = S.integrate(S.SolidMotor(N=12, **base), 3.0, 301)
    assert b["Pc"] == pytest.approx(a["Pc"], rel=2e-3, abs=2e-3)


# ------------------------------------------------------------------------------------------- erosive burning (2.3)
def test_lenoir_robillard_implicit_equation_and_limits():
    prop = S.SolidProp()
    r0 = np.full(5, 5e-3)
    G = np.array([1e-9, 20.0, 100.0, 300.0, 800.0])
    D = np.full(5, 0.05)
    r = S.lenoir_robillard(r0, G, D, prop)
    res = r - r0 - prop.alpha * G**0.8 * D**-0.2 * np.exp(-prop.beta * r * prop.rho_p / G)
    assert np.max(np.abs(res)) < 1e-12
    assert r[0] == pytest.approx(r0[0]) and np.all(np.diff(r) >= 0) and r[-1] > 1.05 * r0[-1]
    assert np.all(S.lenoir_robillard(r0, G, D, S.SolidProp(alpha=0.0)) == r0)


# ------------------------------------------------------------------------------------------- 2-D sections
def test_section_exact_cases():
    b = S2.section("circle", 60.0, R_p=20.0)
    R = 20.0 + b.d
    ok = R < 58.0
    assert np.max(np.abs(b.area[ok] / (np.pi * R[ok] ** 2) - 1)) < 0.004
    assert np.max(np.abs(b.perim[ok] / (2 * np.pi * R[ok]) - 1)) < 0.003
    n, rt, rv, Ro = 6, 0.7, 0.3, 60.0
    star = S2.section("star", Ro, n=n, r_tip=rt, r_valley=rv)
    assert star.area[0] == pytest.approx(0.5 * 2 * n * rt * Ro * rv * Ro * np.sin(np.pi / n), rel=2e-3)


@pytest.mark.parametrize("kind", S2.KINDS)
def test_section_conservation(kind):
    b = S2.section(kind, 60.0)
    assert np.all(np.diff(b.area) >= 0) and b.area[-1] == pytest.approx(b.A_case, rel=1e-9)
    assert b.remain[0] == pytest.approx(1.0) and b.remain[-1] == pytest.approx(0.0, abs=1e-9)
    assert b.perim[0] > 0 and b.perim[-1] == 0.0


# ------------------------------------------------------------------------------------------- level set (2.4)
def test_levelset_mathkit_circular_port_20_to_50_mm():
    """401^2 grid, port 20 -> 50 mm (30 mm burned): max |phi - exact| < 1 cell (MathKit 0.17 mm), perimeter 0.9977."""
    n, Rg = 401, 100.0
    xs = np.linspace(-Rg, Rg, n)
    X, Y = np.meshgrid(xs, xs)
    h = xs[1] - xs[0]
    ls = LS.LevelSet(np.hypot(X, Y) - 20.0, np.hypot(X, Y) <= Rg, h)
    ls.run(30.0, n_samples=4)
    exact = np.hypot(X, Y) - 50.0
    band = np.abs(exact) < 3 * h
    err = float(np.max(np.abs(ls.phi - exact)[band]))
    assert err < h and err == pytest.approx(0.17, abs=0.03)
    assert ls.burning_area() / (2 * np.pi * 50.0) == pytest.approx(0.9977, abs=2e-3)


def test_godunov_gradient_sign_branch():
    """For F > 0 the Osher-Sethian branch max(D-,0)^2 + min(D+,0)^2 gives |grad| = 1 on a distance function."""
    x = np.linspace(-1, 1, 201)
    X, Y = np.meshgrid(x, x)
    phi = np.hypot(X, Y) - 0.5
    g = LS.godunov_np(phi, x[1] - x[0])
    band = np.abs(phi) < 0.3
    assert np.max(np.abs(g[band] - 1.0)) < 0.05
    # the opposite branch (F < 0) would give the wrong answer on the inside of the port: guard against swapping them
    gn = LS.godunov_np(-phi, x[1] - x[0])
    assert np.max(np.abs(gn[band] - 1.0)) < 0.05


@pytest.mark.skipif(not LS.HAVE_NUMBA, reason="numba not installed")
def test_numba_matches_numpy():
    phi, act, h = LS.phi_extruded(lambda Y, Z: np.hypot(Y, Z) - 10.0, 30.0, 40.0, 2.0)
    a = LS.LevelSet(phi, act, h, use_numba=True)
    b = LS.LevelSet(phi, act, h, use_numba=False)
    for _ in range(5):
        a.step(1.0)
        b.step(1.0)
    assert np.max(np.abs(a.phi - b.phi)) < 1e-4


def test_levelset_3d_converges_at_first_order_and_conserves():
    """Extruded circular port: area and gas volume approach 2 pi R L and pi R^2 L; the error shrinks with the cell."""
    R0, w, L, Ro = 20.0, 15.0, 100.0, 60.0
    errs, verr = [], []
    for h in (5.0, 2.5, 1.25):
        phi, act, _ = LS.phi_extruded(lambda Y, Z: np.hypot(Y, Z) - R0, Ro, L, h)
        ls = LS.LevelSet(phi, act, h)
        tab = ls.run(w, n_samples=2)
        errs.append(abs(tab["A_b"][-1] / (2 * np.pi * (R0 + w) * L) - 1))
        verr.append(abs(tab["V_gas"][-1] / (np.pi * (R0 + w) ** 2 * L) - 1))
    assert errs[2] < errs[1] < errs[0] and verr[2] < verr[1] < verr[0]
    assert errs[1] / errs[0] < 0.65 and errs[2] / errs[1] < 0.65                     # roughly first order (halving)
    assert errs[2] < 0.01 and verr[2] < 0.012
    assert ls.surface_area_marching_cubes() == pytest.approx(2 * np.pi * (R0 + w) * L, rel=0.01)   # independent check
    fig, (ax,) = new_fig()
    ax.loglog([5.0, 2.5, 1.25], errs, "o-", color="#ff5a3c", label=r"burning area error")
    ax.loglog([5.0, 2.5, 1.25], verr, "s-", color="#4ea1ff", label=r"gas volume error")
    ax.loglog([5.0, 2.5, 1.25], [errs[0] * h / 5.0 for h in (5.0, 2.5, 1.25)], "k--", lw=0.8, label=r"first order")
    ax.set_xlabel(r"cell size $\Delta x$ (mm)")
    ax.set_ylabel(r"relative error (–)")
    ax.legend()
    save(fig, "solid_levelset_convergence")


def test_levelset_128_cubed_within_budget_and_cancel():
    """Brief: a 3-D level-set burnback at ~128^3 under 30 s (Numba). Smaller here to keep the suite fast; cancel works."""
    import time
    h = 2.5
    phi, act, _ = LS.phi_extruded(lambda Y, Z: np.hypot(Y, Z) - 15.0, 50.0, 120.0, h)
    ls = LS.LevelSet(phi, act, h)
    t = time.time()
    ls.run(15.0, n_samples=4)
    assert time.time() - t < 30.0
    ev = threading.Event()
    ev.set()
    with pytest.raises(InterruptedError):
        LS.LevelSet(phi, act, h).run(15.0, cancel=ev)


# ------------------------------------------------------------------------------------------- Run models
@pytest.fixture(scope="module")
def burn():
    p = SolidBallisticsParams()
    return p, SolidBallistics().run(p, lambda f: None, threading.Event())


def test_ballistics_run_physical_and_conserving(burn):
    p, res = burn
    c, t, m = res.channels, res.t, res.meta
    assert 0.5 * m["Pc_eq_MPa"] < c["Pc"].max() < 1.8 * m["Pc_eq_MPa"] and np.all(c["Pc"] >= 0)
    assert c["Pc"][-1] < 0.02 * c["Pc"].max()                                   # tail-off: the chamber empties
    # gas mass balance: igniter + generated gas = nozzle outflow + change of stored gas
    motor = p.to_si()
    rho = c["Pc"] * 1e6 / motor.prop.RT
    stored = rho[-1] * c["Vc"][-1] - rho[0] * c["Vc"][0]
    mass_in = motor.m_ig + np.trapezoid(c["mdot_gen"], t)
    mass_out = np.trapezoid(c["mdot"], t)
    assert mass_in == pytest.approx(mass_out + stored, rel=2e-3)
    # propellant: burned mass = rho_p * port volume gained
    burned = np.trapezoid(c["mdot_gen"], t)
    dV = c["Vc"][-1] - c["Vc"][0]
    assert burned == pytest.approx(motor.prop.rho_p * dV, rel=2e-3)


def test_erosive_burning_and_flame_spreading_effects():
    base = SolidBallisticsParams(t_end=5.0)
    off = SolidBallistics().run(base.model_copy(update={"alpha": 0.0}), lambda f: None, threading.Event())
    on = SolidBallistics().run(base, lambda f: None, threading.Event())
    assert on.meta["erosive_peak"] > 1.05 and off.meta["erosive_peak"] == pytest.approx(1.0)
    assert on.meta["Pc_max_MPa"] > off.meta["Pc_max_MPa"]
    assert on.channels["rdot_aft"].max() > on.channels["rdot_head"].max()                 # aft sees more core flow
    slow = SolidBallistics().run(base.model_copy(update={"v_fs": 5.0}), lambda f: None, threading.Event())
    i = 12                                                                                 # early frame
    assert slow.channels["lit_frac"][i] < on.channels["lit_frac"][i]                      # slow flame spread lights fewer cells


def test_temperature_changes_pressure_and_burn_time():
    cold = SolidBallistics().run(SolidBallisticsParams(T_i=253.0, t_end=6.0), lambda f: None, threading.Event())
    hot = SolidBallistics().run(SolidBallisticsParams(T_i=333.0, t_end=6.0), lambda f: None, threading.Event())
    assert hot.meta["Pc_eq_MPa"] > cold.meta["Pc_eq_MPa"] and hot.meta["t_burn"] < cold.meta["t_burn"]


def test_burnback_3d_run_matches_section_and_reports_tables():
    p = Solid3DParams(kind="circle", R_p=15.0, cells=48.0, L_g=0.3)
    res = SolidBurnback3D().run(p, lambda f: None, threading.Event())
    c, m = res.channels, res.meta
    assert res.n >= 29 and res.arrays["phi_snaps"].shape[0] == res.n
    main = c["w"] <= 0.8 * m["web_mm"]
    assert np.mean(np.abs(c["A_b"][main] / c["A_b_section"][main] - 1)) < 0.04
    assert np.mean(np.abs(c["V_gas"][main] / c["V_gas_section"][main] - 1)) < 0.03
    assert np.all(np.diff(c["propellant_left"]) <= 1e-9) and c["propellant_left"][-1] < 0.02
    assert m["sliver_frac"] < 0.05 and m["progressivity"] > 1.0                    # circular bore burns progressively


def test_partial_fin_grain_is_genuinely_three_dimensional():
    full = Solid3DParams(kind="finocyl", cells=40.0, L_g=0.3, fin_frac=1.0)
    part = full.model_copy(update={"fin_frac": 0.4})
    a = SolidBurnback3D().run(full, lambda f: None, threading.Event()).channels["A_b"]
    b = SolidBurnback3D().run(part, lambda f: None, threading.Event()).channels["A_b"]
    assert b[0] < a[0]                                                             # fewer fin walls at ignition
