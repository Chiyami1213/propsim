"""Hybrid core physics vs the MATHKIT.md Section 3 reference values, plus conservation checks on the Run models."""
from __future__ import annotations

import threading

import numpy as np
import pytest

from propsim.core import feed
from propsim.core.ballistics import hybrid1d as h1
from propsim.core.flowfields import bidirectional_mantle, mantle_asymptotic, streamline, stream_function, taylor_culick
from propsim.core.regression import (POWER_LAWS, arrhenius_rdot, marxman_rdot, port_radius_analytic, rdot_radiation,
                                     thermal_wave)
from propsim.core.stability import freq_estimate, ilfi_freq, ilfi_tau_bl2, similarity_scale
from propsim.sims.hybrid_burn1d import (HybridBurn1D, HybridBurn1DParams, N2OBlowdown, N2OBlowdownParams)

IN, LBM, PSI = 0.0254, 0.45359237, 6894.757293168


def test_regression_reference_values():
    assert marxman_rdot(200.0, 0.5, 6.0, 8e-5, 920.0) * 1e3 == pytest.approx(0.713, abs=5e-4)
    # MATHKIT 3.1: Qc = 1 MW/m2, Qrad = 0.3 MW/m2, dHv,eff = 1.81 MJ/kg -> 0.625 mm/s (rho_f = 920)
    assert rdot_radiation(1e6, 0.3e6, 920.0, 1.81e6) * 1e3 == pytest.approx(0.625, abs=1.5e-3)
    a, n = POWER_LAWS["Paraffin / GOX"]
    assert a == pytest.approx(0.1171e-3, rel=1e-6) and n == 0.62
    assert float(arrhenius_rdot(700.0, 1e4, 1.2e5)) < float(arrhenius_rdot(800.0, 1e4, 1.2e5))
    x = np.linspace(0, 1e-3, 50)
    prof = thermal_wave(x, 1e-3, 1e-7)
    assert prof[0] == 1.0 and np.all(np.diff(prof) < 0) and thermal_wave(1e-4, 1e-3, 1e-7) == pytest.approx(np.exp(-1))


def test_taylor_culick_wall_conditions():
    ur, uz = taylor_culick(1.0, 0.7)
    assert ur == pytest.approx(-1.0, abs=1e-12) and uz == pytest.approx(0.0, abs=1e-12)
    # streamline: psi constant along it; headwall injection shifts it
    z, r = streamline(0.5, 0.2, 2.0, u_h=0.1)
    psi = stream_function(r, z, 0.1)
    assert np.allclose(psi, psi[0], rtol=1e-9)
    ur0, _ = taylor_culick(0.0, 1.0)
    assert float(ur0) == 0.0


def test_mantle_beta_reference_values():
    assert bidirectional_mantle(0.01, 1e-12) == pytest.approx(1 / np.sqrt(2), abs=1e-6)
    assert bidirectional_mantle(0.01, 0.001) == pytest.approx(0.717888, abs=2e-6)
    assert bidirectional_mantle(0.01, 0.005) == pytest.approx(0.753469, abs=2e-6)
    assert mantle_asymptotic(0.01, 0.001) == pytest.approx(0.717896, abs=2e-6)


def test_ilfi_nasa_ames_4nf04():
    tau = ilfi_tau_bl2(45 * IN, 568.5 * PSI, 0.48 * LBM / IN**2, 3.01, 6.38e5)
    assert tau * 1e3 == pytest.approx(18.29, abs=0.02)
    assert ilfi_freq(tau) == pytest.approx(26.24, abs=0.02)
    # the brief's quick estimate agrees to within ~10 % with the delay-based frequency
    f2 = freq_estimate(45 * IN, 568.5 * PSI, 0.48 * LBM / IN**2, 3.01, 6.38e5)
    assert f2 == pytest.approx(ilfi_freq(tau), rel=0.15)


def test_similarity_scaling():
    out = similarity_scale({"mdot_ox": 1.8, "thrust": 2000.0, "t_burn": 10.0, "freq": 26.0}, 2.0)
    assert out == {"mdot_ox": 7.2, "thrust": 8000.0, "t_burn": 20.0, "freq": 13.0}


def test_n2o_esdu_and_spi():
    assert float(feed.psat(293.15)) / 1e6 == pytest.approx(5.060, abs=5e-4)
    assert float(feed.rho_l(293.15)) == pytest.approx(786.6, abs=0.1)
    assert float(feed.rho_v(293.15)) == pytest.approx(158.1, abs=0.1)
    assert float(feed.psat(feed.TC)) / 1e6 == pytest.approx(7.251, abs=1e-9)
    assert feed.mdot_spi(0.65, np.pi * 0.75e-3**2, 786.0, 2e6) == pytest.approx(0.0644, abs=5e-5)


def test_n2o_coolprop_consistency():
    from CoolProp.CoolProp import PropsSI
    for T in (250.0, 280.0, 300.0):
        assert float(feed.psat(T)) == pytest.approx(PropsSI("P", "T", T, "Q", 0, "NitrousOxide"), rel=0.01)
        assert float(feed.rho_l(T)) == pytest.approx(PropsSI("D", "T", T, "Q", 0, "NitrousOxide"), rel=0.01)


@pytest.mark.parametrize("T0", [260.0, 285.0, 293.15, 300.0])
@pytest.mark.parametrize("fill", [0.1, 0.5, 0.9])
def test_tank_state_roundtrip(T0, fill):
    V = 0.02
    m, U = feed.tank_initial(T0, V, fill)
    st = feed.tank_state(m, U, V)
    assert st.liquid and st.T == pytest.approx(T0, abs=1e-5)
    assert st.m_liq + st.m_vap == pytest.approx(m, rel=1e-12)
    assert st.m_liq / float(feed.rho_l(T0)) + st.m_vap / float(feed.rho_v(T0)) == pytest.approx(V, rel=1e-9)


def test_tank_drains_through_depletion_monotonically():
    V = 0.02
    m, U = feed.tank_initial(293.15, V, 0.8)
    p_prev, liquid = np.inf, True
    for _ in range(300):
        st = feed.tank_state(m, U, V)
        if not st.liquid:
            liquid = False
            break
        assert st.p <= p_prev + 1e-6
        p_prev = st.p
        m -= 0.05
        U -= 0.05 * float(feed.h_l(st.T))
    assert not liquid and st.m_vap > 0.5            # depletes with residual vapour, no exception on the way


def test_dyer_limits():
    A = 8 * np.pi * (1.25e-3) ** 2
    T = 289.0
    ps = float(feed.psat(T))
    spi = feed.mdot_spi(0.65, A, float(feed.rho_l(T)), 0.95 * ps - 1e6)
    dy = feed.mdot_dyer(0.65, A, T, 0.95 * ps, 1e6, ps)
    assert 0.2 * spi < dy < spi                       # two-phase flashing reduces flow below SPI
    # no flashing when the downstream pressure is at or above the vapour pressure: pure single-phase liquid flow
    assert feed.mdot_dyer(0.65, A, T, 1.2 * ps, 1.1 * ps, ps) == pytest.approx(
        feed.mdot_spi(0.65, A, float(feed.rho_l(T)), 0.1 * ps), rel=1e-12)
    assert feed.mdot_dyer(0.65, A, T, 1e6, 2e6, ps) == 0.0      # no flow against the pressure gradient


def _motor(**kw):
    d = dict(pair="GOX/HTPB", L=0.75, R0=0.025, R_out=0.06, rho_f=920.0, law="power", a=0.0541e-3, n=0.647,
             B=6.0, mu=8e-5, Dt=0.032, eps=6.0, eta_cstar=0.95, N=20)
    d.update(kw)
    return h1.Motor1D(**d)


def test_quasi1d_single_cell_equals_analytic_port():
    """With one axial cell the march reduces to MATHKIT 3.2: R(10 s) = 50.846 mm (fast valve removes ramp error)."""
    sup = h1.ValveSupply(1.8, 10.0, tau_open=1e-4, tau_close=1e-4)
    ch = h1.integrate(_motor(N=1), sup, 10.0, 301)
    ref = port_radius_analytic(10.0, 0.025, 0.0541e-3, 0.647, 1.8) * 1e3
    assert ref == pytest.approx(50.846, abs=5e-4)
    assert ch["R"][-1] == pytest.approx(ref, abs=5e-3)


def test_quasi1d_axial_gradient_and_mass_conservation():
    p = _motor()
    ch = h1.integrate(p, h1.ValveSupply(1.8, 10.0), 10.0, 601)
    t = np.linspace(0, 10.0, 601)
    # downstream stations see more mass flux (fuel adds to G) -> larger radius at the aft end
    assert ch["Rp_19"][-1] > ch["Rp_00"][-1] + 1.0
    # fuel mass: grain volume lost x rho_f equals integrated fuel flow (to quadrature accuracy)
    R = np.array([ch[f"Rp_{i:02d}"] for i in range(p.N)]) * 1e-3
    burned = p.rho_f * np.pi * np.sum(R[:, -1] ** 2 - p.R0**2) * p.dx
    assert np.trapezoid(ch["mdot_f"], t) == pytest.approx(burned, rel=2e-3)


def test_throttle_and_throat_erosion():
    base = h1.integrate(_motor(), h1.ValveSupply(1.8, 10.0), 10.0, 301)
    thr = h1.integrate(_motor(), h1.ValveSupply(1.8, 10.0, throttle=0.5, t_th1=3.0, t_th2=6.0), 10.0, 301)
    t = np.linspace(0, 10.0, 301)
    i = int(np.argmin(np.abs(t - 4.5)))
    assert thr["mdot_ox"][i] == pytest.approx(0.5 * 1.8, rel=0.02)
    assert thr["Pc"][i] < 0.62 * base["Pc"][i]
    er = h1.integrate(_motor(kt=0.05e-3), h1.ValveSupply(1.8, 10.0), 10.0, 301)
    assert er["Rt"][-1] > er["Rt"][0] + 0.2 and er["Pc"][-1] < base["Pc"][-1]


def test_n2o_blowdown_conserves_oxidizer_and_is_physical():
    P = N2OBlowdownParams()
    res = N2OBlowdown().run(P, lambda f: None, threading.Event())
    t, c = res.t, res.channels
    m_used = np.trapezoid(c["mdot_ox"], t)
    assert m_used == pytest.approx(c["m_tank"][0] - c["m_tank"][-1], rel=1e-4)          # tank <-> injector
    assert np.all(np.diff(c["p_tank"][5:-20]) < 1e3)                                     # blowdown: pressure falls
    assert np.all(c["T_tank"][5:-20] <= c["T_tank"][0] + 1e-6)                           # evaporative cooling
    assert np.all(c["p_tank"][5:-20] > c["Pc"][5:-20] * 1e6)                             # feed always above chamber
    assert 4.0 <= np.nanmin(c["OF"][50:550]) and np.nanmax(c["OF"][50:550]) <= 12.0
    assert 150 < res.meta["Isp_avg"] < 280


def test_valve_fed_run_model_log_and_meta():
    res = HybridBurn1D().run(HybridBurn1DParams(throttle=0.6), lambda f: None, threading.Event())
    assert res.n == 601 and len(res.meta["x_cells"]) == 20
    assert any("throttle" in s for s in res.log)
    of_mid = res.channels["OF"][60:540]                     # skip the valve start-up / shutdown transients
    assert np.all(np.isfinite(of_mid)) and 1.3 <= of_mid.min() and of_mid.max() <= 6.0
    assert 150 < res.meta["Isp_avg"] < 330
