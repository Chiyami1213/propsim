"""Nuclear thermal propulsion against the worked examples of Emrich, *Principles of Nuclear Rocket Propulsion*, 2nd ed.
(numbers only; page references in each test). Emrich works in cm, g and W·cm⁻³; the core is SI, so the tests convert."""
import threading

import numpy as np
import pytest

from propsim.core.nozzle import exit_state
from propsim.core.nuclear import criticality as cr
from propsim.core.nuclear import cycles as cy
from propsim.core.nuclear import kinetics as kn
from propsim.core.nuclear import performance as pf
from propsim.core.nuclear import poisons as po
from propsim.core.nuclear import thermal as th


# ---------------------------------------------------------------------------------------------- Ch. 2 performance
def test_isp_hydrogen_area_ratio_200():
    """Emrich Ch. 2 example: H2 (m_w 2, gamma 1.4), area ratio 200, P_e/P_0 = 9.53e-5, about 918 s (919 s with
    g_c = 9.8)."""
    _, pe = exit_state(200.0, 1.4)
    assert pe == pytest.approx(9.53e-5, rel=2e-3)
    assert pf.isp_thermo(3000.0, 2.0, 1.4, pe) == pytest.approx(918.0, abs=1.0)


def test_isp_infinite_expansion_ntr_vs_chemical():
    """Eq. 2.20: H2 at 3000 K gives 944 s, a m_w 10 / gamma 1.33 exhaust at 3500 K gives 494 s, ratio 1.9."""
    a = pf.isp_thermo(3000.0, 2.0, 1.41)
    b = pf.isp_thermo(3500.0, 10.0, 1.33)
    assert a == pytest.approx(944.0, abs=1.0) and b == pytest.approx(494.0, abs=1.0)
    assert a / b == pytest.approx(1.9, abs=0.02)


def test_performance_consistency():
    o = pf.performance(2750.0, 7e6, 200.0, 2.016, 1.41, 111e3, fluid="Hydrogen")
    assert o["isp_v"] == pytest.approx(o["cstar"] * o["cf_v"] / 9.80665, rel=1e-12)
    assert o["mdot"] * o["isp_v"] * 9.80665 == pytest.approx(111e3, rel=1e-12)
    assert o["isp_v"] < o["isp_max"]                                   # finite nozzle below the infinite limit
    # the real-gas enthalpy rise of hydrogen exceeds the frozen c_p estimate (c_p grows with temperature)
    frozen = pf.performance(2750.0, 7e6, 200.0, 2.016, 1.41, 111e3)
    assert o["P_R"] > frozen["P_R"]


# ---------------------------------------------------------------------------------------------- Ch. 8 criticality
def test_bare_sphere_critical_radius_and_mass():
    """Emrich Ch. 8 example (illustrative one-group data): nuSf 0.13, Sa 0.10 cm⁻¹, D 1 cm, 10 g·cm⁻³ gives
    k_inf 1.30, R_c 16.14 cm and M_c 176 kg."""
    a = cr.analyse(13.0, 10.0, 0.01, "sphere", 0.2, 1.0, 0.0, 0.005, 0.05, 1e4)
    assert a["kinf"] == pytest.approx(1.30) and a["Bm2"] * 1e-4 == pytest.approx(0.03)
    assert a["Rc"] * 100 == pytest.approx(16.14, abs=0.01)
    assert a["Mc"] == pytest.approx(176.0, abs=0.5)


def test_keff_is_one_at_the_critical_size():
    for geom in ("sphere", "cylinder", "slab"):
        Rc = cr.critical_size(geom, 300.0, 0.02, 0.8)
        Bg2 = cr.geometric_buckling(geom, Rc, 2 * 0.8 * Rc, 0.02)
        assert cr.k_eff(13.0, 10.0, 0.01, Bg2) == pytest.approx(1.0, abs=1e-9)


def test_reflected_slab_thick_reflector_limit():
    """Eq. 8.41 with a thick reflector: coth -> 1, so tan(alpha a) = D_r / (D_c alpha L_r). Reflector savings grow
    with thickness and saturate after a few diffusion lengths."""
    Bm2, Dc, Dr, Lr = 300.0, 0.01, 0.005, 0.05
    al = np.sqrt(Bm2)
    a = cr.reflected_slab_half_thickness(Bm2, Dc, Dr, Lr, 1.0)
    assert np.tan(al * a) == pytest.approx(Dr / (Dc * al * Lr), rel=1e-6)
    sav = [cr.reflector_savings(Bm2, Dc, Dr, Lr, b) for b in (0.01, 0.05, 0.1, 0.3)]
    assert all(np.diff(sav) > 0) and sav[3] - sav[2] < 0.02 * sav[3]


# ---------------------------------------------------------------------------------------------- Ch. 9 fuel element
CELL = dict(ri=0.0015, ro=0.0035, L=1.0, a=3.0, cp=15200.0, k=30.0, hc=1e5)   # Emrich Ch. 9 example in SI


def test_cell_resistance_and_textbook_constant():
    """R_tot = 0.1230 K·cm³·W⁻¹ and C1/C2 = 5.6016 / mdot (Emrich pp. 153-157)."""
    c = CELL
    R = th.cell_resistance(c["ri"], c["ro"], c["k"], c["hc"])
    assert R * 1e6 == pytest.approx(0.1230, abs=5e-5)
    Af = np.pi * (c["ro"] ** 2 - c["ri"] ** 2)
    ratio = Af / (c["a"] * 1e-3 * c["cp"] * R)                        # mdot = 1 g/s
    assert ratio == pytest.approx(5.6016, abs=2e-3)


def test_axial_cell_peak_temperature_example():
    """T_max,pk = 3100 K at mdot = 1 g/s needs q0 = 2019 W·cm⁻³ (P_ave 1342), the peak sits at 96.5 cm and the
    propellant leaves at 3075 K."""
    c = CELL
    q0 = th.q0_for_peak(3100.0, 300.0, c["ri"], c["ro"], c["L"], c["a"], 1e-3, c["cp"], c["k"], c["hc"])
    assert q0 / 1e6 == pytest.approx(2019.0, abs=1.0)
    P_ave = q0 * 2 * np.sin(0.5 * c["a"] * c["L"]) / (c["a"] * c["L"])
    assert P_ave / 1e6 == pytest.approx(1342.0, abs=1.0)
    assert th.q0_from_average(P_ave, c["a"], c["L"]) == pytest.approx(q0, rel=1e-12)
    zpk, Tpk = th.peak_closed_form(c["ri"], c["ro"], c["L"], c["a"], q0, 1e-3, c["cp"], c["k"], 300.0, c["hc"])
    assert zpk * 100 == pytest.approx(96.5, abs=0.1) and Tpk == pytest.approx(3100.0, abs=1e-6)
    o = th.axial_cell(c["ri"], c["ro"], c["L"], c["a"], q0, 1e-3, c["cp"], c["k"], 300.0, hc=c["hc"], n=2001)
    assert o["Tout"] == pytest.approx(3075.0, abs=1.0)
    assert o["Tpk"] == pytest.approx(3100.0, abs=0.5) and o["zpk"] * 100 == pytest.approx(96.5, abs=0.1)
    # doubling the flow at the same peak raises the power density to 2625 W·cm⁻³ and drops the outlet to 3013 K
    q2 = th.q0_for_peak(3100.0, 300.0, c["ri"], c["ro"], c["L"], c["a"], 2e-3, c["cp"], c["k"], c["hc"])
    assert q2 * 2 * np.sin(1.5) / 3.0 / 1e6 == pytest.approx(2625.0, abs=3.0)
    o2 = th.axial_cell(c["ri"], c["ro"], c["L"], c["a"], q2, 2e-3, c["cp"], c["k"], 300.0, hc=c["hc"], n=2001)
    assert o2["Tout"] == pytest.approx(3013.0, abs=2.0)


def test_dittus_boelter_example():
    """Emrich Ch. 9 transpiration example, no-blowing part: Re 10 000, Pr 0.7 gives Nu 31.6 and h 0.27 W·cm⁻²·K⁻¹."""
    Re, Pr, k, D = 1e4, 0.7, 0.017, 2.0
    Nu = 0.023 * Re**0.8 * Pr**0.4
    assert Nu == pytest.approx(31.6, abs=0.05) and Nu * k / D == pytest.approx(0.27, abs=0.005)


def test_pressure_drop_falls_along_the_channel():
    c = CELL
    o = th.axial_cell(c["ri"], c["ro"], c["L"], c["a"], 2.0e9, 1e-3, c["cp"], c["k"], 300.0, hc=None, p_in=7e6)
    assert np.all(np.diff(o["p"]) < 0) and 0 < o["dp"] < 0.5e6
    assert o["hc"][-1] != pytest.approx(o["hc"][0])


# ---------------------------------------------------------------------------------------------- Ch. 3 cycles
def test_expander_pump_outlet_pressure():
    """Emrich Ch. 3 example (pp. 26-28): saturated H2 at 20 K, isentropic pump and turbine, warmed to 100 K, reactor
    inlet 7 MPa: pump outlet about 9.67 MPa, 24 K."""
    o = cy.cycle("expander", 7e6, 2750.0, T1=20.0, p_tank=0.0, T3=100.0, dp_core=0.0, dp_regen=0.0, eta_p=1.0,
                 eta_t=1.0)
    assert o["p2"] / 1e6 == pytest.approx(9.67, abs=0.03)
    assert o["T2"] == pytest.approx(24.0, abs=0.5)
    st = {s["name"]: s for s in o["states"]}
    assert st["2"]["h"] - st["1"]["h"] == pytest.approx(st["3"]["h"] - st["4"]["h"], rel=1e-9)   # Eq. 3.3


def test_bleed_cycle_power_balances():
    for kind in ("hot", "cold"):
        o = cy.cycle(kind, 7e6, 2750.0)
        st = {s["name"]: s for s in o["states"]}
        wp = st["2"]["h"] - st["1"]["h"]
        tin, tout = ("5", "6") if kind == "hot" else ("3", "4")
        assert o["fb"] * (st[tin]["h"] - st[tout]["h"]) == pytest.approx(wp, rel=1e-9)          # Eqs. 3.1, 3.2
        assert o["isp"] < o["isp_main"]
    hot, cold = cy.cycle("hot", 7e6, 2750.0), cy.cycle("cold", 7e6, 2750.0)
    assert hot["fb"] < cold["fb"]                         # the hot bleed needs far less turbine flow
    assert cy.cycle("expander", 7e6, 2750.0)["fb"] == 0.0


# ---------------------------------------------------------------------------------------------- Ch. 11-12 kinetics
def test_delayed_neutron_data():
    """Table 11.1: beta = 0.0065 and composite lambda = 0.0764 s⁻¹."""
    assert kn.BETA == pytest.approx(0.0065) and kn.LAM_1G == pytest.approx(0.0764, abs=1e-4)


def test_inhour_doubling_time_for_ten_cents():
    """Ch. 11 example: 10 cents with Lambda = 50 us gives omega_0 = 0.01011 s⁻¹ (doubling 68.5 s); one group gives
    0.00849 s⁻¹ (81.7 s)."""
    w = kn.inhour_root(0.1 * kn.BETA, 5e-5)
    assert w == pytest.approx(0.01011, abs=5e-6) and np.log(2) / w == pytest.approx(68.5, abs=0.1)
    w1 = 0.1 * kn.BETA * kn.LAM_1G / (0.9 * kn.BETA)
    assert np.log(2) / w1 == pytest.approx(81.7, abs=0.2)


def test_point_kinetics_matches_inhour_period():
    """Without feedback the simulated asymptotic period equals the inhour root."""
    core = kn.Core(Lam=5e-5, alpha_f=0.0, alpha_m=0.0, beta_p=0.0)
    o = kn.simulate(core, kn.Program(n0=1.0, rho1=0.1 * kn.BETA), 600.0, 601)
    w_sim = np.log(o["n"][-1] / o["n"][-101]) / (o["t"][-1] - o["t"][-101])
    assert w_sim == pytest.approx(kn.inhour_root(0.1 * kn.BETA, 5e-5), rel=1e-3)


def test_shutdown_cooling_propellant():
    """Ch. 11 example: 1000 MW for 600 s, scram at -$5: Q_sd = 16.4 GJ and 2.7 % of the propellant; -$1 gives
    4.3 %."""
    q = kn.shutdown_energy_ratio(600.0, -5 * kn.BETA, 5e-5)
    assert q * 1e9 / 1e9 == pytest.approx(16.39, abs=0.02)
    assert kn.cooldown_fraction(600.0, -5 * kn.BETA, 5e-5) == pytest.approx(0.027, abs=0.001)
    assert kn.cooldown_fraction(600.0, -1 * kn.BETA, 5e-5) == pytest.approx(0.043, abs=0.001)
    assert kn.decay_heat_fraction(1.0, 600.0) == pytest.approx(0.066 * (1 - 601**-0.2))


def test_nerva_stability_example():
    """Ch. 12 example (pp. 236-238): tau_f 0.063, tau_m 0.091, tau_p 0.0031 s; A_f 0.031, A_m 0.028 cm³·K·W⁻¹;
    B_p 1.11e-8 g·W⁻¹; net 7.7e-7 (stable); critical beta_p about 169-170 cm³·g⁻¹; low-frequency gain 122 dB."""
    c = kn.Core(R_p=4200.0, cp_p=16800.0)
    k = c.constants()
    assert k["tau_f"] == pytest.approx(0.063, abs=5e-4) and k["tau_m"] == pytest.approx(0.091, abs=5e-4)
    assert k["tau_p"] == pytest.approx(0.0031, abs=1e-4)
    assert k["A_f"] * 1e6 == pytest.approx(0.031, abs=5e-4) and k["A_m"] * 1e6 == pytest.approx(0.028, abs=5e-4)
    assert k["B_p"] * 1e3 == pytest.approx(1.11e-8, rel=0.01)                 # kg·m³·W⁻¹·m⁻³ -> g·W⁻¹
    assert k["net"] * 1e6 == pytest.approx(7.7e-7, rel=0.03) and k["net"] > 0
    assert k["beta_p_crit"] * 1e3 == pytest.approx(170.0, abs=2.0)
    H = c.transfer(np.array([1e-4]))[0] / 1e6
    assert 20 * np.log10(abs(H)) == pytest.approx(122.0, abs=0.5)
    unstable = kn.Core(beta_p=0.2).constants()
    assert unstable["net"] < 0


def test_kinetics_run_model_scenarios():
    from propsim.sims.nuclear_run import KineticsParams, ReactorKinetics
    for sc in ("startup", "step", "shutdown"):
        r = ReactorKinetics().run(KineticsParams(scenario=sc), lambda f: None, threading.Event())
        assert np.all(np.isfinite(r.channels["P_tot"])) and r.n == ReactorKinetics.N_FRAMES
        if sc == "startup":
            assert r.channels["P_tot"][-1] == pytest.approx(1.0, abs=0.05)
        if sc == "shutdown":
            assert r.channels["n"][-1] < 0.01 and r.channels["decay"][-1] > 0


# ---------------------------------------------------------------------------------------------- Ch. 13 poisons
def test_xenon_equilibrium_and_post_shutdown_peak():
    """Ch. 13 example: equilibrium Sigma_Xe / Sigma_f = 0.0621 at 1e14 and 0.0665 at 1e15; after shutdown the peak
    comes at 10.1 h (4.5 x equilibrium) and 11 h (38.6 x); at 1e12 there is almost no rise (1.002)."""
    assert po.xenon_equilibrium(1e14) == pytest.approx(0.0621, abs=1e-4)
    assert po.xenon_equilibrium(1e15) == pytest.approx(0.0665, abs=1e-4)
    t, k = po.post_shutdown_peak(1e14)
    assert t == pytest.approx(10.1, abs=0.1) and k == pytest.approx(4.5, abs=0.05)
    t, k = po.post_shutdown_peak(1e15)
    assert t == pytest.approx(11.0, abs=0.1) and k == pytest.approx(38.6, abs=0.2)
    assert po.post_shutdown_peak(1e12)[1] == pytest.approx(1.002, abs=1e-3)


def test_xenon_history_reaches_equilibrium():
    h = po.history([(200 * po.HOUR, 1e14)], n=400)
    assert h["xe"][-1] == pytest.approx(po.xenon_equilibrium(1e14), rel=1e-3)
    # samarium equilibrium gamma_Nd / sigma_Sm * sigma_Sm = gamma_Nd (Sigma_a^Sm / Sigma_f = 0.0113)
    h = po.history([(2000 * po.HOUR, 1e15)], n=400)
    assert h["sm"][-1] == pytest.approx(0.0113, rel=1e-3)
