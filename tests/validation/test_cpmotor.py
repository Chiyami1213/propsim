"""Quasi-steady CP motor: area formulas, equilibrium pressure, mass conservation, throat erosion, ideal thrust coefficient,
and agreement with an independent plain-loop implementation of the same textbook algorithm in US units."""
import threading

import numpy as np
import pytest

from propsim.core.ballistics import cpmotor as cp
from propsim.core.ballistics import solid as sol
from propsim.core.nozzle import CF, exit_state
from propsim.sims.cp_run import CPMotorRun, CPParamsSI, CPParamsUS

IN, PSI, LBF, LBM, FT = 0.0254, 6894.757293168, 4.4482216152605, 0.45359237, 0.3048


def motor(**kw):
    d = dict(r_o=3 * IN, r_i=1.25 * IN, L=12 * IN, rho_p=0.064 * LBM / IN**3, cstar=5000 * FT, n=0.4, r_ref=0.35 * IN,
             p_ref=1000 * PSI, R_t=0.4 * IN, R_e=1.6 * IN, dw=0.004 * IN)
    d.update(kw)
    return cp.CPMotor(**d)


def test_burn_area_and_volume_formulas():
    m, w = motor(ends=True), np.array([0.0, 0.01, 0.02])
    ri = m.r_i + w
    assert cp.burn_area(m, w) == pytest.approx(2 * np.pi * ri * (m.L - 2 * w) + 2 * np.pi * (m.r_o**2 - ri**2), rel=1e-14)
    assert cp.propellant_volume(m, w) == pytest.approx(np.pi * (m.r_o**2 - ri**2) * (m.L - 2 * w), rel=1e-14)
    inh = motor(ends=False)
    assert cp.burn_area(inh, w) == pytest.approx(2 * np.pi * ri * inh.L, rel=1e-14)       # bore only
    assert cp.burn_area(m, 0.0) > cp.burn_area(inh, 0.0)                                   # end faces add area
    assert motor(ends=True, L=0.05).web == pytest.approx(0.025)                             # end faces meet before the bore


def test_equilibrium_pressure_matches_the_solid_core():
    m = motor()
    d = cp.march(m)
    prop = sol.SolidProp(rho_p=m.rho_p, a=m.a, n=m.n, sigma_p=0.0, cstar=m.cstar)
    assert d["Pc"][0] == pytest.approx(sol.p_equilibrium(prop, d["Kn"][0]), rel=1e-12)
    assert d["mdot_in"] == pytest.approx(d["mdot"], rel=1e-12)                              # mass in = mass out at every step


@pytest.mark.parametrize("ends", [False, True])
@pytest.mark.parametrize("erosion", ["none", "const", "pressure"])
def test_propellant_mass_is_conserved(ends, erosion):
    m = motor(ends=ends, erosion=erosion, edot_ref=0.006 * IN, dw=0.002 * IN)
    d = cp.march(m)
    burnt = float(np.sum(d["mdot_in"] * d["dt"]))
    assert burnt == pytest.approx(d["m_p"][0] - d["m_p"][-1], rel=6e-3)                    # forward-Euler error at this step
    fine = cp.march(cp.CPMotor(**{**m.__dict__, "dw": m.dw / 4}))
    assert abs(np.sum(fine["mdot_in"] * fine["dt"]) / (fine["m_p"][0] - fine["m_p"][-1]) - 1) < abs(burnt / (d["m_p"][0] - d["m_p"][-1]) - 1)


def test_throat_erosion_modes():
    base = cp.march(motor())
    c = cp.march(motor(erosion="const", edot_ref=0.006 * IN))
    p = cp.march(motor(erosion="pressure", edot_ref=0.006 * IN))
    assert np.all(base["Rt"] == base["Rt"][0])
    assert np.all(np.diff(c["Rt"]) > 0) and np.all(np.diff(p["Rt"]) > 0)
    # constant rate: radius grows by edot * elapsed time exactly
    assert c["Rt"][-1] == pytest.approx(c["Rt"][0] + 0.006 * IN * c["t"][-1], rel=0.01)
    assert p["Pc"].max() < base["Pc"].max() and c["eps"][-1] < c["eps"][0]               # a growing throat lowers p_c and eps
    assert cp.summary(motor(erosion="const", edot_ref=0.006 * IN), c)["Rt_end"] > motor().R_t


def test_ideal_vacuum_thrust_coefficient():
    # gamma = 1.2, eps = 12: C_F,vac = 1.7645 (Sutton Eq. 3-30 with p_a = 0)
    d = cp.march(motor(R_t=1.6 * IN / np.sqrt(12.0)))
    assert d["CFv"][0] == pytest.approx(1.76451, abs=5e-5)
    g = 1.2
    Me, pe = exit_state(12.0, g)
    mom = np.sqrt(2 * g**2 / (g - 1) * (2 / (g + 1)) ** ((g + 1) / (g - 1)) * (1 - pe ** ((g - 1) / g)))
    assert d["CFv"][0] == pytest.approx(mom + pe * 12.0, rel=2e-5)
    assert np.all(d["CFsl"] < d["CFv"])                                                    # ambient pressure costs thrust


def reference_us(r_o, r_i, L, rho, cstar_fts, n, rdot_ref, p_ref, R_t, R_e, dw, ends, edot_ref, mode, gamma=1.2):
    """Independent plain-loop version of the algorithm in US customary units (in, psi, lbm, ft/s)."""
    g = 32.174
    a = rdot_ref / p_ref**n
    web = min(r_o - r_i, L / 2) if ends else r_o - r_i
    N = int(np.ceil(web / dw)) + 1
    w = np.linspace(0, web, N)
    d = w[1] - w[0]
    Rt = R_t
    Pc, t, F, Ab = [], [0.0], [], []
    for i in range(N):
        e = 1.0 if ends else 0.0
        ab = 2 * np.pi * (r_i + w[i]) * (L - 2 * e * w[i]) + e * 2 * np.pi * (r_o**2 - (r_i + w[i]) ** 2)
        At = np.pi * Rt**2
        pc = (ab / At * rho * cstar_fts * a / g) ** (1 / (1 - n))
        rd = rdot_ref * (pc / p_ref) ** n
        dt = d / rd
        eps = np.pi * R_e**2 / At
        pe = exit_state(eps, gamma)[1]
        Pc.append(pc)
        Ab.append(ab)
        F.append(float(CF(gamma, pe, 0.0, eps)) * At * pc)
        t.append(t[-1] + dt)
        if mode == "const":
            Rt += dt * edot_ref
        elif mode == "pressure":
            Rt += dt * edot_ref * (pc / p_ref) ** 0.8
    return np.array(Pc), np.array(Ab), np.array(F), np.array(t[:-1])


@pytest.mark.parametrize("ends", [False, True])
@pytest.mark.parametrize("mode", ["none", "const", "pressure"])
def test_agrees_with_an_independent_us_unit_implementation(ends, mode):
    p = CPParamsUS(ends="burning" if ends else "inhibited", erosion=mode, dw=0.004)
    m = p.motor()
    d = cp.march(m)
    Pc, Ab, F, t = reference_us(p.r_o, p.r_i, p.L, p.rho_p, p.cstar, p.n, p.r_ref, p.p_ref, p.R_t, p.R_e, p.dw, ends,
                                p.edot_ref, mode)
    assert d["Pc"] / PSI == pytest.approx(Pc, rel=2e-4)                                    # g = 32.174 vs 9.80665/0.3048
    assert d["Ab"] / IN**2 == pytest.approx(Ab, rel=1e-9)
    assert d["F_vac"] / LBF == pytest.approx(F, rel=3e-4)
    assert d["t"] == pytest.approx(t, rel=3e-4)


def test_si_and_us_parameter_sets_describe_the_same_motor():
    a, b = CPParamsSI().motor(), CPParamsUS().motor()
    for k in ("r_o", "r_i", "L", "R_t", "R_e"):
        assert getattr(a, k) == pytest.approx(getattr(b, k), rel=5e-3), k                       # SI defaults are rounded to 0.1 mm
    assert a.rho_p == pytest.approx(b.rho_p, rel=1e-3) and a.cstar == pytest.approx(b.cstar, rel=1e-3)


def test_run_driver_sweep_and_cancel():
    r = CPMotorRun().run(CPParamsSI(sweep="r_i", span=20.0, erosion="pressure"), lambda f: None, threading.Event())
    assert r.meta["n_cases"] == 3 and r.meta["sweep_labels"][1] == "31.8"
    t_burn = [r.arrays[f"c{c}_m0_t"][-1] for c in range(3)]
    assert t_burn[0] > t_burn[1] > t_burn[2]                                              # a bigger bore leaves less web
    r0 = CPMotorRun().run(CPParamsSI(), lambda f: None, threading.Event())
    assert r0.meta["n_cases"] == 3 and r0.meta["modes"] == ["none", "const", "pressure"]
    assert r0.channels["w"][-1] == pytest.approx(r0.meta["web"], rel=1e-2) and len(r0.t) == 601
    ev = threading.Event()
    ev.set()
    with pytest.raises(InterruptedError):
        CPMotorRun().run(CPParamsSI(), lambda f: None, ev)
    with pytest.raises(ValueError, match="bore radius"):
        CPMotorRun().run(CPParamsSI(r_i=70.0, r_o=60.0), lambda f: None, threading.Event())
