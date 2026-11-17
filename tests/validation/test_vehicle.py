"""Vehicle core: U.S. Standard Atmosphere 1976 values, MathKit Section 6 equations, conservation, drag, staging, orbit."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from propsim.core.constants import G0
from propsim.core.trajectory import flight as fl
from propsim.core.trajectory import profiles
from propsim.core.trajectory.atmosphere import R0, atmosphere

NOBURN = fl.StageSpec(0.0, 0.0, 0.0, 300.0, 0.0, 1.0)            # a vehicle that only coasts


def coast_vehicle(m=100.0):
    return fl.Vehicle((NOBURN,), m)


# --------------------------------------------------------------------------------------------- atmosphere
@pytest.mark.parametrize("H, T, p, rho", [(0, 288.15, 101325.0, 1.225), (11, 216.65, 22632.06, 0.36391),
                                           (20, 216.65, 5474.889, 0.088035), (32, 228.65, 868.0187, 0.013225),
                                           (47, 270.65, 110.9063, 0.0014275), (51, 270.65, 66.93887, 0.00086160),
                                           (71, 214.65, 3.956420, 6.4210e-5)])
def test_standard_atmosphere_layer_values(H, T, p, rho):
    Z = R0 * H * 1e3 / (R0 - H * 1e3)                            # geopotential -> geometric
    a = atmosphere(Z)
    assert a["T"][0] == pytest.approx(T, abs=1e-3)
    assert a["p"][0] == pytest.approx(p, rel=2e-6)
    assert a["rho"][0] == pytest.approx(rho, rel=2e-4)


def test_atmosphere_high_altitude_and_continuity():
    assert atmosphere(100e3)["rho"][0] == pytest.approx(5.604e-7, rel=2e-3)        # 1976 table
    assert atmosphere(86e3)["rho"][0] == pytest.approx(6.958e-6, rel=1e-3)
    h = np.linspace(0, 1.0e6, 5000)                              # held constant above 1000 km
    rho = atmosphere(h)["rho"]
    assert np.all(np.diff(np.log(rho)) < 0)                                          # strictly decreasing
    a, b = atmosphere(85999.0), atmosphere(86001.0)
    assert b["rho"][0] == pytest.approx(a["rho"][0], rel=1e-3) and b["T"][0] == pytest.approx(a["T"][0], abs=0.2)
    assert atmosphere(R0 * 11e3 / (R0 - 11e3))["a"][0] == pytest.approx(295.07, abs=0.05)      # speed of sound at H = 11 km


# --------------------------------------------------------------------------------------------- equations
def test_polar_integration_matches_mathkit_equations():
    """Unpowered, no drag: the polar free-flight integrator and the (V, gamma, h, x) form of MathKit agree."""
    V0, g0_deg = 1500.0, 60.0
    f = fl.simulate(coast_vehicle(), fl.Guidance(), fl.Aero(scale=0.0), fl.Wind(), t_max=600.0,
                    initial=dict(h=0.0, V=V0, gamma_deg=g0_deg), atmosphere_on=False)
    ev = lambda t, y, *a: y[2]
    ev.terminal, ev.direction = True, -1
    s = solve_ivp(fl.mathkit_rhs, (0.0, 600.0), [V0, np.radians(g0_deg), 0.0, 0.0, 100.0], events=ev, rtol=1e-11,
                  atol=1e-9, args=(0.0, 300.0, 0.0, 0.0, 1.0), dense_output=True)
    t_ap = np.linspace(0, s.t[-1], 20001)
    h_max = s.sol(t_ap)[2].max()
    assert f.meta["apogee_km"] * 1e3 == pytest.approx(h_max, rel=1e-5)
    assert f.meta["range_km"] * 1e3 == pytest.approx(s.y[3, -1], rel=1e-5)
    assert f.meta["t_end"] == pytest.approx(s.t[-1], rel=1e-6)


def test_circular_orbit_conserves_energy_and_angular_momentum():
    h0 = 300e3
    V = float(np.sqrt(fl.MU / (fl.R_E + h0)))
    f = fl.simulate(coast_vehicle(), fl.Guidance(), fl.Aero(), fl.Wind(), t_max=2 * np.pi * (fl.R_E + h0) / V,
                    initial=dict(h=h0, V=V, gamma_deg=0.0), atmosphere_on=False)
    r = fl.R_E + f.ch["h"] * 1e3
    E = 0.5 * f.ch["V"] ** 2 - fl.MU / r
    Lz = r * f.ch["V"] * np.cos(np.radians(f.ch["gamma"]))
    assert np.ptp(E) / abs(E[0]) < 1e-9 and np.ptp(Lz) / Lz[0] < 1e-9
    assert f.ch["x"][-1] * 1e3 / fl.R_E == pytest.approx(2 * np.pi, rel=1e-6)       # one revolution per period
    assert f.meta["perigee_km"] == pytest.approx(300.0, abs=1e-3)


def test_rocket_equation_in_vacuum():
    veh = fl.Vehicle((fl.StageSpec(100.0, 400.0, 60e3, 300.0, 0.0, 1.0),), 0.0)
    f = fl.simulate(veh, fl.Guidance(), fl.Aero(), fl.Wind(), t_max=19.0, initial=dict(h=0.0, V=0.0, gamma_deg=90.0),
                    atmosphere_on=False)
    tb = 400.0 / (60e3 / (G0 * 300.0))
    f = fl.simulate(veh, fl.Guidance(), fl.Aero(), fl.Wind(), t_max=tb - 1e-6, initial=dict(h=0.0, V=0.0, gamma_deg=90.0),
                    atmosphere_on=False)
    v_ideal = 300.0 * G0 * np.log(500.0 / 100.0) - G0 * tb                            # Tsiolkovsky minus constant-g loss
    assert f.ch["V"][-1] == pytest.approx(v_ideal, rel=0.01)
    assert f.ch["m"][-1] == pytest.approx(100.0, abs=0.05)


def test_terminal_velocity_with_drag():
    aero = fl.Aero(mach=(0.0, 30.0), cd=(1.0, 1.0))
    body = fl.Vehicle((fl.StageSpec(0.0, 0.0, 0.0, 300.0, 0.0, 0.5),), 10.0)
    f = fl.simulate(body, fl.Guidance(), aero, fl.Wind(), t_max=300.0, n_out=3001,
                    initial=dict(h=3000.0, V=0.0, gamma_deg=-90.0 + 1e-9))
    i = int(np.argmin(np.abs(f.ch["h"] - 1.5)))                                       # 1500 m
    rho = atmosphere(1500.0)["rho"][0]
    v_term = np.sqrt(2 * 10.0 * fl.MU / (fl.R_E + 1500.0) ** 2 / (rho * 1.0 * np.pi * 0.25 ** 2))
    assert f.ch["V"][i] == pytest.approx(v_term, rel=0.03)
    assert f.meta["impact"]


def test_staging_masses_and_events():
    st = (fl.StageSpec(1100.0, 10000.0, 260e3, 300.0, 0.2, 1.4, 5.0), fl.StageSpec(400.0, 3200.0, 45e3, 345.0, 0.05, 1.0))
    veh = fl.Vehicle(st, 100.0, 50.0, 110e3)
    g = fl.Guidance(rail=20.0, h_kick=250.0, kick=np.radians(2.5), mode="level", h_level=70e3, perigee_cut=200e3)
    f = fl.simulate(veh, g, fl.Aero(), fl.Wind(), t_max=700.0)
    ev = dict(f.events)
    names = [n for n, _ in f.events]
    assert names[:2] == ["Liftoff", "Rail exit"] and "Max-Q" in ev and "Mach 1" in ev
    assert ev["Stage 1 burnout"] < ev["Stage 2 ignition"] < ev["Fairing jettison"] < ev["Orbit-insertion cutoff"]
    assert ev["Stage 2 ignition"] - ev["Stage 1 burnout"] == pytest.approx(5.0, abs=1e-6)       # coast time
    m0 = fl.stack_mass(veh, 0)
    assert f.ch["m"][0] == pytest.approx(m0, rel=1e-9)
    j = int(np.searchsorted(f.t, ev["Stage 1 burnout"]))
    assert f.ch["m"][j] == pytest.approx(fl.stack_mass(veh, 1), rel=2e-3)              # stage 1 gone, fairing still on
    assert f.ch["m"][-1] == pytest.approx(f.meta["m_burnout"], rel=1e-9) or f.ch["m"][-1] <= f.ch["m"][j]
    assert sum(fl.ideal_dv(veh)) == pytest.approx(f.meta["dv_ideal"])


def test_demo_vehicle_reaches_circular_orbit():
    st = (fl.StageSpec(1100.0, 10000.0, 260e3, 300.0, 0.2, 1.4, 5.0), fl.StageSpec(400.0, 3200.0, 45e3, 345.0, 0.05, 1.0))
    veh = fl.Vehicle(st, 100.0, 50.0, 110e3)
    g = fl.Guidance(rail=20.0, h_kick=250.0, kick=np.radians(2.5), mode="level", h_level=70e3, perigee_cut=200e3)
    f = fl.simulate(veh, g, fl.Aero(), fl.Wind(), t_max=1500.0)
    m = f.meta
    assert m["orbit"] and not m["impact"]
    assert m["perigee_km"] == pytest.approx(200.0, abs=0.5) and 200.0 <= m["apogee_orbit_km"] < 230.0
    assert m["v_burnout"] == pytest.approx(np.sqrt(fl.MU / (fl.R_E + 200e3)), rel=0.01)
    assert 30e3 < m["maxq_kpa"] * 1e3 < 60e3 and 5.0 < m["max_g"] < 12.0
    assert f.ch["perigee"][-1] == pytest.approx(f.ch["perigee"][-2], abs=0.05)       # coasting on a fixed orbit


def test_pad_and_wind_effects():
    heavy = fl.Vehicle((fl.StageSpec(1000.0, 9000.0, 80e3, 300.0, 0.0, 1.0),), 0.0)
    with pytest.raises(ValueError, match="thrust-to-weight"):
        fl.simulate(heavy, fl.Guidance(), fl.Aero(), fl.Wind())
    snd = fl.Vehicle((fl.StageSpec(30.0, 70.0, 6e3, 220.0, 0.0, 0.2),), 5.0)
    g = fl.Guidance(elevation=np.radians(85.0), rail=5.0)
    calm = fl.simulate(snd, g, fl.Aero(), fl.Wind())
    tail = fl.simulate(snd, g, fl.Aero(), fl.Wind(w_surf=15.0, w_jet=30.0))
    assert calm.meta["impact"] and tail.meta["range_km"] != pytest.approx(calm.meta["range_km"], rel=1e-3)
    assert calm.meta["apogee_km"] > 1.0


def test_thrust_profile_exchange(tmp_path):
    from propsim.core.result import Result
    t = np.linspace(0.0, 4.0, 401)
    F = np.where((t > 0.5) & (t < 3.0), 5.0 * np.sin(np.pi * (t - 0.5) / 2.5) + 0.5, 0.0)
    res = Result(t, {"F": F, "mdot_ox": 0.4 * np.ones_like(t), "mdot_f": 0.1 * np.ones_like(t)})
    prof = profiles.from_result("hybrid.burn0d", res)
    assert prof["t"][0] == 0.0 and max(prof["F"]) == pytest.approx(5.5e3)             # kN -> N, trimmed to the burn
    I, m = profiles.impulse(prof)
    assert m == pytest.approx(0.5 * (prof["t"][-1]), rel=1e-6)
    p = profiles.save(prof, tmp_path / "stage1.json")
    assert profiles.load(p)["F"] == prof["F"] and profiles.find(tmp_path, 1) == p
    with pytest.raises(ValueError):
        profiles.from_result("ep.ion", res)
    # a stage flying that profile burns for its duration and uses its propellant flow
    stage = fl.StageSpec(10.0, 5.0, 0.0, 250.0, 0.0, 0.2, profile=profiles.arrays(prof))
    veh = fl.Vehicle((stage,), 2.0)
    f = fl.simulate(veh, fl.Guidance(elevation=np.radians(88.0), rail=3.0), fl.Aero(), fl.Wind(), t_max=120.0)
    assert dict(f.events)["Stage 1 burnout"] == pytest.approx(prof["t"][-1], abs=0.02)
