"""Electric propulsion: MathKit reference values, Edelbaum spiral, ion optics trend, Hall discharge mass balance."""
import threading

import numpy as np
import pytest

from propsim.core import ep
from propsim.core.ep import hall as hl
from propsim.core.ep import mission as mi
from propsim.core.ep.optics import Optics, OpticsGeom, perveance_max

XE = ep.PROPELLANTS["Xe"] * ep.AMU


def test_child_langmuir_mathkit():
    assert ep.child_langmuir_j(1000.0, 1e-3, XE) == pytest.approx(150.9, abs=0.05)


def test_ion_thrust_mathkit():
    assert ep.ion_thrust(1.76, 1100.0, XE) * 1e3 == pytest.approx(96.3, abs=0.05)


def test_edelbaum_mathkit():
    v0, v1 = ep.circular_v(ep.R_EARTH + 400e3), ep.circular_v(42164.137e3)
    assert ep.edelbaum_dv(v0, v1, np.radians(28.5)) / 1e3 == pytest.approx(5.898, abs=1e-3)


def test_stuhlinger_mathkit():
    assert ep.stuhlinger_payload(5e3, 20e3, 20e3) == pytest.approx(0.5576, abs=1e-4)
    ve, pl = ep.optimum_ve(5e3, 20e3)
    assert pl >= ep.stuhlinger_payload(5e3, 20e3, 20e3) and pl >= ep.stuhlinger_payload(5e3, 0.95 * ve, 20e3)


def test_spiral_reaches_target_and_matches_rocket_equation():
    s = mi.Spiral(r0=ep.R_EARTH + 400e3, r1=42164.137e3, di=np.radians(28.5), m0=2000.0, P0=10e3, eta=0.6, isp=1800.0,
                  degr=0.0)
    o = mi.simulate(s)
    assert o["reached"] and o["dv_tot"] / 1e3 == pytest.approx(5.898, abs=1e-3)
    assert o["v"][-1] == pytest.approx(ep.circular_v(42164.137e3), rel=1e-6)
    assert np.degrees(o["inc"][-1]) == pytest.approx(28.5, abs=1e-6)
    assert o["m_prop"] == pytest.approx(2000.0 * (1 - np.exp(-o["dv_tot"] / (9.80665 * 1800.0))), rel=1e-6)
    slow = mi.simulate(mi.Spiral(**{**s.__dict__, "degr": 0.05}))
    assert slow["t_end"] > o["t_end"] and slow["m_prop"] == pytest.approx(o["m_prop"], rel=1e-6)


def test_ion_optics_focusing_trend():
    g = OpticsGeom(n=32)
    o = Optics(g)
    I_max = perveance_max(g) * (g.V_s - g.V_a) ** 1.5
    lo = o.run(0.2 * I_max, iters=6)
    mid = o.run(0.6 * I_max, iters=6)
    assert lo["converged"] and mid["converged"]
    assert np.degrees(mid["theta_med"]) < np.degrees(lo["theta_med"])          # under-perveance over-focuses
    assert mid["impinge"] < 0.15


def test_hall_discharge_mass_balance_and_fife():
    p = hl.HallParams()
    out = hl.simulate(p, 1.0e-3, 201)
    err = (out["mass_in"] - out["mass_out"] - out["mass_stored"]) / out["mass_in"]
    assert abs(err) < 5e-3
    assert out["Ib"][-1] == pytest.approx(out["Id"][-1], rel=0.02)             # steady: production = outflow
    assert 5e3 < hl.fife_frequency(p) < 40e3


@pytest.mark.parametrize("cls_name", ["IonOptics", "HallDischarge", "LowThrustSpiral"])
def test_run_drivers_cancel(cls_name):
    from propsim.sims import ep_run as R
    sim = getattr(R, cls_name)()
    params = {"IonOptics": R.IonOpticsParams, "HallDischarge": R.HallRunParams, "LowThrustSpiral": R.SpiralParams}[cls_name]
    ev = threading.Event()
    ev.set()
    if cls_name == "LowThrustSpiral":                     # closed-form fast path: completes regardless
        assert sim.run(params(), lambda f: None, ev).meta["reached"]
        return
    with pytest.raises(InterruptedError):
        sim.run(params(sweep="on") if cls_name == "IonOptics" else params(), lambda f: None, ev)
