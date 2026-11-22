"""Engine start-up transient: valve schedule, shaft spin-up, propellant depletion and throttle on the library cycles."""
import pytest

from propsim.core.engine_graph import edit as E
from propsim.core.engine_graph.transient import Sequence, ValveStep, default_sequence, run, wheel_inertia


def test_valve_step_schedule():
    v = ValveStep(t_open=0.2, ramp_open=0.4, t_close=1.0, ramp_close=0.5, position=0.8)
    assert v.at(0.1) == 0 and v.at(0.4) == pytest.approx(0.4) and v.at(0.7) == pytest.approx(0.8)
    assert v.at(1.25) == pytest.approx(0.4) and v.at(2.0) == 0
    s = Sequence({}, throttle=[(0.0, 1.0), (1.0, 0.5)])
    assert s.throttle_at(0.5) == pytest.approx(0.75)


def test_wheel_inertia_scales_with_diameter():
    assert wheel_inertia(0.2) == pytest.approx(wheel_inertia(0.1) * 2**5)


def test_pressure_fed_start_reaches_the_steady_chamber_pressure():
    d = E.template("pressure_fed_kerolox")
    r = run(d, default_sequence(d, 1.5))
    pc = r.series("chamber", "p_c")
    assert r.message.startswith("ran to") and len(r.t) > 10
    assert pc[0] < 0.2 * pc[-1] and all(b >= a - 1e-3 * pc[-1] for a, b in zip(pc, pc[1:]))
    from propsim.core.engine_graph.network import Network
    steady = Network(d["graph"]).solve().components["chamber"]["p_c"]
    assert pc[-1] == pytest.approx(steady, rel=0.02)
    assert r.tank_mass["ox_tank"][-1] < r.tank_mass["ox_tank"][0]


def test_throttle_lowers_chamber_pressure():
    d = E.template("pressure_fed_kerolox")
    seq = default_sequence(d, 1.6)
    seq.throttle = [(0.0, 1.0), (1.0, 0.15)]
    r = run(d, seq)
    pc = r.series("chamber", "p_c")
    assert pc[-1] < 0.8 * max(pc)


def test_small_propellant_load_runs_dry():
    d = E.template("pressure_fed_kerolox")
    seq = default_sequence(d, 3.0)
    seq.loads = {"fuel_tank": 0.05}
    r = run(d, seq)
    assert "ran dry" in r.message and r.tank_mass["fuel_tank"][-1] < 0.05


def test_fast_valve_closure_reports_the_water_hammer_surge():
    d = E.template("pressure_fed_kerolox")
    seq = default_sequence(d, 1.5)
    seq.valves["ox_valve"].t_close, seq.valves["ox_valve"].ramp_close = 1.0, 0.001
    r = run(d, seq)
    assert any(n.startswith("ox_valve closes") and "surge" in n for n in r.notes)
