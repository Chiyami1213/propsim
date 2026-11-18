"""Hybrid fuel grains: custom ports with inhibitors, one fuel per grain, user chemistry tables."""
import json

import numpy as np
import pytest

from propsim.core.ballistics import hybrid1d as h1
from propsim.sims.hybrid_burn1d import HybridBurn1DParams


def _run(**kw):
    p = HybridBurn1DParams(t_burn=6.0, **kw)
    m, s = p.to_si(), p.supply()
    t = np.linspace(0.0, 6.0, 121)
    ch = h1.integrate(m, s, 6.0, 121, n_chunks=12)
    return m, ch, h1.summary(m, t, ch)


@pytest.fixture(scope="module")
def circle():
    return _run()


def test_custom_circle_reproduces_the_circular_port_solver(circle):
    m, ch, s = _run(kind="custom")
    assert m.grains and not circle[0].grains
    for k in ("I_t_kNs", "Pc_max_MPa", "m_fuel", "web_burned"):
        assert s[k] == pytest.approx(circle[2][k], rel=2e-3), k


def test_custom_port_and_inhibited_bore():
    wagon = {"R_out": 60, "shapes": [{"kind": "circle", "r": 25}, {"kind": "slot", "r1": 40, "w": 6, "repeat": 6}]}
    m, ch, s = _run(kind="custom", section_json=json.dumps(wagon))
    m0, _, s0 = _run(kind="custom")
    assert m.grains[0].table.P[0] > 2 * m0.grains[0].table.P[0]           # slots more than double the surface
    assert s["m_fuel"] > 1.3 * s0["m_fuel"]
    coated = {"R_out": 60, "shapes": [{"kind": "circle", "r": 25}, {"kind": "circle", "r": 6, "cx": 45,
                                                                      "inhibited": True}]}
    m1, _, s1 = _run(kind="custom", section_json=json.dumps(coated))
    assert m1.grains[0].table.P[0] == pytest.approx(m0.grains[0].table.P[0], rel=5e-3)   # coated: no burning surface
    assert m1.grains[0].table.A_p[0] > m0.grains[0].table.A_p[0]          # but it is open port area: lower flux,
    assert 0.94 * s0["m_fuel"] < s1["m_fuel"] < s0["m_fuel"]              # so slightly less fuel


def test_two_fuels_and_a_user_chemistry_table(circle):
    fast = {"rho_f": 900.0, "a": 0.1171, "n": 0.62}
    _, ch, s = _run(stack_json=json.dumps([{"L": 0.375}, {"L": 0.375, "prop": fast}]))
    assert s["m_fuel"] > 1.2 * circle[2]["m_fuel"]                        # the faster-regressing grain adds fuel
    of = [1.0, 2.0, 3.0]
    chem = {"of": of, "cstar": [1400.0, 1500.0, 1450.0], "Tc": [2800.0, 3200.0, 3100.0], "gamma": [1.22, 1.2, 1.2]}
    _, ch2, s2 = _run(stack_json=json.dumps([{"L": 0.75, "prop": {"chem": chem}}]))
    ok = np.isfinite(ch2["OF"])
    np.testing.assert_allclose(ch2["cstar"][ok], 0.95 * np.interp(ch2["OF"][ok], of, chem["cstar"]), rtol=1e-9)
    mixed = _run(stack_json=json.dumps([{"L": 0.375}, {"L": 0.375, "prop": {"chem": chem}}]))[1]
    lo = np.minimum(ch2["cstar"], circle[1]["cstar"])
    assert np.nanmin(mixed["cstar"][10:] - 0.98 * lo[10:]) > 0               # mixed c* lies near or between the two


@pytest.mark.parametrize("bad", ['[{"L": 0.3, "prop": {"law": "magic"}}]', '[{"L": 0.3, "prop": {"pair": "XX"}}]',
                                 '[{"L": 0.3, "prop": {"chem": {"of": [2, 1], "cstar": [1, 1], "Tc": [1, 1], '
                                 '"gamma": [1.2, 1.2]}}}]', '[{"L": 0.3, "prop": {"rho_p": 1}}]'])
def test_bad_fuel_overrides_are_rejected(bad):
    with pytest.raises(ValueError):
        HybridBurn1DParams(stack_json=bad).to_si()
