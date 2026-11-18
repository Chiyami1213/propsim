"""User-built grain sections, inhibitors, grain stacks with per-grain propellant, and the single-grain regression."""
import json

import numpy as np
import pytest

from propsim.core.ballistics import solid as sl
from propsim.core.grain import builder as gb
from propsim.core.grain import stack as gs
from propsim.sims.solid_burn import SolidBallisticsParams


def _sec(shapes, outer=False, R=60.0):
    return gb.build_section(gb.spec_from_dict({"R_out": R, "outer_burn": outer, "shapes": shapes}))


def test_circle_bore_matches_analytic():
    b = _sec([{"kind": "circle", "r": 14}])
    assert b.web == pytest.approx(46.0, abs=0.3)
    k = 15
    assert b.perim[k] == pytest.approx(2 * np.pi * (14 + b.d[k]), rel=2e-3)


def test_outer_burning_tube_is_neutral():
    b = _sec([{"kind": "circle", "r": 14}], outer=True)
    assert b.web == pytest.approx(23.0, abs=0.3)
    assert b.perim[10] == pytest.approx(2 * np.pi * 74, rel=2e-3)      # inner + outer perimeter stays 2 pi (R_in + R_out)
    assert b.perim[0] == pytest.approx(2 * np.pi * 74, rel=3e-3)


def test_inhibited_shape_does_not_burn_or_regress():
    b = _sec([{"kind": "circle", "r": 10}, {"kind": "circle", "r": 8, "cx": 35, "inhibited": True}])
    assert b.perim[0] == pytest.approx(2 * np.pi * 10, rel=2e-3)      # the coated bore adds no burning perimeter
    assert b.area[0] == pytest.approx(np.pi * (10**2 + 8**2), rel=2e-3)
    plain = _sec([{"kind": "circle", "r": 10}])
    k = 3
    assert b.area[k] - plain.area[k] == pytest.approx(np.pi * 64, rel=0.02)    # static void adds its own area only


def test_polar_repeat_and_json_roundtrip():
    spec = gb.spec_from_dict({"R_out": 60, "shapes": [{"kind": "circle", "r": 14},
                                                      {"kind": "slot", "r1": 30, "w": 8, "repeat": 6}]})
    assert gb.spec_from_json(spec.to_json()) == spec
    b = gb.build_section(spec)
    one = _sec([{"kind": "circle", "r": 14}, {"kind": "slot", "r1": 30, "w": 8}])
    assert b.perim[0] > one.perim[0] + 4 * 2 * 14                       # five more slots add two flanks each


@pytest.mark.parametrize("bad", [
    {"R_out": 60, "shapes": []}, {"R_out": 60, "shapes": [{"kind": "blob"}]},
    {"R_out": 60, "shapes": [{"kind": "circle"}]}, {"R_out": 60, "shapes": [{"kind": "circle", "r": 5, "zz": 1}]},
    {"R_out": 60, "shapes": [{"kind": "polygon", "pts": [[0, 0], [1, 1]]}]},
    {"R_out": 60, "shapes": [{"kind": "slot", "r1": 10, "r0": 20, "w": 4}]},
    {"R_out": 60, "shapes": [{"kind": "circle", "r": 5, "repeat": 100}]}, {"shapes": []}])
def test_invalid_sections_are_rejected(bad):
    with pytest.raises(ValueError):
        gb.spec_from_dict(bad)


def test_nothing_burns_is_an_error():
    with pytest.raises(ValueError):
        _sec([{"kind": "circle", "r": 10, "inhibited": True}])


def test_stack_parser():
    assert gs.parse_stack("") == []
    d = gs.parse_stack(json.dumps([{"L": 0.2, "burn_head": True, "prop": {"n": 0.4}}]))
    assert d[0].burn_head and dict(d[0].prop) == {"n": 0.4}
    for bad in ["[", "[]", '[{"x": 1}]', '[{"L": 9}]', '[{"L": 0.2, "prop": {"zz": 1}}]']:
        with pytest.raises(ValueError):
            gs.parse_stack(bad)


BASE = dict(kind="circle", R_p=20.0, R_out=60.0, L_g=0.6, D_t=36.0, N=9, t_end=14.0, alpha=0.0)


def _run(stack="", **kw):
    p = SolidBallisticsParams(stack_json=stack, **{**BASE, **kw})
    m = p.to_si()
    t = np.linspace(0.0, p.t_end, 281)
    ch = sl.integrate(m, p.t_end, 281, n_chunks=14)
    return m, ch, t, sl.summary(m, t, ch)


def _ideal(m):
    return sum(g.prop.rho_p * (g.table.A_case - g.table.A_p[0]) * g.L for g in m.stack) + m.m_ig


def test_single_grain_reproduces_the_previous_solver():
    """Reference values from the solver before grain stacks were added (default propellant, wagon section)."""
    p = SolidBallisticsParams(N=8, t_end=6.0)
    tab = p.section_table()
    m = sl.SolidMotor(prop=sl.SolidProp(), table=sl.SectionTable(tab.d, tab.A_p, tab.P), L=0.6, At=1.0e-3, eps=8, N=8)
    ch = sl.integrate(m, 6.0, 201, n_chunks=10)
    assert ch["Pc"].max() == pytest.approx(6.654791, rel=1e-4)
    assert np.trapezoid(ch["F"], np.linspace(0, 6, 201)) == pytest.approx(22.67681, rel=1e-4)


def test_bates_stack_burns_the_ends_and_conserves_mass():
    single = _run()
    m, ch, _, s = _run(json.dumps([{"L": 0.2, "burn_head": True, "burn_aft": True}] * 3))
    assert m.nc == 9 and len(m.stack) == 3
    assert ch["Ab"][2] > 1.5 * single[1]["Ab"][2]                       # six end faces add burning area
    assert s["m_prop"] == pytest.approx(_ideal(m), rel=0.04)
    assert ch["L_eff_0"][-1] < ch["L_eff_0"][0]                         # the grain shortened from its ends
    assert s["Pc_max_MPa"] < single[3]["Pc_max_MPa"]                    # lower Kn swing than the single long grain


def test_inhibited_ends_equal_a_single_grain_of_the_same_total_length():
    a = _run()[3]
    b = _run(json.dumps([{"L": 0.3}, {"L": 0.3}]))[3]
    assert b["m_prop"] == pytest.approx(a["m_prop"], rel=0.02)
    assert b["Pc_max_MPa"] == pytest.approx(a["Pc_max_MPa"], rel=0.03)


def test_different_propellant_per_grain():
    fast = {"a_mm": 9.0, "cstar": 1400.0, "gamma": 1.25, "rho_p": 1650.0}
    m, ch, _, s = _run(json.dumps([{"L": 0.3}, {"L": 0.3, "prop": fast}]))
    assert s["mixed"] and np.isnan(s["Pc_eq_MPa"])
    assert s["m_prop"] == pytest.approx(_ideal(m), rel=0.04)
    assert 1400.0 < ch["cstar_eff"].min() and ch["cstar_eff"].max() <= 1500.0 + 1e-6   # between the two propellants
    base_pmax = _run()[3]["Pc_max_MPa"]
    assert s["Pc_max_MPa"] > base_pmax                                  # the faster grain raises the pressure


def test_custom_kind_runs_through_the_form():
    spec = {"R_out": 60, "shapes": [{"kind": "circle", "r": 14}, {"kind": "slot", "r1": 30, "w": 8, "repeat": 4}]}
    p = SolidBallisticsParams(kind="custom", section_json=json.dumps(spec), N=4, t_end=3.0)
    m = p.to_si()
    assert m.table.A_p[0] > np.pi * 0.014**2 and m.table.A_case == pytest.approx(np.pi * 0.06**2, rel=0.01)
    ch = sl.integrate(m, 3.0, 61, n_chunks=6)
    assert ch["Pc"].max() > 1.0
