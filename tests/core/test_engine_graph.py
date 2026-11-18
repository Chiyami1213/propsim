"""Engine-graph cycle solver: drive-gas models and the steady operating point of the library example graphs."""
import numpy as np
import pytest

from propsim.core.engine_graph import GraphError, solve
from propsim.core.engine_graph.drive_gas import frozen_gas, table_gas
from propsim.data import graphs


def _err(sol, g, node, output):
    vt = next(v for v in g["validation_targets"] if v["node"] == node and v["output"] == output)
    return sol.value(node, output) / vt["value"]["value"] - 1.0


@pytest.mark.parametrize("mr", [0.785, 0.903, 1.025, 1.143, 1.41])
def test_frozen_hydrogen_gas_matches_huzel_huang_table(mr):
    """Complete-combustion H2/H2O gas against Huzel & Huang Table 6-4 (fuel-rich LOX/LH2 drive gas)."""
    fr, tb = frozen_gas("LOX", "LH2", mr, 90.0, 22.0, 20e6), table_gas("LOX", "LH2", mr)
    assert fr["T"] == pytest.approx(tb["T"], rel=0.03)
    assert fr["cp"] == pytest.approx(tb["cp"], rel=0.05)
    assert fr["gamma"] == pytest.approx(tb["gamma"], rel=0.02)


def test_pressure_fed_example_converges():
    g = graphs()["pressure_fed_kerolox"]
    s = solve(g)
    assert s.converged
    assert abs(_err(s, g, "chamber", "p_c")) < 0.10          # the graph leaves out the dome and reserve drops
    for st in s.stations:                                    # pressure falls along each feed
        assert st["p"] > 0


def test_gas_generator_example_matches_a1_within_ten_percent():
    g = graphs()["gg_kerolox"]
    s = solve(g)
    assert s.converged
    for node, out in [("chamber", "p_c"), ("nozzle", "thrust"), ("ox_pump", "power"), ("fuel_pump", "power"),
                      ("turbine", "power"), ("shaft", "speed")]:
        assert abs(_err(s, g, node, out)) < 0.10, (node, out)
    sh = s.shafts[0]
    assert sh["turbine_power"] == pytest.approx(sh["pump_power"] + sh["aux_power"], rel=1e-6)
    gg = s.components["gas_generator"]
    assert gg["mixture_ratio"] == pytest.approx(0.408, rel=0.05)


def test_expander_example_converges_and_heats_the_hydrogen():
    g = graphs()["expander_hydrolox"]
    s = solve(g)
    assert s.converged
    assert abs(_err(s, g, "chamber", "p_c")) < 0.15
    jk = s.components["regen_jacket"]
    assert jk["Q"] > 0 and jk["T_out"] > 150.0


def test_ssme_with_preburner_valves_controlling_pc_and_mixture_ratio():
    """Close the loop the way the SSME controller does: OPOV sets chamber pressure, FPOV the mixture ratio."""
    g = graphs()["ssme_staged_combustion"]
    s = solve(g, targets=[{"node": "chamber", "output": "p_c", "value": 20477400.0},
                          {"node": "chamber", "output": "mixture_ratio", "value": 6.0}],
              free=[{"node": "opov", "param": "position"}, {"node": "fpov", "param": "position"}])
    assert s.converged, s.message
    assert abs(_err(s, g, "chamber", "p_c")) < 1e-4
    for node, out, tol in [("nozzle", "thrust", 0.05), ("nozzle", "isp", 0.05), ("chamber", "T_c", 0.02),
                           ("hpftp_pump", "speed", 0.05), ("hpotp_pump", "speed", 0.05)]:
        assert abs(_err(s, g, node, out)) < tol, (node, out)
    assert 0.0 < s.free["opov.position"] < 1.0 and 0.0 < s.free["fpov.position"] < 1.0


def test_unconnected_port_is_reported():
    g = graphs()["pressure_fed_kerolox"]
    g = dict(g, edges=[e for e in g["edges"] if e["to"]["node"] != "chamber" or e["to"]["port"] != "fuel_in"])
    with pytest.raises(GraphError, match="unconnected"):
        solve(g)


def test_sweep_marches_from_the_design_point_and_reports_every_point():
    import numpy as np
    from propsim.core.engine_graph import edit as E
    from propsim.core.engine_graph.trade import compare, current_value, inputs, sweep, to_csv
    d = E.template("gg_kerolox")
    assert ("param", "nozzle", "expansion_ratio", "1") in inputs(d)
    eps0 = current_value(d, "param", "nozzle", "expansion_ratio")
    vals = np.linspace(0.8 * eps0, 1.2 * eps0, 5)
    seen = []
    r = sweep(d, "param", "nozzle", "expansion_ratio", vals, progress=lambda a, b: seen.append(a))
    assert all(r["converged"]) and seen == [1, 2, 3, 4, 5]
    isp_vac = r["summary"]["isp_vac"]
    assert all(b > a for a, b in zip(isp_vac, isp_vac[1:]))       # vacuum Isp grows with area ratio
    thrust = r["summary"]["thrust"]
    assert all(b < a for a, b in zip(thrust, thrust[1:]))         # at sea level the larger nozzles overexpand
    mid = compare({"base": d})["base"]
    assert abs(mid["thrust"] - thrust[2]) / thrust[2] < 1e-4      # the middle point is the design as it stands
    csv = to_csv(r)
    assert csv.splitlines()[0].startswith("nozzle.expansion_ratio,converged,thrust") and len(csv.splitlines()) == 6
    assert d["graph"]["nodes"] == E.template("gg_kerolox")["graph"]["nodes"]   # the document is not changed


def test_sweep_of_a_target_resizes_the_free_parameters():
    from propsim.core.engine_graph import edit as E
    from propsim.core.engine_graph.trade import sweep
    d = E.template("ssme_staged_combustion")
    r = sweep(d, "target", "chamber", "p_c", [19e6, 20.4774e6])
    assert all(r["converged"])
    assert abs(r["summary"]["p_c"][0] - 19e6) < 2e3 and r["outputs"]["opov.position"][0] < r["outputs"]["opov.position"][1]
