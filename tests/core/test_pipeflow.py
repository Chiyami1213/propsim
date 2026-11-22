"""Pipe-flow losses against the check values worked from Rennels & Hudson, Pipe Flow (2012), and the sandbox lines."""
import math

import pytest

from propsim.core.engine_graph import edit as E
from propsim.core.engine_graph import pipeflow as PF
from propsim.core.engine_graph.network import Network


def test_churchill_spans_laminar_and_turbulent():
    assert PF.churchill(500) == pytest.approx(64 / 500, rel=1e-3)
    for Re, e in ((1e5, 0.0), (1e6, 1e-4), (1e7, 1e-3)):           # Colebrook-White, solved by iteration
        f = 0.02
        for _ in range(50):
            f = (-2 * math.log10(e / 3.7 + 2.51 / (Re * math.sqrt(f)))) ** -2
        assert PF.churchill(Re, e) == pytest.approx(f, rel=0.03)
    f = [PF.churchill(r) for r in (1800, 2300, 3000, 4000)]           # smooth through the critical zone
    assert all(b > 0 for b in f)


def test_fitting_coefficients_match_the_book_check_values():
    assert PF.contraction_sharp(0.0) == pytest.approx(0.570, abs=1e-3)      # flush sharp entrance
    assert PF.entrance_rounded(0.0) == pytest.approx(0.570, abs=1e-3)
    assert PF.entrance_rounded(1.0) == pytest.approx(0.030, abs=1e-3)
    assert PF.entrance_reentrant(0.0) == pytest.approx(1.12)
    assert PF.entrance_reentrant(0.05) == pytest.approx(0.570, abs=2e-3)
    assert PF.expansion_sudden(0.0) == 1.0 and PF.expansion_sudden(1.0) == 0.0
    assert PF.orifice_K(0.0, 0.0) == pytest.approx(2.814, abs=1e-3)          # sharp orifice in a wall
    assert PF.orifice_K(0.0, 1.4) == pytest.approx(1.570, abs=1e-3)          # reattached: contraction + expansion
    assert PF.cd_from_K(PF.orifice_K(0.0, 0.0)) == pytest.approx(0.596, abs=2e-3)
    assert PF.orifice_K(0.0, 1.4) == pytest.approx(PF.orifice_K(0.0, 1.4 + 1e-9), abs=1e-6)
    assert PF.miter(90) == pytest.approx(0.42 * math.sqrt(0.5) + 2.56 * 0.5 ** 1.5)
    assert 0.1 < PF.bend(90, 1.5, 0.018) < 0.4


def test_water_hammer_book_example_in_si():
    # rigid pipe, B = 300,000 psi water: 1439 m/s (Rennels & Hudson ch. 21 check value)
    B, rho = 300000 * 6894.76, 998.0
    a = PF.wave_speed(math.sqrt(B / rho), rho)
    assert a == pytest.approx(1439, rel=0.01)
    dp, t = PF.surge(rho, a, 3.0, 100.0)
    assert dp == pytest.approx(rho * a * 3.0) and t == pytest.approx(200 / a)
    assert PF.surge(rho, a, 3.0, 100.0, t_close=10 * t)[0] == pytest.approx(dp / 10)


def test_line_fittings_and_injector_edge_change_the_solution():
    d = E.template("pressure_fed_kerolox")
    base = Network(d["graph"]).solve()
    line = next(n for n in d["graph"]["nodes"] if n["id"] == "ox_line")
    line.setdefault("options", {}).update(entrance="sharp", bends="4", exit="none")
    sol = Network(d["graph"]).solve()
    c0, c1 = base.components["ox_line"], sol.components["ox_line"]
    assert c1["K_total"] > c0["K_total"] + 0.57 and sol.components["chamber"]["p_c"] < base.components["chamber"]["p_c"]
    assert c1["surge_dp"] > 0 and c1["round_trip_time"] > 0
    inj = next(n for n in d["graph"]["nodes"] if n["id"] == "ox_injector")
    inj.setdefault("options", {}).update(edge="thick", thickness_ratio="1.4")
    sol2 = Network(d["graph"]).solve()
    assert sol2.components["ox_injector"]["Cd"] == pytest.approx(1 / math.sqrt(1.570), abs=1e-3)


def test_isothermal_gas_line_matches_incompressible_at_small_drop():
    G, RT, K = 50.0, 300.0 * 287.0, 10.0
    p1 = 5e6
    rho = p1 / RT
    dp_inc = K * G * G / (2 * rho)
    p2 = p1 - dp_inc
    for _ in range(30):                                     # solve the isothermal residual for p2
        r = PF.isothermal_residual(p1, p2, G, RT, K)
        p2 += r
    assert abs(PF.isothermal_residual(p1, p2, G, RT, K)) < 1.0
    assert (p1 - p2) == pytest.approx(dp_inc, rel=0.05) and (p1 - p2) > dp_inc     # density falls along the line
