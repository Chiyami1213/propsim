"""Hybrid 0-D burn driven by CEA: lookup accuracy/speed, chamber mass balance, plausibility of the demo case."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from propsim.core.thermo.cea import PAIRS, Thermo, chamber_lut
from propsim.sims.hybrid_burn import HybridBurn0D, HybridBurnParams

pytest.importorskip("rocketcea")


def test_chamber_lut_matches_direct_cea():
    th = Thermo("GOX/HTPB", "cea")
    lut = chamber_lut("GOX/HTPB")
    rng = np.random.default_rng(3)
    for _ in range(40):
        mr = rng.uniform(1.4, 5.9)
        pc = float(np.exp(rng.uniform(np.log(0.6e6), np.log(19e6))))
        cs, tc, g, mw = lut(mr, pc)
        ref = th.chamber(mr, pc)
        assert cs == pytest.approx(float(ref.cstar), rel=3e-3)
        assert tc == pytest.approx(float(ref.Tc), rel=3e-3)
        assert g == pytest.approx(float(ref.gamma), rel=3e-3)
        assert mw == pytest.approx(float(ref.Mw), rel=3e-3)


def test_chamber_lut_is_fast_and_clamps():
    lut = chamber_lut("GOX/HTPB")
    t = time.perf_counter()
    for _ in range(2000):
        lut(3.0, 3e6)
    assert (time.perf_counter() - t) / 2000 < 1e-4          # < 100 µs per RHS call
    # below / above the table: clamped to the edge value, finite
    lo, hi = lut(3.0, 1e3), lut(3.0, 1e9)
    assert lo == lut(3.0, np.exp(chamber_lut("GOX/HTPB").lnp[0])) and np.all(np.isfinite(hi))


@pytest.fixture(scope="module")
def burn():
    p = HybridBurnParams()
    return p, HybridBurn0D().run(p, lambda f: None, threading.Event())


def test_quasi_steady_chamber_mass_balance(burn):
    """At quasi-steady state P_c A_t = eta c*(O/F, P_c) (mdot_ox + mdot_f) with c* from CEA."""
    p, res = burn
    si = p.to_si()
    ch, t = res.channels, res.t
    i = int(np.argmin(np.abs(t - 5.0)))
    cs_cea = chamber_lut("GOX/HTPB")(float(ch["OF"][i]), ch["Pc"][i] * 1e6)[0]
    mdot = ch["mdot_ox"][i] + ch["mdot_f"][i]
    assert ch["Pc"][i] * 1e6 * si.At == pytest.approx(si.eta_cstar * cs_cea * mdot, rel=2e-3)
    assert ch["cstar"][i] == pytest.approx(si.eta_cstar * cs_cea, rel=1e-9)


def test_demo_case_is_physical(burn):
    p, res = burn
    ch = res.channels
    assert 1.05 < np.nanmin(ch["gamma"]) and np.nanmax(ch["gamma"]) < 1.3
    assert 2000.0 < np.nanmin(ch["Tc"][ch["mdot_ox"] > 0.1]) and np.nanmax(ch["Tc"]) < 4000.0
    assert 1.3 <= np.nanmin(res.channels["OF"]) and np.nanmax(res.channels["OF"]) <= PAIRS["GOX/HTPB"].mr_hi
    assert 150.0 < res.meta["Isp_avg"] < 330.0
    assert res.meta["gamma_mean"] == pytest.approx(np.nanmean(ch["gamma"]))
