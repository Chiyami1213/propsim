import dataclasses
import threading
import time

import numpy as np
import pytest

from propsim.core import io as pio
from propsim.core.ballistics import hybrid0d as h0
from propsim.core.nozzle import (CF, contour, eps_from_pratio, mach_along_contour, mach_from_area,
                                 mach_from_area_array)
from propsim.core.regression import marxman_rdot, port_radius_analytic
from propsim.core.result import Result
from propsim.modules.nozzle_live import NozzleLiveModel, NozzleLiveParams


def test_ideal_cf_validation_case():
    eps, Me = eps_from_pratio(1 / 34, 1.2)
    assert eps == pytest.approx(5.305, abs=1e-3)
    assert Me == pytest.approx(2.828, abs=1e-3)
    assert CF(1.2, 1 / 34, 1 / 34, eps) == pytest.approx(1.4977, abs=1e-4)


@pytest.mark.parametrize("g", [1.15, 1.2, 1.4])
def test_vectorised_mach_matches_brentq(g):
    eps = np.array([1.05, 2.0, 6.0, 25.0, 80.0])
    vec = mach_from_area_array(eps, g, True)
    ref = np.array([mach_from_area(e, g, True) for e in eps])
    np.testing.assert_allclose(vec, ref, rtol=2e-4)
    sub_eps = eps[:3]
    vec = mach_from_area_array(sub_eps, g, False)
    ref = np.array([mach_from_area(e, g, False) for e in sub_eps])
    np.testing.assert_allclose(vec, ref, rtol=2e-4)


def test_contour_geometry():
    Rt, eps = 0.016, 6.0
    x, r = contour(Rt, eps, Rc=0.06)
    assert r.min() == pytest.approx(Rt, rel=1e-9)
    assert x[np.argmin(r)] == pytest.approx(0.0, abs=1e-12)
    assert r[-1] == pytest.approx(np.sqrt(eps) * Rt, rel=1e-9)
    assert r[0] == pytest.approx(0.06, rel=1e-9)
    assert len(contour(Rt, 30.0, Rc=0.06)[0]) == len(x)          # fixed topology -> in-place mesh updates
    M = mach_along_contour(x, r, Rt, 1.2)
    assert M[x < 0].max() <= 1.0 and M[-1] == pytest.approx(mach_from_area(eps, 1.2), rel=1e-3)


def test_marxman_validation_case():
    assert marxman_rdot(200, 0.5, 6, 8e-5, 920) * 1e3 == pytest.approx(0.713, abs=1e-3)


def _burn(**kw):
    p = h0.HybridBurnSI(mdot_ox=1.8, p_tank=5.5e6, R0=.025, L=.75, R_out=.06, rho_f=920, a=0.0541e-3, n=.647,
                        Dt=.032, eps=6, t_burn=10, eta_cstar=.95)
    return dataclasses.replace(p, **kw)


def test_hybrid_port_radius_matches_analytic():
    q = _burn(tau_open=1e-4, tau_close=1e-4, R_out=1.0)
    ch = h0.integrate(q)
    Ra = port_radius_analytic(10, .025, 0.0541e-3, .647, 1.8) * 1e3
    assert Ra == pytest.approx(50.846, abs=1e-3)
    assert ch["R"][-1] == pytest.approx(Ra, rel=1e-4)


def test_hybrid_mass_conservation_progress_cancel():
    p = _burn(R_out=1.0)
    seen = []
    ch = h0.integrate(p, n_frames=6001, progress=seen.append)       # fine frames: trapezoid error ~1e-6
    t = np.linspace(0, p.t_burn, 6001)
    m_fuel_geo = p.rho_f * np.pi * ((ch["R"][-1] * 1e-3) ** 2 - p.R0**2) * p.L
    m_fuel_int = np.trapezoid(ch["mdot_f"], t)
    assert m_fuel_int == pytest.approx(m_fuel_geo, rel=1e-5)
    assert seen[-1] == pytest.approx(1.0) and np.all(np.diff(seen) > 0)
    ev = threading.Event()
    ev.set()
    with pytest.raises(InterruptedError):
        h0.integrate(p, cancel=ev)


def test_live_eval_under_50ms():
    m, p = NozzleLiveModel(), NozzleLiveParams()
    m.evaluate(p)
    t0 = time.perf_counter()
    for _ in range(10):
        r = m.evaluate(p)
    assert (time.perf_counter() - t0) / 10 < 0.05
    assert r["cf"] > 1.0 and r["Isp"] > 150


def test_io_roundtrip(tmp_path):
    res = Result(np.linspace(0, 1, 5), {"a": np.arange(5.0)}, {"x": 1.5}, ["hello"])
    pio.save_result_h5(tmp_path / "r.psim.h5", res, "m.key", {"k": 1})
    r2, model, params = pio.load_result_h5(tmp_path / "r.psim.h5")
    assert model == "m.key" and params == {"k": 1} and r2.meta["x"] == 1.5 and r2.log == ["hello"]
    np.testing.assert_array_equal(r2.channels["a"], res.channels["a"])
    pio.save_case(tmp_path / "c.psim.json", "m.key", {"k": 2})
    assert pio.load_case(tmp_path / "c.psim.json") == ("m.key", {"k": 2})
    from scipy.io import loadmat
    pio.export_mat(tmp_path / "e.mat", {"t (s)": np.arange(3.0), "1x": np.ones(3)})
    m = loadmat(tmp_path / "e.mat")
    assert "t__s_" in m and "v_1x" in m
    pio.export_csv(tmp_path / "e.csv", {"a": np.arange(3.0), "b": np.arange(2.0)})
    assert (tmp_path / "e.csv").read_text().splitlines()[0] == "a,b"
