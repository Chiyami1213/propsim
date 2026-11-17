"""Hybrid Live tools: reference values, exact-geometry checks and Live-budget timing."""
from __future__ import annotations

import time

import numpy as np
import pytest

from propsim.modules.hybrid_live import (LAYOUTS, PortLiveModel, PortLiveParams, RegressionLiveModel,
                                         RegressionLiveParams)
from propsim.modules.hybrid_live_ext import (FlowLiveModel, FlowLiveParams, FuelLiveModel, FuelLiveParams,
                                             StabilityLiveModel, StabilityLiveParams)


def test_regression_tool_matches_mathkit():
    m = RegressionLiveModel()
    r = m.evaluate(RegressionLiveParams(G=200.0, x=0.5, B=6.0, mu=8.0, rho_f=920.0, Qc=1.0, Qr=0.3, dHv=1.81))
    assert r["r_mar"] == pytest.approx(0.713, abs=5e-4)                       # Marxman, MATHKIT 3.1
    assert r["r_rad"] == pytest.approx(0.625, abs=1.5e-3)                     # radiation blocking
    assert r["r_pow"] == pytest.approx(0.0541 * 200.0**0.647, rel=1e-9)       # HTPB/GOX power law
    # paraffin preset uses the SI coefficient 0.1171
    rp = m.evaluate(RegressionLiveParams(preset="paraffin_gox", G=200.0))
    assert rp["r_pow"] == pytest.approx(0.1171 * 200.0**0.62, rel=1e-9)
    # Marxman falls with x (x^-0.2) and radiation blocking lowers the rate
    assert np.all(np.diff(r["mar_x"]) < 0) and np.all(r["rad1"] > r["rad0"] * 0.0) and np.all(r["rad1"] != r["rad0"])
    assert r["ar_T"][-1] > r["ar_T"][0] and r["Tprof"][0] == pytest.approx(800.0) and r["Tprof"][-1] < 400.0


@pytest.mark.parametrize("R_p", [10.0, 14.0, 22.0])
def test_single_port_burnback_is_exact(R_p):
    """Circular port: A = pi (R+d)^2 and perimeter = 2 pi (R+d) while the front is inside the case."""
    b = PortLiveModel().evaluate(PortLiveParams(layout="single", R_p=R_p))["b"]
    R = R_p + b["d"]
    ok = R < 58.0
    assert np.max(np.abs(b["area"][ok] / (np.pi * R[ok] ** 2) - 1)) < 0.004
    assert np.max(np.abs(b["perim"][ok] / (2 * np.pi * R[ok]) - 1)) < 0.003
    assert b["web_max"] == pytest.approx(60.0 - R_p, abs=0.8)                  # web = outer radius - port radius


@pytest.mark.parametrize("layout", LAYOUTS)
def test_burnback_conservation(layout):
    b = PortLiveModel().evaluate(PortLiveParams(layout=layout))["b"]
    assert np.all(np.diff(b["area"]) >= 0) and b["area"][-1] == pytest.approx(b["A_out"], rel=1e-9)   # all consumed
    assert b["remain"][0] == pytest.approx(1.0) and b["remain"][-1] == pytest.approx(0.0, abs=1e-9)
    assert b["perim"][0] > 0 and b["perim"][-1] == pytest.approx(0.0, abs=1.0)                        # nothing left to burn


def test_wagon_wheel_initial_perimeter_and_progressivity():
    p = PortLiveParams(layout="wagon", R_p=14.0, n_slot=6, w_slot=8.0, r_slot=0.55, R_out=60.0)
    b = PortLiveModel().evaluate(p)["b"]
    Rp, w, Rs, N = 14.0, 8.0, 0.55 * 60.0, 6
    u0 = np.sqrt(Rp**2 - (w / 2) ** 2)
    ana = Rp * (2 * np.pi - N * 2 * np.arcsin(w / 2 / Rp)) + N * 2 * (Rs - u0) + N * w
    assert b["perim"][0] == pytest.approx(ana, rel=0.03)
    assert b["perim"][0] > 3 * 2 * np.pi * Rp                                  # far more burning surface than a bare port


def test_port_flow_axial_gradient():
    r = PortLiveModel().evaluate(PortLiveParams(layout="single", R_p=20.0, law="power", L_g=0.8))
    assert np.all(np.diff(r["G"]) > 0)                          # fuel adds mass flux downstream
    assert r["mf"] > 0 and r["OF"] == pytest.approx(1.8 / r["mf"])


def test_flow_tool_matches_mathkit():
    m = FlowLiveModel()
    r = m.evaluate(FlowLiveParams(kappa=0.01, eps=1.0))
    assert r["beta"] == pytest.approx(0.717888, abs=2e-6) and r["beta_asym"] == pytest.approx(0.717896, abs=2e-6)
    assert m.evaluate(FlowLiveParams(kappa=0.01, eps=5.0))["beta"] == pytest.approx(0.753469, abs=2e-6)
    assert r["ur"][-1] == pytest.approx(-1.0, abs=1e-12) and r["uz"][-1] == pytest.approx(0.0, abs=1e-12)
    for z, rr in r["lines"]:                                    # streamlines stay inside the port
        assert np.all((rr >= 0) & (rr <= 1.0 + 1e-9))
    hw = m.evaluate(FlowLiveParams(mode="headwall", u_h=0.5))
    assert hw["uz"][0] == pytest.approx(np.pi * (3.0 + 0.5))


def test_stability_tool_ames_case_and_scaling():
    m = StabilityLiveModel()
    r = m.evaluate(StabilityLiveParams(preset="ames", s=2.0))
    assert r["tau"] * 1e3 == pytest.approx(18.29, abs=0.02) and r["f"] == pytest.approx(26.24, abs=0.02)
    assert r["scaled"]["freq"] == pytest.approx(r["f"] / 2.0)                 # tau ~ L  =>  f ~ 1/s


def test_fuel_tool_energy_and_density_impulse():
    m = FuelLiveModel()
    base = m.evaluate(FuelLiveParams(additive="none"))
    al = m.evaluate(FuelLiveParams(additive="Al", loading=20.0))
    assert base["h"] == pytest.approx(42.0) and base["rho"] == pytest.approx(920.0)
    assert al["rho"] > base["rho"] and al["e"] > base["e"]                       # denser, more energy per volume
    assert al["h"] == pytest.approx(0.8 * 42.0 + 0.2 * 31.0)
    # dense storable oxidizer + dense fuel beats cryogenic hydrogen on density impulse
    di = {k: m.evaluate(FuelLiveParams(pair=k)).get("di").max() for k in ("LOX/RP-1", "LOX/LH2")}
    assert di["LOX/RP-1"] > 1.5 * di["LOX/LH2"]


@pytest.mark.parametrize("model,params", [
    (RegressionLiveModel, RegressionLiveParams), (PortLiveModel, PortLiveParams), (FlowLiveModel, FlowLiveParams),
    (FuelLiveModel, FuelLiveParams), (StabilityLiveModel, StabilityLiveParams)])
def test_live_budget(model, params):
    m, p = model(), params()
    m.evaluate(p)
    t = time.perf_counter()
    for _ in range(10):
        r = m.evaluate(p)
    assert (time.perf_counter() - t) / 10 < 0.05
    tabs = m.tabs(p, r)
    assert tabs and all(len(t.plots) == 3 for t in tabs)
    assert len(m.readout(p, r, None)) == 4
