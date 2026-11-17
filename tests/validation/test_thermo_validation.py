"""Thermochemistry validation: CEA vs Chiaverini Table 1, tables vs direct CEA, fallback behaviour, Live model."""
from __future__ import annotations

import importlib
import sys
import time

import numpy as np
import pytest

from propsim.core.constants import G0
from propsim.core.nozzle import eps_from_pratio
from propsim.core.thermo import cea as cea_mod
from propsim.core.thermo.cea import PAIRS, PSI, TableBackend, Thermo
from propsim.modules import engine_live
from propsim.modules.engine_live import EngineLiveModel, EngineLiveParams

from .figs import new_fig, save

FT = 0.3048
pytest.importorskip("rocketcea")

# Chiaverini Table 1, LOX/HTPB at 500 psia (quoted in MATHKIT.md Section 2.4)
TABLE1 = dict(OF=1.9, isp_sl=280.0, cstar_fts=5972.0)


@pytest.fixture(scope="module")
def lox_htpb():
    return Thermo("LOX/HTPB", "cea")


def _optimum_expansion_isp_sl(th, mr, pc, pa=101325.0):
    """Sea-level Isp with the nozzle at optimum expansion (pe = pa) for this gas."""
    g = th.chamber(mr, pc).gamma
    eps = float(eps_from_pratio(pa / pc, float(g))[0])
    return float(th.isp_sl(mr, pc, eps, pa)), eps


def test_lox_htpb_vs_chiaverini_table1(lox_htpb):
    """CEA's built-in HTPB at 500 psia vs the book. MathKit expects ~1 %, noting Table 1 used its own HTPB
    formulation and enthalpy; measured agreement is within 2 %."""
    pc = 500.0 * PSI
    mr = np.linspace(1.3, 3.0, 171)
    cs = np.array([float(lox_htpb.chamber(m, pc).cstar) for m in mr]) / FT
    isp = np.array([_optimum_expansion_isp_sl(lox_htpb, m, pc)[0] for m in mr])
    mr_cs, cs_max = mr[cs.argmax()], cs.max()
    isp_at_19 = _optimum_expansion_isp_sl(lox_htpb, TABLE1["OF"], pc)[0]
    d_cs = cs_max / TABLE1["cstar_fts"] - 1
    d_isp = isp.max() / TABLE1["isp_sl"] - 1

    fig, (a1, a2) = new_fig(2)
    a1.plot(mr, cs, color="#ff9d3c", label="CEA, built-in HTPB")
    a1.plot([TABLE1["OF"]], [TABLE1["cstar_fts"]], "o", color="#ff5a3c", label="Chiaverini Table 1")
    a1.set_xlabel(r"$\frac{O}{F}$ (–)")
    a1.set_ylabel(r"$c^*$ (ft·s$^{-1}$)")
    a1.legend()
    a2.plot(mr, isp, color="#4ea1ff", label="CEA, optimum expansion")
    a2.plot([TABLE1["OF"]], [TABLE1["isp_sl"]], "o", color="#ff5a3c", label="Chiaverini Table 1")
    a2.set_xlabel(r"$\frac{O}{F}$ (–)")
    a2.set_ylabel(r"$I_{sp}$ at sea level (s)")
    a2.legend()
    save(fig, "thermo_lox_htpb_vs_table1")

    print(f"\nLOX/HTPB 500 psia: c*max {cs_max:.0f} ft/s at O/F {mr_cs:.2f} ({d_cs:+.1%} vs 5972); "
          f"Isp max {isp.max():.1f} s ({d_isp:+.1%} vs 280); Isp(O/F=1.9) {isp_at_19:.1f} s")
    assert abs(mr_cs - TABLE1["OF"]) < 0.15              # c* peaks near the book's optimum O/F
    assert abs(d_cs) < 0.025
    assert abs(d_isp) < 0.02


@pytest.mark.xfail(strict=True, reason="CEA's built-in HTPB is 1.7 % low in c* and 1.2 % low in Isp vs Chiaverini "
                                       "Table 1 (different HTPB formulation/enthalpy); the 1 % target is not met")
def test_lox_htpb_within_one_percent_of_table1(lox_htpb):
    pc = 500.0 * PSI
    mr = np.linspace(1.3, 3.0, 171)
    cs_max = max(float(lox_htpb.chamber(m, pc).cstar) for m in mr) / FT
    assert abs(cs_max / TABLE1["cstar_fts"] - 1) < 0.01


def test_ambient_isp_identity_matches_rocketcea():
    """Isp(Pa) = Ivac - Pa eps c*/(Pc g0) must reproduce RocketCEA's own ambient estimate (unseparated nozzles)."""
    from rocketcea.cea_obj import CEA_Obj
    for pair in ("LOX/HTPB", "LOX/RP-1", "LOX/LH2"):
        p = PAIRS[pair]
        c = CEA_Obj(oxName=p.ox, fuelName=p.fuel)
        th = Thermo(pair, "cea")
        for mr in (0.5 * (p.mr_lo + p.mr_hi), 0.7 * p.mr_lo + 0.3 * p.mr_hi):
            for eps in (4.0, 8.0):
                ours = float(th.isp_sl(mr, 500 * PSI, eps))
                ref = float(c.estimate_Ambient_Isp(Pc=500.0, MR=mr, eps=eps, Pamb=14.696)[0])
                assert ours == pytest.approx(ref, rel=2e-4), (pair, mr, eps)


def test_tables_vs_direct_cea_all_pairs():
    """Interpolated tables reproduce direct CEA within 0.3 % at random off-grid points, equilibrium and frozen."""
    rng = np.random.default_rng(7)
    worst = {}
    for key, pr in PAIRS.items():
        for frozen in (False, True):
            tt, cc = Thermo(key, "tables", frozen), Thermo(key, "cea", frozen)
            errs = []
            for _ in range(25):
                mr = rng.uniform(pr.mr_lo, pr.mr_hi)
                pc = float(np.exp(rng.uniform(np.log(0.6e6), np.log(19e6))))
                eps = float(np.exp(rng.uniform(np.log(1.6), np.log(95.0))))
                a, b = tt.point(mr, pc, eps), cc.point(mr, pc, eps)
                errs.append(max(abs(a[q] / b[q] - 1) for q in ("cstar", "Tc", "gamma", "Mw", "isp_vac", "cf_vac")))
            worst[(key, frozen)] = max(errs)
    fig, (ax,) = new_fig(1, 7.0, 3.8)
    labels = [f"{k}{' fr' if f else ''}" for k, f in worst]
    ax.bar(range(len(worst)), [100 * v for v in worst.values()], color="#4ea1ff")
    ax.axhline(0.3, color="#ff5a3c", ls="--", label="limit 0.3 %")
    ax.set_xticks(range(len(worst)), labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(r"max relative error (%)")
    ax.legend()
    save(fig, "thermo_tables_vs_cea")
    assert max(worst.values()) < 0.003, worst


def test_frozen_vs_equilibrium_ordering():
    """Frozen expansion cannot beat equilibrium: Isp_frozen < Isp_eq and c*_frozen < c*_eq for every pair."""
    for key, pr in PAIRS.items():
        mr, pc, eps = 0.5 * (pr.mr_lo + pr.mr_hi), 3.5e6, 20.0
        eq, fr = Thermo(key, "tables", False).point(mr, pc, eps), Thermo(key, "tables", True).point(mr, pc, eps)
        assert fr["isp_vac"] < eq["isp_vac"], key
        assert fr["cstar"] < eq["cstar"], key
        assert abs(fr["Tc"] / eq["Tc"] - 1) < 1e-9, key         # same chamber state


def test_table_integrity():
    z = np.load(cea_mod.DATA, allow_pickle=False)
    for key, pr in PAIRS.items():
        assert z[f"{key}__mr"][0] == pytest.approx(pr.mr_lo) and z[f"{key}__mr"][-1] == pytest.approx(pr.mr_hi)
        for n in ("cstar", "cstar_fr", "Tc", "gamma", "Mw", "ivac_eq", "ivac_fr"):
            a = z[f"{key}__{n}"]
            assert np.all(np.isfinite(a)) and np.all(a > 0), (key, n)
        assert 1.05 < z[f"{key}__gamma"].min() and z[f"{key}__gamma"].max() < 1.4


def test_direct_cea_refuses_silent_zero():
    """CEA returns Isp = c* = 0 for frozen flow of fuel-rich HTPB; the wrapper must raise, not return zeros."""
    th = Thermo("LOX/HTPB", "cea", True)
    with pytest.raises(RuntimeError, match="no solution"):
        th.point(1.03, 2.9e6, 12.0)


def test_fallback_without_rocketcea(monkeypatch):
    """With rocketcea unavailable the app still runs from the shipped tables (direct request falls back)."""
    monkeypatch.setitem(sys.modules, "rocketcea", None)
    monkeypatch.setitem(sys.modules, "rocketcea.cea_obj", None)
    assert not cea_mod.cea_available()
    cea_mod.get_backend.cache_clear()
    engine_live._THERMO.clear()
    try:
        assert isinstance(cea_mod.get_backend("auto"), TableBackend)
        with pytest.raises(RuntimeError):
            cea_mod.get_backend("cea")
        m = EngineLiveModel()
        r = m.evaluate(EngineLiveParams(source="cea"))      # asks for CEA, gets tables
        assert "using tables" in r["src"] and r["Isp"] > 200
    finally:
        monkeypatch.undo()
        importlib.invalidate_caches()
        cea_mod.get_backend.cache_clear()
        engine_live._THERMO.clear()


def test_live_engine_budget_and_consistency():
    """Live physics < 50 ms with tables; tables and direct CEA give the same headline numbers."""
    m = EngineLiveModel()
    p = EngineLiveParams(pair="LOX/HTPB", mr=2.0, p_c=3.45, eps=8.0)
    m.evaluate(p)
    t = time.perf_counter()
    for _ in range(10):
        r_t = m.evaluate(p)
    dt = (time.perf_counter() - t) / 10
    assert dt < 0.05, f"{dt * 1e3:.1f} ms"
    r_c = m.evaluate(p.model_copy(update={"source": "cea"}))
    for k in ("cs", "Isp", "Isp_vac", "cf", "g"):
        assert r_t[k] == pytest.approx(r_c[k], rel=3e-3), k
    # thrust identity: F = C_F Pc At = Isp g0 mdot
    assert r_t["F"] == pytest.approx(r_t["Isp"] * G0 * r_t["mdot"], rel=1e-9)


def test_flow_separation_gives_physical_thrust():
    """Strongly over-expanded nozzle at sea level: thrust uses the separated (truncated) nozzle, never negative."""
    m = EngineLiveModel()
    r = m.evaluate(EngineLiveParams(eps=100.0, p_c=0.5))
    assert r["separated"] and r["Isp"] > 0 and r["F"] > 0
    ok = m.evaluate(EngineLiveParams(eps=6.0, p_c=3.5))
    assert not ok["separated"]
    assert np.all(r["cfa_c"] > 0)
