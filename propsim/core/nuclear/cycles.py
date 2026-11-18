r"""NTR engine cycles: hot bleed, cold bleed and expander (Emrich Ch. 3), with real hydrogen from CoolProp.

State numbering follows Emrich Figs. 3.1-3.3: 1 tank (saturated liquid), 2 pump outlet, 3 after regenerative heating in
the nozzle, reflector and structure. The turbine power always equals the pump power:

* hot bleed (NERVA, Timberwind): :math:`\dot m_{tot}(h_2-h_1)=\dot m_{bleed}(h_5-h_6)`, where the turbine inlet 5 mixes
  warm gas (state 3) with hot gas from the chamber plenum (state 0) down to the turbine temperature limit
  (Emrich Eqs. 3.1-3.2);
* cold bleed: :math:`\dot m_{tot}(h_2-h_1)=\dot m_{bleed}(h_3-h_4)` with the warm bleed dumped after the turbine
  (Eq. 3.2);
* expander: all the flow passes through the turbine before the core, :math:`h_2-h_1=h_3-h_4` (Eq. 3.3), so the pump
  must over-pressurize the hydrogen enough to leave the core inlet pressure after the turbine.

Bleed gas leaves through a small auxiliary nozzle; the delivered specific impulse is the flow-weighted mean of the main
and auxiliary nozzles. Hydrogen above 1000 K uses CoolProp's extrapolation of the Leachman equation of state (its
ideal-gas part is valid to high temperature).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from .performance import isp_vs_eps

FLUID = "Hydrogen"
CYCLES = ["hot", "cold", "expander"]
CYCLE_LABELS = ["Hot bleed (NERVA)", "Cold bleed", "Expander"]


_AS = None
_PAIRS = {}


def _flash(k1: str, v1: float, k2: str, v2: float) -> dict:
    """One CoolProp update; returns ``T, p, h, s``."""
    _ps("T", k1, v1, k2, v2)
    return dict(T=float(_AS.T()), p=float(_AS.p()), h=float(_AS.hmass()), s=float(_AS.smass()))


def _ps(out: str, k1: str, v1: float, k2: str, v2: float, fluid: str = FLUID) -> float:
    """``PropsSI``-style call through one cached CoolProp ``AbstractState`` (about 20 times faster, which keeps the
    Live model under its time budget). Supports the input pairs used here."""
    global _AS
    import CoolProp as CP
    if _AS is None:
        _AS = CP.AbstractState("HEOS", fluid)
        _PAIRS.update({("P", "T"): CP.PT_INPUTS, ("H", "P"): CP.HmassP_INPUTS, ("P", "S"): CP.PSmass_INPUTS,
                       ("H", "S"): CP.HmassSmass_INPUTS, ("Q", "T"): CP.QT_INPUTS})
    key = tuple(sorted((k1, k2)))
    vals = dict(((k1, v1), (k2, v2)))
    _AS.update(_PAIRS[key], vals[key[0]], vals[key[1]])
    return float({"H": _AS.hmass, "S": _AS.smass, "T": _AS.T, "P": _AS.p}[out]())


@lru_cache(maxsize=1)
def saturation_dome(n: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hydrogen saturation line ``(T, s_liquid, s_vapour)`` from the triple point to just below critical."""
    from CoolProp.CoolProp import PropsSI
    Tt = PropsSI("Ttriple", FLUID) + 0.01
    Tc = PropsSI("Tcrit", FLUID) - 0.05
    T = np.linspace(Tt, Tc, n)
    return T, PropsSI("S", "T", T, "Q", np.zeros(n), FLUID), PropsSI("S", "T", T, "Q", np.ones(n), FLUID)


@lru_cache(maxsize=256)
def _isp(T: float, eps: float, mw: float, g: float) -> float:
    return float(isp_vs_eps(T, mw, g, np.array([eps]))[0])


def _state(name, T=None, p=None, h=None, s=None):
    st = _flash("T", T, "P", p) if T is not None else _flash("H", h, "P", p)
    return dict(name=name, T=st["T"], p=p, h=st["h"], s=st["s"])


def _expand(st: dict, p_out: float, eta: float) -> dict:
    """Turbine (or nozzle) expansion from state ``st`` to ``p_out`` with isentropic efficiency ``eta``."""
    hs = _ps("H", "P", p_out, "S", st["s"], FLUID)
    return _state("", p=p_out, h=st["h"] - eta * (st["h"] - hs))


def _pump(st: dict, p_out: float, eta: float) -> dict:
    hs = _ps("H", "P", p_out, "S", st["s"], FLUID)
    return _state("", p=p_out, h=st["h"] + (hs - st["h"]) / eta)


def _turbine_outlet_pressure(st3: dict, dh: float, eta: float) -> float:
    """Pressure at which a turbine with efficiency ``eta`` extracts ``dh`` from state 3."""
    hs = st3["h"] - dh / eta
    return _ps("P", "H", hs, "S", st3["s"])


def cycle(kind: str, pc: float, T0: float, T1: float = 20.0, p_tank: float = 0.2e6, T3: float = 200.0,
          dp_core: float = 0.1, dp_regen: float = 0.1, eta_p: float = 0.7, eta_t: float = 0.7, PR_t: float = 20.0,
          T_turb: float = 800.0, eps: float = 200.0, eps_aux: float = 50.0, mw: float = 2.016,
          g: float = 1.41) -> dict:
    """Solve one cycle. ``pc`` chamber pressure (Pa), ``T0`` chamber temperature (K), ``T1`` tank temperature,
    ``p_tank`` tank pressure, ``T3`` temperature after regenerative heating, ``dp_core`` and ``dp_regen`` pressure losses
    as fractions of ``pc``, ``eta_p`` / ``eta_t`` pump and turbine efficiencies, ``PR_t`` bleed-turbine pressure ratio,
    ``T_turb`` hot-bleed turbine inlet temperature limit, ``eps`` / ``eps_aux`` main and auxiliary nozzle area ratios.

    Returns the states (list of dicts with ``name, T, p, h, s``), the bleed fraction, pump specific work, delivered and
    main-nozzle vacuum Isp, and the reactor heat per unit total flow."""
    p_core_in = pc * (1.0 + dp_core)
    psat = _ps("P", "T", T1, "Q", 0, FLUID)
    p1 = max(p_tank, psat * 1.0001)
    s1 = _state("1", T=T1, p=p1) if p1 > psat * 1.0001 else dict(name="1", T=T1, p=p1, h=_ps("H", "T", T1, "Q", 0, FLUID),
                                                                   s=_ps("S", "T", T1, "Q", 0, FLUID))
    isp_main = _isp(T0, eps, mw, g)
    st0 = _state("0", T=T0, p=pc)
    out = dict(kind=kind, isp_main=isp_main, p_core_in=p_core_in)
    if kind == "expander":
        def gap(p2):
            s2 = _pump(s1, p2, eta_p)
            s3 = _state("3", T=T3, p=p2 - dp_regen * pc)
            return _turbine_outlet_pressure(s3, s2["h"] - s1["h"], eta_t) - p_core_in

        lo = p_core_in * 1.0001 + dp_regen * pc
        # the outlet pressure is nearly linear in the pump pressure (Emrich's worked example converges in three
        # tries), so a secant from two guesses usually lands in 3-4 evaluations; bracketed brentq is the fallback
        x0, x1 = 1.3 * lo, 1.6 * lo
        f0, f1 = gap(x0), gap(x1)
        p2 = None
        for _ in range(8):
            if f1 == f0:
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            if not np.isfinite(x2) or x2 <= lo:
                break
            x0, f0, x1 = x1, f1, x2
            f1 = gap(x1)
            if abs(f1) < 500.0:
                p2 = x1
                break
        if p2 is None:
            hi = lo * 1.5
            while gap(hi) < 0 and hi < 60 * pc:
                hi *= 1.5
            if gap(hi) < 0:
                raise ValueError("expander cycle cannot close: raise T3 or the turbine efficiency")
            p2 = brentq(gap, lo, hi, xtol=500.0)
        s2 = _pump(s1, p2, eta_p)
        s3 = _state("3", T=T3, p=p2 - dp_regen * pc)
        s4 = _state("4", p=p_core_in, h=s3["h"] - (s2["h"] - s1["h"]))
        fb = 0.0
        q_core = st0["h"] - s4["h"]
        states = [s1, {**s2, "name": "2"}, s3, s4, {**_state("5", T=T0, p=pc)}]
        paths = [["1", "2", "3", "4", "5"]]
        isp = isp_main
        T_tin = T3
    else:
        p2 = p_core_in + dp_regen * pc
        s2 = _pump(s1, p2, eta_p)
        s3 = _state("3", T=T3, p=p_core_in)
        wp = s2["h"] - s1["h"]
        if kind == "cold":
            s4 = _expand(s3, s3["p"] / PR_t, eta_t)
            fb = wp / (s3["h"] - s4["h"])
            T_aux = s4["T"]
            q_core = (1.0 - fb) * (st0["h"] - s3["h"])
            states = [s1, {**s2, "name": "2"}, s3, {**s4, "name": "4"}, {**_state("5", T=T0, p=pc)}]
            paths = [["1", "2", "3", "5"], ["3", "4"]]
            T_tin = T3
        else:
            T5 = min(T_turb, T0)
            s5 = _state("5", T=T5, p=pc)
            x_hot = (s5["h"] - s3["h"]) / (st0["h"] - s3["h"])        # hot share of the turbine flow (Eq. 3.2)
            s6 = _expand(s5, pc / PR_t, eta_t)
            fb = wp / (s5["h"] - s6["h"])
            T_aux = s6["T"]
            q_core = (1.0 - fb * (1.0 - x_hot)) * (st0["h"] - s3["h"])
            states = [s1, {**s2, "name": "2"}, s3, {**st0, "name": "4"}, s5, {**s6, "name": "6"}]
            paths = [["1", "2", "3", "4"], ["5", "6"]]
            out["x_hot"] = x_hot
            T_tin = T5
        isp_aux = _isp(round(max(T_aux, 30.0), 3), eps_aux, mw, g)
        isp = (1.0 - fb) * isp_main + fb * isp_aux
        out["isp_aux"] = isp_aux
    out.update(states=states, paths=paths, fb=float(fb), p2=float(p2), w_pump=float(s2["h"] - s1["h"]), isp=float(isp),
               q_core=float(q_core), T_turb_in=float(T_tin), T2=float(s2["T"]))
    return out


def isobar(p: float, T_lo: float, T_hi: float, n: int = 40) -> tuple[np.ndarray, np.ndarray]:
    """Entropy along an isobar between two temperatures (for the T-s diagram). Temperature is the independent
    variable because CoolProp's pressure-entropy flash stops at 1500 K while the pressure-temperature one extrapolates."""
    T = np.geomspace(T_lo, T_hi, n)
    out = []
    for x in T:
        try:
            out.append(_flash("P", p, "T", x)["s"])
        except ValueError:                                    # inside the dome: skip the point
            out.append(np.nan)
    return np.array(out), T


def path_curve(states: list[dict], names: list[str], n: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """``(s, T)`` along a chain of states: heat addition at (nearly) constant pressure follows the isobar, pumps and
    turbines are drawn as straight segments."""
    by = {st["name"]: st for st in states}
    S, T = [], []
    for a, b in zip(names[:-1], names[1:]):
        sa, sb = by[a], by[b]
        if abs(sa["p"] - sb["p"]) <= 0.25 * sb["p"] and sb["T"] > sa["T"]:
            ss, tt = isobar(sb["p"], sa["T"], sb["T"], n)
            ss[0] = sa["s"]
        else:
            ss, tt = np.array([sa["s"], sb["s"]]), np.array([sa["T"], sb["T"]])
        S.append(ss)
        T.append(tt)
    return np.concatenate(S), np.concatenate(T)
