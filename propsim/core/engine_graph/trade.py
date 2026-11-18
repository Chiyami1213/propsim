r"""Trade studies on the cycle sandbox: parametric sweeps of one input and side-by-side design comparison.

A sweep varies one input over a list of values and solves the engine at each:

* a component parameter (``kind="param"``: a line diameter, a pump tip diameter, a turbine pressure ratio, a nozzle
  expansion ratio, ...) or a valve position;
* a design target (``kind="target"``: chamber pressure or mixture ratio with the free parameters re-sized at each
  point, as the controller would do).

The point nearest the design's own value is solved first from scratch, then the sweep marches outward on both sides,
each point starting from its neighbour's solution (so a sweep of twenty points costs little more than a few cold
solves).  A point that will not converge from its neighbour is retried cold; one that still fails is kept, flagged,
with NaN outputs.

``summary`` reduces a solution to the engine-level numbers a trade is judged by (thrust, Isp, pc, O/F, pump and
turbine power, peak wall temperature, warnings); ``compare`` runs it on several sandbox documents.
"""
from __future__ import annotations

import copy
import math
from typing import Callable

import numpy as np

from .edit import param_value, set_param
from .network import GraphError, Network, Solution

SUMMARY_KEYS = [  # (key, label, scale, unit)
    ("thrust", "Thrust", 1e-3, "kN"), ("isp", "Isp", 1.0, "s"), ("isp_vac", "Isp (vacuum)", 1.0, "s"),
    ("p_c", "Chamber pressure", 1e-6, "MPa"), ("mixture_ratio", "Mixture ratio O/F", 1.0, ""),
    ("mdot", "Engine flow", 1.0, "kg/s"), ("pump_power", "Pump power", 1e-6, "MW"),
    ("turbine_power", "Turbine power", 1e-6, "MW"), ("turbine_flow", "Turbine flow", 1.0, "kg/s"),
    ("T_turbine_max", "Turbine inlet T (max)", 1.0, "K"), ("T_wg_max", "Gas-side wall T (max)", 1.0, "K"),
    ("q_max", "Peak heat flux", 1e-6, "MW/m²"), ("Q_jacket", "Jacket heat load", 1e-6, "MW"),
    ("p_pump_max", "Highest pump discharge", 1e-6, "MPa"), ("npsh_margin_min", "Lowest NPSH available", 1.0, "m"),
    ("warnings", "Warnings", 1.0, ""),
]


def summary(sol: Solution, graph: dict) -> dict[str, float]:
    """Engine-level numbers of a solution (NaN where the engine has no such part)."""
    nan = float("nan")
    comps = sol.components
    by_type: dict[str, list[dict]] = {}
    for c in comps.values():
        by_type.setdefault(c.get("type"), []).append(c)

    def first(t, k):
        v = [c.get(k) for c in by_type.get(t, []) if isinstance(c.get(k), (int, float))]
        return float(v[0]) if v else nan

    def total(t, k):
        v = [c.get(k) for c in by_type.get(t, []) if isinstance(c.get(k), (int, float))]
        return float(sum(v)) if v else nan

    def peak(t, k, f=max):
        v = [c.get(k) for c in by_type.get(t, []) if isinstance(c.get(k), (int, float))]
        return float(f(v)) if v else nan

    stations = {}
    for s in sol.stations:
        for p in s["ports"]:
            stations[p] = s
    p_dis = [stations[f"{nid}.outlet"]["p"] for nid, c in comps.items()
             if c.get("type") == "pump_centrifugal" and f"{nid}.outlet" in stations]
    return {
        "thrust": first("nozzle_bell", "thrust"), "isp": first("nozzle_bell", "isp"),
        "isp_vac": first("nozzle_bell", "isp_vac"), "p_c": first("chamber", "p_c"),
        "mixture_ratio": first("chamber", "mixture_ratio_overall"), "mdot": first("nozzle_bell", "mdot"),
        "pump_power": total("pump_centrifugal", "power"), "turbine_power": total("turbine_impulse", "power"),
        "turbine_flow": total("turbine_impulse", "mdot"), "T_turbine_max": peak("turbine_impulse", "T_inlet"),
        "T_wg_max": peak("regen_jacket", "T_wg_max"), "q_max": peak("regen_jacket", "q_max"),
        "Q_jacket": total("regen_jacket", "Q"), "p_pump_max": float(max(p_dis)) if p_dis else nan,
        "npsh_margin_min": peak("pump_centrifugal", "npsh_available", min), "warnings": float(len(sol.warnings)),
    }


def flat_outputs(sol: Solution) -> dict[str, float]:
    """Every numeric component output as ``node.output``, sized free parameters as ``node.param``, and shaft speeds."""
    out: dict[str, float] = {}
    for nid, c in sol.components.items():
        for k, v in c.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[f"{nid}.{k}"] = float(v)
    for k, v in sol.free.items():
        out[k] = float(v)
    return out


def inputs(doc: dict) -> list[tuple[str, str, str, str]]:
    """What a sweep can vary: ``(kind, node, name, unit)`` for every numeric parameter, valve positions and targets."""
    from .edit import library
    out = []
    lib = library()
    for n in doc["graph"]["nodes"]:
        for prm in lib[n["type"]]["parameters"]:
            v = param_value(n, prm["name"])[0]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(("param", n["id"], prm["name"], prm.get("unit") or ""))
        if n["type"].startswith("valve"):
            out.append(("param", n["id"], "position", "1"))
    for t in doc.get("targets", []):
        out.append(("target", t["node"], t["output"], ""))
    return out


def current_value(doc: dict, kind: str, node: str, name: str) -> float | None:
    if kind == "target":
        t = next((t for t in doc.get("targets", []) if t["node"] == node and t["output"] == name), None)
        return float(t["value"]) if t else None
    n = next(n for n in doc["graph"]["nodes"] if n["id"] == node)
    if name == "position":
        return float(n.get("options", {}).get("position", 1.0))
    v = param_value(n, name)[0]
    return float(v) if isinstance(v, (int, float)) else None


def _apply(doc: dict, kind: str, node: str, name: str, value: float) -> dict:
    d = copy.deepcopy(doc)
    if kind == "target":
        for t in d["targets"]:
            if t["node"] == node and t["output"] == name:
                t["value"] = float(value)
        return d
    n = next(n for n in d["graph"]["nodes"] if n["id"] == node)
    if name == "position":
        n.setdefault("options", {})["position"] = float(value)
        d["free"] = [f for f in d.get("free", []) if not (f["node"] == node and f["param"] == "position")]
        return d
    from .edit import library
    unit = next((p.get("unit") for p in library()[n["type"]]["parameters"] if p["name"] == name), None)
    set_param(n, name, float(value), unit)
    d["free"] = [f for f in d.get("free", []) if not (f["node"] == node and f["param"] == name)]
    return d


def sweep(doc: dict, kind: str, node: str, name: str, values, progress: Callable[[int, int], None] | None = None,
          cancelled: Callable[[], bool] | None = None) -> dict:
    """Solve ``doc`` at each of ``values`` of one input.

    Returns ``{"values": [...], "converged": [...], "summary": {key: [...]}, "outputs": {node.output: [...]},
    "messages": [...]}``, every list in the order of ``values``."""
    values = [float(v) for v in values]
    if not values:
        raise ValueError("no sweep values")
    if kind == "param" and name != "position" and doc.get("free"):
        # a parameter swept while a target sizes the same parameter would be overwritten: refuse it plainly
        for f in doc["free"]:
            if f["node"] == node and f["param"] == name:
                raise GraphError(f"{node}.{name} is sized by the design targets; sweep the target instead")
    base = current_value(doc, kind, node, name)
    order = sorted(range(len(values)), key=lambda i: abs(values[i] - base) if base is not None else i)
    start = order[0]
    res: list[tuple[Solution | None, str] | None] = [None] * len(values)
    xs: list[np.ndarray | None] = [None] * len(values)
    refs: dict[int, tuple[float, float]] = {}
    done = 0
    total = len(values)

    def run(i: int, x0: np.ndarray | None, ref):
        d = _apply(doc, kind, node, name, values[i])
        net = Network(d["graph"], d.get("targets") or None, d.get("free") or None)
        if x0 is not None:
            net.m_ref, net.P_ref = ref                   # the neighbour's scales, as its solution was found with them
            sol = net.solve(x0=x0)
        else:
            sol = net.solve()
        if not sol.converged and x0 is not None:
            net = Network(d["graph"], d.get("targets") or None, d.get("free") or None)
            sol = net.solve()
        return sol, (net.x if sol.converged else None), d, (net.m_ref, net.P_ref)

    def step(i, x0, ref=None):
        nonlocal done
        if cancelled and cancelled():
            return None
        try:
            sol, x, d, rf = run(i, x0, ref)
            res[i] = (sol, d)
            xs[i] = x
            refs[i] = rf
        except (GraphError, ValueError, ArithmeticError) as e:
            res[i] = (None, str(e))
            xs[i] = None
        done += 1
        if progress:
            progress(done, total)
        return xs[i]

    step(start, None)
    for direction in (1, -1):                            # march outward from the design point on both sides
        idx = sorted(range(len(values)), key=lambda i: values[i])
        pos = idx.index(start)
        prev = start
        j = pos + direction
        while 0 <= j < len(idx):
            if cancelled and cancelled():
                break
            step(idx[j], xs[prev], refs.get(prev))
            if xs[idx[j]] is not None:
                prev = idx[j]                            # a failed point is skipped: the next starts from the last good one
            j += direction
    out_vals: dict[str, list[float]] = {}
    summ: dict[str, list[float]] = {k: [] for k, *_ in SUMMARY_KEYS}
    conv, msgs = [], []
    for i in range(len(values)):
        sol, d = res[i] if res[i] is not None else (None, "cancelled")
        ok = sol is not None and sol.converged
        conv.append(bool(ok))
        msgs.append(sol.message if sol is not None else str(d))
        s = summary(sol, d["graph"]) if ok else {}
        for k in summ:
            summ[k].append(s.get(k, math.nan))
        f = flat_outputs(sol) if ok else {}
        for k in set(out_vals) | set(f):
            out_vals.setdefault(k, [math.nan] * i)
            out_vals[k].append(f.get(k, math.nan))
    for k, v in out_vals.items():
        v.extend([math.nan] * (len(values) - len(v)))
    return {"kind": kind, "node": node, "name": name, "values": values, "converged": conv, "summary": summ,
            "outputs": out_vals, "messages": msgs}


def compare(docs: dict[str, dict]) -> dict[str, dict[str, float]]:
    """``{design name: summary}`` of each sandbox document, solved as it stands (design targets included)."""
    out = {}
    for name, d in docs.items():
        sol = Network(d["graph"], d.get("targets") or None, d.get("free") or None).solve()
        s = summary(sol, d["graph"]) if sol.converged else {k: math.nan for k, *_ in SUMMARY_KEYS}
        s["converged"] = float(sol.converged)
        out[name] = s
    return out


def to_csv(result: dict) -> str:
    """A sweep as CSV: the swept input, convergence, the summary and every component output."""
    head = [f"{result['node']}.{result['name']}", "converged"] + [k for k, *_ in SUMMARY_KEYS] + sorted(result["outputs"])
    rows = [",".join(head)]
    for i, v in enumerate(result["values"]):
        cells = [repr(v), "1" if result["converged"][i] else "0"]
        cells += [repr(result["summary"][k][i]) for k, *_ in SUMMARY_KEYS]
        cells += [repr(result["outputs"][k][i]) for k in sorted(result["outputs"])]
        rows.append(",".join(cells))
    return "\n".join(rows) + "\n"
