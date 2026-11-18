"""Designs: the engine or motor the user builds in a designer, and how the analysis modules start from it.

A *design* is a small JSON document (``*.psimdesign.json``) with a ``kind`` (liquid, solid, hybrid, nuclear, electric), a
name, where it came from (a library engine or the user) and the data of its designer:

* ``liquid``  the cycle-sandbox document (graph, design targets and free parameters)
* ``solid``   the grain and propellant parameters of the solid motor designer
* ``hybrid``  the port and grain parameters of the hybrid motor designer

``updates(design, model_key)`` gives the parameter values an analysis module takes from the design (display units, the
names of that module's own parameters); anything the design does not define keeps the module's default.  No Qt here.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

SUFFIX = ".psimdesign.json"
FORMAT = "propsim-design"
KINDS = ("liquid", "solid", "hybrid", "nuclear", "electric")

# the analysis models that start from each design kind
MODELS_FOR = {
    "liquid": ("liquid.feed", "liquid.cycle", "liquid.chugging"),
    "solid": ("solid.ballistics", "solid.burnback3d", "solid.burnrate", "solid.cpmotor", "solid.cpmotor_us"),
    "hybrid": ("hybrid.burn0d", "hybrid.burn1d", "hybrid.n2o"),
}


@dataclass
class Design:
    kind: str
    name: str
    data: dict = field(default_factory=dict)
    source: str = "user"                                   # "user" or a library engine / template id
    note: str = ""

    def to_dict(self) -> dict:
        return {"format": FORMAT, "version": 1, "kind": self.kind, "name": self.name, "source": self.source,
                "note": self.note, "data": self.data}

    @classmethod
    def from_dict(cls, d: dict) -> "Design":
        if d.get("format") != FORMAT or d.get("kind") not in KINDS:
            raise ValueError("not a PropSim design file")
        return cls(d["kind"], d.get("name", "design"), d.get("data", {}), d.get("source", "user"), d.get("note", ""))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        if not p.name.endswith(SUFFIX):
            p = p.with_name(p.name.split(".")[0] + SUFFIX)
        p.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Design":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def applies_to(design: Design | None, model_key: str) -> bool:
    return design is not None and model_key in MODELS_FOR.get(design.kind, ())


# ------------------------------------------------------------------------------------------------ liquid
def liquid_summary(graph: dict, targets=None, free=None) -> dict:
    """Solve the engine graph and collect the numbers the liquid analysis modules need (SI)."""
    from .core.engine_graph import edit as E
    from .core.engine_graph import solve
    _pv = lambda node, name: E.param_value(node, name)[0]          # (value, provenance, is_default) -> value
    sol = solve(graph, targets or None, free or None)
    nodes = {n["id"]: n for n in graph["nodes"]}
    typ = lambda t: [n for n in graph["nodes"] if n["type"] == t]
    out: dict = {"converged": bool(sol.converged)}
    ch = typ("chamber")
    noz = typ("nozzle_bell")
    if ch:
        c = sol.components.get(ch[0]["id"], {})
        out["Pc"], out["MR"] = c.get("p_c"), c.get("mixture_ratio")
        out["At"] = _pv(ch[0], "A_t")
        out["L_star"] = _pv(ch[0], "Lstar")
        out["contraction"] = _pv(ch[0], "contraction_ratio")
    if noz:
        out["thrust"] = sol.components.get(noz[0]["id"], {}).get("thrust")
        out["isp"] = sol.components.get(noz[0]["id"], {}).get("isp")
        out["eps"] = _pv(noz[0], "expansion_ratio")
    inj = [sol.components.get(n["id"], {}) for n in typ("injector")]
    drops = [i["dp"] / out["Pc"] for i in inj if i.get("dp") and out.get("Pc")]
    if drops:
        out["dp_inj"] = float(sum(drops) / len(drops))
    tanks = [_pv(n, "p_set") for n in typ("tank_regulated")]
    tanks = [t for t in tanks if t]
    if tanks:
        out["p_tank"] = float(max(tanks))
    shafts = [sol.components.get(n["id"], {}) for n in typ("shaft")]
    if shafts and shafts[0].get("speed"):
        out["N0"] = float(shafts[0]["speed"]) * 60.0 / (2.0 * math.pi)
    pumps = [_pv(n, "eta_design") for n in typ("pump_centrifugal")]
    pumps = [e for e in pumps if e]
    if pumps:
        out["eta_p"] = float(sum(pumps) / len(pumps))
    tb = typ("turbine_impulse")
    if tb:
        out["eta_t"] = _pv(tb[0], "eta_design")
        out["PR_t"] = _pv(tb[0], "pressure_ratio")
        out["T_t"] = _pv(tb[0], "T_inlet")
    mixed = [sol.components.get(n["id"], {}) for n in typ("gas_generator")]
    if mixed and mixed[0].get("mixture_ratio"):
        out["MR_gg"] = mixed[0]["mixture_ratio"]
    return {k: v for k, v in out.items() if v is not None}


def _liquid_updates(d: Design) -> dict:
    s = liquid_summary(d.data["graph"], d.data.get("targets"), d.data.get("free"))
    u: dict = {}
    if s.get("Pc"):
        u["Pc"] = s["Pc"] * 1e-6
    for k in ("MR", "eps", "L_star", "contraction", "eta_p", "eta_t", "PR_t", "T_t", "MR_gg", "N0"):
        if s.get(k):
            u[k] = s[k]
    if s.get("At"):
        u["D_t"] = 2.0 * math.sqrt(s["At"] / math.pi) * 1e3             # mm
    if s.get("dp_inj"):
        u["dp_inj"] = s["dp_inj"] * 100.0                                # %
    if s.get("p_tank"):
        u["p_tank"] = s["p_tank"] * 1e-6
    return u


# ------------------------------------------------------------------------------------------------ solid and hybrid
def _solid_updates(d: Design) -> dict:
    return {k: v for k, v in d.data.items() if k not in ("burn",)}


def _hybrid_updates(d: Design) -> dict:
    x = d.data
    u: dict = {}
    if "R_p" in x:
        u["R_p0"] = x["R_p"]
    if "R_out" in x and "R_p" in x:
        u["web"] = max(x["R_out"] - x["R_p"], 1.0)
    for k in ("L_g", "mdot_ox", "law"):
        if k in x:
            u[k] = x[k]
    return u


def updates(design: Design | None, model_key: str, fields: set[str] | None = None) -> dict:
    """Parameter values (display units) a model takes from ``design``; restricted to ``fields`` when given."""
    if not applies_to(design, model_key):
        return {}
    u = {"liquid": _liquid_updates, "solid": _solid_updates, "hybrid": _hybrid_updates}[design.kind](design)
    return {k: v for k, v in u.items() if fields is None or k in fields}


def fit(values: dict, params_cls) -> tuple[dict, list[str]]:
    """Keep only the parameters ``params_cls`` has and clamp each to that module's slider range; returns the values and
    the names that had to be clamped (a design can be bigger than a module's range, for example a large throat)."""
    out, clamped = {}, []
    for k, v in values.items():
        f = params_cls.model_fields.get(k)
        if f is None:
            continue
        extra = f.json_schema_extra or {}
        lo, hi = extra.get("lo"), extra.get("hi")
        if isinstance(v, (int, float)) and lo is not None and hi is not None:
            c = min(max(float(v), lo), hi)
            if c != float(v):
                clamped.append(k)
            v = c
        out[k] = v
    return out, clamped
