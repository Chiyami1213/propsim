"""Graph editing for the cycle sandbox (no Qt): add, connect and remove components, infer edge media, templates and
the sandbox file format.

A sandbox file is ``{"format": "propsim-engine-sandbox", "version": 1, "graph": <engine graph>, "targets": [...],
"free": [...]}``; the graph inside follows ``schemas/graph.schema.json`` so it can be shared with the library.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

FORMAT = "propsim-engine-sandbox"

# palette groups (library category -> heading) and readable names
CATEGORY_LABELS = {"tank": "Tanks", "line": "Lines", "valve": "Valves and orifices", "volume": "Manifolds",
                   "turbomachinery": "Turbomachinery", "combustion": "Combustion", "nozzle": "Nozzles",
                   "heat_transfer": "Cooling"}
TYPE_LABELS = {"tank_regulated": "Regulated tank", "tank_blowdown": "Blowdown tank", "line": "Feed line",
               "valve_ball": "Ball valve", "valve_butterfly": "Butterfly valve", "valve_poppet": "Poppet valve",
               "check_valve": "Check valve", "orifice": "Orifice", "cavitating_venturi": "Cavitating venturi",
               "manifold": "Manifold", "injector": "Injector", "chamber": "Combustion chamber",
               "nozzle_bell": "Bell nozzle", "pump_centrifugal": "Centrifugal pump", "inducer": "Inducer",
               "turbine_impulse": "Turbine", "gas_generator": "Gas generator / preburner", "shaft": "Shaft",
               "regen_jacket": "Regenerative jacket"}

# categorical options the solver reads, with their choices (first = default)
OPTIONS: dict[str, dict[str, list[str]]] = {
    "tank_regulated": {"fluid": []}, "tank_blowdown": {"fluid": []},
    "valve_ball": {"position": ["1.0"]}, "valve_butterfly": {"position": ["1.0"]}, "valve_poppet": {"position": ["1.0"]},
    "turbine_impulse": {"flow_model": ["auto", "pressure_ratio", "choked"],
                        "arrangement": ["impulse_single_stage", "impulse_pressure_compounded_2stage",
                                        "impulse_velocity_compounded_2row", "reaction_2stage", "hydraulic_6stage"]},
    "gas_generator": {"gas_model": ["auto", "table", "frozen", "user"], "injector_dp_fraction": ["0.0"]},
    "regen_jacket": {"section": ["full", "chamber", "nozzle"], "split_area_ratio": ["5"],
                     "hydraulics": ["friction", "none"]},
    "chamber": {"cooling": ["regenerative", "film", "ablative", "radiative", "heat_sink"],
                "rayleigh_loss": ["on", "off"]},
    "nozzle_bell": {"cooling": ["regenerative", "radiative", "ablative", "uncooled"], "emissivity": ["0.8"],
                    "split_area_ratio": ["5"]},
    "pump_centrifugal": {"shutoff_ratio": ["1.25"]},
    "injector": {"type": ["coaxial", "impinging", "showerhead", "pintle"]},
}
OPTION_HELP = {
    "fluid": "propellant in the tank (fluids.csv)", "position": "valve opening, 1 = fully open",
    "flow_model": "turbine flow: fixed pressure ratio, or choked nozzle (needs A_nozzle)",
    "gas_model": "drive gas: Huzel & Huang Table 6-4, frozen complete combustion, or your own T, cp, gamma",
    "injector_dp_fraction": "built-in injector drop as a fraction of the generator pressure",
    "section": "part of the wall the jacket cools", "split_area_ratio": "area ratio where the chamber and nozzle "
    "sections meet", "hydraulics": "friction: the jacket's own channel loss; none: a separate orifice carries it",
    "cooling": "how this wall is cooled", "rayleigh_loss": "injector-end to nozzle-stagnation pressure loss",
    "emissivity": "radiating wall emissivity", "shutoff_ratio": "shut-off head over design head",
    "type": "injector element type", "arrangement": "turbine staging"}


def library() -> dict[str, dict]:
    from ...data import components
    return components()


def palette() -> dict[str, list[str]]:
    """Heading -> component types, in the library's categories."""
    out: dict[str, list[str]] = {}
    for t, c in sorted(library().items(), key=lambda kv: (list(CATEGORY_LABELS).index(kv[1]["category"])
                                                           if kv[1]["category"] in CATEGORY_LABELS else 99, kv[0])):
        out.setdefault(CATEGORY_LABELS.get(c["category"], c["category"].title()), []).append(t)
    return out


def blank_graph(ox: str = "LOX", fuel: str = "RP1", name: str = "My engine") -> dict:
    return {"schema_version": "1.0", "graph_id": "sandbox", "name": name, "description": "Built in the cycle sandbox",
            "cycle": "custom", "reference_engine": None, "propellants": {"oxidizer": ox, "fuel": fuel},
            "nodes": [], "edges": [], "boundary_conditions": [], "sequence": []}


def unique_id(graph: dict, base: str) -> str:
    base = re.sub(r"[^a-z0-9_]", "_", base.lower()).strip("_") or "node"
    if not base[0].isalpha():
        base = "n_" + base
    ids = {n["id"] for n in graph["nodes"]}
    if base not in ids:
        return base
    k = 2
    while f"{base}_{k}" in ids:
        k += 1
    return f"{base}_{k}"


def add_node(graph: dict, ntype: str, x: float = 0.0, y: float = 0.0, nid: str | None = None) -> dict:
    """New component with every library default (``user_input`` parameters copied as empty values to fill in)."""
    spec = library()[ntype]
    nid = nid or unique_id(graph, {"pump_centrifugal": "pump", "turbine_impulse": "turbine",
                                   "tank_regulated": "tank", "tank_blowdown": "tank", "nozzle_bell": "nozzle",
                                   "gas_generator": "gas_generator", "regen_jacket": "jacket"}.get(ntype, ntype))
    node = {"id": nid, "type": ntype, "label": TYPE_LABELS.get(ntype, ntype), "params": {}, "ui": {"x": x, "y": y}}
    opts = {k: v[0] for k, v in OPTIONS.get(ntype, {}).items() if v}
    if ntype.startswith("tank"):
        opts["fluid"] = graph["propellants"]["fuel" if "fuel" in nid else "oxidizer"]
    if opts:
        node["options"] = opts
    graph["nodes"].append(node)
    return node


def remove_node(graph: dict, nid: str) -> None:
    graph["nodes"] = [n for n in graph["nodes"] if n["id"] != nid]
    graph["edges"] = [e for e in graph["edges"] if nid not in (e["from"]["node"], e["to"]["node"])]
    graph["boundary_conditions"] = [b for b in graph.get("boundary_conditions", []) if b["node"] != nid]


def rename_node(graph: dict, old: str, new: str) -> str:
    new = unique_id(graph, new) if new != old else old
    for n in graph["nodes"]:
        if n["id"] == old:
            n["id"] = new
    for e in graph["edges"]:
        for side in ("from", "to"):
            if e[side]["node"] == old:
                e[side]["node"] = new
        e["id"] = f"{e['from']['node']}.{e['from']['port']}->{e['to']['node']}.{e['to']['port']}"
    for b in graph.get("boundary_conditions", []):
        if b["node"] == old:
            b["node"] = new
    return new


def port(graph: dict, nid: str, pname: str) -> dict:
    n = next(n for n in graph["nodes"] if n["id"] == nid)
    return next(p for p in library()[n["type"]]["ports"] if p["name"] == pname)


def can_connect(graph: dict, a: tuple[str, str], b: tuple[str, str]) -> tuple[bool, str, tuple, tuple]:
    """Whether ports ``a`` and ``b`` may be joined; returns ``(ok, reason, from, to)`` in flow order."""
    if a[0] == b[0]:
        return False, "a component cannot connect to itself", a, b
    pa, pb = port(graph, *a), port(graph, *b)
    if pa["kind"] != pb["kind"]:
        return False, f"{pa['kind']} port cannot join a {pb['kind']} port", a, b
    if pa["kind"] == "fluid":
        if pa["direction"] == pb["direction"]:
            return False, f"both ports are {pa['direction']}lets", a, b
        if pa["direction"] == "in":
            a, b = b, a
    for e in graph["edges"]:
        ends = {(e["from"]["node"], e["from"]["port"]), (e["to"]["node"], e["to"]["port"])}
        if ends == {a, b}:
            return False, "already connected", a, b
    return True, "", a, b


def connect(graph: dict, a: tuple[str, str], b: tuple[str, str]) -> dict:
    ok, why, a, b = can_connect(graph, a, b)
    if not ok:
        raise ValueError(why)
    kind = port(graph, *a)["kind"]
    e = {"id": f"{a[0]}.{a[1]}->{b[0]}.{b[1]}", "from": {"node": a[0], "port": a[1]},
         "to": {"node": b[0], "port": b[1]}, "kind": kind,
         "medium": {"fluid": "combustion_gas", "shaft": "torque", "thermal": "heat"}.get(kind, "heat")}
    graph["edges"].append(e)
    infer_media(graph)
    return e


def disconnect(graph: dict, edge_id: str) -> None:
    graph["edges"] = [e for e in graph["edges"] if e["id"] != edge_id]


def infer_media(graph: dict) -> None:
    """Fluid edge media from the tanks downstream: a propellant until a combustor, combustion gas after it."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    out: dict[str, list[dict]] = {}
    for e in graph["edges"]:
        if e["kind"] == "fluid":
            out.setdefault(e["from"]["node"], []).append(e)
    med: dict[str, str] = {}
    queue = []
    for n in graph["nodes"]:
        if n["type"].startswith("tank"):
            med[n["id"]] = n.get("options", {}).get("fluid") or graph["propellants"]["oxidizer"]
            queue.append(n["id"])
        elif n["type"] in ("chamber", "gas_generator"):
            med[n["id"]] = "combustion_gas"
            queue.append(n["id"])
    while queue:
        a = queue.pop(0)
        for e in out.get(a, []):
            e["medium"] = med[a]
            b = e["to"]["node"]
            if b not in nodes or nodes[b]["type"] in ("chamber", "gas_generator") or b in med:
                continue
            med[b] = med[a]
            queue.append(b)


def unconnected(graph: dict) -> list[str]:
    """Fluid ports left open (the solver needs every one connected or bounded)."""
    lib = library()
    used = {(e[s]["node"], e[s]["port"]) for e in graph["edges"] for s in ("from", "to")}
    bc = {(b["node"], b["port"]) for b in graph.get("boundary_conditions", [])}
    out = []
    for n in graph["nodes"]:
        for p in lib[n["type"]]["ports"]:
            if p["kind"] == "fluid" and p["name"] != "pressurant_in" and (n["id"], p["name"]) not in used | bc:
                out.append(f"{n['id']}.{p['name']}")
    return out


def set_param(node: dict, name: str, value: float | None, unit: str) -> None:
    """A value typed in the sandbox: provenance ``USER`` (estimated, the user's design choice)."""
    if value is None:
        node["params"].pop(name, None)
        return
    node["params"][name] = {"value": float(value), "unit": unit, "quality": "estimated", "source_id": "USER",
                            "location": "cycle sandbox", "formula": "entered by the user"}


def param_value(node: dict, name: str):
    """``(value, provenance text, from_library_default)`` of a parameter of ``node``."""
    if name in node.get("params", {}):
        v = node["params"][name]
        return v["value"], f"{v.get('quality') or ''} · {v.get('source_id') or ''} {v.get('location') or ''}".strip(), False
    for prm in library()[node["type"]]["parameters"]:
        if prm["name"] == name:
            d = prm["default"]
            if d["value"] is None:
                return None, "required" if prm.get("user_input") else "not set", True
            return d["value"], f"library default · {d['source_id']} {d['location']}", True
    return None, "", True


def set_ambient(graph: dict, pa: float) -> None:
    for n in graph["nodes"]:
        if n["type"] == "nozzle_bell":
            graph["boundary_conditions"] = [b for b in graph["boundary_conditions"]
                                            if not (b["node"] == n["id"] and b["kind"] == "ambient_pressure")]
            graph["boundary_conditions"].append({"node": n["id"], "port": "inlet", "kind": "ambient_pressure",
                                                 "value": {"value": float(pa), "unit": "Pa", "quality": "estimated",
                                                           "source_id": "USER", "location": "cycle sandbox"}})


def set_exhaust(graph: dict, nid: str, pname: str, p: float | None) -> None:
    """Fixed-pressure exhaust on an open fluid port (turbine exhaust dumped overboard); ``None`` removes it."""
    graph["boundary_conditions"] = [b for b in graph["boundary_conditions"]
                                    if not (b["node"] == nid and b["port"] == pname and b["kind"] == "fixed_pressure")]
    if p is not None:
        graph["boundary_conditions"].append({"node": nid, "port": pname, "kind": "fixed_pressure",
                                             "value": {"value": float(p), "unit": "Pa", "quality": "estimated",
                                                       "source_id": "USER", "location": "cycle sandbox"}})


# --------------------------------------------------------------------------------------------- templates and files
TEMPLATE_CONTROLS = {
    "ssme_staged_combustion": (
        [{"node": "chamber", "output": "p_c", "value": 20477400.0},
         {"node": "chamber", "output": "mixture_ratio", "value": 6.0}],
        [{"node": "opov", "param": "position"}, {"node": "fpov", "param": "position"}]),
}


def templates() -> dict[str, str]:
    from ...data import graphs
    return {gid: g["name"] for gid, g in graphs().items()}


def template(gid: str) -> dict:
    """Sandbox document of a library example: the graph, plus the controller targets the engine runs with."""
    from ...data import graphs
    g = copy.deepcopy(graphs()[gid])
    for n in g["nodes"]:
        if n["type"].startswith("tank"):
            n.setdefault("options", {})
            n["options"].setdefault("fluid", _tank_fluid(g, n["id"]))
    if overlapping(g):
        auto_layout(g)
    targets, free = TEMPLATE_CONTROLS.get(gid, ([], []))
    return {"format": FORMAT, "version": 1, "graph": g, "targets": copy.deepcopy(targets), "free": copy.deepcopy(free)}


def _tank_fluid(g: dict, nid: str) -> str:
    for e in g["edges"]:
        if e["from"]["node"] == nid and e.get("medium") not in (None, "combustion_gas"):
            return e["medium"]
    return g["propellants"]["fuel" if "fuel" in nid else "oxidizer"]


def new_document(ox: str = "LOX", fuel: str = "RP1") -> dict:
    return {"format": FORMAT, "version": 1, "graph": blank_graph(ox, fuel), "targets": [], "free": []}


def save(doc: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(doc, indent=1), encoding="utf-8")


def load(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if "nodes" in d and "edges" in d:                         # a bare library graph
        d = {"format": FORMAT, "version": 1, "graph": d, "targets": [], "free": []}
    if d.get("format") != FORMAT:
        raise ValueError("not a PropSim engine sandbox file")
    return d


def node_height(ntype: str) -> float:
    ports = library()[ntype]["ports"]
    n_in = sum(p["kind"] == "fluid" and p["direction"] == "in" for p in ports)
    n_out = sum(p["kind"] == "fluid" and p["direction"] == "out" for p in ports)
    return max(58.0, 30.0 + 16.0 * max(n_in, n_out, 1)) + 30.0


def auto_layout(graph: dict, dx: float = 230.0, gap: float = 34.0) -> None:
    """Left-to-right flow diagram: columns by the longest fluid path from the tanks (loops broken), rows ordered by
    the mean row of each component's upstream neighbours; shafts sit under the machines they join."""
    nodes = {n["id"]: n for n in graph["nodes"]}
    succ: dict[str, set[str]] = {k: set() for k in nodes}
    pred: dict[str, set[str]] = {k: set() for k in nodes}
    for e in graph["edges"]:
        if e["kind"] == "fluid":
            a, b = e["from"]["node"], e["to"]["node"]
            succ[a].add(b)
            pred[b].add(a)
    rank = {k: 0 for k in nodes}
    # depth-first order to find back edges (recirculation loops) and ignore them for ranking
    state, back = {}, set()

    def dfs(u):
        state[u] = 1
        for v in sorted(succ[u]):
            if state.get(v) == 1:
                back.add((u, v))
            elif v not in state:
                dfs(v)
        state[u] = 2
    starts = [k for k in nodes if not pred[k]] + list(nodes)
    for k in starts:
        if k not in state:
            dfs(k)
    for _ in range(len(nodes)):
        changed = False
        for a in nodes:
            for b in succ[a]:
                if (a, b) not in back and rank[b] < rank[a] + 1:
                    rank[b] = rank[a] + 1
                    changed = True
        if not changed:
            break
    shaft_of: dict[str, list[str]] = {}
    for e in graph["edges"]:
        if e["kind"] == "shaft":
            a, b = e["from"]["node"], e["to"]["node"]
            shaft_of.setdefault(a, []).append(b)
            shaft_of.setdefault(b, []).append(a)
    for k, n in nodes.items():
        if n["type"] == "shaft" and shaft_of.get(k):
            rank[k] = max(rank[m] for m in shaft_of[k] if nodes[m]["type"] != "shaft") if any(
                nodes[m]["type"] != "shaft" for m in shaft_of[k]) else rank[k]
    cols: dict[int, list[str]] = {}
    for k in nodes:
        cols.setdefault(rank[k], []).append(k)
    y_of = {k: float(nodes[k].get("ui", {}).get("y", 0.0)) for k in nodes}
    for r in sorted(cols):
        def key(k):
            ps = [p for p in pred[k] if rank[p] < r] or [m for m in shaft_of.get(k, [])]
            return (sum(y_of[p] for p in ps) / len(ps) if ps else y_of[k], y_of[k])
        cols[r].sort(key=key)
        y = 0.0
        for k in cols[r]:
            y_of[k] = y
            nodes[k]["ui"] = {"x": r * dx, "y": y}
            y += node_height(nodes[k]["type"]) + gap


def overlapping(graph: dict) -> bool:
    boxes = [(float(n.get("ui", {}).get("x", 0)), float(n.get("ui", {}).get("y", 0)), node_height(n["type"]))
             for n in graph["nodes"]]
    for i, (x1, y1, h1) in enumerate(boxes):
        for x2, y2, h2 in boxes[i + 1:]:
            if abs(x1 - x2) < 175 and (y1 < y2 + h2 and y2 < y1 + h1):
                return True
    return False
