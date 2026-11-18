"""Component and property library: sourced, machine-readable data files under ``propsim/data/library``.

Every number in the library carries its provenance (``quality``, ``source_id``, ``location``; see
``library/README.md``). This module reads the files and checks their integrity; it does no physics.

    >>> from propsim import data
    >>> data.fluid_table("LOX")["rho"][:3]
"""

from __future__ import annotations

import csv
import json
from functools import cache
from pathlib import Path

import numpy as np

LIBRARY = Path(__file__).resolve().parent / "library"
FLUIDS = LIBRARY / "fluids"
COMPONENTS = LIBRARY / "components"
ENGINES = LIBRARY / "engines"
CYCLES = LIBRARY / "cycles"
CONDITIONS = {"sl", "vac", "n/a", "unstated"}
EXAMPLES = LIBRARY / "examples"
DASHBOARDS = LIBRARY / "dashboards"
MATERIALS = LIBRARY / "materials"
HYBRID = LIBRARY / "hybrid"
ELECTRIC = LIBRARY / "electric"
G0 = 9.80665  # standard gravity, m/s2 (exact by definition)
FIT_FORMS = {"log10_poly", "poly_floor", "cu_rational"}
MATERIAL_UNITS = {
    "k": "W/(m K)",
    "cp": "J/(kg K)",
    "E": "Pa",
    "thermal_strain": "1",
    "alpha": "1/K",
    "poisson": "1",
    "yield_strength": "Pa",
    "ultimate_strength": "Pa",
    "shear_modulus": "Pa",
    "density": "kg/m3",
    "T_solidus": "K",
    "T_liquidus": "K",
    "alpha_mean": "1/K",
}
# valve timing parameters may come from the graph sequence instead of the node
SEQUENCED = {"t_open", "ramp_open", "t_close", "ramp_close"}

QUALITIES = {"published", "derived", "estimated"}
# SI unit strings allowed in ``unit`` columns. "1" marks a dimensionless value.
UNITS = {
    "N m",
    "W/m2",
    "m3/s",
    "kg",
    "1",
    "K",
    "Pa",
    "Pa s",
    "kg/m3",
    "kg/mol",
    "kg/s",
    "J/mol",
    "J/kg",
    "J/(kg K)",
    "W/(m K)",
    "m/s",
    "N/m",
    "s",
    "m",
    "m2",
    "m3",
    "N",
    "W",
    "rad/s",
    "kg m2",
    "1/K",
    "W/(m2 K)",
}
TABLE_COLUMNS = ["T", "p", "phase", "rho", "mu", "cp", "cv", "k", "a_sound", "h", "s"]
SAT_COLUMNS = ["T", "p_sat", "rho_l", "rho_v", "h_l", "h_v", "cp_l", "mu_l", "k_l", "sigma"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@cache
def sources() -> dict[str, dict[str, str]]:
    """``source_id`` -> row of ``sources.csv``."""
    return {r["source_id"]: r for r in _read_csv(LIBRARY / "sources.csv")}


@cache
def fluids() -> dict[str, dict[str, str]]:
    """``fluid_id`` -> row of ``fluids/fluids.csv`` (identity, property backend and table provenance)."""
    return {r["fluid_id"]: r for r in _read_csv(FLUIDS / "fluids.csv")}


def fluid_scalars(fluid_id: str | None = None, preferred_only: bool = False) -> list[dict[str, str]]:
    """Rows of ``fluid_scalars.csv`` (one sourced scalar per row), optionally one fluid / preferred only."""
    rows = _read_csv(FLUIDS / "fluid_scalars.csv")
    return [
        r
        for r in rows
        if (fluid_id is None or r["fluid_id"] == fluid_id) and (not preferred_only or r["preferred"] == "1")
    ]


def scalar(fluid_id: str, prop: str) -> float:
    """The preferred value of one scalar property (SI). Raises ``KeyError`` when the library has none."""
    for r in fluid_scalars(fluid_id, preferred_only=True):
        if r["property"] == prop and r["value"] != "":
            return float(r["value"])
    raise KeyError(f"{fluid_id}.{prop} is not in the library (see gaps.md)")


def _numeric_table(path: Path, text_cols: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    rows = _read_csv(path)
    out: dict[str, np.ndarray] = {}
    for c in rows[0].keys():
        if c in text_cols:
            out[c] = np.array([r[c] for r in rows])
        else:
            out[c] = np.array([float(r[c]) if r[c] != "" else np.nan for r in rows])
    return out


@cache
def fluid_table(fluid_id: str) -> dict[str, np.ndarray]:
    """Single-phase (T, p) grid of a CoolProp fluid as column arrays; missing cells are NaN."""
    f = fluids()[fluid_id]
    if not f["table_file"]:
        raise KeyError(f"{fluid_id} has no generated table (model={f['model']!r})")
    return _numeric_table(FLUIDS / f["table_file"], text_cols=("phase",))


@cache
def sat_table(fluid_id: str) -> dict[str, np.ndarray]:
    """Saturation line of a pure CoolProp fluid as column arrays."""
    f = fluids()[fluid_id]
    if not f["sat_file"]:
        raise KeyError(f"{fluid_id} has no saturation table")
    return _numeric_table(FLUIDS / f["sat_file"])


def cea_species() -> dict[str, dict[str, str]]:
    return {r["species_id"]: r for r in _read_csv(FLUIDS / "cea_species.csv")}


def cea_mixtures() -> dict[str, dict[str, str]]:
    return {r["mixture_id"]: r for r in _read_csv(FLUIDS / "cea_mixtures.csv")}


def cea_validation() -> list[dict[str, str]]:
    return _read_csv(FLUIDS / "cea_validation.csv")


@cache
def components() -> dict[str, dict]:
    """``type`` -> component record (``components/<type>.json``, schema ``schemas/component.schema.json``)."""
    out = {}
    for p in sorted(COMPONENTS.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        out[rec["type"]] = rec
    return out


def component_table(name: str) -> list[dict[str, str]]:
    """Rows of ``components/tables/<name>.csv``."""
    return _read_csv(COMPONENTS / "tables" / f"{name}.csv")


@cache
def engines() -> dict[str, dict[str, str]]:
    """``engine_id`` -> row of ``engines/engines.csv`` (identity, cycle and propellants)."""
    return {r["engine_id"]: r for r in _read_csv(ENGINES / "engines.csv")}


def engine_params(engine_id: str | None = None, preferred_only: bool = False) -> list[dict[str, str]]:
    """Rows of ``engines/engine_params.csv`` (one sourced value per row), optionally filtered."""
    return [
        r
        for r in _read_csv(ENGINES / "engine_params.csv")
        if (engine_id is None or r["engine_id"] == engine_id)
        and (not preferred_only or r["preferred"] == "1")
    ]


def engine_value(engine_id: str, subsystem: str, prop: str, condition: str | None = None) -> float:
    """Preferred SI value of one parameter, e.g. ``engine_value("HH_A1", "thrust_chamber", "cstar")``."""
    for r in engine_params(engine_id, preferred_only=True):
        if r["subsystem"] == subsystem and r["property"] == prop and condition in (None, r["condition"]):
            return float(r["value"])
    raise KeyError(f"{engine_id}.{subsystem}.{prop} is not in the library (see gaps.md)")


@cache
def cycles() -> dict[str, dict]:
    """``cycle_id`` -> cycle record of ``cycles/cycles.json`` (schema ``schemas/cycle.schema.json``)."""
    return {
        c["cycle_id"]: c for c in json.loads((CYCLES / "cycles.json").read_text(encoding="utf-8"))["cycles"]
    }


def cycle_table(name: str) -> list[dict[str, str]]:
    """Rows of ``cycles/<name>.csv``."""
    return _read_csv(CYCLES / f"{name}.csv")


@cache
def graphs() -> dict[str, dict]:
    """``graph_id`` -> example engine graph (``examples/*.graph.json``, ``schemas/graph.schema.json``)."""
    out = {}
    for p in sorted(EXAMPLES.glob("*.graph.json")):
        g = json.loads(p.read_text(encoding="utf-8"))
        out[g["graph_id"]] = g
    return out


@cache
def default_panels() -> dict[str, list[dict]]:
    """Component type -> standard panels (``dashboards/default_panels.json``); ``$node`` is a placeholder."""
    return json.loads((DASHBOARDS / "default_panels.json").read_text(encoding="utf-8"))["panels_by_type"]


@cache
def materials() -> dict[str, dict[str, str]]:
    """``material_id`` -> row of ``materials/materials.csv``."""
    return {r["material_id"]: r for r in _read_csv(MATERIALS / "materials.csv")}


@cache
def material_fits() -> list[dict[str, str]]:
    """Rows of ``materials/property_fits.csv``; evaluate with ``propsim.data.matprops.material_property``."""
    return _read_csv(MATERIALS / "property_fits.csv")


def material_points(material_id: str | None = None) -> list[dict[str, str]]:
    """Rows of ``materials/material_points.csv`` (single published values with a temperature range)."""
    return [
        r for r in _read_csv(MATERIALS / "material_points.csv") if material_id in (None, r["material_id"])
    ]


def hybrid_fuels() -> dict[str, dict[str, str]]:
    """``fuel_id`` -> row of ``hybrid/hybrid_fuels.csv``."""
    return {r["fuel_id"]: r for r in _read_csv(HYBRID / "hybrid_fuels.csv")}


def regression_laws() -> dict[str, dict[str, str]]:
    """``law_id`` -> row of ``hybrid/regression_laws.csv``; ``a`` is SI (r in m/s with G in kg/(m2 s))."""
    return {r["law_id"]: r for r in _read_csv(HYBRID / "regression_laws.csv")}


def regression_rate(law_id: str, G_ox: float) -> float:
    """Fuel regression rate (m/s) from a library law at oxidizer mass flux ``G_ox`` (kg/(m2 s))."""
    r = regression_laws()[law_id]
    return float(r["a"]) * G_ox ** float(r["n"])


def hotfire_cases() -> list[dict[str, str]]:
    """Rows of ``hybrid/hotfire_cases.csv`` (published hybrid motor tests, SI)."""
    return _read_csv(HYBRID / "hotfire_cases.csv")


def _check_hybrid(problems: list[str]) -> None:
    fuels = hybrid_fuels()
    oxid = set(fluids()) | {"GOX"}
    for name, rows in (
        ("regression_laws.csv", list(regression_laws().values())),
        ("hotfire_cases.csv", hotfire_cases()),
    ):
        _check_provenance(rows, f"hybrid/{name}", problems, unit_col=None, value_col=None)
        for i, r in enumerate(rows, start=2):
            if r["fuel"] not in fuels:
                problems.append(f"hybrid/{name}:{i}: unknown fuel {r['fuel']!r}")
            if r["oxidizer"] not in oxid:
                problems.append(f"hybrid/{name}:{i}: unknown oxidizer {r['oxidizer']!r}")
    for fid, f in fuels.items():
        if f["source_id"] not in sources():
            problems.append(f"hybrid/hybrid_fuels.csv {fid}: unknown source {f['source_id']!r}")


def thrusters() -> dict[str, dict[str, str]]:
    """``thruster_id`` -> row of ``electric/thrusters.csv``."""
    return {r["thruster_id"]: r for r in _read_csv(ELECTRIC / "thrusters.csv")}


def throttle_table(thruster_id: str | None = None, table: str | None = None) -> list[dict[str, str]]:
    """Rows of ``electric/throttle_tables.csv`` (SI), optionally for one thruster and one table."""
    rows = _read_csv(ELECTRIC / "throttle_tables.csv")
    return [
        r
        for r in rows
        if (thruster_id is None or r["thruster_id"] == thruster_id) and (table is None or r["table"] == table)
    ]


def thruster_power(row: dict[str, str]) -> float:
    """Electrical power into the thruster (W): printed, or PPU input times PPU efficiency."""
    if row["P_in_thruster"]:
        return float(row["P_in_thruster"])
    return float(row["P_in_ppu"]) * float(row["eta_ppu"])


def _check_electric(problems: list[str]) -> None:
    th = thrusters()
    rows = throttle_table()
    _check_provenance(rows, "electric/throttle_tables.csv", problems, unit_col=None, value_col=None)
    for tid, t in th.items():
        if t["source_id"] not in sources():
            problems.append(f"electric/thrusters.csv {tid}: unknown source {t['source_id']!r}")
    for i, r in enumerate(rows, start=2):
        tag = f"electric/throttle_tables.csv:{i}"
        if r["thruster_id"] not in th:
            problems.append(f"{tag}: unknown thruster {r['thruster_id']!r}")
            continue
        if not (r["P_in_thruster"] or (r["P_in_ppu"] and r["eta_ppu"])):
            problems.append(f"{tag}: no thruster or PPU input power")
            continue
        m, F, isp = float(r["mdot"]), float(r["thrust"]), float(r["Isp"])
        if abs(m * isp * G0 / F - 1) > 0.01:
            problems.append(f"{tag}: thrust differs from mdot Isp g0 by more than 1 %")


def _check_materials(problems: list[str]) -> None:
    mats = materials()
    fits = material_fits()
    _check_provenance(fits, "materials/property_fits.csv", problems, value_col=None)
    pts = material_points()
    _check_provenance(pts, "materials/material_points.csv", problems)
    for name, rows in (("property_fits.csv", fits), ("material_points.csv", pts)):
        for i, r in enumerate(rows, start=2):
            tag = f"materials/{name}:{i}"
            if r["material_id"] not in mats:
                problems.append(f"{tag}: unknown material {r['material_id']!r}")
            if MATERIAL_UNITS.get(r["property"]) != r["unit"]:
                problems.append(f"{tag}: property {r['property']!r} with unit {r['unit']!r}")
            if r["T_min"] and r["T_max"] and float(r["T_min"]) > float(r["T_max"]):
                problems.append(f"{tag}: T_min > T_max")
    for i, r in enumerate(fits, start=2):
        if r["form"] not in FIT_FORMS:
            problems.append(f"materials/property_fits.csv:{i}: unknown form {r['form']!r}")


def _check_sv(v: dict, unit: str | None, tag: str, problems: list[str]) -> None:
    if unit is not None and v["unit"] != unit:
        problems.append(f"{tag}: unit {v['unit']!r}, expected {unit!r}")
    if v["unit"] not in UNITS:
        problems.append(f"{tag}: unit {v['unit']!r} not allowed")
    if v["value"] is None:
        problems.append(f"{tag}: value is null")
        return
    if v["quality"] not in QUALITIES or v["source_id"] not in sources() or not v["location"]:
        problems.append(f"{tag}: incomplete provenance")
    if v["quality"] in ("derived", "estimated") and not v.get("formula"):
        problems.append(f"{tag}: {v['quality']} value without formula")


def _check_graph(g: dict, problems: list[str]) -> None:
    where = f"examples/{g['graph_id']}.graph.json"
    comps = components()
    fl = fluids()
    nodes = {n["id"]: n for n in g["nodes"]}
    if len(nodes) != len(g["nodes"]):
        problems.append(f"{where}: duplicate node ids")
    if g["cycle"] not in cycles():
        problems.append(f"{where}: unknown cycle {g['cycle']!r}")
    if g.get("reference_engine") and g["reference_engine"] not in engines():
        problems.append(f"{where}: unknown reference engine {g['reference_engine']!r}")
    for role, fid in g["propellants"].items():
        if fid not in fl:
            problems.append(f"{where}: {role} {fid!r} not in fluids.csv")
    sequenced = {s["node"] for s in g["sequence"]}

    def ports(nid: str) -> dict[str, dict]:
        return {p["name"]: p for p in comps[nodes[nid]["type"]]["ports"]}

    def outputs(nid: str) -> set[str]:
        return {o["name"] for o in comps[nodes[nid]["type"]]["outputs_for_plots"]}

    for nid, n in nodes.items():
        if n["type"] not in comps:
            problems.append(f"{where} {nid}: unknown component type {n['type']!r}")
            continue
        spec = {p["name"]: p for p in comps[n["type"]]["parameters"]}
        for name, v in n["params"].items():
            if name not in spec:
                problems.append(f"{where} {nid}: {n['type']} has no parameter {name!r}")
                continue
            _check_sv(v, spec[name]["unit"], f"{where} {nid}.{name}", problems)
            lo, hi = spec[name].get("min"), spec[name].get("max")
            if v["value"] is not None and (
                (lo is not None and v["value"] < lo) or (hi is not None and v["value"] > hi)
            ):
                problems.append(f"{where} {nid}.{name} = {v['value']} outside [{lo}, {hi}]")
        for name, prm in spec.items():
            if prm.get("user_input") and name not in n["params"]:
                if not (name in SEQUENCED and nid in sequenced):
                    problems.append(f"{where} {nid}: user-input parameter {name!r} missing")
    connected = set()
    for e in g["edges"]:
        ends = []
        for side in ("from", "to"):
            nid, port = e[side]["node"], e[side]["port"]
            if nid not in nodes or nodes[nid]["type"] not in comps:
                problems.append(f"{where} edge {e['id']}: unknown node {nid!r}")
                break
            p = ports(nid).get(port)
            if p is None:
                problems.append(f"{where} edge {e['id']}: {nid} has no port {port!r}")
                break
            ends.append(p)
            connected.add((nid, port))
        if len(ends) != 2:
            continue
        a, b = ends
        if a["kind"] != e["kind"] or b["kind"] != e["kind"]:
            problems.append(f"{where} edge {e['id']}: port kinds {a['kind']}/{b['kind']} vs edge {e['kind']}")
        if e["kind"] == "fluid":
            if a["direction"] == "in" or b["direction"] == "out":
                problems.append(f"{where} edge {e['id']}: flows against port directions")
            if e["medium"] not in fl and e["medium"] != "combustion_gas":
                problems.append(f"{where} edge {e['id']}: unknown medium {e['medium']!r}")
    for bc in g["boundary_conditions"]:
        tag = f"{where} boundary {bc['node']}.{bc['port']}"
        if bc["node"] not in nodes or bc["port"] not in ports(bc["node"]):
            problems.append(f"{tag}: unknown node or port")
            continue
        _check_sv(
            bc["value"],
            "K" if bc["kind"] == "fixed_temperature" else "kg/s" if bc["kind"] == "fixed_mass_flow" else "Pa",
            tag,
            problems,
        )
    for nid, n in nodes.items():
        for p in comps[n["type"]]["ports"]:
            if (
                p["kind"] == "fluid"
                and (nid, p["name"]) not in connected
                and not any(bc["node"] == nid and bc["port"] == p["name"] for bc in g["boundary_conditions"])
            ):
                if p["name"] != "pressurant_in":
                    problems.append(f"{where} {nid}.{p['name']}: fluid port neither connected nor bounded")
    for i, st in enumerate(g["sequence"]):
        tag = f"{where} sequence[{i}]"
        if st["node"] != "*" and st["node"] not in nodes:
            problems.append(f"{tag}: unknown node {st['node']!r}")
        for k in ("t", "value", "ramp"):
            if k in st:
                _check_sv(st[k], "s" if k != "value" else None, f"{tag}.{k}", problems)
        if "when" in st:
            w = st["when"]
            if w["node"] not in nodes or w["output"] not in outputs(w["node"]):
                problems.append(f"{tag}: trigger on unknown output {w['node']}.{w['output']}")
            else:
                unit = next(
                    o["unit"]
                    for o in comps[nodes[w["node"]]["type"]]["outputs_for_plots"]
                    if o["name"] == w["output"]
                )
                _check_sv(w["value"], unit, f"{tag}.when", problems)
    for i, vt in enumerate(g.get("validation_targets", [])):
        tag = f"{where} validation_targets[{i}]"
        if vt["node"] not in nodes or vt["output"] not in outputs(vt["node"]):
            problems.append(f"{tag}: unknown output {vt['node']}.{vt['output']}")
            continue
        unit = next(
            o["unit"]
            for o in comps[nodes[vt["node"]]["type"]]["outputs_for_plots"]
            if o["name"] == vt["output"]
        )
        _check_sv(vt["value"], unit, tag, problems)
    for k, v in g.get("design_point", {}).items():
        _check_sv(v, None, f"{where} design_point.{k}", problems)


def _check_panels(problems: list[str]) -> None:
    comps = components()
    for t, panels in default_panels().items():
        if t not in comps:
            problems.append(f"default_panels.json: unknown component type {t!r}")
            continue
        outs = {o["name"]: o["unit"] for o in comps[t]["outputs_for_plots"]}
        for pn in panels:
            for s in pn.get("y_left", []) + pn.get("y_right", []):
                if s["output"] not in outs:
                    problems.append(f"default_panels.json {pn['id']}: {t} has no output {s['output']!r}")
                elif s["unit"] != outs[s["output"]]:
                    problems.append(f"default_panels.json {pn['id']}: {s['output']} unit {s['unit']!r}")
    missing = set(comps) - set(default_panels())
    if missing:
        problems.append(f"default_panels.json: no panels for {sorted(missing)}")


def _check_component(rec: dict, where: str, problems: list[str]) -> None:
    src = sources()

    def value(v: dict, tag: str) -> None:
        if v["unit"] not in UNITS:
            problems.append(f"{tag}: unit {v['unit']!r} not allowed")
        if v["value"] is None:
            return
        if v["quality"] not in QUALITIES or v["source_id"] not in src or not v["location"]:
            problems.append(f"{tag}: default value without complete provenance")
        if v["quality"] in ("derived", "estimated") and not v.get("formula"):
            problems.append(f"{tag}: {v['quality']} default without formula")

    for prm in rec["parameters"]:
        tag = f"{where} parameter {prm['name']}"
        if prm["unit"] != prm["default"]["unit"]:
            problems.append(f"{tag}: unit differs from its default's unit")
        value(prm["default"], tag)
        if prm.get("table") and not (LIBRARY / prm["table"]).exists():
            problems.append(f"{tag}: table {prm['table']} missing")
    for item in rec["state_variables"] + rec["outputs_for_plots"]:
        if item["unit"] not in UNITS:
            problems.append(f"{where} {item['name']}: unit {item['unit']!r} not allowed")
    for e in rec["governing_equations"]:
        if e["source_id"] not in src:
            problems.append(f"{where} equation {e['name']!r}: unknown source_id {e['source_id']!r}")
    names = [p["name"] for p in rec["ports"]]
    if len(set(names)) != len(names):
        problems.append(f"{where}: duplicate port names")


def _check_provenance(
    rows: list[dict[str, str]],
    where: str,
    problems: list[str],
    unit_col: str | None = "unit",
    value_col: str | None = "value",
) -> None:
    src = sources()
    for i, r in enumerate(rows, start=2):  # line 1 is the header
        tag = f"{where}:{i}"
        if r["quality"] not in QUALITIES:
            problems.append(f"{tag}: quality {r['quality']!r} not in {sorted(QUALITIES)}")
        if r["source_id"] not in src:
            problems.append(f"{tag}: source_id {r['source_id']!r} missing from sources.csv")
        if not r["location"]:
            problems.append(f"{tag}: empty location")
        if r["quality"] in ("derived", "estimated") and not r.get("formula"):
            problems.append(f"{tag}: {r['quality']} value without formula")
        if unit_col and r[unit_col] not in UNITS:
            problems.append(f"{tag}: unit {r[unit_col]!r} is not an allowed SI unit string")
        if value_col and r[value_col] != "":
            try:
                float(r[value_col])
            except ValueError:
                problems.append(f"{tag}: value {r[value_col]!r} is not a number")


def validate() -> list[str]:
    """Integrity problems of the library files (empty list when clean). Used by the test suite."""
    problems: list[str] = []
    fl = fluids()
    for fid, f in fl.items():
        if f["model"] not in ("coolprop", "rocketprops", "none"):
            problems.append(f"fluids.csv {fid}: unknown model {f['model']!r}")
        if f["model"] == "coolprop":
            if not f["table_file"] or not (FLUIDS / f["table_file"]).exists():
                problems.append(f"fluids.csv {fid}: CoolProp fluid without a generated table")
            elif list(_read_csv(FLUIDS / f["table_file"])[0].keys()) != TABLE_COLUMNS:
                problems.append(f"{f['table_file']}: columns differ from {TABLE_COLUMNS}")
            if f["table_source_id"] not in sources():
                problems.append(
                    f"fluids.csv {fid}: table_source_id {f['table_source_id']!r} not in sources.csv"
                )
        if f["sat_file"] and list(_read_csv(FLUIDS / f["sat_file"])[0].keys()) != SAT_COLUMNS:
            problems.append(f"{f['sat_file']}: columns differ from {SAT_COLUMNS}")

    sc = fluid_scalars()
    _check_provenance(sc, "fluid_scalars.csv", problems)
    for i, r in enumerate(sc, start=2):
        if r["fluid_id"] not in fl:
            problems.append(f"fluid_scalars.csv:{i}: unknown fluid_id {r['fluid_id']!r}")
    seen: dict[tuple[str, str], int] = {}
    for r in sc:
        if r["preferred"] == "1":
            key = (r["fluid_id"], r["property"])
            seen[key] = seen.get(key, 0) + 1
    problems += [f"fluid_scalars.csv: {k[0]}.{k[1]} has {n} preferred rows" for k, n in seen.items() if n > 1]

    sp = cea_species()
    _check_provenance(list(sp.values()), "cea_species.csv", problems, unit_col="dHf_unit", value_col="dHf")
    for mid, m in cea_mixtures().items():
        parts = [c.split(":") for c in m["components"].split(";")]
        total = sum(float(x) for _, x in parts)
        if abs(total - 1.0) > 1e-9:
            problems.append(f"cea_mixtures.csv {mid}: fractions sum to {total}")
        problems += [f"cea_mixtures.csv {mid}: unknown species {s!r}" for s, _ in parts if s not in sp]
        if mid not in fl:
            problems.append(f"cea_mixtures.csv {mid}: no fluids.csv entry")
    _check_provenance(cea_validation(), "cea_validation.csv", problems, unit_col=None, value_col=None)

    for t, rec in components().items():
        _check_component(rec, f"components/{t}.json", problems)
    for p in sorted((COMPONENTS / "tables").glob("*.csv")):
        rows = _read_csv(p)
        unit_col = "unit" if "unit" in rows[0] else None
        _check_provenance(rows, f"components/tables/{p.name}", problems, unit_col=unit_col, value_col=None)

    eng = engines()
    pr = engine_params()
    _check_provenance(pr, "engines/engine_params.csv", problems)
    count: dict[tuple[str, str, str, str], int] = {}
    for i, r in enumerate(pr, start=2):
        if r["engine_id"] not in eng:
            problems.append(f"engines/engine_params.csv:{i}: unknown engine_id {r['engine_id']!r}")
        if r["condition"] not in CONDITIONS:
            problems.append(
                f"engines/engine_params.csv:{i}: condition {r['condition']!r} not in {sorted(CONDITIONS)}"
            )
        if r["preferred"] == "1":
            key = (r["engine_id"], r["subsystem"], r["property"], r["condition"])
            count[key] = count.get(key, 0) + 1
    problems += [
        f"engine_params.csv: {'.'.join(k)} has {n} preferred rows" for k, n in count.items() if n > 1
    ]
    cyc = cycles()
    for eid, e in eng.items():
        if e["cycle"] and (e["cycle"] not in cyc or e["cycle_source_id"] not in sources()):
            problems.append(f"engines.csv {eid}: cycle {e['cycle']!r} unknown or without source")
    for cid, c in cyc.items():
        problems += [
            f"cycles.json {cid}: unknown engine {x!r}" for x in c["reference_engines"] if x not in eng
        ]
        problems += [
            f"cycles.json {cid}: unknown component type {x!r}"
            for x in c["component_types"]
            if x not in components()
        ]
        for k in ("cycle_efficiency_min", "cycle_efficiency_max", "turbine_inlet_temperature_max"):
            if c.get(k):
                if c[k]["unit"] not in UNITS or c[k]["source_id"] not in sources() or not c[k]["location"]:
                    problems.append(f"cycles.json {cid}.{k}: incomplete provenance")
        problems += [
            f"cycles.json {cid}: equation source {e['source_id']!r} unknown"
            for e in c["governing_equations"]
            if e["source_id"] not in sources()
        ]
    for p in sorted(CYCLES.glob("*.csv")):
        _check_provenance(_read_csv(p), f"cycles/{p.name}", problems, unit_col=None, value_col=None)

    for g in graphs().values():
        _check_graph(g, problems)
    _check_panels(problems)
    _check_materials(problems)
    _check_hybrid(problems)
    _check_electric(problems)

    for p in LIBRARY.rglob("*.json"):
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            problems.append(f"{p.relative_to(LIBRARY)}: invalid JSON ({e})")
    return problems
