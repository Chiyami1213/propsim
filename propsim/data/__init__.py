"""Component and property library: sourced, machine-readable data files under ``propsim/data/library``.

Every number in the library carries its provenance (``quality``, ``source_id``, ``location``; see
``library/README.md``). This module reads the files and checks their integrity; it does no physics.

    >>> from propsim import data
    >>> data.fluid_table("LOX")["rho"][:3]
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

LIBRARY = Path(__file__).resolve().parent / "library"
FLUIDS = LIBRARY / "fluids"

QUALITIES = {"published", "derived", "estimated"}
# SI unit strings allowed in ``unit`` columns. "1" marks a dimensionless value.
UNITS = {
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
}
TABLE_COLUMNS = ["T", "p", "phase", "rho", "mu", "cp", "cv", "k", "a_sound", "h", "s"]
SAT_COLUMNS = ["T", "p_sat", "rho_l", "rho_v", "h_l", "h_v", "cp_l", "mu_l", "k_l", "sigma"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=None)
def sources() -> dict[str, dict[str, str]]:
    """``source_id`` -> row of ``sources.csv``."""
    return {r["source_id"]: r for r in _read_csv(LIBRARY / "sources.csv")}


@lru_cache(maxsize=None)
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


@lru_cache(maxsize=None)
def fluid_table(fluid_id: str) -> dict[str, np.ndarray]:
    """Single-phase (T, p) grid of a CoolProp fluid as column arrays; missing cells are NaN."""
    f = fluids()[fluid_id]
    if not f["table_file"]:
        raise KeyError(f"{fluid_id} has no generated table (model={f['model']!r})")
    return _numeric_table(FLUIDS / f["table_file"], text_cols=("phase",))


@lru_cache(maxsize=None)
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

    for p in LIBRARY.rglob("*.json"):
        try:
            json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            problems.append(f"{p.relative_to(LIBRARY)}: invalid JSON ({e})")
    return problems
