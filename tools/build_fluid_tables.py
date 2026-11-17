"""Generate the CoolProp fluid tables of the component library (``propsim/data/library/fluids``).

Run from the repository root::

    python tools/build_fluid_tables.py

Writes, for every fluid whose ``model`` is ``coolprop`` in ``fluids.csv``:

* ``tables/<fluid_id>.csv`` - single-phase grid over (T, p):
  ``T, p, phase, rho, mu, cp, cv, k, a_sound, h, s``
* ``sat/<fluid_id>.csv``    - saturation line (pure fluids only):
  ``T, p_sat, rho_l, rho_v, h_l, h_v, cp_l, mu_l, k_l, sigma``

and rewrites the CoolProp-derived rows of ``fluid_scalars.csv`` (critical point, triple point, normal boiling
point, molar mass). All values are SI (K, Pa, kg/m3, Pa s, J/(kg K), W/(m K), m/s, J/kg, N/m). A cell CoolProp
cannot evaluate (no transport model, too close to the critical point) is left empty, never filled in.

The CoolProp version and the equation-of-state references of each fluid are written to ``fluids.csv``
(``table_source_id``, ``table_location``), so every generated number is traceable to its EOS paper.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

try:
    import CoolProp
    import CoolProp.CoolProp as CP
except ImportError:  # pragma: no cover
    sys.exit("CoolProp is required: pip install CoolProp")

ROOT = Path(__file__).resolve().parents[1] / "propsim" / "data" / "library" / "fluids"
P_GRID = [0.1e6, 0.2e6, 0.5e6, 1e6, 2e6, 3e6, 5e6, 7e6, 10e6, 15e6, 20e6, 25e6, 30e6, 40e6]  # (Pa)
N_T_COLD = 50  # temperatures from T_min to 1.3 T_crit (liquid and near-critical region, finely spaced)
N_T_HOT = 20  # temperatures from 1.3 T_crit to T_max (gas region)
N_SAT = 40  # points on the saturation line
SRC = f"COOLPROP-{CoolProp.__version__}"

# fluid_id -> (CoolProp string, T_min, T_max). The T range covers tank, line, pump, jacket and
# pressurant use; the upper limit stays inside each EOS's validity range.
SPECS: dict[str, tuple[str, float, float]] = {
    "LOX": ("Oxygen", 60.0, 700.0),
    "LN2": ("Nitrogen", 65.0, 700.0),
    "LH2": ("ParaHydrogen", 14.0, 700.0),
    "LCH4": ("Methane", 91.0, 620.0),
    "RP1_SURR": ("n-Dodecane", 265.0, 700.0),
    "ETOH": ("Ethanol", 160.0, 640.0),
    "ETOH95": ("HEOS::Ethanol[{x95}]&Water[{w95}]", 200.0, 420.0),  # mixture flash fails nearer T_crit
    "ETOH75": ("HEOS::Ethanol[{x75}]&Water[{w75}]", 200.0, 420.0),
    "H2O": ("Water", 274.0, 1000.0),
    "N2O": ("NitrousOxide", 183.0, 520.0),
    "GHE": ("Helium", 20.0, 700.0),
}
# The pressurants GN2, GH2 and GOX use the same tables as LN2, LH2 and LOX (one EOS covers both phases).

M_ETOH, M_H2O = 46.06844e-3, 18.01528e-3


def mole_fraction_ethanol(w: float) -> float:
    """Ethanol mole fraction of an ethanol-water blend of ethanol mass fraction ``w``."""
    n_e, n_w = w / M_ETOH, (1.0 - w) / M_H2O
    return n_e / (n_e + n_w)


def fluid_string(fid: str) -> str:
    s = SPECS[fid][0]
    x95, x75 = mole_fraction_ethanol(0.95), mole_fraction_ethanol(0.75)
    return s.format(x95=f"{x95:.6f}", w95=f"{1 - x95:.6f}", x75=f"{x75:.6f}", w75=f"{1 - x75:.6f}")


def _p(prop: str, *args) -> float | None:
    try:
        v = CP.PropsSI(prop, *args)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _fmt(v) -> str:
    return "" if v is None else f"{v:.7g}"


def t_grid(fs: str, t_lo: float, t_hi: float) -> np.ndarray:
    """Fine spacing up to 1.3 T_crit, where the liquid properties change fastest; coarse above."""
    t_c = _p("Tcrit", fs)  # mixtures have no tabulated T_crit: use one fine grid
    t_mid = t_hi if t_c is None else min(1.3 * t_c, t_hi)
    cold = np.linspace(t_lo, t_mid, N_T_COLD)
    hot = np.linspace(t_mid, t_hi, N_T_HOT + 1)[1:] if t_hi > t_mid else np.array([])
    return np.concatenate([cold, hot])


def phase_name(fs: str, T: float, p: float) -> str:
    try:
        return CP.PhaseSI("T", T, "P", p, fs)
    except Exception:
        return ""


def build_table(fid: str) -> int:
    fs = fluid_string(fid)
    _, t_lo, t_hi = SPECS[fid]
    rows = []
    for T in t_grid(fs, t_lo, t_hi):
        for p in P_GRID:
            rho = _p("D", "T", T, "P", p, fs)
            if rho is None:
                continue
            phase = phase_name(fs, T, p)
            vals = [_p(q, "T", T, "P", p, fs) for q in ("V", "C", "O", "L", "A", "H", "S")]
            cp, a = vals[1], vals[4]
            if phase.startswith("twophase") or (cp is not None and cp <= 0) or (a is not None and a <= 0):
                continue  # failed flash (seen for the ethanol-water mixtures): drop the point
            rows.append([f"{T:.4f}", f"{p:.6g}", phase, _fmt(rho), *(_fmt(v) for v in vals)])
    path = ROOT / "tables" / f"{fid}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["T", "p", "phase", "rho", "mu", "cp", "cv", "k", "a_sound", "h", "s"])
        w.writerows(rows)
    return len(rows)


def build_sat(fid: str) -> int | None:
    fs = fluid_string(fid)
    if "&" in fs or fid == "GHE":
        return None  # mixtures have no single saturation curve; helium is not a propellant
    t_tr = max(CP.PropsSI("Ttriple", fs), CP.PropsSI("Tmin", fs))
    t_c = CP.PropsSI("Tcrit", fs)
    rows = []
    for T in np.linspace(t_tr + 0.01, t_tr + 0.995 * (t_c - t_tr), N_SAT):

        def g(prop: str, q: int, T: float = T) -> float | None:
            return _p(prop, "T", T, "Q", q, fs)

        rows.append(
            [
                f"{T:.4f}",
                _fmt(g("P", 0)),
                _fmt(g("D", 0)),
                _fmt(g("D", 1)),
                _fmt(g("H", 0)),
                _fmt(g("H", 1)),
                _fmt(g("C", 0)),
                _fmt(g("V", 0)),
                _fmt(g("L", 0)),
                _fmt(g("I", 0)),
            ]
        )
    path = ROOT / "sat" / f"{fid}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["T", "p_sat", "rho_l", "rho_v", "h_l", "h_v", "cp_l", "mu_l", "k_l", "sigma"])
        w.writerows(rows)
    return len(rows)


def eos_refs(fid: str) -> str:
    fs = fluid_string(fid)
    names = [s.split("[")[0] for s in fs.replace("HEOS::", "").split("&")]
    parts = []
    for n in names:
        refs = {k: CP.get_fluid_param_string(n, f"BibTeX-{k}") for k in ("EOS", "VISCOSITY", "CONDUCTIVITY")}
        parts.append(
            f"{n}: EOS {refs['EOS'] or 'n/a'}; mu {refs['VISCOSITY'] or 'none'}; "
            f"k {refs['CONDUCTIVITY'] or 'none'}"
        )
    if len(names) > 1:
        parts.append(
            "binary Ethanol-Water from REFPROP 9.1 via CoolProp; mixture transport by CoolProp " "estimation"
        )
    return " | ".join(parts)


SCALARS = [  # (property, CoolProp key or callable, unit)
    ("molar_mass", "molar_mass", "kg/mol"),
    ("T_crit", "Tcrit", "K"),
    ("p_crit", "pcrit", "Pa"),
    ("rho_crit", "rhomass_critical", "kg/m3"),
    ("T_triple", "Ttriple", "K"),
]


def coolprop_scalars(fid: str) -> list[dict]:
    fs = fluid_string(fid)
    if "&" in fs:
        return []
    out = []
    loc = f"EOS {CP.get_fluid_param_string(fs, 'BibTeX-EOS')}"
    for prop, key, unit in SCALARS:
        v = _p(key, fs)
        if v is None:
            continue
        out.append(
            dict(
                fluid_id=fid,
                property=prop,
                value=_fmt(v),
                unit=unit,
                quality="derived",
                source_id=SRC,
                location=loc,
                uncertainty="",
                T_cond="",
                p_cond="",
                formula=f"CoolProp PropsSI('{key}','{fs}')",
                preferred="1",
            )
        )
    tnbp = _p("T", "P", 101325.0, "Q", 0, fs)
    if tnbp is not None and fid != "GHE":
        out.append(
            dict(
                fluid_id=fid,
                property="T_nbp",
                value=_fmt(tnbp),
                unit="K",
                quality="derived",
                source_id=SRC,
                location=loc,
                uncertainty="",
                T_cond="",
                p_cond="101325",
                formula=f"CoolProp PropsSI('T','P',101325,'Q',0,'{fs}')",
                preferred="1",
            )
        )
    return out


def main() -> None:
    fluids_csv = ROOT / "fluids.csv"
    with fluids_csv.open(encoding="utf-8") as f:
        fluids = list(csv.DictReader(f))
    only = set(sys.argv[1:])  # optional fluid ids: rebuild only these tables
    for row in fluids:
        fid = row["fluid_id"]
        if row["model"] != "coolprop" or fid not in SPECS or (only and fid not in only):
            continue
        n = build_table(fid)
        ns = build_sat(fid)
        row["coolprop_name"] = fluid_string(fid)
        row["table_file"] = f"tables/{fid}.csv"
        row["sat_file"] = f"sat/{fid}.csv" if ns else ""
        row["table_source_id"] = SRC
        row["table_location"] = eos_refs(fid)
        print(f"{fid:9s} {n:4d} table rows, {ns or 0:3d} saturation rows")
    with fluids_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fluids[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(fluids)

    sc_path = ROOT / "fluid_scalars.csv"
    with sc_path.open(encoding="utf-8") as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames
        kept = [r for r in rd if not r["source_id"].startswith("COOLPROP-")]
    new = [r for fid in SPECS for r in coolprop_scalars(fid)]
    with sc_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(new + kept)
    print(f"fluid_scalars.csv: {len(new)} CoolProp rows + {len(kept)} literature rows")


if __name__ == "__main__":
    main()
