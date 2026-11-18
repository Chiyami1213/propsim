"""Transcribe NIST-JANAF thermochemical tables (NIST SRD 13, https://janaf.nist.gov) for hot-gas species.

    python tools/build_janaf_tables.py [--cache DIR] [species ...]

For each species the raw tab-separated table (``tables/<ID>.txt``) is read exactly as published: T (K), Cp
(J/(mol K)), S (J/(mol K)), H - H(Tr) (kJ/mol) and Delta_f H (kJ/mol), at the tabulated temperatures only (no
interpolation).  The library stores SI units: kJ/mol is multiplied by 1000 (written in the ``formula``
column).  Before a
file is written the transcription is checked against itself: S and H must integrate from Cp
(``dH = int Cp dT``, ``dS = int Cp/T dT`` over each tabulated interval).  Output:

    propsim/data/library/fluids/janaf/<species_id>.csv     long format, one row per (T, property)
    propsim/data/library/fluids/janaf_species.csv           species index

NIST terms: SRD 13 is a compilation whose copyright is secured by the U.S. Secretary of Commerce (15 U.S.C.
290e);
see ``sources.csv`` (NIST-JANAF) for the statement and the note about redistribution.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "propsim" / "data" / "library" / "fluids"
BASE = "https://janaf.nist.gov/"
SOURCE_ID = "NIST-JANAF"
T_REF = 298.15

# species_id -> (first element of the formula = index page, Hill formula in the JANAF index, phase)
SPECIES = {
    "H2": ("H", "H2", "ref"),
    "O2": ("O", "O2", "ref"),
    "H2O": ("H", "H2O", "g"),
    "CO": ("C", "CO", "g"),
    "CO2": ("C", "CO2", "g"),
    "N2": ("N", "N2", "ref"),
    "OH": ("H", "HO", "g"),
    "H": ("H", "H", "g"),
    "O": ("O", "O", "g"),
    "NO": ("N", "NO", "g"),
    "CH4": ("C", "CH4", "g"),
    "CH": ("C", "CH", "g"),
}
PROPS = [  # (property id, source column, printed unit, SI unit, factor to SI, formula text)
    ("cp", "Cp", "J/(mol K)", "J/(mol K)", 1.0, "as printed (J/(mol K))"),
    ("S", "S", "J/(mol K)", "J/(mol K)", 1.0, "as printed (J/(mol K))"),
    ("H_minus_H298", "H-H(Tr)", "kJ/mol", "J/mol", 1000.0, "printed kJ/mol * 1000"),
    ("dfH", "delta-f H", "kJ/mol", "J/mol", 1000.0, "printed kJ/mol * 1000"),
]


def fetch(url: str, cache: Path) -> str:
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / url.replace(BASE, "").replace("/", "_")
    if not f.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "propsim-library-builder"})
        f.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    return f.read_text(encoding="utf-8", errors="replace")


def index_rows(element: str, cache: Path) -> list[dict]:
    html = fetch(f"{BASE}tables/{element}-index.html", cache)
    rows = []
    for tr in re.findall(r"<tr>(.*?)</tr>", html, flags=re.S):
        cells = re.findall(r"<td>(.*?)</td>", tr, flags=re.S)
        if len(cells) >= 5:
            link = re.search(r'href="([A-Za-z]+-\d+)\.html"', cells[4])
            if link:
                rows.append(
                    dict(
                        cas=cells[0].strip(),
                        formula=cells[1].strip(),
                        name=cells[2].strip(),
                        phase=cells[3].strip(),
                        table_id=link.group(1),
                    )
                )
    return rows


def parse_table(text: str) -> tuple[str, str, dict[str, np.ndarray]]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    title, jformula = (lines[0].split("\t") + [""])[:2]
    header = lines[1].split("\t")
    cols: dict[str, list[float]] = {h: [] for h in header}
    for ln in lines[2:]:
        cells = ln.split("\t")
        if len(cells) != len(header):
            raise ValueError(f"unexpected row: {ln!r}")
        for h, c in zip(header, cells):
            try:
                cols[h].append(float(c))
            except ValueError:  # INFINITE, notes: not stored
                cols[h].append(np.nan)
    return title.strip(), jformula.strip(), {h: np.array(v) for h, v in cols.items()}


def check_consistency(T, cp, S, H) -> list[str]:
    """Cp must integrate to H and S.  A monotone cubic through the tabulated Cp is integrated over each
    interval (the
    spacing is 100 K, so the check is looser below 300 K where Cp is not smooth) and cumulatively from
    298.15 K."""
    from scipy.interpolate import PchipInterpolator

    ok = ~np.isnan(cp)
    f = PchipInterpolator(T[ok], cp[ok])
    bad = []
    i0 = int(np.argmin(np.abs(T - T_REF)))
    for i in range(1, len(T)):
        if T[i - 1] < 100.0:
            continue  # the 0 K row is not a smooth interval
        t = np.linspace(T[i - 1], T[i], 400)
        dH = np.trapezoid(f(t), t) / 1000.0  # kJ/mol
        dS = np.trapezoid(f(t) / t, t)
        tol = 0.06 if T[i - 1] < 300.0 else 0.015
        if abs(H[i] - H[i - 1] - dH) > tol * abs(dH) + 0.03:
            bad.append(f"H interval to {T[i]:g} K: table {H[i] - H[i - 1]:.3f} vs int cp dT {dH:.3f} kJ/mol")
        if abs(S[i] - S[i - 1] - dS) > tol * abs(dS) + 0.15:
            bad.append(
                f"S interval to {T[i]:g} K: table {S[i] - S[i - 1]:.3f} vs int cp/T dT {dS:.3f} J/(mol K)"
            )
    for i in range(i0 + 1, len(T)):  # cumulative from the reference temperature
        t = np.linspace(T[i0], T[i], 2000)
        dH = np.trapezoid(f(t), t) / 1000.0
        dS = np.trapezoid(f(t) / t, t)
        if abs(H[i] - H[i0] - dH) > 0.006 * abs(dH) + 0.05:
            bad.append(f"cumulative H to {T[i]:g} K: table {H[i] - H[i0]:.3f} vs {dH:.3f} kJ/mol")
        if abs(S[i] - S[i0] - dS) > 0.006 * abs(dS) + 0.1:
            bad.append(f"cumulative S to {T[i]:g} K: table {S[i] - S[i0]:.3f} vs {dS:.3f} J/(mol K)")
    return bad


def build(ids: list[str], cache: Path) -> None:
    (OUT / "janaf").mkdir(parents=True, exist_ok=True)
    index_cache: dict[str, list[dict]] = {}
    index_out = []
    for sid in ids:
        el, hill, phase = SPECIES[sid]
        rows = index_cache.setdefault(el, index_rows(el, cache))
        hit = [r for r in rows if r["formula"] == hill and r["phase"] == phase]
        if len(hit) != 1:
            raise SystemExit(f"{sid}: {len(hit)} index rows for formula {hill!r} phase {phase!r}")
        ent = hit[0]
        title, jformula, tab = parse_table(fetch(f"{BASE}tables/{ent['table_id']}.txt", cache))
        T = tab["T(K)"]
        if np.any(np.diff(T) <= 0):
            raise SystemExit(f"{sid}: temperatures not increasing")
        bad = check_consistency(T, tab["Cp"], tab["S"], tab["H-H(Tr)"])
        if bad:
            raise SystemExit(f"{sid}: transcription inconsistent with itself:\n  " + "\n  ".join(bad))
        out = OUT / "janaf" / f"{sid}.csv"
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(
                [
                    "species_id",
                    "T",
                    "property",
                    "value",
                    "unit",
                    "quality",
                    "source_id",
                    "location",
                    "formula",
                ]
            )
            for i, t in enumerate(T):
                for pid, col, pu, su, fac, ftxt in PROPS:
                    v = tab[col][i]
                    if np.isnan(v):
                        continue
                    w.writerow(
                        [
                            sid,
                            f"{t:g}",
                            pid,
                            f"{v * fac:.10g}",
                            su,
                            "published",
                            SOURCE_ID,
                            f"NIST-JANAF table {ent['table_id']} ({jformula}), row T = {t:g} K, column {col}",
                            ftxt,
                        ]
                    )
        index_out.append(
            [
                sid,
                ent["name"],
                jformula,
                phase,
                ent["cas"],
                ent["table_id"],
                f"{T_REF}",
                "1e5",
                f"{T[0]:g}",
                f"{T[-1]:g}",
                len(T),
            ]
        )
        print(
            f"{sid:4s} {ent['table_id']}  {jformula:10s} {len(T):3d} rows  T {T[0]:g}..{T[-1]:g} K  checks ok"
        )
    idx = OUT / "janaf_species.csv"
    old = {}
    if idx.exists():
        old = {r[0]: r for r in csv.reader(idx.open(encoding="utf-8"))}
    for r in index_out:
        old[r[0]] = [str(x) for x in r]
    old.pop("species_id", None)
    with idx.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            [
                "species_id",
                "name",
                "janaf_formula",
                "phase",
                "cas",
                "table_id",
                "T_ref",
                "p_ref_Pa",
                "T_min",
                "T_max",
                "n_rows",
            ]
        )
        for sid in SPECIES:
            if sid in old:
                w.writerow(old[sid])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("species", nargs="*", help="species ids (default: all)")
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "propsim_janaf"))
    a = ap.parse_args()
    build(a.species or list(SPECIES), Path(a.cache))
    return 0


if __name__ == "__main__":
    sys.exit(main())
