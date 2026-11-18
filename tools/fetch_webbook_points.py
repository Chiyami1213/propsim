"""Fetch NIST Chemistry WebBook fluid properties (SRD 69, ``fluid.cgi``) at state points that lie exactly
on the grids of
the library's CoolProp tables, to check those tables:

    python tools/fetch_webbook_points.py [--cache DIR]

Ten (T, p) points per fluid (compressed liquid, near-saturation liquid, vapour, supercritical) are chosen
from the
``fluids/tables`` grids, so the comparison needs no interpolation.  The WebBook values are written as
printed (density,
heat capacities, speed of sound, viscosity, thermal conductivity, converted to SI) to
``propsim/data/library/fluids/webbook_check_points.csv``; ``tools/check_library.py`` compares them with the
CoolProp tables.
Enthalpy and entropy are not compared (different reference states).
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from propsim import data  # noqa: E402

OUT = data.FLUIDS / "webbook_check_points.csv"
URL = "https://webbook.nist.gov/cgi/fluid.cgi"
SOURCE_ID = "NIST-WEBBOOK-FLUID"
FLUIDS = {
    "LOX": "C7782447",
    "LH2": "B5000001",
    "LCH4": "C74828",
    "N2O": "C10024972",
}  # WebBook ids (LH2: parahydrogen)
# (p in MPa, index into the table's temperature grid): liquid, near-saturation, vapour and supercritical
# states
PICKS = [
    (0.5, 8),
    (0.5, 40),
    (1.0, 18),
    (3.0, 28),
    (5.0, 24),
    (5.0, 48),
    (10.0, 26),
    (10.0, 52),
    (20.0, 14),
    (20.0, 62),
]
# WebBook column -> (property id, SI unit, factor)
COLS = {
    "Density (kg/m3)": ("rho", "kg/m3", 1.0),
    "Cp (J/g*K)": ("cp", "J/(kg K)", 1000.0),
    "Cv (J/g*K)": ("cv", "J/(kg K)", 1000.0),
    "Sound Spd. (m/s)": ("a_sound", "m/s", 1.0),
    "Viscosity (uPa*s)": ("mu", "Pa s", 1e-6),
    "Therm. Cond. (W/m*K)": ("k", "W/(m K)", 1.0),
}


def fetch(fid: str, T: float, p_MPa: float, cache: Path) -> dict[str, str]:
    q = dict(
        TUnit="K",
        PUnit="MPa",
        DUnit="kg/m3",
        HUnit="kJ/kg",
        WUnit="m/s",
        VisUnit="uPa*s",
        STUnit="N/m",
        Type="IsoBar",
        Digits="6",
        P=f"{p_MPa:g}",
        THigh=f"{T:.4f}",
        TLow=f"{T:.4f}",
        TInc="1",
        RefState="DEF",
        ID=fid,
        Action="Data",
    )
    url = URL + "?" + urllib.parse.urlencode(q)
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / (f"{fid}_{p_MPa:g}_{T:.4f}.tsv")
    if not f.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "propsim-library-builder"})
        f.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    lines = [ln for ln in f.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    if len(lines) < 2 or "\t" not in lines[0]:
        raise RuntimeError(f"no data for {fid} at {T} K, {p_MPa} MPa: {lines[:1]}")
    head = lines[0].split("\t")
    return dict(zip(head, lines[1].split("\t")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "propsim_webbook"))
    a = ap.parse_args()
    rows = []
    for fluid, wid in FLUIDS.items():
        tab = data.fluid_table(fluid)
        Tg = np.unique(tab["T"])
        pg = np.unique(tab["p"])
        for pi, (p_MPa, ti) in enumerate(PICKS, start=1):
            T = float(Tg[ti])
            if not np.any(np.isclose(pg, p_MPa * 1e6)):
                raise SystemExit(f"{fluid}: {p_MPa} MPa is not on the table grid")
            rec = fetch(wid, T, p_MPa, Path(a.cache))
            pid = f"{fluid}_{pi:02d}"
            for col, (prop, unit, fac) in COLS.items():
                raw = rec.get(col, "").strip()
                try:
                    v = float(raw)
                except ValueError:
                    continue  # the WebBook gives no value (e.g. N2O transport)
                rows.append(
                    [
                        pid,
                        fluid,
                        f"{T:.4f}",
                        f"{p_MPa * 1e6:g}",
                        prop,
                        f"{v * fac:.8g}",
                        unit,
                        rec.get("Phase", ""),
                        "published",
                        SOURCE_ID,
                        f"NIST Chemistry WebBook fluid.cgi, ID={wid}, IsoBar P = {p_MPa:g} MPa, T = "
                        f"{T:.4f} K, "
                        f"column '{col}'",
                        "" if fac == 1.0 else f"printed value * {fac:g}",
                    ]
                )
            print(f"{pid} T={T:9.4f} K p={p_MPa:5.1f} MPa  {rec.get('Phase', '')}")
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            [
                "point_id",
                "fluid_id",
                "T",
                "p",
                "property",
                "value",
                "unit",
                "phase_webbook",
                "quality",
                "source_id",
                "location",
                "formula",
            ]
        )
        w.writerows(rows)
    print(f"{len(rows)} values -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
