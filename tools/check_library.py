"""Sanity checks of the component library; writes ``propsim/data/library/checks.md``.

Run from the repository root::

    python tools/check_library.py            # writes checks.md, exit code 1 if an integrity check fails

Checks (brief rule 5, the parts that apply to the blocks delivered so far):

1. File integrity: provenance fields, SI unit strings, ids, mixtures (``propsim.data.validate``).
2. Published theoretical performance (Huzel & Huang Tables 1-8 to 1-10) against the CEA tables shipped in
   ``propsim/core/thermo/data/cea_tables.npz``.
3. Scalars given by more than one source (critical point, boiling point, ...): spread between sources.
4. Generated fluid tables: positive density, heat capacity and sound speed; empty cells per column.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from propsim import data  # noqa: E402
from propsim.core.thermo.cea import Thermo  # noqa: E402

OUT = data.LIBRARY / "checks.md"
TOL_CEA = 2.0  # (%) flag limit for theoretical-performance comparisons
TOL_SCALAR = 1.0  # (%) flag limit for disagreement between sources of one scalar
PAIR_OF = {("LOX", "RP1"): "LOX/RP-1", ("LOX", "LH2"): "LOX/LH2", ("N2O4", "MMH"): "N2O4/MMH"}
NOT_SHIPPED = {
    ("LOX", "ETOH95"): "shipped LOX/Ethanol pair is 100 % ethanol, the case is 95 % ethanol",
    ("LOX", "N2H4"): "no LOX/N2H4 pair shipped",
    ("N2O4", "UDMH"): "no N2O4/UDMH pair shipped",
}


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def cea_section() -> tuple[list[str], int]:
    lines = [
        "## 2. Theoretical performance: Huzel & Huang vs. shipped CEA tables",
        "",
        "Conditions of the published cases: P_c = 1000 psia, optimum expansion to 14.7 psia, infinite "
        "contraction ratio, shifting equilibrium. The tables are evaluated at the same P_c and O/F; the "
        "optimum sea-level Isp is the maximum over area ratio of `Thermo.isp_sl` with P_a = P_e. M is the "
        "product molar mass (Huzel & Huang: chamber; tables: throat, so a small difference is expected). "
        f"Differences above {TOL_CEA:g} % are flagged.",
        "",
        "| case | T_c (K) pub / tab / Δ% | M (g/mol) pub / tab / Δ% | c* (m/s) pub / tab / Δ% | "
        "Isp_opt (s) pub / tab / Δ% | flag |",
        "|---|---|---|---|---|---|",
    ]
    n_flag = 0
    skipped = []
    for r in data.cea_validation():
        key = (r["oxidizer"], r["fuel"])
        if key not in PAIR_OF:
            skipped.append(f"- `{r['case_id']}`: not compared ({NOT_SHIPPED.get(key, 'pair not shipped')}).")
            continue
        th = Thermo(PAIR_OF[key], source="tables")
        mr, pc, pe = float(r["OF"]), float(r["pc"]), float(r["pe"])
        ch = th.chamber(mr, pc)
        eps = np.linspace(1.5, 100.0, 4000)
        isp_opt = float(np.max(th.isp_sl(mr, pc, eps, pa=pe)))
        pub = dict(
            Tc=float(r["Tc"]),
            cstar=float(r["cstar"]),
            isp=float(r["Isp_opt"]),
            M=float(r["M_products"]) if r["M_products"] else None,
        )
        tab = dict(Tc=float(ch.Tc), cstar=float(ch.cstar), isp=isp_opt, M=float(ch.Mw))
        d = {k: pct(tab[k], pub[k]) for k in ("Tc", "cstar", "isp")}
        dM = pct(tab["M"], pub["M"]) if pub["M"] else None
        bad = [k for k, v in d.items() if abs(v) > TOL_CEA]
        n_flag += bool(bad)
        m_cell = (
            f"{pub['M'] * 1e3:.1f} / {tab['M'] * 1e3:.1f} / {dM:+.1f}"
            if dM is not None
            else f"n/a / {tab['M'] * 1e3:.1f} / n/a"
        )
        lines.append(
            f"| `{r['case_id']}` | {pub['Tc']:.0f} / {tab['Tc']:.0f} / {d['Tc']:+.1f} | {m_cell} | "
            f"{pub['cstar']:.0f} / {tab['cstar']:.0f} / {d['cstar']:+.1f} | "
            f"{pub['isp']:.0f} / {tab['isp']:.1f} / {d['isp']:+.1f} | "
            f"{'**' + ', '.join(bad) + '**' if bad else 'ok'} |"
        )
    return lines + [""] + skipped + [""], n_flag


def scalar_section() -> tuple[list[str], int]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in data.fluid_scalars():
        if r["value"]:
            groups[(r["fluid_id"], r["property"])].append(r)
    lines = [
        "## 3. Scalars with more than one source",
        "",
        f"Spread = (max - min) / preferred value. Above {TOL_SCALAR:g} % the pair is flagged and discussed "
        "in `conflicts.md`.",
        "",
        "| fluid | property | values (source) | spread % | flag |",
        "|---|---|---|---|---|",
    ]
    n_flag = 0
    for (fid, prop), rows in sorted(groups.items()):
        if len(rows) < 2:
            continue
        vals = [float(r["value"]) for r in rows]
        ref = next((float(r["value"]) for r in rows if r["preferred"] == "1"), vals[0])
        spread = 100.0 * (max(vals) - min(vals)) / abs(ref)
        flag = spread > TOL_SCALAR
        n_flag += flag
        desc = "; ".join(
            f"{float(r['value']):.6g} ({r['source_id']}{', preferred' if r['preferred'] == '1' else ''})"
            for r in rows
        )
        lines.append(f"| {fid} | {prop} | {desc} | {spread:.2f} | {'**check**' if flag else 'ok'} |")
    return lines + [""], n_flag


def table_section() -> list[str]:
    lines = [
        "## 4. Generated fluid tables",
        "",
        "| fluid | rows | rho, cp, a > 0 | empty cells |",
        "|---|---|---|---|",
    ]
    for fid, f in data.fluids().items():
        if not f["table_file"]:
            continue
        t = data.fluid_table(fid)
        ok = all(np.all(t[c][np.isfinite(t[c])] > 0) for c in ("rho", "cp", "a_sound"))
        empty = {
            c: int(np.isnan(t[c]).sum())
            for c in data.TABLE_COLUMNS
            if c not in ("phase",) and np.isnan(t[c]).any()
        }
        lines.append(
            f"| {fid} | {len(t['T'])} | {'ok' if ok else '**fail**'} | "
            f"{', '.join(f'{c}: {n}' for c, n in empty.items()) or 'none'} |"
        )
    return lines + [""]


def main() -> int:
    problems = data.validate()
    integrity = (
        ["## 1. File integrity", ""]
        + (
            [f"- {p}" for p in problems]
            if problems
            else ["All provenance fields, unit strings, ids and mixtures pass."]
        )
        + [""]
    )
    cea, n_cea = cea_section()
    sca, n_sca = scalar_section()
    head = [
        "# Library checks",
        "",
        "Generated by `tools/check_library.py`; do not edit by hand.",
        "",
        f"Summary: {len(problems)} integrity problems, {n_cea} theoretical-performance cases outside "
        f"{TOL_CEA:g} %, {n_sca} multi-source scalars spreading more than {TOL_SCALAR:g} %.",
        "",
        "Engine checks (c* and thrust closure, pump/turbine power balance) start with the engines block.",
        "",
    ]
    OUT.write_text("\n".join(head + integrity + cea + sca + table_section()), encoding="utf-8")
    print(f"wrote {OUT}: {len(problems)} integrity problems, {n_cea} CEA flags, {n_sca} scalar flags")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
