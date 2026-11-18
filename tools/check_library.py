"""Sanity checks of the component library; writes ``propsim/data/library/checks.md``.

Run from the repository root::

    python tools/check_library.py            # writes checks.md, exit code 1 if an integrity check fails

Checks (brief rule 5, the parts that apply to the blocks delivered so far):

1. File integrity: provenance fields, SI unit strings, ids, mixtures (``propsim.data.validate``).
2. Published theoretical performance (Huzel & Huang Tables 1-8 to 1-10) against the CEA tables shipped in
   ``propsim/core/thermo/data/cea_tables.npz``.
3. Scalars given by more than one source (critical point, boiling point, ...): spread between sources.
4. Generated fluid tables: positive density, heat capacity and sound speed; empty cells per column.
5. Reference engines: c*, thrust, C_F, mixture-ratio, pump/turbine power and cycle-efficiency closure
   (``propsim.data.closure``).
6. Turbine drive gas (Huzel & Huang Table 6-4): R against c_p (gamma - 1) / gamma.
7. Example graphs: count of published, derived and estimated values per graph.
8. Material fits: continuity at segment joins and at T_low, zero thermal strain at 293 K; material points:
   monotonic trends and yield below ultimate (transcription checks).
9. Hybrid hot-fire cases: initial oxidizer flux against mdot_ox / port area (column alignment), and the
   published regression law against the measured average regression rate (information).
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from propsim import data  # noqa: E402
from propsim.data import closure, matprops  # noqa: E402
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


def engine_section() -> tuple[list[str], int]:
    cl = closure.all_closures()
    lines = [
        "## 5. Reference engines: closure of published values",
        "",
        "Each row computes one quantity two ways from the published values of `engines/engine_params.csv` "
        "(`propsim.data.closure`). Δ% = (left - right) / right.",
        "",
        "| engine | check | left / right | Δ% | tol % | flag |",
        "|---|---|---|---|---|---|",
    ]
    for c in cl:
        lines.append(
            f"| {c.engine_id} | {c.detail} | {c.lhs:.5g} / {c.rhs:.5g} | {100 * c.diff:+.2f} | "
            f"{100 * c.tol:g} | "
            f"{'ok' if c.ok else '**check**'} |"
        )
    checked = {c.engine_id for c in cl}
    rest = [e for e in data.engines() if e not in checked]
    lines += [
        "",
        f"Not checked (the sources give no flow rates or throat area): {', '.join(rest)}.",
        "",
    ]
    return lines, sum(not c.ok for c in cl)


def turbine_gas_section() -> tuple[list[str], int]:
    bad = closure.turbine_gas_consistency(tol=0.06)
    lines = [
        "## 6. Turbine drive gas (Huzel & Huang Table 6-4)",
        "",
        "R against c_p (γ - 1) / γ; the printed columns agree within about 5 % (rounding). Rows outside 6 %:",
        "",
    ]
    lines += [
        f"- {r['propellants']} at {float(r['T_inlet']):.1f} K: R = {float(r['R']):.1f} J/(kg K), "
        f"{100 * d:+.1f} % (see `conflicts.md`)"
        for r, d in bad
    ] or ["None."]
    return lines + [""], len(bad)


def graph_section() -> list[str]:
    lines = [
        "## 7. Example graphs: parameter provenance",
        "",
        "Every node parameter, boundary value and sequence time is a sourced value (integrity: section 1). "
        "Counts per quality:",
        "",
        "| graph | nodes | edges | published | derived | estimated |",
        "|---|---|---|---|---|---|",
    ]
    for gid, g in data.graphs().items():
        vals = [v for n in g["nodes"] for v in n["params"].values()]
        vals += [b["value"] for b in g["boundary_conditions"]]
        vals += [st[k] for st in g["sequence"] for k in ("t", "ramp", "value") if k in st]
        vals += [st["when"]["value"] for st in g["sequence"] if "when" in st]
        q = {k: sum(v["quality"] == k for v in vals) for k in ("published", "derived", "estimated")}
        lines.append(
            f"| `{gid}` | {len(g['nodes'])} | {len(g['edges'])} | {q['published']} | {q['derived']} | "
            f"{q['estimated']} |"
        )
    return lines + [""]


def material_section() -> tuple[list[str], int]:
    """Transcription checks: a wrong coefficient breaks continuity or the 293 K zero of thermal strain."""
    lines = [
        "## 8. Material fits: transcription checks",
        "",
        "| material | property | check | value | flag |",
        "|---|---|---|---|---|",
    ]
    n_flag = 0
    fits = data.material_fits()

    def row(m: str, p: str, what: str, val: float, ok: bool) -> None:
        nonlocal n_flag
        n_flag += not ok
        lines.append(f"| {m} | {p} | {what} | {val:+.3g} | {'ok' if ok else '**check**'} |")

    for r in fits:
        if r["property"] == "thermal_strain":
            v = matprops._eval(r, 293.0)
            row(r["material_id"], "thermal_strain", "value at 293 K", v, abs(v) < 2e-6)
        if r["T_low"]:
            T = float(r["T_low"])
            jump = matprops._eval(r, T + 1e-9) / float(r["scale"]) - float(r["y_low"])
            row(
                r["material_id"],
                r["property"],
                f"jump at T_low = {T:g} K (printed units)",
                jump,
                abs(jump) < 0.2,
            )
    for r in fits:
        if r["segment"] == "low":
            hi = next(
                x
                for x in fits
                if x["material_id"] == r["material_id"]
                and x["property"] == r["property"]
                and x["segment"] == "high"
            )
            T = float(r["T_max"])
            d = matprops._eval(r, T) / matprops._eval(hi, T) - 1
            row(r["material_id"], r["property"], f"low/high fits at {T:g} K", d, abs(d) < 0.01)
    # tables read from PDF text layers: trends and ordering confirm the column assignment
    pts = data.material_points()
    trend = {"E": -1, "shear_modulus": -1, "k": +1, "alpha_mean": +1}
    groups: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for r in pts:
        if r["property"] in trend and r["T_max"]:
            groups[(r["material_id"], r["property"], r["condition"])].append(
                (float(r["T_max"]), float(r["value"]))
            )
    for (m, prop, _), xy in sorted(groups.items()):
        if len(xy) < 3:
            continue
        ys = [y for _, y in sorted(xy)]
        steps = [trend[prop] * (b - a) for a, b in zip(ys, ys[1:])]
        row(
            m,
            prop,
            f"monotonic {'falling' if trend[prop] < 0 else 'rising'} over {len(ys)} points",
            min(steps) / max(abs(y) for y in ys),
            min(steps) >= 0,
        )
    ys_ut: dict[tuple[str, str, str], dict[str, float]] = defaultdict(dict)
    for r in pts:
        if r["property"] in ("yield_strength", "ultimate_strength") and r["T_max"]:
            ys_ut[(r["material_id"], r["T_max"], r["condition"])][r["property"]] = float(r["value"])
    for (m, T, _), v in sorted(ys_ut.items()):
        if len(v) == 2:
            ratio = v["yield_strength"] / v["ultimate_strength"]
            row(m, "strength", f"yield / ultimate at {float(T):.0f} K", ratio, 0 < ratio <= 1)
    return lines + [""], n_flag


def hybrid_section() -> tuple[list[str], int]:
    lines = [
        "## 9. Hybrid hot-fire cases",
        "",
        "Flux check: G_ox0 against mdot_ox / (pi D_port0^2 / 4), flagged beyond 10 % (throttled tests "
        "excepted). Law column: SP-1a/GOX law at the published average flux against the measured average "
        "regression rate "
        "(information, not flagged).",
        "",
        "| case | flux Δ% | law / measured - 1 (%) | flag |",
        "|---|---|---|---|",
    ]
    n_flag = 0
    for c in data.hotfire_cases():
        area = math.pi * float(c["D_port0"]) ** 2 / 4
        d_flux = float(c["mdot_ox"]) / area / float(c["G_ox0"]) - 1
        law = next(
            (
                lid
                for lid, r in data.regression_laws().items()
                if (r["fuel"], r["oxidizer"]) == (c["fuel"], c["oxidizer"])
            ),
            None,
        )
        d_law = (
            data.regression_rate(law, float(c["G_ox_avg"])) / float(c["rdot_avg"]) - 1
            if law
            else float("nan")
        )
        throttled = "Thr" in c["case_id"]
        bad = abs(d_flux) > 0.10 and not throttled
        n_flag += bad
        lines.append(
            f"| {c['case_id']} | {100 * d_flux:+.1f}{' (throttled)' if throttled else ''} | "
            f"{100 * d_law:+.1f} | {'**check**' if bad else 'ok'} |"
        )
    return lines + [""], n_flag


def electric_section() -> tuple[list[str], int]:
    lines = [
        "## 10. Electric thruster throttle tables",
        "",
        "Per point: thrust against mdot Isp g0 (1 %), thruster efficiency against F^2 / (2 mdot P_thruster) "
        "(3 %; P_thruster = P_PPU eta_PPU where only PPU power is printed), lifetime impulse against "
        "throughput Isp g0 (2 %) and system efficiency against eta_PPU eta_thruster (2 %). The printed "
        "values carry two or three significant figures, so differences under about 1 % are rounding.",
        "",
        "| thruster | point | P_thruster (W) | thrust Δ% | eta_t Δ% | impulse Δ% | eta_sys Δ% | flag |",
        "|---|---|---|---|---|---|---|---|",
    ]
    n_flag = 0
    for r in data.throttle_table():
        m, F, isp = float(r["mdot"]), float(r["thrust"]), float(r["Isp"])
        P = data.thruster_power(r)
        d_F = m * isp * data.G0 / F - 1
        d_eta = F**2 / (2 * m * P) / float(r["eta_thruster"]) - 1
        d_I = float(r["throughput"]) * isp * data.G0 / float(r["total_impulse"]) - 1
        d_sys = (
            float(r["eta_ppu"]) * float(r["eta_thruster"]) / float(r["eta_system"]) - 1
            if r["eta_system"]
            else None
        )
        bad = abs(d_F) > 0.01 or abs(d_eta) > 0.03 or abs(d_I) > 0.02
        bad = bad or (d_sys is not None and abs(d_sys) > 0.02)
        n_flag += bad
        sys_txt = "n/a" if d_sys is None else f"{100 * d_sys:+.1f}"
        lines.append(
            f"| {r['thruster_id']} | {r['point']} | {P:.0f} | {100 * d_F:+.1f} | {100 * d_eta:+.1f} | "
            f"{100 * d_I:+.1f} | {sys_txt} | {'**check**' if bad else 'ok'} |"
        )
    return lines + [""], n_flag


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
    eng, n_eng = engine_section()
    tg, n_tg = turbine_gas_section()
    mat, n_mat = material_section()
    hyb, n_hyb = hybrid_section()
    ele, n_ele = electric_section()
    head = [
        "# Library checks",
        "",
        "Generated by `tools/check_library.py`; do not edit by hand.",
        "",
        f"Summary: {len(problems)} integrity problems, {n_cea} theoretical-performance cases outside "
        f"{TOL_CEA:g} %, {n_sca} multi-source scalars spreading more than {TOL_SCALAR:g} %, {n_eng} engine "
        f"closure checks outside tolerance, {n_tg} turbine-gas rows flagged, {n_mat} material-fit checks "
        f"flagged, {n_hyb} hybrid cases flagged, {n_ele} electric throttle points flagged.",
        "",
    ]
    OUT.write_text(
        "\n".join(
            head + integrity + cea + sca + table_section() + eng + tg + graph_section() + mat + hyb + ele
        ),
        encoding="utf-8",
    )
    print(
        f"wrote {OUT}: {len(problems)} integrity problems, {n_cea} CEA flags, {n_sca} scalar flags, "
        f"{n_eng} engine flags, {n_tg} turbine-gas flags, {n_mat} material flags, {n_hyb} hybrid flags, "
        f"{n_ele} electric flags"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
