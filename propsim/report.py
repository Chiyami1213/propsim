"""Report export: the current model, parameters, results and every plot tab as a self-contained HTML page, a PDF, or
LaTeX tables with the figures as PNG files.  Math is typeset as LaTeX everywhere (rendered with Matplotlib mathtext in
HTML and PDF; written as LaTeX source in the .tex file).  Figures use the light theme so they print well."""
from __future__ import annotations

import base64
import datetime as _dt
import html
import io
import re
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import mathtext  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

from . import __version__  # noqa: E402
from .ui.export import tab_to_figure  # noqa: E402
from .ui.text import _MATH_SPLIT, apa, dims_tex, to_math  # noqa: E402
from .ui.theme import TOKENS  # noqa: E402

LIGHT = TOKENS["light"]
INK = "#132033"


# ------------------------------------------------------------------------------------------- content
def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def parameter_rows(ctrl, p) -> list[dict]:
    """One row per slider/choice: symbol (LaTeX), description, value text, units (LaTeX), dimensions (LaTeX), group."""
    rows = []
    for m in ctrl.Params.fields_meta():
        v = getattr(p, m["name"])
        if m.get("kind") == "choice":
            val = m["labels"][m["choices"].index(v)] if v in m["choices"] else str(v)
        else:
            val = _fmt(v)
        rows.append(dict(symbol=f"${m['symbol']}$", desc=m["desc"], value=val,
                         units=f"({m['units']})" if m.get("units") else "", dims=m.get("dims", "1"),
                         group=apa(m.get("group", ""))))
    return rows


def result_rows(ctrl, p, r) -> tuple[list[tuple[str, str]], str, str]:
    """``(rows of (quantity, value), summary line, caption)`` from the readouts at the final frame."""
    live = ctrl.mode == "live"
    frame = None if live else r.n - 1
    rows = []
    for line in ctrl.readout(p, r, frame):
        if " = " in line:
            sym, val = line.split(" = ", 1)
            rows.append((sym, val))
        elif line.strip():
            rows.append((line, ""))
    summary = "" if live else ctrl.summary_line(p, r)
    return rows, summary, ctrl.caption(p, r, frame)


def event_rows(r) -> list[tuple[str, str]]:
    ev = getattr(r, "meta", {}).get("events") if not isinstance(r, dict) else None
    return [(str(n), f"{t:.2f}") for n, t in (ev or [])]


def figures(ctrl, p, r, dpi: int = 130):
    """One light-theme Matplotlib figure per plot tab (cursor at the end of a Run)."""
    tabs = [t for t in ctrl.tabs(p, r) if t.plots and any(ps.curves for ps in t.plots)]
    cur = None if ctrl.mode == "live" else float(r.t[-1])
    return [(apa(t.name), tab_to_figure(t, cur, dpi, tokens=LIGHT)) for t in tabs]


# ------------------------------------------------------------------------------------------- math in HTML
@lru_cache(maxsize=2048)
def _math_img(tex: str, size_pt: float = 10.0, dpi: int = 220) -> str:
    """``$...$`` -> inline ``<img>`` (base64 PNG, baseline-aligned) so the page needs no fonts or scripts."""
    buf = io.BytesIO()
    prop = FontProperties(size=size_pt)
    try:
        mathtext.math_to_image(tex, buf, prop=prop, dpi=dpi, format="png", color=INK)
        parser = mathtext.MathTextParser("agg")
        depth = parser.parse(tex, dpi=dpi, prop=prop).depth
    except ValueError:
        return html.escape(tex.strip("$"))
    from PIL import Image
    buf.seek(0)
    w, h = Image.open(buf).size
    pt = lambda px: px / dpi * 72.0
    return (f'<img alt="{html.escape(tex)}" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" '
            f'style="height:{pt(h):.2f}pt;vertical-align:-{pt(depth):.2f}pt">')


def html_text(s: str) -> str:
    """Normalize (units, symbols) to LaTeX, escape the words and typeset the math."""
    out = []
    for part in _MATH_SPLIT.split(to_math(s or "")):
        out.append(_math_img(part) if (part.startswith("$") and part.endswith("$") and len(part) >= 2)
                   else html.escape(part))
    return "".join(out)


def _fig_b64(fig, dpi: int = 130) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    return base64.b64encode(buf.getvalue()).decode()


CSS = """
body{font-family:'Segoe UI',Arial,sans-serif;color:#132033;max-width:980px;margin:2rem auto;padding:0 1.2rem;line-height:1.45}
h1{font-size:1.7rem;margin-bottom:.2rem} h2{font-size:1.15rem;margin-top:2rem;border-bottom:2px solid #1d6cb0;padding-bottom:.2rem}
.meta{color:#5b6779;font-size:.9rem} table{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.92rem}
th,td{border:1px solid #d6dce5;padding:.3rem .5rem;text-align:left;vertical-align:middle} th{background:#e9eef5}
tr.group td{background:#f3f5f8;font-weight:600} figure{margin:1rem 0} figure img{max-width:100%;border:1px solid #d6dce5}
figcaption{color:#5b6779;font-size:.9rem} .summary{background:#eaf1fa;border-left:4px solid #1d6cb0;padding:.5rem .8rem}
pre{background:#f3f5f8;padding:.6rem;overflow:auto;font-size:.82rem}
"""


def build_html(ctrl, p, r, log: list[str] | None = None) -> str:
    title = apa(ctrl.title)
    rows = parameter_rows(ctrl, p)
    res, summary, caption = result_rows(ctrl, p, r)
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(title)} Report</title>"
             f"<style>{CSS}</style></head><body>",
             f"<h1>{html.escape(title)}</h1><p class='meta'>{html.escape(apa(ctrl.module))} module, "
             f"{'Live' if ctrl.mode == 'live' else 'Run'} model <code>{html.escape(ctrl.key)}</code> · PropSim {__version__} · "
             f"{_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>"]
    if summary:
        parts.append(f"<p class='summary'>{html_text(summary)}</p>")
    if res:
        parts.append("<h2>Results</h2><table><tr><th>Quantity</th><th>Value</th></tr>" + "".join(
            f"<tr><td>{html_text(q)}</td><td>{html_text(v)}</td></tr>" for q, v in res) + "</table>")
        if caption:
            parts.append(f"<p class='meta'>{html_text(caption)}</p>")
    ev = event_rows(r)
    if ev:
        parts.append("<h2>Events</h2><table><tr><th>Event</th><th>$t$ (s)</th></tr>".replace("$t$ (s)", "")
                     + "".join(f"<tr><td>{html.escape(n)}</td><td>{t}</td></tr>" for n, t in ev) + "</table>")
        parts[-1] = parts[-1].replace("<th></th>", f"<th>{html_text(r'$t$ (s)')}</th>")
    parts.append("<h2>Parameters</h2><table><tr><th>Symbol</th><th>Description</th><th>Value</th><th>Units</th>"
                 "<th>Dimensions</th></tr>")
    group = None
    for row in rows:
        if row["group"] != group:
            group = row["group"]
            parts.append(f"<tr class='group'><td colspan='5'>{html.escape(group)}</td></tr>")
        parts.append(f"<tr><td>{html_text(row['symbol'])}</td><td>{html.escape(row['desc'])}</td>"
                     f"<td>{html_text(row['value'])}</td><td>{html_text(row['units'])}</td>"
                     f"<td>{html_text(dims_tex(row['dims']))}</td></tr>")
    parts.append("</table>")
    figs = figures(ctrl, p, r)
    if figs:
        parts.append("<h2>Plots</h2>")
    for name, fig in figs:
        parts.append(f"<figure><img alt='{html.escape(name)}' src='data:image/png;base64,{_fig_b64(fig)}'>"
                     f"<figcaption>{html.escape(name)}</figcaption></figure>")
        plt.close(fig)
    if log:
        parts.append("<h2>Run Log</h2><div>" + "".join(f"<div>{html_text(l)}</div>" for l in log) + "</div>")
    parts.append("</body></html>")
    return "".join(parts)


# ------------------------------------------------------------------------------------------- PDF
def build_pdf(path: str | Path, ctrl, p, r, log: list[str] | None = None) -> None:
    title = apa(ctrl.title)
    rows = parameter_rows(ctrl, p)
    res, summary, caption = result_rows(ctrl, p, r)
    with plt.rc_context({"mathtext.fontset": "cm", "font.family": "DejaVu Sans"}), PdfPages(str(path)) as pdf:
        # ---- page 1: header, summary, results
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.07, 0.95, title, fontsize=20, fontweight="bold", color=INK)
        fig.text(0.07, 0.925, f"{apa(ctrl.module)} module, {'Live' if ctrl.mode == 'live' else 'Run'} model · PropSim "
                 f"{__version__} · {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=9, color="#5b6779")
        y = 0.89
        if summary:
            fig.text(0.07, y, to_math(summary), fontsize=10.5, color="#1d6cb0")
            y -= 0.035
        fig.text(0.07, y, "Results", fontsize=13, fontweight="bold", color=INK)
        y -= 0.03
        for q, v in res:
            fig.text(0.08, y, to_math(q), fontsize=11, color=INK)
            fig.text(0.40, y, to_math(v), fontsize=11, color=INK)
            y -= 0.028
        ev = event_rows(r)
        if ev:
            y -= 0.015
            fig.text(0.07, y, "Events", fontsize=13, fontweight="bold", color=INK)
            y -= 0.028
            for n, t in ev[:18]:
                fig.text(0.08, y, n, fontsize=9.5, color=INK)
                fig.text(0.40, y, f"t = {t} s".replace("t =", "$t$ ="), fontsize=9.5, color=INK)
                y -= 0.022
        if caption:
            fig.text(0.07, 0.04, to_math(caption), fontsize=8, color="#5b6779")
        pdf.savefig(fig)
        plt.close(fig)
        # ---- parameter table pages
        per = 38
        for start in range(0, len(rows), per):
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.07, 0.95, "Parameters", fontsize=14, fontweight="bold", color=INK)
            cells = [[row["symbol"], row["value"], row["units"], row["desc"][:62]] for row in rows[start:start + per]]
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.86])
            ax.axis("off")
            tb = ax.table(cellText=[[to_math(c) for c in r_] for r_ in cells], colLabels=["Symbol", "Value", "Units", "Description"],
                          colWidths=[0.14, 0.2, 0.16, 0.5], loc="upper center", cellLoc="left")
            tb.auto_set_font_size(False)
            tb.set_fontsize(8.5)
            tb.scale(1, 1.55)
            for (ri, ci), cell in tb.get_celld().items():
                cell.set_edgecolor("#d6dce5")
                if ri == 0:
                    cell.set_facecolor("#e9eef5")
            pdf.savefig(fig)
            plt.close(fig)
        # ---- one page per plot tab
        for name, f in figures(ctrl, p, r, dpi=110):
            f.suptitle(name, color=INK, fontsize=12)
            f.set_size_inches(f.get_size_inches()[0], f.get_size_inches()[1] + 0.3)
            pdf.savefig(f, facecolor=f.get_facecolor())
            plt.close(f)
        if log:
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.07, 0.95, "Run Log", fontsize=14, fontweight="bold", color=INK)
            yy = 0.92
            for line in log[:60]:
                fig.text(0.07, yy, to_math(line)[:120], fontsize=7.5, color=INK, family="DejaVu Sans")
                yy -= 0.0145
            pdf.savefig(fig)
            plt.close(fig)


# ------------------------------------------------------------------------------------------- LaTeX
def _tex_escape(s: str) -> str:
    return re.sub(r"([#%&_{}])", r"\\\1", s)


def _tex_math_text(s: str) -> str:
    """Words escaped for LaTeX, ``$...$`` kept as math."""
    out = []
    for part in _MATH_SPLIT.split(to_math(s or "")):
        out.append(part if (part.startswith("$") and part.endswith("$") and len(part) >= 2) else _tex_escape(part))
    return "".join(out).replace("–", "--")


def build_latex(path: str | Path, ctrl, p, r) -> list[Path]:
    """``<name>.tex`` (booktabs tables and figure includes) plus ``<name>_fig<k>.png``; returns the files written."""
    path = Path(path)
    stem = re.sub(r"\W", "_", path.stem)
    rows = parameter_rows(ctrl, p)
    res, summary, _ = result_rows(ctrl, p, r)
    files = [path]
    tex = ["% PropSim report: " + apa(ctrl.title), r"\documentclass[11pt]{article}",
           r"\usepackage[margin=2.2cm]{geometry}\usepackage{booktabs,longtable,graphicx,amsmath}",
           r"\begin{document}", rf"\section*{{{_tex_escape(apa(ctrl.title))}}}"]
    if summary:
        tex.append(_tex_math_text(summary) + r"\par")
    if res:
        tex += [r"\begin{table}[h]\centering", r"\begin{tabular}{ll}\toprule Quantity & Value \\ \midrule"]
        tex += [f"{_tex_math_text(q)} & {_tex_math_text(v)} \\\\" for q, v in res]
        tex += [r"\bottomrule\end{tabular}", r"\caption{Results.}", r"\end{table}"]
    tex += [r"\begin{longtable}{llll}", r"\toprule Symbol & Value & Units & Description \\ \midrule \endhead"]
    for row in rows:
        tex.append(f"{_tex_math_text(row['symbol'])} & {_tex_math_text(row['value'])} & {_tex_math_text(row['units'])} & "
                   f"{_tex_math_text(row['desc'])} \\\\")
    tex += [r"\bottomrule\caption{Parameters.}\end{longtable}"]
    for k, (name, fig) in enumerate(figures(ctrl, p, r), 1):
        png = path.with_name(f"{stem}_fig{k}.png")
        fig.savefig(png, dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        files.append(png)
        tex += [r"\begin{figure}[h]\centering", rf"\includegraphics[width=\linewidth]{{{png.name}}}",
                rf"\caption{{{_tex_escape(name)}.}}", r"\end{figure}"]
    tex.append(r"\end{document}")
    path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    return files


def export(kind: str, path: str | Path, ctrl, p, r, log: list[str] | None = None) -> list[Path]:
    """``kind`` is ``html``, ``pdf`` or ``tex``."""
    path = Path(path)
    if kind == "html":
        path.write_text(build_html(ctrl, p, r, log), encoding="utf-8")
        return [path]
    if kind == "pdf":
        build_pdf(path, ctrl, p, r, log)
        return [path]
    if kind == "tex":
        return build_latex(path, ctrl, p, r)
    raise ValueError(f"unknown report kind {kind!r}")
