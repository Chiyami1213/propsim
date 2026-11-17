"""Export a model as runnable code: a Python script (reruns the model through ``propsim`` and redraws every plot with
Matplotlib) or a MATLAB script (parameters as a struct, every plot redrawn natively from a .mat data file, optional
rerun from MATLAB through Python).

Command line (also what the MATLAB script calls)::

    python -m propsim.export_code <model key> <params.json> <out.mat>
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import sys
import threading
from pathlib import Path

import numpy as np

# light-background colours for exported figures (theme quantity colours + readable neutrals)
EXPORT_COLORS = dict(oxidizer="#e07b12", fuel="#2e9d4f", hot="#d93a1f", pressure="#8a4fd8", thrust="#1f6fd1",
                     text="#132033", dim="#6b778a", accent="#1d6cb0", faint="#9aa5b5", ok="#2e8b57")


def export_color(c: str) -> str:
    return c if c.startswith("#") else EXPORT_COLORS.get(c, "#132033")


# ------------------------------------------------------------------------------------------- evaluation
def evaluate(key: str, params: dict):
    """Controller, parameter object, result (Live dict or Run ``Result``) and the plot tabs."""
    from .modules import registry
    ctrl = registry.get(key)
    p = ctrl.Params(**params)
    if ctrl.mode == "live":
        r = ctrl.evaluate(p)
    else:
        r = ctrl.sim.run(p, lambda f: None, threading.Event())
    return ctrl, p, r, ctrl.tabs(p, r)


def plot_arrays(tabs) -> dict[str, np.ndarray]:
    """Flat, MATLAB-friendly variables: ``T1P2C3x``/``y`` per curve and ``T1P2M`` (n x 2) per marker set."""
    out: dict[str, np.ndarray] = {}
    for i, tab in enumerate(tabs, 1):
        for j, ps in enumerate(tab.plots, 1):
            for k, c in enumerate(ps.curves, 1):
                out[f"T{i}P{j}C{k}x"] = np.asarray(c.x, float).ravel()
                out[f"T{i}P{j}C{k}y"] = np.asarray(c.y, float).ravel()
            if ps.markers:
                out[f"T{i}P{j}M"] = np.array([[float(x), float(y)] for x, y, _ in ps.markers])
    return out


def run_to_mat(key: str, params: dict | str, path: str | Path) -> Path:
    """Run/evaluate ``key`` and write the plot data (+ run channels) to a .mat file."""
    from scipy.io import savemat
    if isinstance(params, str):
        params = json.loads(params)
    ctrl, p, r, tabs = evaluate(key, params)
    data = plot_arrays(tabs)
    if ctrl.mode == "run":
        data["t"] = np.asarray(r.t, float)
        for k, v in r.channels.items():
            data["ch_" + re.sub(r"\W", "_", k)] = np.asarray(v, float)
    savemat(str(path), data, do_compression=True)
    return Path(path)


# ------------------------------------------------------------------------------------------- text helpers
def _tex(s: str) -> str:
    """App label -> LaTeX-normalized string (units, formulas) with Unicode dashes as LaTeX text."""
    from .ui.text import to_math
    return to_math(s or "").replace("–", "--").replace("—", "---")


def _apa(s: str) -> str:
    from .ui.text import apa
    return apa(s)


def _m_str(s: str) -> str:
    """MATLAB char literal for the LaTeX interpreter (quotes doubled, text-mode specials escaped outside math)."""
    from .ui.text import _MATH_SPLIT
    parts = []
    for seg in _MATH_SPLIT.split(_tex(s)):
        if seg.startswith("$") and seg.endswith("$") and len(seg) >= 2:
            parts.append(seg)
        else:
            parts.append(re.sub(r"(?<!\\)([%&#_])", r"\\\1", seg))
    return "'" + "".join(parts).replace("'", "''") + "'"


def _py_literal(v) -> str:
    return repr(float(v)) if isinstance(v, float) else repr(v)


def _param_lines(ctrl, params: dict, style: str) -> list[str]:
    meta = {m["name"]: m for m in ctrl.Params.fields_meta()}
    lines, group = [], None
    for name, value in params.items():
        m = meta.get(name, {})
        if m.get("group") and m["group"] != group:
            group = m["group"]
            lines.append(("    # " if style == "py" else "% ") + _apa(group))
        unit = f" ({m['units']})" if m.get("units") else ""
        rng = f" [{m['lo']:g} .. {m['hi']:g}]" if "lo" in m else ""
        choices = f" one of {m['choices']}" if m.get("choices") else ""
        desc = f"{m.get('desc', '')}{unit}{rng}{choices}"
        if style == "py":
            lines.append(f"    {name!r}: {_py_literal(value)},  # {desc}")
        else:
            val = f"'{value}'" if isinstance(value, str) else f"{float(value):.10g}"
            lines.append(f"P.{name} = {val};  % {desc}")
    return lines


# ------------------------------------------------------------------------------------------- Python
def python_script(key: str, params: dict) -> str:
    from .modules import registry
    ctrl = registry.get(key)
    when = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    body = "\n".join(_param_lines(ctrl, params, "py"))
    return f'''"""PropSim export: {_apa(ctrl.title)} ({ctrl.mode.capitalize()} model, key "{key}"), generated {when}.

Reruns the model with the parameters below through the propsim package and redraws every plot tab with Matplotlib
(major + minor grid, LaTeX labels).  Edit PARAMS and run:  python {re.sub(r'[^A-Za-z0-9]+', '_', ctrl.title.lower()).strip('_')}.py
Requires propsim (pip install -e . in the PropSim folder, or the packaged app's Python environment).
"""
import threading

import matplotlib.pyplot as plt
import numpy as np

from propsim.export_code import export_color
from propsim.modules import registry
from propsim.ui.text import apa, to_math

MODEL = "{key}"
PARAMS = {{
{body}
}}


def run(params=PARAMS):
    ctrl = registry.get(MODEL)
    p = ctrl.Params(**params)
    if ctrl.mode == "live":
        r = ctrl.evaluate(p)
    else:
        r = ctrl.sim.run(p, progress=lambda f: None, cancel=threading.Event())
        for line in r.log:
            print(line)
        print(ctrl.summary_line(p, r))
    return ctrl, p, r


def plot(ctrl, p, r):
    plt.rcParams["mathtext.fontset"] = "cm"
    for tab in ctrl.tabs(p, r):
        n = max(len(tab.plots), 1)
        fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 3.8), squeeze=False)
        fig.canvas.manager.set_window_title(apa(tab.name))
        for ax, ps in zip(axes[0], tab.plots):
            for c in ps.curves:
                ax.plot(c.x, c.y, color=export_color(c.color), lw=1.8, ls="--" if c.style == "dash" else "-",
                        label=to_math(c.label) if c.label else None)
            for x, y, col in ps.markers:
                ax.plot([x], [y], "o", color=export_color(col), ms=6)
            ax.grid(True, which="major", lw=0.7, alpha=0.6)
            ax.minorticks_on()
            ax.grid(True, which="minor", lw=0.4, alpha=0.3)
            ax.set_xlabel(to_math(ps.xlabel))
            ax.set_ylabel(to_math(ps.ylabel))
            ax.set_title(to_math(apa(ps.title)))
            if any(c.label for c in ps.curves):
                ax.legend(fontsize=8)
        fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    ctrl, p, r = run()
    data = ctrl.export_data(p, r)
    np.savez("{key.replace('.', '_')}_data.npz", **{{k: np.asarray(v) for k, v in data.items()}})
    plot(ctrl, p, r)
'''


# ------------------------------------------------------------------------------------------- MATLAB
def matlab_script(key: str, params: dict, tabs, data_file: str) -> str:
    from .modules import registry
    ctrl = registry.get(key)
    when = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    L = [f"%% PropSim export: {_apa(ctrl.title)} ({ctrl.mode.capitalize()} model, key \"{key}\"), generated {when}",
         "% Every plot of the app redrawn natively from the data file (grid on, LaTeX labels).",
         "% Set rerun = true to recompute with the parameters in P (needs Python with propsim: either MATLAB's",
         "% Python interface, see pyenv, or a 'python' executable on the system path).",
         "clear; clc;", "",
         "%% Parameters (units as shown in the app)"]
    L += _param_lines(ctrl, params, "m")
    L += ["", f"model = '{key}';", f"dataFile = '{data_file}';", "rerun = false;", "",
          "if rerun", "    dataFile = propsim_rerun(model, P, dataFile);", "end",
          "if ~isfile(dataFile)", "    error('PropSim:data', 'data file %s not found (keep it next to this script)', dataFile);", "end",
          "D = load(dataFile);", ""]
    for i, tab in enumerate(tabs, 1):
        n = max(len(tab.plots), 1)
        L += [f"%% Tab {i}: {_apa(tab.name)}",
              f"figure('Name', {_m_str(_apa(tab.name))}, 'Color', 'w', 'Position', [100 100 {460 * n} 380]);",
              f"tiledlayout(1, {n}, 'TileSpacing', 'compact', 'Padding', 'compact');"]
        for j, ps in enumerate(tab.plots, 1):
            L += ["nexttile; hold on; box on; grid on; grid minor;"]
            names = []
            for k, c in enumerate(ps.curves, 1):
                ls = "--" if c.style == "dash" else "-"
                col = export_color(c.color)
                rgb = "[" + " ".join(f"{int(col[q:q + 2], 16) / 255:.3f}" for q in (1, 3, 5)) + "]"
                disp = f", 'DisplayName', {_m_str(c.label)}" if c.label else ", 'HandleVisibility', 'off'"
                L.append(f"plot(D.T{i}P{j}C{k}x, D.T{i}P{j}C{k}y, '{ls}', 'Color', {rgb}, 'LineWidth', 1.6{disp});")
                names.append(bool(c.label))
            if ps.markers:
                L.append(f"plot(D.T{i}P{j}M(:, 1), D.T{i}P{j}M(:, 2), 'o', 'MarkerSize', 6, 'MarkerFaceColor', "
                         f"[0.85 0.23 0.12], 'MarkerEdgeColor', 'none', 'HandleVisibility', 'off');")
            L += [f"xlabel({_m_str(ps.xlabel)}, 'Interpreter', 'latex');",
                  f"ylabel({_m_str(ps.ylabel)}, 'Interpreter', 'latex');",
                  f"title({_m_str(_apa(ps.title))}, 'Interpreter', 'latex');"]
            if any(names):
                L.append("legend('Interpreter', 'latex', 'Location', 'best');")
        L.append("")
    L += ["%% Local function: recompute through Python and reload",
          "function out = propsim_rerun(model, P, dataFile)",
          "    [folder, name] = fileparts(dataFile);",
          "    if isempty(folder), folder = pwd; end",
          "    out = fullfile(folder, [name '_rerun.mat']);",
          "    params = jsonencode(P);",
          "    try",
          "        py.propsim.export_code.run_to_mat(model, params, out);",
          "    catch",
          "        jf = [tempname '.json'];",
          "        fid = fopen(jf, 'w'); fwrite(fid, params, 'char'); fclose(fid);",
          "        status = system(sprintf('python -m propsim.export_code \"%s\" \"%s\" \"%s\"', model, jf, out));",
          "        delete(jf);",
          "        if status ~= 0, error('PropSim:rerun', 'python -m propsim.export_code failed'); end",
          "    end",
          "end", ""]
    return "\n".join(L)


def export(key: str, params: dict, path: str | Path, tabs=None) -> list[Path]:
    """Write ``path`` (.py or .m); a .m also writes ``<stem>_data.mat`` next to it.  Returns the written files."""
    path = Path(path)
    if path.suffix.lower() == ".py":
        path.write_text(python_script(key, params), encoding="utf-8")
        return [path]
    if path.suffix.lower() != ".m":
        raise ValueError("export path must end in .py or .m")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", path.stem):         # MATLAB script names must be identifiers
        path = path.with_name(re.sub(r"\W", "_", path.stem).lstrip("_0123456789") or "propsim_export").with_suffix(".m")
    from scipy.io import savemat
    if tabs is None:
        _, _, _, tabs = evaluate(key, params)
    mat = path.with_name(path.stem + "_data.mat")
    savemat(str(mat), plot_arrays(tabs), do_compression=True)
    path.write_text(matlab_script(key, params, tabs, mat.name), encoding="utf-8")
    return [path, mat]


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: python -m propsim.export_code <model key> <params.json> <out.mat>")
    run_to_mat(sys.argv[1], json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")), sys.argv[3])
