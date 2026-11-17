"""Code export: the Python script runs and reproduces the model; the MATLAB script and its data file are consistent."""
import ast
import re
import subprocess
import sys

import pytest
from scipy.io import loadmat

from propsim.export_code import export, matlab_script, plot_arrays, python_script, run_to_mat
from propsim.modules import registry


def _matlab_strings_closed(line: str) -> bool:
    """Scan one MATLAB line: quote opens a char literal unless it is a transpose; '' escapes; % outside = comment."""
    i, inside, prev = 0, False, " "
    while i < len(line):
        ch = line[i]
        if inside:
            if ch == "'":
                if i + 1 < len(line) and line[i + 1] == "'":
                    i += 2
                    continue
                inside, prev = False, "'"
        elif ch == "%":
            return True
        elif ch == "'" and not (prev.isalnum() or prev in ")]}_.'"):
            inside = True
        elif not ch.isspace():
            prev = ch
        i += 1
    return not inside


@pytest.mark.parametrize("key", ["common.engine", "ep.spiral", "liquid.chugging"])
def test_python_script_is_valid_and_complete(key):
    ctrl = registry.get(key)
    params = ctrl.Params().model_dump()
    src = python_script(key, params)
    tree = ast.parse(src)
    d = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == "PARAMS")
    assert ast.literal_eval(d.value) == params                      # every parameter, exact values


def test_python_script_runs(tmp_path):
    out = export("ep.spiral", registry.get("ep.spiral").Params().model_dump(), tmp_path / "spiral.py")
    code = out[0].read_text(encoding="utf-8").replace("plot(ctrl, p, r)\n", "print('PLOTTABS', len(ctrl.tabs(p, r)))\n")
    out[0].write_text(code, encoding="utf-8")
    res = subprocess.run([sys.executable, str(out[0])], cwd=tmp_path, capture_output=True, text=True, timeout=300,
                         env={**__import__("os").environ, "MPLBACKEND": "Agg", "PYTHONUTF8": "1"})
    assert res.returncode == 0, res.stderr[-2000:]
    assert "PLOTTABS 3" in res.stdout and (tmp_path / "ep_spiral_data.npz").exists()


def test_matlab_script_matches_data(tmp_path):
    key = "ep.trades"
    ctrl = registry.get(key)
    files = export(key, ctrl.Params().model_dump(), tmp_path / "mission trades.m")
    m, mat = files
    assert m.name == "mission_trades.m" and mat.name == "mission_trades_data.mat"
    src = m.read_text(encoding="utf-8")
    data = loadmat(str(mat))
    used = set(re.findall(r"D\.(T\d+P\d+(?:C\d+[xy]|M))", src))
    assert used and used <= set(data)                               # every referenced variable exists
    assert src.count("nexttile") == sum(len(t.plots) for t in ctrl.tabs(ctrl.Params(), ctrl.evaluate(ctrl.Params())))
    assert "grid on; grid minor" in src and "'Interpreter', 'latex'" in src
    for line in src.splitlines():                                   # MATLAB char literals all closed
        assert _matlab_strings_closed(line), line
    assert "function out = propsim_rerun" in src


def test_run_to_mat_cli(tmp_path):
    import json
    pj = tmp_path / "p.json"
    pj.write_text(json.dumps(registry.get("ep.spiral").Params().model_dump()), encoding="utf-8")
    out = tmp_path / "o.mat"
    res = subprocess.run([sys.executable, "-m", "propsim.export_code", "ep.spiral", str(pj), str(out)],
                         capture_output=True, text=True, timeout=300, env={**__import__("os").environ, "PYTHONUTF8": "1"})
    assert res.returncode == 0, res.stderr[-2000:]
    d = loadmat(str(out))
    assert "t" in d and "ch_dv" in d and "T1P1C1x" in d
