"""Case (JSON), result (HDF5), MATLAB (.mat) and CSV I/O."""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat

from .result import Result

CASE_SCHEMA = 1


def save_case(path: str | Path, model: str, params: dict) -> None:
    Path(path).write_text(json.dumps({"schema": CASE_SCHEMA, "model": model, "params": params}, indent=2),
                          encoding="utf-8")


def load_case(path: str | Path) -> tuple[str, dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("schema") != CASE_SCHEMA:
        raise ValueError(f"unsupported case schema {d.get('schema')!r}")
    return d["model"], d["params"]


def save_result_h5(path: str | Path, res: Result, model: str, params: dict) -> None:
    with h5py.File(path, "w") as f:
        f.attrs["schema"] = CASE_SCHEMA
        f.attrs["model"] = model
        f.attrs["params"] = json.dumps(params)
        f.attrs["meta"] = json.dumps(res.meta, default=float)
        f.attrs["log"] = json.dumps(res.log)
        f.create_dataset("t", data=res.t, compression="gzip")
        g = f.create_group("channels")
        for k, v in res.channels.items():
            g.create_dataset(k, data=v, compression="gzip")


def load_result_h5(path: str | Path) -> tuple[Result, str, dict]:
    with h5py.File(path, "r") as f:
        t = f["t"][()]
        ch = {k: f["channels"][k][()] for k in f["channels"]}
        res = Result(t, ch, json.loads(f.attrs["meta"]), json.loads(f.attrs["log"]))
        return res, str(f.attrs["model"]), json.loads(f.attrs["params"])


def export_h5_dict(path: str | Path, data: dict[str, np.ndarray], attrs: dict | None = None) -> None:
    """Plain named-array HDF5 (used for Live-mode exports that have no time history)."""
    with h5py.File(path, "w") as f:
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
        for k, v in data.items():
            f.create_dataset(k, data=np.asarray(v), compression="gzip")


def export_mat(path: str | Path, data: dict[str, np.ndarray]) -> None:
    """MATLAB-safe variable names (letters, digits, underscore; no leading digit)."""
    savemat(path, {_mat_name(k): np.asarray(v) for k, v in data.items()})


def export_csv(path: str | Path, data: dict[str, np.ndarray]) -> None:
    """Columns of equal length are written side by side; shorter ones are padded with blanks."""
    keys = list(data)
    n = max(len(np.atleast_1d(data[k])) for k in keys)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(",".join(keys) + "\n")
        for i in range(n):
            row = []
            for k in keys:
                a = np.atleast_1d(data[k])
                row.append(repr(float(a[i])) if i < len(a) else "")
            f.write(",".join(row) + "\n")


def _mat_name(k: str) -> str:
    s = "".join(c if c.isalnum() or c == "_" else "_" for c in k)
    return s if s and s[0].isalpha() else "v_" + s
