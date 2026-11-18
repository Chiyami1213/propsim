"""Thrust profiles exchanged between modules: a hybrid, solid or liquid Run result becomes a stage thrust curve of the
vehicle model.  A profile is ``dict(t=[s], F=[N], mdot=[kg·s⁻¹] or None)`` starting at ignition (t = 0).

Stored as ``<data folder>/profiles/stage<k>.json`` (or a plain ``stage<k>.csv`` with columns ``t,F,mdot`` in SI that you
can write by hand) and read by the vehicle model when a stage's thrust source is *profile*.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Run-model key -> (thrust channel, thrust unit factor to N, mass-flow channels summed)
SOURCES = {
    "hybrid.burn0d": ("F", 1e3, ("mdot_ox", "mdot_f")),
    "hybrid.burn1d": ("F", 1e3, ("mdot_ox", "mdot_f")),
    "hybrid.n2o": ("F", 1e3, ("mdot_ox", "mdot_f")),
    "solid.ballistics": ("F", 1e3, ("mdot",)),
    "liquid.feed": ("F", 1e3, ("mdot_o", "mdot_f")),
    "liquid.cycle": ("F", 1e3, ("mdot",)),
}


def from_result(model_key: str, res, threshold: float = 0.01) -> dict:
    """Thrust profile of a Run result: thrust channel in N from the first to the last sample above ``threshold`` of the
    peak, time re-zeroed at the first, mass flow summed over the propellant streams."""
    if model_key not in SOURCES:
        raise ValueError(f"model {model_key!r} has no thrust channel to send to the vehicle")
    ch, k, mdots = SOURCES[model_key]
    F = np.asarray(res.channels[ch], float) * k
    on = np.where(F > threshold * F.max())[0]
    if on.size == 0:
        raise ValueError("the run has no thrust")
    i0, i1 = int(on[0]), int(on[-1])
    t = np.asarray(res.t, float)[i0:i1 + 1]
    md = sum(np.asarray(res.channels[c], float) for c in mdots)[i0:i1 + 1]
    return dict(t=(t - t[0]).tolist(), F=F[i0:i1 + 1].tolist(), mdot=md.tolist(), source=model_key)


def arrays(profile: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    t = np.asarray(profile["t"], float)
    F = np.asarray(profile["F"], float)
    md = profile.get("mdot")
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("thrust profile times must be strictly increasing")
    return t, F, (np.asarray(md, float) if md is not None else None)


def impulse(profile: dict) -> tuple[float, float]:
    """``(total impulse N·s, propellant mass kg or nan)`` of a profile."""
    t, F, md = arrays(profile)
    return float(np.trapezoid(F, t)), float(np.trapezoid(md, t)) if md is not None else float("nan")


def save(profile: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path


def load(path: str | Path) -> dict:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        d = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
        names = d.dtype.names or ()
        prof = dict(t=d["t"].tolist(), F=d["F"].tolist(), mdot=d["mdot"].tolist() if "mdot" in names else None,
                    source=path.name)
    else:
        prof = json.loads(path.read_text(encoding="utf-8"))
    arrays(prof)                                                             # validate
    return prof


def load_drag(path: str | Path) -> dict:
    """Drag table from a CSV with columns ``mach,cd`` (Mach strictly increasing)."""
    d = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    names = d.dtype.names or ()
    if "mach" not in names or "cd" not in names:
        raise ValueError("drag.csv needs the columns: mach,cd")
    m, c = np.atleast_1d(d["mach"]), np.atleast_1d(d["cd"])
    if len(m) < 2 or np.any(np.diff(m) <= 0) or np.any(c < 0):
        raise ValueError("drag.csv: at least two rows, Mach strictly increasing, Cd not negative")
    return dict(mach=m.tolist(), cd=c.tolist())


def find(profiles_dir: str | Path, stage: int) -> Path | None:
    for ext in (".json", ".csv"):
        p = Path(profiles_dir) / f"stage{stage}{ext}"
        if p.is_file():
            return p
    return None
