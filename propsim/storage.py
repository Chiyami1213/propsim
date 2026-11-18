"""Where PropSim keeps local data: saved runs, cases, exports and backups.

Default: ``%APPDATA%\\PropSim`` on Windows (``$XDG_DATA_HOME/PropSim`` or ``~/.local/share/PropSim`` elsewhere).
A folder chosen by the user in the app (File > Data Folder) takes precedence, then the ``PROPSIM_HOME`` environment
variable, then the default.  Layout::

    <home>/runs/      saved Run results (*.psim.h5)
    <home>/cases/     saved cases (*.psim.json)
    <home>/exports/   PNG / CSV / .mat / .h5 / code exports
    <home>/backups/   automatic backups: overwritten cases and the last session
    <home>/profiles/  thrust profiles sent to the vehicle model (stage1.json ...)
    <home>/screenshots/  window screenshots taken with F2
    <home>/designs/   saved designs (*.psimdesign.json) from the designers
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

APP = "PropSim"
SUBDIRS = ("runs", "cases", "exports", "backups", "profiles", "screenshots", "designs")
KEEP_BACKUPS = 20


def platform_default() -> Path:
    if os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / APP
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP


def resolve_home(user_choice: str | os.PathLike | None = None) -> Path:
    """User-chosen folder > ``PROPSIM_HOME`` > platform default."""
    if user_choice:
        return Path(user_choice)
    if os.environ.get("PROPSIM_HOME"):
        return Path(os.environ["PROPSIM_HOME"])
    return platform_default()


def ensure_layout(home: Path) -> Path:
    for d in SUBDIRS:
        (home / d).mkdir(parents=True, exist_ok=True)
    return home


def backup_file(path: str | os.PathLike, home: Path, keep: int = KEEP_BACKUPS) -> Path | None:
    """Copy an existing file into ``<home>/backups`` with a timestamp; keep the newest ``keep`` copies per file."""
    src = Path(path)
    if not src.is_file():
        return None
    bdir = home / "backups"
    bdir.mkdir(parents=True, exist_ok=True)
    stem = src.name
    dst = bdir / f"{stem}.{time.strftime('%Y%m%d-%H%M%S')}.bak"
    k = 1
    while dst.exists():
        dst = bdir / f"{stem}.{time.strftime('%Y%m%d-%H%M%S')}-{k}.bak"
        k += 1
    shutil.copy2(src, dst)
    old = sorted(bdir.glob(f"{stem}.*.bak"), key=lambda p: p.stat().st_mtime)
    for p in old[:-keep] if keep > 0 else []:
        p.unlink(missing_ok=True)
    return dst


def copy_data(src: Path, dst: Path) -> int:
    """Copy runs and cases from one data folder to another (existing files are kept). Returns files copied."""
    n = 0
    for sub in ("runs", "cases"):
        if not (src / sub).is_dir():
            continue
        (dst / sub).mkdir(parents=True, exist_ok=True)
        for f in (src / sub).iterdir():
            if f.is_file() and not (dst / sub / f.name).exists():
                shutil.copy2(f, dst / sub / f.name)
                n += 1
    return n


def migrate_legacy(home: Path) -> int:
    """One-time copy of runs from the pre-1.0 location ``~/.propsim`` into an empty data folder."""
    legacy = Path.home() / ".propsim"
    if legacy.resolve() == home.resolve() or not (legacy / "runs").is_dir():
        return 0
    if any((home / "runs").glob("*.psim.h5")):
        return 0
    return copy_data(legacy, home)
