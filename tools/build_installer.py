"""Build the Windows installer (Inno Setup 6): python tools/build_installer.py [--rebuild-exe] [--fast]

Steps: (optionally) PyInstaller build -> wizard artwork -> ISCC compile.
Output: dist/installer/PropSim-<version>-Setup.exe.  Needs Inno Setup 6 (ISCC.exe on PATH or in a default location).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Inno Setup 6/ISCC.exe",
              Path("C:/Program Files (x86)/Inno Setup 6/ISCC.exe"), Path("C:/Program Files/Inno Setup 6/ISCC.exe")]


def find_iscc() -> Path:
    w = shutil.which("ISCC")
    if w:
        return Path(w)
    for c in CANDIDATES:
        if c.exists():
            return c
    raise SystemExit("Inno Setup 6 not found: install it (winget install JRSoftware.InnoSetup) or put ISCC on PATH")


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from propsim import HELP_URL, REPO_URL, __version__
    if "--rebuild-exe" in sys.argv:
        subprocess.run([sys.executable, str(ROOT / "tools" / "build_exe.py")], check=True)
    dist = ROOT / "dist" / "PropSim"
    if not (dist / "PropSim.exe").exists():
        raise SystemExit("dist/PropSim/PropSim.exe missing: run tools/build_exe.py first (or pass --rebuild-exe)")
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_installer_art.py")], check=True)
    out = ROOT / "dist" / "installer"
    out.mkdir(parents=True, exist_ok=True)
    defs = dict(AppVersion=__version__, DistDir=dist, OutDir=out, ArtDir=ROOT / "build" / "installer",
                IconFile=ROOT / "propsim" / "assets" / "icon.ico", ReqFile=ROOT / "tools" / "installer" / "requirements.txt",
                RepoURL=REPO_URL, HelpURL=HELP_URL)
    if "--fast" in sys.argv:                                     # quick test build (larger file)
        defs["Compress"] = "lzma2/fast"
    cmd = [str(find_iscc()), "/Qp"] + [f"/D{k}={v}" for k, v in defs.items()] + [str(ROOT / "tools" / "installer" / "propsim.iss")]
    subprocess.run(cmd, check=True)
    print(f"installer -> {out / f'PropSim-{__version__}-Setup.exe'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
