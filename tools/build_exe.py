"""Build the Windows executable (one-folder). Run: python tools/build_exe.py

Output: dist/PropSim/PropSim.exe (ship the whole dist/PropSim folder). Needs `pip install pyinstaller`.
Bundled data: example cases, app icon and the pre-tabulated CEA thermochemistry grids.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEP = ";" if sys.platform == "win32" else ":"


def main() -> int:
    import PyInstaller.__main__ as pyi

    data = [("cases", "cases"), ("propsim/assets", "propsim/assets"),
            ("propsim/core/thermo/data", "propsim/core/thermo/data")]
    args = [str(ROOT / "run.py"), "--name", "PropSim", "--noconfirm", "--clean", "--windowed",
            "--icon", str(ROOT / "propsim" / "assets" / "icon.ico"),
            "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build" / "pyinstaller"),
            "--specpath", str(ROOT / "build"),
            "--collect-submodules", "propsim",
            "--hidden-import", "pyvistaqt", "--hidden-import", "h5py",
            "--hidden-import", "CoolProp", "--hidden-import", "CoolProp.CoolProp",
            "--exclude-module", "tkinter", "--exclude-module", "pytest", "--exclude-module", "rocketcea"]
    for src, dst in data:
        args += ["--add-data", f"{ROOT / src}{SEP}{dst}"]
    pyi.run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
