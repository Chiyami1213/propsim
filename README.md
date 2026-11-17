# PropSim

Interactive 3D rocket propulsion simulator for Windows: liquid, hybrid, solid and electric propulsion in one desktop
application, with live parameter studies, time-domain simulations, synchronized 3D views and publication-style plots.

## Features
- **Live mode**: physics recomputes in under 50 ms while you drag a slider; plots, 3D scene and readouts follow.
- **Run mode**: longer simulations run in a separate process with progress and Cancel, then play back with a
  scrubber synchronized to the plots, the 3D scene and the readouts. Saved runs can be overlaid for comparison.
- **Modules**
  - *Common*: NASA CEA thermochemistry (eight propellant pairs, equilibrium or frozen) and nozzle performance.
  - *Liquid*: injector and chamber sizing, regenerative cooling (Bartz), feed-network transients (water hammer,
    valve timing), pump-fed cycles (gas generator, expander, staged combustion), chugging and L\* stability.
  - *Hybrid*: regression laws, port and grain, core flow fields, fuel and oxidizer, instability and scaling,
    0-D and quasi-1D burns, N₂O self-pressurized blowdown.
  - *Solid*: grain designer (star, wagon wheel, finocyl and more), burn rate and K_n, 3D level-set burnback,
    internal ballistics with erosive burning.
  - *Electric*: mission and power trades (Stuhlinger), electrothermal thrusters, gridded ion and Hall thrusters,
    ion-optics particle simulation, Hall discharge, low-thrust Edelbaum spiral with power degradation.
- **All math in LaTeX**: plot labels, tables, readouts and hover help; every plot has a grid.
- **Export**: PNG figures, CSV, MATLAB `.mat`, HDF5, and runnable code:
  - *Python script* (`.py`): rebuilds the model with your parameters through the `propsim` package and redraws
    every plot with Matplotlib.
  - *MATLAB script* (`.m` + `_data.mat`): every plot redrawn natively (grid, LaTeX labels); set `rerun = true`
    to recompute from MATLAB through Python.

## System requirements

|                  | Minimum                                          | Recommended                         |
|------------------|--------------------------------------------------|-------------------------------------|
| Operating system | Windows 10 64-bit (version 1809) or Windows 11   | Windows 11 64-bit                   |
| Processor        | 64-bit x86 (x86-64), 2 cores                     | 4 or more cores                     |
| Memory           | 4 GB RAM                                         | 8 GB RAM or more (large 3D runs)    |
| Graphics         | OpenGL 3.2 or newer, up-to-date driver           | Recent integrated or discrete GPU   |
| Disk space       | 1.5 GB free (about 0.85 GB installed)            |                                     |
| Display          | 1366 × 768                                       | 1920 × 1080 or larger               |

The installed application needs no Python and works fully offline. Windows on ARM (x64 emulation) is not tested.
Running from source needs Python 3.11 or newer.

## Installation
**Windows installer**: run `PropSim-<version>-Setup.exe` from the
[Releases](https://github.com/Chiyami1213/propsim/releases) page. It installs for your account (or for all users),
adds Start-menu shortcuts for PropSim and this help page, and can be removed from Windows *Apps & features*.

**From source**:
```powershell
pip install -e .
python run.py
```
Optional extras: `pip install -e ".[cea]"` queries NASA CEA directly through RocketCEA (otherwise the bundled
CEA tables are used); `pip install -e ".[rao]"` enables Rao optimum-bell angles from the Huzel & Huang chart via
pygasflow (GPL-3.0, not bundled; without it the bell contour uses labelled placeholder angles).

## Getting started
1. Pick a model in the **Modules** tree: *(Live)* models update as you move the sliders, *(Run)* models start with
   **Run** (Ctrl+R).
2. Hover over a parameter to see its meaning and dimensions.
3. Use the 3D toolbar for cutaway, colour field, colour bar and snapshots; the plot tabs below show the results.
4. Export from the bottom-right buttons or **File > Export Code**. **Help > Online Help** (F1) opens this page.

## Where your data is stored
Saved runs, cases, exports and automatic backups are kept in `%APPDATA%\PropSim` by default:
`runs\`, `cases\`, `exports\`, `backups\` (previous versions of overwritten cases and your last session).
Choose another location with **File > Data Folder**, which can also copy existing runs and cases there.
**File > Restore Last Session** reopens the model and parameters from your previous session.

## Model notes and limitations
- Combustion properties come from NASA CEA tables (about 0.14 % from direct CEA over the tabulated range).
- Without pygasflow, Rao bell angles are placeholders and labelled as such.
- The Hall discharge model has fixed field and electron-temperature profiles: the discharge settles and the
  breathing mode is not reproduced; the breathing frequency is given by the Fife (predator-prey) estimate.
- Ion optics uses a fixed emission plane (no plasma-meniscus solve) and flags runs near the perveance limit that
  do not converge; read the focusing curve as a trend.
- The solid internal-ballistics 3D view uses an equivalent circular port.

## Building the Windows package
```powershell
pip install pyinstaller
python tools/build_exe.py          # dist/PropSim/PropSim.exe
python tools/build_installer.py    # dist/installer/PropSim-<version>-Setup.exe (needs Inno Setup 6)
```

## Help and feedback
Questions and bug reports: [GitHub Issues](https://github.com/Chiyami1213/propsim/issues).
