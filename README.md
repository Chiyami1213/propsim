# PropSim

Interactive 3D rocket-propulsion simulator (liquid, hybrid, solid, electric) for Windows 11 / Python 3.11+.

## Rules the UI follows
1. **Live mode** (< 50 ms) recomputes while a slider drags. **Run mode** executes in a background thread with
   progress and Cancel, then plays back with a scrubber synchronised to the plots, the 3D scene and the readout.
   Both modes call the same physics in `propsim/core/`.
2. Every 2D plot has a grid; all math is LaTeX (mathtext), units are in parentheses, dimensions in `[ ]`.

## Install and run
```powershell
pip install -e ".[dev]"
python run.py                               # opens cases/hybrid_demo.psim.json (no install needed beyond deps)
python run.py cases/nozzle_demo.psim.json
python tools/make_icon.py                   # regenerate propsim/assets/icon.png/.ico
python tools/build_exe.py                   # Windows build: dist/PropSim/PropSim.exe (needs pyinstaller)
python -m pytest                            # reference checks, core physics, validation cases, UI acceptance
```
Headless / CI: set `QT_QPA_PLATFORM=offscreen` (the 3D view then renders through an off-screen VTK plotter).

## Thermochemistry
Combustion properties (c\*, T_c, γ, molar mass, I_sp) come from NASA CEA for eight propellant pairs
(LOX with RP-1, LH₂, CH₄, ethanol or HTPB; GOX/HTPB; N₂O/HTPB; N₂O₄/MMH), equilibrium or frozen flow.
Pre-tabulated CEA grids ship with the package, so the app runs without CEA installed. Install the optional
`rocketcea` (`pip install -e ".[cea]"`) to query CEA directly; the *Thermochemistry & nozzle* model can switch
between the two. `python tools/build_thermo_tables.py` regenerates the grids.

## Layout
```
propsim/core/      pure physics (no Qt): thermo (CEA), nozzle, ballistics, regression, schema, result, io
propsim/sims/      Run-mode drivers (Simulation ABC, hybrid 0-D burn, spring-mass toy)
propsim/modules/   model controllers: params + plots + 3D scene per model; registry = left-dock tree
propsim/ui/        main window, LaTeX pixmaps, gridded plots, param forms, playback, 3D viewport, theme
cases/             example .psim.json cases        tests/reference/  mathkit_check.py (24 PASS)
```
Add a model: write a `ParamModel` (one `P(...)` per parameter drives slider, tooltip and validation), a
`ModelController`, and register it in `modules/registry.py`.
