# PropSim

Interactive 3D rocket-propulsion simulator (liquid, hybrid, solid, electric) for Windows 11 / Python 3.11+.

**Status: Phase 1 (skeleton) complete.** See [docs/PROGRESS.md](docs/PROGRESS.md).

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
python -m pytest                            # 43 tests: 24 reference checks, core physics, UI acceptance
```
Headless / CI: set `QT_QPA_PLATFORM=offscreen` (the 3D view then renders through an off-screen VTK plotter).

## Layout
```
propsim/core/      pure physics (no Qt): nozzle, ballistics, regression, schema, result, io
propsim/sims/      Run-mode drivers (Simulation ABC, hybrid 0-D burn, spring-mass toy)
propsim/modules/   model controllers: params + plots + 3D scene per model; registry = left-dock tree
propsim/ui/        main window, LaTeX pixmaps, gridded plots, param forms, playback, 3D viewport, theme
cases/             example .psim.json cases        tests/reference/  mathkit_check.py (24 PASS)
```
Add a model: write a `ParamModel` (one `P(...)` per parameter drives slider, tooltip and validation), a
`ModelController`, and register it in `modules/registry.py`.
