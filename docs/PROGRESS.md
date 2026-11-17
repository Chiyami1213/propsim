# PropSim progress

## Phase 1 — Skeleton: DONE (awaiting go-ahead for Phase 2)

**Done**
- Repo skeleton (Section 3), `pyproject.toml`, README; `mathkit_check.py` in `tests/reference/` (24 PASS under pytest).
- Main window per Section 5: left/right docks, 55:45 centre splitter, 40 px bottom playback bar, dark/light theme
  tokens + QSS, dock layout saved via `QSettings`.
- LaTeX → cached `QPixmap` labels/buttons/tab captions (transparent background, theme-aware).
- Gridded pyqtgraph plots with LaTeX axes + shared playback cursor; Matplotlib PNG export (major+minor grid, mathtext).
- Schema → slider forms (`P(...)`: symbol, units, dims, bounds, step); hover box with `[dimensions]`.
- Background Run worker (QThread) with progress + Cancel; `PlaybackModel` (60 Hz) driving scrubber, cursors, 3D, readout.
- JSON cases, HDF5 results, `.mat` / CSV / `.h5` export, saved-runs list, compare overlay.
- Live demo: nozzle area ratio → C_F plot, Mach and pressure distributions, 3D bell (Mach/T/p colouring). ~1 ms/eval.
- Run demo: 0-D hybrid burn (HTPB/GOX power law) → port growth in 3D, P_c(t), F(t), O/F(t); integrator verified
  against the analytic port radius (50.846 mm) and fuel mass conservation.
- Toy Run model (spring-mass) to prove the pipeline independent of propulsion physics.

**Next (Phase 2):** RocketCEA wrapper + tabulated fallback, Rao-chart contour angles, validation tests, MOC contour.

**Known issues / differences from the mockup**
- Combustion properties are an analytic **placeholder** (`core/thermo/surrogate.py`), not CEA. With the mockup's
  inputs the 0-D model gives O/F ≈ 1.4→4.5 and P_c ≈ 3.7 MPa, not the mockup's illustrative 1.9→2.5 and 3.42 MPa.
- The Run demo is 0-D (single port radius), so the port colouring is uniform along x (the mockup's axial gradient
  needs the Phase 3 quasi-1D march). Tracers are illustrative (motion tied to cumulative oxidizer throughput).
- `theta_n/theta_e` of the bell contour are a placeholder trend, not the Rao chart table.
- Native OS title bar instead of the mockup's drawn one; colour bar is a Qt overlay, not a VTK scalar bar.
- `QtInteractor` cannot create a GL context under `QT_QPA_PLATFORM=offscreen`; the viewport falls back to an
  off-screen VTK plotter shown in a QLabel (screenshots/tests only; real runs use `QtInteractor`).
- Debugging note: pyqtgraph crashed (access violation) painting the dashed `InfiniteLine` cursor on an *empty* plot;
  the cursor is now hidden until a plot has data. **Residual risk:** after that fix 29 of 30 full-suite runs were clean and
  one crashed early on with an access violation whose trace was not captured (25 consecutive runs afterwards were clean).
  If it recurs, capture `faulthandler` output and check pyqtgraph paint paths (cursor/scatter items) first.
- Remaining style nits (E501/E731/E702 in a few lines) are not yet cleaned; `black`/`ruff format` pass pending.
