# Changelog

New changes go above this line, newest first.

## 1.0.0

- Live mode (physics recomputed in under 50 ms while a slider drags) and Run mode (background process with progress,
  Cancel and synchronized playback of plots, 3D scene and readouts); saved runs can be overlaid for comparison.
- Common: NASA CEA thermochemistry for eight propellant pairs (equilibrium or frozen flow) and nozzle performance.
- Liquid: injector and chamber sizing, regenerative cooling, feed-network transients, pump-fed cycles
  (gas generator, expander, staged combustion), chugging and L* stability.
- Hybrid: regression laws, port and grain, core flow fields, fuel and oxidizer properties, instability and scaling,
  0-D and quasi-1D burns, N2O blowdown.
- Solid: grain designer, burn rate and K_n, 3D level-set burnback, internal ballistics with erosive burning.
- Electric: mission and power trades, electrothermal thrusters, gridded ion and Hall thrusters, ion-optics particle
  simulation, Hall discharge, low-thrust Edelbaum spiral.
- All math in LaTeX; every plot has a grid; dockable panels with a restorable default layout.
- Five color themes (Graphite, Ocean, Forest, Plum, Light) with a palette picker that collapses to names when narrow.
- Vehicle: staging and atmosphere budget; point-mass ascent with the U.S. Standard Atmosphere 1976, staging, wind,
  drag curves or a drag table, level-off guidance, and thrust profiles sent from hybrid, solid and liquid runs.
- Panels with drag grips and close buttons, a View > Panels menu and a restorable default layout; guided tutorial
  navigated with the arrow keys; solid scrollbars; crisp text at fractional display scaling.
- Reports (HTML, PDF, LaTeX tables with figures) and a user guide.
- Export to PNG, CSV, MATLAB `.mat`, HDF5, and runnable Python and MATLAB scripts.
- Windows installer with a system-requirements page and a link to the online help.
