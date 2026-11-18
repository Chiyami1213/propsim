# Changelog

New changes go above this line, newest first.

## Unreleased

- Component library, block A (fluids and propellants): `propsim/data/library` with sourced fluid scalars, CoolProp
  property tables for 11 fluids, CEA reactant cards and published theoretical-performance cases, read by
  `propsim.data`; checks in `tools/check_library.py` and `tests/data`.
- Component library, block B (component catalog): 19 component types with ports, sourced parameter defaults,
  plot outputs and cited equations, plus valve/fitting loss coefficients, roughness, L*, inducer ranges and digitized
  pump and turbine efficiency curves from Huzel & Huang.
- Component library, blocks C and D (cycles and reference engines): Huzel & Huang example engines A-1 to A-4 with
  full feed, turbopump and gas-generator data, top-level data for RL10 variants, RS-25, F-1 and J-2, five cycle
  records with sourced efficiency ranges, fuel-rich turbine gas properties, and closure checks (c*, thrust, C_F,
  pump and turbine power, cycle efficiency) in `propsim.data.closure`.

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
- Export to PNG, CSV, MATLAB `.mat`, HDF5, and runnable Python and MATLAB scripts.
- Windows installer with a system-requirements page and a link to the online help.
