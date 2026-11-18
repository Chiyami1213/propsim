# Changelog

New changes go above this line, newest first.

## Unreleased

- Solid motors: grain design and internal ballistics extended. New sections: moon burner, C-slot, D-grain, X-core and
  end burner (no port, burns its aft face); any section can be repeated as identical segments with inhibited or
  burning end faces (a BATES motor is a circular bore with both ends burning), also in the 3D burnback. The Grain
  Designer now plots the burning area and K_n with segments and end faces. The burn rate can follow pressure bands
  (your own table, or Nakka's 1999 KNSB and KNDX bands, now in the library under `solid/`). Internal Ballistics adds
  c* efficiency, the conical divergence loss, other C_F losses, ambient pressure, a two-phase gamma for the nozzle,
  throat erosion, Isp(t), C_F(t), design checks (port-to-throat ratio, peak mass flux, volumetric loading, L/D),
  the motor impulse class, and a *File, Export Thrust Curve (.eng)* export. Validated against Nakka's Kappa-SB static
  test KSB-002 (burn time, C_F and total impulse within 6 %) and the Shuttle RSRM burn law (AIAA 2000-3803).

- Cycle sandbox: a *3D Model* panel shows the selected component, sized from its parameters (lines, valves at their
  solved position, pumps by impeller kind and stages, turbines by arrangement, tanks with their fill, chamber and Rao
  nozzle contour, jacket channels, injectors, manifolds). Cutaway, a readout of the solved state, and wall
  temperature, heat flux and coolant temperature from the jacket march on the chamber, nozzle and jacket.
- Hybrid motors (quasi-1D burn and N2O blowdown): the port can be *Custom* (the same shape editor as the solid grain,
  with inhibited shapes), and a *fuel grain stack* gives each grain its own length, port, fuel density, regression law
  (power or Marxman) and chemistry: the motor's CEA pair, another CEA pair, or your own table of c*, Tc and gamma
  versus O/F. Several fuels are mixed by fuel mass flow (frozen composition). A custom circular port reproduces the
  circular-port solver within 0.1 %. The 3D view draws the equivalent circular port.
- Nuclear thermal propulsion, a new module with six models following Emrich, *Principles of Nuclear Rocket
  Propulsion* (2nd ed.): NTR performance against LOX / LH₂ with reactor power and core size; one-group reactor
  criticality (bare or reflected sphere, cylinder, slab); fuel-element thermal hydraulics (chopped-cosine power, peak
  fuel temperature, Dittus–Boelter with CoolProp hydrogen, channel pressure drop); hot bleed, cold bleed and expander
  cycles with real hydrogen and a T–s diagram; xenon and samarium over a burn, coast and restart; and a Run model for
  six-group reactor kinetics with fuel, moderator and hydrogen-density feedback (startup under outlet temperature
  control, reactivity step, scram with decay heat and cooldown propellant, closed-loop Bode plot). Physics in
  `propsim/core/nuclear`; `tests/validation/test_nuclear.py` reproduces the book's worked examples (Isp at area ratio
  200, critical sphere, fuel cell peak temperature, expander pump pressure, inhour period, shutdown cooling, NERVA-type
  stability margin, xenon peak after shutdown).
- Liquid engines: an engine cycle sandbox (*Liquid, Cycle sandbox*, Ctrl+E). Build any cycle on a flow diagram from
  the component library, choose propellants, parts and cooling (regenerative jacket, radiative, ablative, uncooled),
  and solve the steady operating point with real-fluid properties (CoolProp): station pressures, temperatures,
  phases and flows, pump, turbine, jacket, gas generator, chamber and nozzle states, shaft speeds and warnings (NPSH,
  injector stiffness, turbine inlet temperature). Design targets size free parameters such as valve positions. The
  RS-25 style staged-combustion template solves at 20.48 MPa and O/F 6.0. Steady state only.
- Solid motors: build your own grain. The section kind *Custom* takes a list of shapes (circle, rectangle, radial slot,
  polygon, star, with polar repeats) and any shape can be marked inhibited, so its surface is coated and does not
  regress. A new *grain stack* describes a motor as several grains, each with its own length, section, propellant
  values and burning or inhibited end faces, and an option lets the outer surface burn too. Different propellants in
  one chamber are mixed by generation mass flow (frozen composition). Both editors are dialogs reached from the
  parameter panel; the data is stored as JSON in the case file. A single grain reproduces the earlier solver. The
  3-D burnback accepts custom sections.
- Cycle sandbox: a Play Flow button animates the flow diagram itself; the Cycle Animation panel redraws the engine as a
  textbook-style schematic (pumps, turbine and shaft, gas generator or preburner, valves, chamber, nozzle and regenerative
  cooling tubes) for pressure-fed, gas-generator, expander and staged-combustion cycles, and falls back to the diagram for
  custom cycles.
- Cycle sandbox: ANSI/ISA-5.1 P&ID symbols (valves, pumps, turbines, vessels, instrument bubbles with tags), a
  connection-line legend, draggable panels, and an animated cycle panel (moving flow, rotating machinery) that exports a
  looping GIF or a PNG for every cycle.
- Grain designer and Port & grain: the grain is shown as a full-length 3D extrusion you can turn with the mouse, and the
  Cutaway button gives a side view of the port.
- F2 saves a screenshot of the window to the data folder (`screenshots`); F11 toggles full screen; a Legend button in the 3D view shows a color key for the solid-color blocks (bottom right);
  the Spring-mass toy draws a real helical spring; the combo box arrow is a plain chevron; the Next Color Theme command
  (Ctrl+T) is removed, themes are picked in View, Color Theme.
- Edit, Preferences: appearance options and editable key bindings; double-click a slider to restore its default;
  Help, Tutorial moves to F1 and Restore Default Layout to F12.
- Loading screen (application icon, progress bar and step, shown for at least one second) and a confirmation before
  quitting.
- Short names (buttons, list and combo items such as `N2O Blowdown`) use the interface font with true sub- and
  superscripts; formulas elsewhere stay in LaTeX.
- Light theme: the plot tab strip and the 3D view edge follow the theme; more room above the parameter panel title.
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
- Component library, block H (graphs and dashboards): JSON Schemas for engine graphs (nodes, typed edges,
  boundary conditions, timed and event-triggered start sequence) and dashboards, default panels for every component
  type, and three example graphs (gas-generator kerolox from the Huzel & Huang A-1, pressure-fed kerolox, expander
  hydrolox) with every value published, derived or flagged estimated; graph wiring and units checked by
  `propsim.data.validate`.
- Component library, block E part 1 (materials): NIST cryogenic property fits (conductivity, specific heat,
  Young's modulus, thermal expansion, 4 to 300 K) for 6061-T6, OFHC copper, Inconel 718, 304, 316 and Ti-6Al-4V,
  Huzel & Huang Inconel X points at chamber-wall temperatures, evaluated by `propsim.data.matprops`.
- Component library, block E part 2: Inconel 718 properties to 2000 degF (Special Metals bulletin) and GRCop-84
  strength and conductivity (NASA/TM-2005-213566), with trend checks on the transcribed tables.
- Component library, block F part 1 (hybrid): regression-rate laws (SP-1a paraffin, HTPB, aluminized HTPB with
  GOX) in SI and 22 published SP-1a/GOX hot-fire cases, read by `propsim.data.regression_rate` and
  `propsim.data.hotfire_cases`.
- Component library, block G (electric): NEXT gridded-ion throttle tables (56 beginning-of-life points, nominal
  with PPU efficiency and extended thruster-only, NASA/TM-2008-214986) in SI, read by `propsim.data.throttle_table`,
  with thrust, efficiency and lifetime-impulse closure checks.
- Component library, SSME example graph: fuel-rich staged combustion with two preburners and four turbopumps at
  rated power level, built only from public NASA documents (1981 Space Shuttle News Reference, NASA TP-3700); chamber
  temperature from the shipped CEA tables matches the published 3590 K.
- Component library, loss-coefficient cross-check: Huzel & Huang Table 7-4 sudden enlargement and contraction
  coefficients compared with the Borda-Carnot and Benedict-fit equations of Rennels and Hudson, Pipe Flow (2012);
  `propsim.data.closure.sudden_expansion_K` and `sharp_contraction_K`; one enlargement value and the contraction
  values disagree and are recorded in `conflicts.md`.
- Component library, heat-transfer validation: PropSim's Bartz film coefficient and property-variation factor
  reproduce the Huzel & Huang A-1 and A-2 throat values (Sample Calculation 4-3) within 4 %; inputs are in
  `components/tables/bartz_validation.csv`, checked in `checks.md` section 12 and by a test.

- Solid: quasi-steady CP motor (SI and US customary units) with optional end-face burning, constant-rate and
  pressure-dependent throat erosion, and one-parameter sweeps drawn on all plots.
- Vehicle stages can be filled from the library's reference engines (RS-25, J-2 and the RL10 variants) with one
  choice; values the library does not have stay empty or are labelled as assumed or derived.

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
