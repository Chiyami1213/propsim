# PropSim

Interactive 3D rocket propulsion simulator for Windows: liquid, hybrid, solid, electric and nuclear thermal propulsion in one desktop
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
    internal ballistics with erosive burning, and a quasi-steady CP motor (end-face burning, three throat-erosion
    treatments, parameter sweeps, SI or US customary units).
  - *Electric*: mission and power trades (Stuhlinger), electrothermal thrusters, gridded ion and Hall thrusters,
    ion-optics particle simulation, Hall discharge, low-thrust Edelbaum spiral with power degradation.
  - *Nuclear*: nuclear thermal rocket performance against LOX / LH₂, one-group reactor criticality with reflectors,
    fuel-element thermal hydraulics (chopped-cosine power, peak fuel temperature, channel pressure drop), hot bleed,
    cold bleed and expander cycles with real hydrogen, xenon and samarium poisoning over burn and restart histories,
    and six-group reactor kinetics with fuel, moderator and hydrogen-density feedback (startup with temperature
    control, reactivity steps, scram with decay heat, Bode stability).
  - *Vehicle*: staging and atmosphere budget, and a 2-D point-mass ascent over a spherical Earth (U.S. Standard
    Atmosphere 1976, drag curves or your own table, wind, up to three stages, pitch kick and level-off guidance,
    perigee cutoff) that can fly thrust curves sent from the hybrid, solid and liquid models.
- **Component library** (`propsim/data/library`): sourced fluid and propellant data, a catalog of 19 liquid-engine
  component types, reference engines and cycles, every value with its source and quality (`published`, `derived`,
  `estimated`), plus documented gaps and source conflicts. Read with `propsim.data`; checked by
  `python tools/check_library.py`. The vehicle model uses it for stage engine presets.
- **All math in LaTeX**: plot labels, tables, readouts and hover help; every plot has a grid.
- **Color themes**: Graphite, Ocean, Forest, Plum and Light, chosen in *View > Color Theme* (F10) and remembered
  between sessions. The picker shows each theme's palette and falls back to the
  theme names alone when the window is too narrow for the palettes.

- **Panels and tutorial**: every panel has a drag grip (top left) and a close button (top right); *View > Panels* and
  F12 bring them back. *Help > Tutorial* (F1) is a guided tour you step through with the left and right
  arrow keys, and it is offered on the first launch.
- **Preferences and shortcuts**: *Edit > Preferences* (F9) sets the color theme, the 3D legend and colorbar, and
  the quit confirmation, and lets you rebind every menu command. F11 toggles full screen, F2 saves a screenshot of the
  window to the data folder (`screenshots`), and double-clicking a slider restores its default. The 3D view has a
  Legend button (bottom right) that names its solid-color blocks. A loading screen shows while PropSim starts.
- **Export**: reports (HTML, PDF, or LaTeX tables with figures) and the exports below, PNG figures, CSV, MATLAB `.mat`,
  HDF5, and runnable code:
  - *Python script* (`.py`): rebuilds the model with your parameters through the `propsim` package and redraws
    every plot with Matplotlib.
  - *MATLAB script* (`.m` + `_data.mat`): every plot redrawn natively (grid, LaTeX labels); set `rerun = true`
    to recompute from MATLAB through Python.

## Models

**Common**

- **Thermochemistry & nozzle** (Live): propellant pair, O/F, P_c and area ratio give CEA performance and the nozzle.

**Liquid**

- **Injector & chamber** (Live): injector and chamber sizing.
- **Regenerative cooling** (Live): regenerative cooling of the chamber and nozzle.
- **Feed network transient** (Run): feed-network transient (valve timing, line dynamics).
- **Pump-fed cycles** (Run): pump-fed cycles (gas generator, expander, staged combustion).
- **Chugging & L* stability** (Run): chugging and L* stability.

**Hybrid**

- **Nozzle thrust coefficient** (Live): nozzle area ratio gives the C_F curve, Mach and pressure distributions, and a 3D bell nozzle.
- **Regression laws** (Live): regression-law explorer.
- **Port & grain** (Live): port and multiport grain.
- **Core flow fields** (Live): core flow fields (Taylor-Culick and bidirectional-vortex mantle).
- **Fuel & oxidizer** (Live): fuel and oxidizer properties.
- **Instability & scaling** (Live): ILFI instability and similarity scaling.
- **0-D burn** (Run): 0-D burn (HTPB/GOX power law) with 3D grain cutaway playback.
- **Quasi-1D burn** (Run): quasi-1D burn: axial port evolution, throttling, throat erosion.
- **N₂O blowdown** (Run): N2O self-pressurized tank blowdown feeding the hybrid motor.

**Solid**

- **Grain designer** (Live): grain designer (2-D section burnback tables).
- **Burn rate & K_n** (Live): burn-rate and K_n tool.
- **3D burnback** (Run): 3-D level-set burnback.
- **Internal ballistics** (Run): internal ballistics.
- **CP motor (quasi-steady)** (Run): quasi-steady cylindrical-perforated motor marched in web: end-face burning, three throat-erosion treatments and parameter sweeps.
- **CP motor (US units)** (Run): the same CP motor with inputs and plots in US customary units (in, psi, lbf, lbm).

**Electric**

- **Mission & power trades** (Live): mission and power trades.
- **Electrothermal thrusters** (Live): electrothermal thrusters.
- **Gridded ion thruster** (Live): gridded ion thruster.
- **Hall thruster** (Live): Hall thruster.
- **Ion optics** (Run): ion-optics beamlet particle simulation.
- **Hall discharge** (Run): 1-D Hall discharge.
- **Low-thrust spiral** (Run): low-thrust Edelbaum spiral with solar-array degradation.

**Nuclear**

- **NTR performance** (Live): chamber temperature, pressure and area ratio give I_sp, flow, reactor power and core size, compared with a LOX / LH₂ engine (CEA) and on stage payload fraction.
- **Reactor criticality** (Live): one-group bare or reflected sphere, cylinder or slab: k_eff, critical size and mass, reflector savings and the flux shape.
- **Fuel element thermal hydraulics** (Live): equivalent fuel cell with a chopped-cosine power shape: propellant, wall and peak fuel temperatures, Dittus–Boelter film coefficient and channel pressure drop.
- **NTR engine cycles** (Live): hot bleed, cold bleed and expander energy balances with CoolProp hydrogen: bleed fraction, delivered I_sp, pump outlet pressure and the T–s diagram.
- **Xenon & samarium** (Live): iodine-xenon and promethium-samarium chains over a burn, coast and restart, with the poison reactivity against the drum excess.
- **Reactor kinetics & startup** (Run): six-group point kinetics with fuel, moderator and propellant-density feedback: startup under outlet temperature control, reactivity steps and a scram with decay heat; closed-loop Bode plot.

**Vehicle**

- **Vehicle & staging** (Live): staging budget (ideal velocity, mass, thrust-to-weight), atmosphere, drag and wind, with a stack sketch.
- **Point-mass ascent** (Run): 2-D point-mass ascent over a spherical Earth with staging, wind, drag and imported thrust profiles.

**Toy**

- **Spring-mass** (Run): damped spring-mass, a proof of the Run pipeline (not a propulsion model).

Planned and not yet implemented: ILFI stability (Run).

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
**Windows installer**: v1.0.0 is not released yet. Build the installer yourself with
`python tools/build_installer.py` (see *Building the Windows package* below); once released, `PropSim-<version>-Setup.exe`
will be on the [Releases](https://github.com/Chiyami1213/propsim/releases) page. It installs for your account (or for all
users), adds Start-menu shortcuts for PropSim and this help page, and can be removed from Windows *Apps & features*.

**From source**:
```powershell
pip install -e .
python run.py
```
Optional extras: `pip install -e ".[cea]"` queries NASA CEA directly through RocketCEA (otherwise the bundled
CEA tables are used); `pip install -e ".[rao]"` enables Rao optimum-bell angles from the Huzel & Huang chart via
pygasflow (GPL-3.0, not bundled; without it the bell contour uses labelled placeholder angles).

## Getting started
The [user guide](GUIDE.md) covers the window, workflows, exports, the data folder and shortcuts.

1. Pick a model in the **Modules** tree: *(Live)* models update as you move the sliders, *(Run)* models start with
   **Run** (Ctrl+R).
2. Hover over a parameter to see its meaning and dimensions.
3. Use the 3D toolbar for cutaway, color field, color bar and snapshots; the plot tabs below show the results.
4. Export from the bottom-right buttons or **File > Export Code**. **Help > Online Help** opens this page.

## Where your data is stored
Saved runs, cases, exports and automatic backups are kept in `%APPDATA%\PropSim` by default:
`runs\`, `cases\`, `exports\`, `backups\` (previous versions of overwritten cases and your last session).
Choose another location with **File > Data Folder**, which can also copy existing runs and cases there.
**File > Restore Last Session** reopens the model and parameters from your previous session.

## Validation
The test suite (`python -m pytest`, headless: set `QT_QPA_PLATFORM=offscreen`) currently reports **236 passed, 1 expected
failure** (xfail). By group: `tests/reference` 25 (the MathKit reference script, `mathkit_check.py`, reports 24 PASS),
`tests/core` 10, `tests/validation` 118, `tests/ui` 83 (Live under 50 ms, text and plot standards, process worker,
window layout, code export).
- **Reference values**: the validation tests reproduce the worked-example values of the project's MathKit document
  (Sections 2, 3 and 5; Bartz heat-transfer values for the liquid module, ideal thrust coefficient, hybrid, solid and
  electric-propulsion examples).
- **Thermochemistry**: LOX/HTPB performance against Chiaverini Table 1, the bundled tables against direct NASA CEA for
  all pairs, and frozen versus equilibrium ordering. One test is a documented expected failure: CEA's built-in HTPB is
  1.7 % low in c\* and 1.2 % low in I_sp against Table 1 (a different HTPB formula), so the 1 % target is not met.
- **Analytic and conservation checks**: Prandtl-Meyer and divergence loss for the nozzle, the Joukowsky water-hammer
  limit and polytropic blowdown for the feed network, pump-turbine power balance for the cycles, chamber mass balance,
  level-set convergence for the solid grain, and the Hall discharge mass balance.
- Not validated here: the Hall breathing mode (not reproduced) and ion-optics focusing (checked for trend only).

## How it was built
I developed PropSim with Claude in VS Code, working from a written specification and the MathKit reference values.
All physics lives in the pure-Python `propsim/core/` package, which has no Qt dependency, so it is tested on its own
(`tests/core`, `tests/validation`); the Run models are drivers in `propsim/sims/`, and `propsim/modules/` and
`propsim/ui/` only connect them to the plots and the 3D view. Each model's parameters are declared once with the schema
in `propsim/core/schema.py`, which also drives the sliders, hover help and validation.

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

## License and credits
PropSim's own code is released under the [MIT License](LICENSE). Third-party notes are in [NOTICE](NOTICE).
- **Thermochemistry**: the shipped tables (`propsim/core/thermo/data/cea_tables.npz`) are numbers computed with
  NASA's Chemical Equilibrium with Applications (Gordon and McBride, NASA RP-1311) through RocketCEA. Neither
  program is distributed with PropSim.
- **Optional extras** are installed separately and keep their own licenses: `rocketcea` (GPL-3.0) and
  `pygasflow` (GPL-3.0). PropSim does not import them unless you install them.
- **Qt for Python (PySide6)** is used under the LGPL-3.0; the Windows build ships the Qt libraries as separate files.
- The remaining runtime libraries (NumPy, SciPy, Matplotlib, PyVista and VTK, pyqtgraph, pydantic, h5py, CoolProp,
  scikit-image, Numba) are permissively licensed; see NOTICE for each.

## Help and feedback
Questions and bug reports: [GitHub Issues](https://github.com/Chiyami1213/propsim/issues).

## Gallery

<table>
  <tr>
    <td><img src="docs/gallery/01-thermochemistry-and-nozzle.png" alt="Thermochemistry & nozzle: propellant pair, O/F, P_c and area ratio give CEA performance and the nozzle"><br><sub><b>Thermochemistry & nozzle</b> (Common, Live)</sub></td>
    <td><img src="docs/gallery/02-injector-and-chamber.png" alt="Injector & chamber: injector and chamber sizing"><br><sub><b>Injector & chamber</b> (Liquid, Live)</sub></td>
  </tr>
  <tr>
    <td><img src="docs/gallery/03-regenerative-cooling.png" alt="Regenerative cooling: regenerative cooling of the chamber and nozzle"><br><sub><b>Regenerative cooling</b> (Liquid, Live)</sub></td>
    <td><img src="docs/gallery/04-feed-network-transient.png" alt="Feed network transient: feed-network transient (valve timing, line dynamics)"><br><sub><b>Feed network transient</b> (Liquid, Run)</sub></td>
  </tr>
  <tr>
    <td><img src="docs/gallery/05-pump-fed-cycles.png" alt="Pump-fed cycles: pump-fed cycles (gas generator, expander, staged combustion)"><br><sub><b>Pump-fed cycles</b> (Liquid, Run)</sub></td>
    <td><img src="docs/gallery/06-chugging-and-l-star-stability.png" alt="Chugging & L* stability: chugging and L* stability"><br><sub><b>Chugging & L* stability</b> (Liquid, Run)</sub></td>
  </tr>
  <tr>
    <td><img src="docs/gallery/07-nozzle-thrust-coefficient.png" alt="Nozzle thrust coefficient: nozzle area ratio gives the C_F curve, Mach and pressure distributions, and a 3D bell nozzle"><br><sub><b>Nozzle thrust coefficient</b> (Hybrid, Live)</sub></td>
    <td><img src="docs/gallery/08-regression-laws.png" alt="Regression laws: regression-law explorer"><br><sub><b>Regression laws</b> (Hybrid, Live)</sub></td>
  </tr>
  <tr>
    <td><img src="docs/gallery/09-port-and-grain.png" alt="Port & grain: port and multiport grain"><br><sub><b>Port & grain</b> (Hybrid, Live)</sub></td>
    <td><img src="docs/gallery/10-core-flow-fields.png" alt="Core flow fields: core flow fields (Taylor-Culick and bidirectional-vortex mantle)"><br><sub><b>Core flow fields</b> (Hybrid, Live)</sub></td>
  </tr>
  <tr>
    <td><img src="docs/gallery/11-fuel-and-oxidizer.png" alt="Fuel & oxidizer: fuel and oxidizer properties"><br><sub><b>Fuel & oxidizer</b> (Hybrid, Live)</sub></td>
    <td><img src="docs/gallery/12-instability-and-scaling.png" alt="Instability & scaling: ILFI instability and similarity scaling"><br><sub><b>Instability & scaling</b> (Hybrid, Live)</sub></td>
  </tr>
  <tr>
    <td><img src="docs/gallery/13-0-d-burn.png" alt="0-D burn: 0-D burn (HTPB/GOX power law) with 3D grain cutaway playback"><br><sub><b>0-D burn</b> (Hybrid, Run)</sub></td>
    <td><img src="docs/gallery/14-quasi-1d-burn.png" alt="Quasi-1D burn: quasi-1D burn: axial port evolution, throttling, throat erosion"><br><sub><b>Quasi-1D burn</b> (Hybrid, Run)</sub></td>
  </tr>
  <tr>
    <td><img src="docs/gallery/15-n2o-blowdown.png" alt="N₂O blowdown: N2O self-pressurized tank blowdown feeding the hybrid motor"><br><sub><b>N₂O blowdown</b> (Hybrid, Run)</sub></td>
    <td><img src="docs/gallery/16-grain-designer.png" alt="Grain designer: grain designer (2-D section burnback tables)"><br><sub><b>Grain designer</b> (Solid, Live)</sub></td>
  </tr>
  <tr>
    <td><img src="docs/gallery/17-burn-rate-and-k-n.png" alt="Burn rate & K_n: burn-rate and K_n tool"><br><sub><b>Burn rate & K_n</b> (Solid, Live)</sub></td>
    <td><img src="docs/gallery/18-3d-burnback.png" alt="3D burnback: 3-D level-set burnback"><br><sub><b>3D burnback</b> (Solid, Run)</sub></td>
  </tr>
  <tr>
    <td><img src="docs/gallery/19-internal-ballistics.png" alt="Internal ballistics: internal ballistics"><br><sub><b>Internal ballistics</b> (Solid, Run)</sub></td>
    <td><img src="docs/gallery/20-cp-motor-quasi-steady.png" alt="CP motor (quasi-steady): quasi-steady cylindrical-perforated motor marched in web: end-face burning, three throat-erosion treatments and parameter sweeps"><br><sub><b>CP motor (quasi-steady)</b> (Solid, Run)</sub></td>
  </tr>
  <tr>
    <td><img src="docs/gallery/21-cp-motor-us-units.png" alt="CP motor (US units): the same CP motor with inputs and plots in US customary units (in, psi, lbf, lbm)"><br><sub><b>CP motor (US units)</b> (Solid, Run)</sub></td>
    <td><img src="docs/gallery/22-mission-and-power-trades.png" alt="Mission & power trades: mission and power trades"><br><sub><b>Mission & power trades</b> (Electric, Live)</sub></td>
  </tr>
  <tr>
    <td><img src="docs/gallery/23-electrothermal-thrusters.png" alt="Electrothermal thrusters: electrothermal thrusters"><br><sub><b>Electrothermal thrusters</b> (Electric, Live)</sub></td>
    <td><img src="docs/gallery/24-gridded-ion-thruster.png" alt="Gridded ion thruster: gridded ion thruster"><br><sub><b>Gridded ion thruster</b> (Electric, Live)</sub></td>
  </tr>
  <tr>
    <td><img src="docs/gallery/25-hall-thruster.png" alt="Hall thruster: Hall thruster"><br><sub><b>Hall thruster</b> (Electric, Live)</sub></td>
    <td><img src="docs/gallery/26-ion-optics.png" alt="Ion optics: ion-optics beamlet particle simulation"><br><sub><b>Ion optics</b> (Electric, Run)</sub></td>
  </tr>
  <tr>
    <td><img src="docs/gallery/27-hall-discharge.png" alt="Hall discharge: 1-D Hall discharge"><br><sub><b>Hall discharge</b> (Electric, Run)</sub></td>
    <td><img src="docs/gallery/28-low-thrust-spiral.png" alt="Low-thrust spiral: low-thrust Edelbaum spiral with solar-array degradation"><br><sub><b>Low-thrust spiral</b> (Electric, Run)</sub></td>
  </tr>
  <tr>
    <td><img src="docs/gallery/29-ntr-performance.png" alt="NTR performance: chamber temperature, pressure and area ratio give I_sp, flow, reactor power and core size against a LOX / LH₂ engine"><br><sub><b>NTR performance</b> (Nuclear, Live)</sub></td>
    <td><img src="docs/gallery/30-reactor-criticality.png" alt="Reactor criticality: one-group bare or reflected reactor, k_eff, critical size and mass, reflector savings and the flux shape"><br><sub><b>Reactor criticality</b> (Nuclear, Live)</sub></td>
  </tr>
  <tr>
    <td><img src="docs/gallery/31-fuel-element-thermal-hydraulics.png" alt="Fuel element thermal hydraulics: chopped-cosine power shape, propellant, wall and peak fuel temperatures and channel pressure drop"><br><sub><b>Fuel element thermal hydraulics</b> (Nuclear, Live)</sub></td>
    <td><img src="docs/gallery/32-ntr-engine-cycles.png" alt="NTR engine cycles: hot bleed, cold bleed and expander energy balances with real hydrogen"><br><sub><b>NTR engine cycles</b> (Nuclear, Live)</sub></td>
  </tr>
  <tr>
    <td><img src="docs/gallery/33-xenon-and-samarium.png" alt="Xenon & samarium: iodine-xenon and promethium-samarium poisoning over a burn, coast and restart"><br><sub><b>Xenon & samarium</b> (Nuclear, Live)</sub></td>
    <td><img src="docs/gallery/34-reactor-kinetics-and-startup.png" alt="Reactor kinetics & startup: six-group point kinetics with feedback, startup under temperature control, reactivity steps and scram"><br><sub><b>Reactor kinetics & startup</b> (Nuclear, Run)</sub></td>
  </tr>
  <tr>
    <td><img src="docs/gallery/35-vehicle-and-staging.png" alt="Vehicle & staging: staging budget (ideal velocity, mass, thrust-to-weight), atmosphere, drag and wind, with a stack sketch"><br><sub><b>Vehicle & staging</b> (Vehicle, Live)</sub></td>
    <td><img src="docs/gallery/36-point-mass-ascent.png" alt="Point-mass ascent: 2-D point-mass ascent over a spherical Earth with staging, wind, drag and imported thrust profiles"><br><sub><b>Point-mass ascent</b> (Vehicle, Run)</sub></td>
  </tr>
  <tr>
    <td><img src="docs/gallery/37-spring-mass.png" alt="Spring-mass: damped spring-mass, a proof of the Run pipeline (not a propulsion model)"><br><sub><b>Spring-mass</b> (Toy, Run)</sub></td>
    <td><img src="docs/gallery/cycle-sandbox.png" alt="Cycle sandbox: flow-diagram editor for building and solving liquid engine cycles"><br><sub><b>Cycle sandbox</b> (Liquid, Editor)</sub></td>
  </tr>
</table>

<p align="center"><img src="docs/gallery/tutorial.png" alt="The guided tutorial highlights the parameters panel and explains it"><br><sub>The guided tutorial (Help &gt; Tutorial): left and right arrow keys move between steps.</sub></p>
