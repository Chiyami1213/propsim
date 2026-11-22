# PropSim User Guide

A short guide to the window, the two kinds of models, the workflows that connect them, and where your files go.
For installation and system requirements see the [README](README.md).

## The window

| Panel | What it does |
|---|---|
| **Modules** | The tree of models, grouped by Common, Liquid, Hybrid, Solid, Electric, Nuclear and Vehicle. *(Live)* and *(Run)* tell you the kind of model. |
| **Saved Runs** | Every Run result is saved automatically. Click one to reload it; tick *Compare Overlay* to draw several saved runs of one model together. |
| **3D View** | Rotate (left mouse), pan (right mouse), zoom (wheel). Buttons: reset, cutaway angle, which field paints the model, colorbar, snapshot. |
| **Plots** | Tabs of gridded plots with LaTeX labels. Run models show a cursor at the playback time. |
| **Parameters** | One slider per input. Hover over a slider to read its meaning and its dimensions in the box below. |
| **Run Log** | Messages, warnings and the summary line of the last run. |

Every panel can be dragged by the grip at its top left (to move it or float it) and closed with the cross at its top
right. *View, Panels* shows or hides them again, and **F12** restores the default arrangement.

## Live and Run models

- **Live** models recompute in under 50 ms while you drag a slider. Use them to explore a design quickly.
- **Run** models simulate over time in a separate process: press **Run** (Ctrl+R), watch the progress bar, **Cancel**
  (Esc) if needed. When it finishes, the scrubber, play button and speed box replay the result in the plots, the 3D
  view and the readouts together. The badge at the bottom left shows *Live* or *Run*.

## Workflows

### Design an engine, then fly it
1. Run a hybrid, solid or liquid Run model (for example *Hybrid, 0-D Burn* or *Solid, Internal Ballistics*).
2. Choose **File, Send Thrust to Vehicle, Stage 1** (or 2, 3). The thrust curve and mass flow of that run are saved as
   the stage's profile.
3. Open **Vehicle, Point-Mass Ascent**, set that stage's *thrust source* to *Imported thrust profile*, set masses and
   the diameter, and Run. The run log reports the burn time, impulse and propellant used.

Imported curves are flown as given (no back-pressure correction). You can also drop your own `stage1.csv` with the
columns `t,F,mdot` (SI units) into the `profiles` folder of the data folder.

### Drag from your own data
Set *Drag* to *Table* and put a file `drag.csv` with the columns `mach,cd` (Mach strictly increasing) in the `profiles`
folder. *File, Open Data Folder* opens it.

### Solid grain design and internal ballistics
*Design, Solid motor designer* (a window) shows the burning area, K_n and equilibrium pressure through the burn of one section
kind: BATES (circular bore), star, wagon wheel, finocyl, centre plus 4 or 6 bores, moon burner (bore offset by $e$),
C-slot, D-grain (flat at $e$ from the axis), X-core, end burner (no port) or a custom shape list. *Segments* repeats
the grain $N_{seg}$ times and *Ends* says which end faces burn, so a classic BATES motor is a circular bore with *Both
ends burn*; an end burner always burns its aft face. The caption gives the web, loading, propellant mass, K_n
progressivity, burn time and the port-to-throat area ratio.

*Solid, Internal Ballistics* (Run) integrates the chamber with the same grain. The *burn-rate law* is either $a$ and
$n$, your own *pressure bands* (a table of $p_{lo}$, $p_{hi}$, $a'$, $n$; the dialog can load a library law), or
Nakka's 1999 KNSB or KNDX bands from the component library. The *Nozzle* group adds the c* efficiency, the conical
divergence loss $\lambda=(1+\cos\alpha)/2$, other $C_F$ losses, the ambient pressure (0 for vacuum) and, for KN and
aluminized propellants, a separate two-phase $\gamma$ for the expansion. *Throat erosion* grows the throat at
$\dot D_t=\dot e_{ref}(p_c/p_{ref})^k$. The *Nozzle & Isp* tab plots Isp, $C_F$, the throat and the aft
port-to-throat ratio; the log lists the design checks (port-to-throat ratio, peak core mass flux, volumetric loading,
L/D) and the motor class. *File, Export Thrust Curve (.eng)* writes the thrust curve for flight simulators.

### Quasi-steady solid motor, sweeps and US units
*Solid, CP Motor (Quasi-Steady)* (SI) and *Solid, CP Motor (US Units)* (in, psi, lbf, lbm) march a hollow-cylinder
grain through its web with the equilibrium chamber pressure at every step. Choose *Both end faces burn* to include the
end faces, and pick a throat-erosion treatment (constant area, constant rate, or a rate that scales with
$(p_c/p_{ref})^{0.8}$); without a sweep the plots overlay all three treatments. *Sweep* draws three values of one length
(the nominal value plus or minus the span) on every plot. There is no ignition transient or tail-off: use *Internal
Ballistics* for those.

### Build your own liquid engine cycle
*Liquid, Engine designer* (or **Design, Liquid engine designer**, Ctrl+E) opens a flow-diagram editor. Pick parts from
the component library on the left (tanks, lines, valves, pumps, inducers, turbines, gas generators, injectors,
chambers, cooling jackets, nozzles, shafts) by double-clicking or dragging, then drag from one port to another to
connect them. Fluid ports join outlet to inlet; shaft ports join a turbine to its pumps. Choose the oxidizer and fuel in
the toolbar; tanks take the propellant of their side. The inspector on the right edits the selected part's parameters
(each shows its library source; an edited value is marked *USER*) and options such as the nozzle and chamber cooling
(regenerative, radiative, ablative, uncooled) or the turbine model. Right-click a part to send its exhaust overboard.

**Solve** (Ctrl+R) finds the steady operating point: every station's pressure, temperature, phase and flow, each
part's state (pump head, power and NPSH margin, turbine work, jacket heat load and wall temperature, chamber pressure,
thrust and Isp) and shaft speeds. Color the diagram by pressure, temperature or flow. In *Design Targets* you can ask
for an output (for example chamber pressure and mixture ratio) and free a parameter (for example a valve position) to
size it, as an engine controller would. The templates include the Huzel & Huang pressure-fed, gas-generator and
expander engines and an RS-25 style staged-combustion cycle. Save and open diagrams as `.json` files.

Away from its design flow a pump's efficiency follows eta_design q(2 - q), where q is the flow ratio at corresponding
points, (Q/Q_design)/(N/N_design) (affinity laws: equal efficiency at equal flow coefficient, Dixon sec. 2.3; the
parabolic shape is PROPSIM-EST), so shut-off and run-out cost power and heat the liquid; choose *Efficiency curve:
constant* on a pump to hold the design value. Each pump and inducer reports the NPSH it requires,
(omega sqrt(Q)/Omega_ss)^(4/3)/g, and its margin. Omega_ss is the pump's own *suction_specific_speed* if given, else an
inducer's on the same shaft (its value, or 7.3, the low end of Huzel & Huang Table 6-5), else a bare impeller's 2.93
(S = 8000 in rpm, gal/min, ft, the low end of the usual 8000-10000). A negative margin is a warning only when the limit
is the pump's own or an inducer's: the bare-impeller figure is far below what inducer-fitted rocket pumps reach.

Lines use Churchill's (1977) friction factor, one smooth curve from laminar (64/Re) through the critical zone to
fully rough flow, so trickle flows during a start-up are no longer treated as turbulent. A line's options add its
fittings, each from Rennels & Hudson's *Pipe Flow* (2012): an entrance from a large volume (sharp 0.57, rounded by
r/d, re-entrant by t/d), a sudden contraction from a larger pipe, smooth bends by angle and r/d (eq. 15.1), miter
bends, a sudden expansion (Borda-Carnot) and free discharge (one velocity head); *K_fittings* adds anything else.
The line reports its Reynolds number, friction factor, total K, the pressure-wave speed (the fluid's sound speed,
the rigid-wall upper bound) and the Joukowsky surge rho a V with the wave's round-trip time 2L/a. Gas lines follow
the isothermal compressible relation with friction (eq. 4.26b). An orifice or injector can take its discharge
coefficient from its edge instead of a typed Cd: sharp (0.60 in a wall) or thick with length t/d (0.80 at 1.4 d,
where the jet has reattached; longer holes add bore friction). In a start-up, a valve that closes faster than the
round trip of the line feeding it gets a note with the surge it would see.

The diagram follows ANSI/ISA-5.1 symbols: process lines are solid (colored by propellant), the shaft link is dash-dot,
and the bubbles (PT, TT, ST, FT, PDT) are field instruments with their tag numbers. The *Cycle Animation* panel shows
the flow moving through the cycle and exports it as a GIF or PNG. Every sandbox panel can be dragged; *Panels* brings
back a closed one.

The *3D Model* panel (beside the results) draws the part you select, built from its own parameters: a line's bore
and length, a valve's bore and its solved position, a pump's impeller tip, stage count and impeller kind, a turbine's
mean blade diameter and arrangement, a tank's volume and fill level, the chamber and nozzle contour from L*, throat
area, contraction ratio, expansion ratio and bell fraction, and a jacket's channel count and size. After a solve the
readout gives the part's operating state, and the chamber, nozzle and jacket can be coloured by the gas-side wall
temperature, heat flux or coolant temperature from the jacket march (*Field*). Sizes the library does not hold
(casings, flanges, actuators, blade counts, injector element pattern, gas generator and shaft size) are drawn to
illustrative proportions, and the caption says which.

The *Trade Study* panel (a tab beside *Results*) runs trades on the engine in the diagram. *Sweep* varies one input
over a range: any numeric part parameter (a line bore, a pump tip diameter, a turbine pressure ratio, the nozzle area
ratio), a valve position, or a design target such as chamber pressure, in which case the free parameters are re-sized
at every point. The sweep starts from the design's own value and steps outward, each point starting from its
neighbour's answer. Plot any engine number (thrust, Isp, pump and turbine power, peak wall temperature, lowest NPSH)
or any part output against the swept input, read the table, or *Export CSV*. *Compare Designs* keeps snapshots of
designs (the diagram, propellants and targets at the time you add them, or a template) and solves them side by side,
so you can compare cycles or propellant pairs.

The *Start-up* panel (a tab beside *Results*) runs the engine up from the closed, pre-start state: the valves open on
a schedule (*Sequence*: open time, ramp, optional close time and ramp, and whether the throttle scales the valve),
the shafts spin up under the torque their turbines make against the pumps, and chamber pressure and every station's
pressure, temperature and flow follow. *Throttle* is a list of (time, fraction) points, *Load* is the propellant mass
in each tank (a run stops with a message when a tank runs dry; blowdown tanks lose pressure as the ullage grows),
and *Starter* adds spin-up power to a shaft for a set time. Valves the design targets size (an engine-controller
valve) hold chamber pressure and mixture ratio from mainstage, scaled by the throttle. Press *Run Start-up*; the
plots (Pressure, Temperature, Speed, Flow, Tanks and Valves) can be scrubbed or played, and the diagram, inspector
and 3D model show the engine at the time on the slider. *Export CSV* writes every series.

The transient is quasi-steady: at each time step the network is solved at the current valve positions, shaft
speeds and tank states, and only the shaft speeds and tank masses are integrated (shaft inertia, I N dN/dt = turbine
power - pump power - accessory load + starter, and propellant mass). Line and volume filling, fluid inertia and
wall thermal lag are not modelled, so the first fraction of a second after a valve cracks open is idealised. Shaft
inertia defaults to steel disks sized from the wheel diameters (PROPSIM-EST; enter the real value on the *Shaft*
if known); the accessory load falls with speed cubed (PROPSIM-EST); a turbine's efficiency follows the impulse-runner
parabola of Dixon eq. 9.3 (zero at standstill, peak at the design blade-speed ratio nu_design, zero again at twice it),
scaled to its design efficiency (the scaling is PROPSIM-EST), which gives it a finite starting torque. A staged-combustion engine may stay below its design speed in a short run: give it a starter or a longer
run. The transient equation set follows Seymour (2003) and Binder et al., NASA TM-107318; start-up data to compare
against are in Bradley, AIAA 97-2687, and Szuch, NASA TN D-3136 (cited, not reproduced).

### Compare designs
Run a model, change a parameter, Run again, then tick *Compare Overlay*: the *Compare* tab overlays the saved runs.

## Exporting

| Where | What you get |
|---|---|
| Buttons at the bottom right | **PNG** of the current plot tab, **CSV**, **.mat** and **.h5** of the data, and **.py** / **.m**: a runnable Python or MATLAB script that rebuilds the model with your parameters and redraws every plot |
| **File, Export Report** | A report with the parameters, results, events and every plot tab: **HTML** (one self-contained file), **PDF**, or **LaTeX tables and figures** (`.tex` plus PNG figures) |
| **File, Export Code** | The same Python and MATLAB scripts as the buttons |

The MATLAB script reads its data from a `.mat` file saved next to it; set `rerun = true` in the script to recompute
through Python.

## Your data

Saved runs, cases, exports, thrust profiles and backups are kept in `%APPDATA%\PropSim`:

| Folder | Contents |
|---|---|
| `runs` | Saved Run results (`.psim.h5`) |
| `cases` | Cases you save with *File, Save Case* (`.psim.json`) |
| `exports` | Default location of exported files |
| `profiles` | Thrust profiles sent to the vehicle, and `drag.csv` |
| `backups` | The previous version of every case you overwrite, and your last session |

Change the location with **File, Data Folder** (it can copy your runs and cases over). **File, Restore Last Session**
reopens your previous model and parameters.

## Color themes

**View, Color Theme** (F10) lists Graphite, Ocean, Forest, Plum and Light with their palettes; the picker
shows only the names when its window is narrow. The choice is remembered.

## Designs and analysis

Build the engine or motor first, then test it. The **Design** menu opens the liquid engine designer (cycle sandbox) and
the solid and hybrid motor designers. **Use in Analysis** makes the current design the active one; opening an analysis
module of the same kind (Liquid: feed transient, pump-fed cycles, chugging; Solid: ballistics, burnback, CP motor;
Hybrid: the burns; Nuclear: the study modules and reactor kinetics) then starts from the design's values, limited to the module's slider range (the log says which).
**File, Open Design** reopens a saved grain design; **Analysis, Use Active Design in This Module** reloads the open
module, and **Clear Active Design** goes back to the module defaults. The **Nuclear thermal designer** has five studies (performance, reactor criticality, fuel element, engine cycle, xenon and
samarium) that keep their values while you switch; the kinetics and startup module takes the core temperature, pressure and
inlet temperature from the performance study. The **Electric propulsion designer** has four studies (mission trades, electrothermal, gridded ion, Hall); the ion
optics run starts from the ion thruster geometry, the Hall discharge run from the Hall thruster and the low-thrust
spiral from the mission trade.

## Preferences

**Edit, Preferences** (F9) has two pages. *Appearance* sets the color theme and whether the 3D view shows the
legend and the colorbar, and whether PropSim asks before quitting. *Key Bindings* lists every menu command with its
shortcut: click a field and press the new keys, clear it to remove the shortcut, or reset one row or all of them. Two
commands cannot share a shortcut. Everything is remembered between sessions.

Double-click a parameter slider to put it back to its default value.

## Keyboard shortcuts

| Keys | Action |
|---|---|
| Ctrl+R / Esc | Run / Cancel |
| Ctrl+O, Ctrl+S | Open case, Save case |
| Ctrl+E | Liquid engine designer |
| F10 | Color theme picker |
| F12 | Restore default panel layout |
| F11 | Full screen on and off |
| F2 | Screenshot of the window, saved in the data folder's screenshots folder |
| F1 | Start the tutorial (Left / Right arrows move between steps, Esc leaves) |
| Ctrl+Q | Quit |

## Model notes

- Units in the sliders are shown in parentheses and dimensions in square brackets, for example `(MPa)` and
  `[M L⁻¹ T⁻²]`; all calculations are in SI.
- Thermochemistry uses bundled NASA CEA tables. The Hall discharge model does not reproduce the breathing mode and the
  ion-optics model uses a fixed emission plane; the README lists these limitations.
- The nuclear models follow Emrich, *Principles of Nuclear Rocket Propulsion* (2nd ed.). Performance uses frozen flow
  (no dissociation); criticality is one-group diffusion, with reflected spheres and cylinders handled through the
  slab reflector savings; the fuel element is an equivalent annular cell; reactor kinetics lumps the core into fuel,
  moderator and propellant with linear reactivity coefficients, so a startup from very low power over-weights the
  hydrogen-density feedback. Hydrogen above 1000 K uses CoolProp's extrapolation.
- The vehicle model is a point mass over a spherical, non-rotating Earth: no lift, a generic drag curve unless you give
  a table, and the 2-D plane is the launch azimuth. The atmosphere is the U.S. Standard Atmosphere 1976.

## Troubleshooting

- **A Run says it failed:** the Run Log shows the reason (for example a vehicle whose thrust-to-weight at lift-off is
  below 1, or a stage set to an imported profile that was never sent).
- **A panel disappeared:** *View, Panels*, or F12.
- **Everything looks tiny or blurry:** PropSim follows the Windows display scaling; restart it after changing the scale.
- **Report or export fails:** the Run Log shows the error; make sure the target folder is writable.

Questions and bug reports: [GitHub Issues](https://github.com/Chiyami1213/propsim/issues).
