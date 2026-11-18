# PropSim User Guide

A short guide to the window, the two kinds of models, the workflows that connect them, and where your files go.
For installation and system requirements see the [README](README.md).

## The window

| Panel | What it does |
|---|---|
| **Modules** | The tree of models, grouped by Common, Liquid, Hybrid, Solid, Electric and Vehicle. *(Live)* and *(Run)* tell you the kind of model. |
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

### Quasi-steady solid motor, sweeps and US units
*Solid, CP Motor (Quasi-Steady)* (SI) and *Solid, CP Motor (US Units)* (in, psi, lbf, lbm) march a hollow-cylinder
grain through its web with the equilibrium chamber pressure at every step. Choose *Both end faces burn* to include the
end faces, and pick a throat-erosion treatment (constant area, constant rate, or a rate that scales with
$(p_c/p_{ref})^{0.8}$); without a sweep the plots overlay all three treatments. *Sweep* draws three values of one length
(the nominal value plus or minus the span) on every plot. There is no ignition transient or tail-off: use *Internal
Ballistics* for those.

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

**View, Color Theme** (Ctrl+Shift+T) lists Graphite, Ocean, Forest, Plum and Light with their palettes; the picker
shows only the names when its window is narrow. The choice is remembered.

## Preferences

**Edit, Preferences** (Ctrl+,) has two pages. *Appearance* sets the color theme and whether the 3D view shows the
legend and the colorbar, and whether PropSim asks before quitting. *Key Bindings* lists every menu command with its
shortcut: click a field and press the new keys, clear it to remove the shortcut, or reset one row or all of them. Two
commands cannot share a shortcut. Everything is remembered between sessions.

Double-click a parameter slider to put it back to its default value.

## Keyboard shortcuts

| Keys | Action |
|---|---|
| Ctrl+R / Esc | Run / Cancel |
| Ctrl+O, Ctrl+S | Open case, Save case |
| Ctrl+Shift+T | Color theme picker |
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
- The vehicle model is a point mass over a spherical, non-rotating Earth: no lift, a generic drag curve unless you give
  a table, and the 2-D plane is the launch azimuth. The atmosphere is the U.S. Standard Atmosphere 1976.

## Troubleshooting

- **A Run says it failed:** the Run Log shows the reason (for example a vehicle whose thrust-to-weight at lift-off is
  below 1, or a stage set to an imported profile that was never sent).
- **A panel disappeared:** *View, Panels*, or F12.
- **Everything looks tiny or blurry:** PropSim follows the Windows display scaling; restart it after changing the scale.
- **Report or export fails:** the Run Log shows the error; make sure the target folder is writable.

Questions and bug reports: [GitHub Issues](https://github.com/Chiyami1213/propsim/issues).
