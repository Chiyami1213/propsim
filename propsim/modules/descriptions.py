"""One-line explanations of every module, shown as a tooltip when the mouse rests on an entry of the module tree.
Keyed by the model key (or tool key); ``MODULES`` explains the top-level groups."""
from __future__ import annotations

MODULES = {
    "Common": "tools shared by every propulsion type",
    "Liquid": "liquid-propellant engines: injector and chamber, cooling, feed systems, cycles and stability",
    "Hybrid": "hybrid motors (solid fuel grain, liquid or gas oxidizer): regression, ports, burns and stability",
    "Solid": "solid rocket motors: grain design, burn rate, burnback and internal ballistics",
    "Electric": "electric propulsion: thrusters, discharge and ion-optics physics, mission trades and spiral transfers",
    "Nuclear": "nuclear thermal rockets: performance, reactor criticality, thermal hydraulics, cycles, poisons and kinetics",
    "Vehicle": "whole-vehicle staging budget and flight to orbit",
    "Toy": "a simple model that proves the run and playback pipeline",
}

MODELS = {
    "common.engine": "chemical equilibrium performance of a propellant pair: isp, c*, thrust coefficient and the nozzle "
                     "for a chosen mixture ratio, chamber pressure and area ratio",
    "liquid.injector": "sizes the injector and chamber: orifices, pressure drop, chamber volume, l* and contraction ratio",
    "liquid.regen": "regenerative cooling of the chamber and nozzle: wall and coolant temperatures and heat flux along "
                    "the jacket",
    "liquid.feed": "transient of the feed network: valve timing, line inertia and the start-up pressure and flow history",
    "liquid.cycle": "power balance of pump-fed cycles (gas generator, expander, staged combustion): turbine and pump "
                    "power, flows and performance",
    "liquid.chugging": "low-frequency combustion stability: chugging against l*, injector drop and feed dynamics",
    "liquid.sandbox": "flow-diagram editor to build, wire and solve any liquid engine cycle, with a guided build, a trade "
                      "study and an animated schematic (opens its own window)",
    "common.nozzle": "nozzle area ratio gives the thrust coefficient curve, mach and pressure along the bell, and a 3d nozzle",
    "hybrid.regression": "compares fuel regression-rate laws against oxidizer mass flux and port size",
    "grain.hybrid": "designs a hybrid port and grain stack (custom ports, inhibitors, per-grain fuel) with a 3d view and "
                    "cross-section plots (opens its own window)",
    "hybrid.flow": "core flow in the port: taylor-culick and bidirectional-vortex velocity fields",
    "hybrid.fuel": "fuel and oxidizer properties and how they set regression and mixture ratio",
    "hybrid.stability": "intrinsic low-frequency instability estimate and geometric similarity scaling of a hybrid motor",
    "hybrid.burn0d": "lumped burn of a hybrid motor: pressure, thrust, mixture ratio shift and port growth over time",
    "hybrid.burn1d": "burn with the port resolved along its length: axial regression, throttling and throat erosion",
    "hybrid.n2o": "nitrous oxide blowdown from a self-pressurizing tank feeding a hybrid burn",
    "solid.burnrate": "burn rate law and the equilibrium chamber pressure against the burning-area ratio k_n",
    "grain.solid": "designs a solid grain section or stack (star, wagon wheel, finocyl, custom shapes, inhibitors) with burnback "
                   "tables and a 3d view (opens its own window)",
    "solid.burnback3d": "level-set burnback of a three-dimensional grain with the burning surface shown over time",
    "solid.ballistics": "internal ballistics of a solid motor: chamber pressure, thrust and burn time from the grain "
                        "geometry and propellant",
    "solid.cpmotor": "quasi-steady march of a cylindrical-port motor with end burning and throat erosion (si units)",
    "solid.cpmotor_us": "the same quasi-steady cylindrical-port motor, entered and shown in us units",
    "ep.trades": "mission and power trades: delta-v, propellant, power and thruster mass for electric propulsion",
    "ep.thermal": "electrothermal thrusters (resistojet, arcjet): heating, specific impulse and efficiency",
    "ep.ion": "gridded ion thruster: beam current, grid voltage, thrust, specific impulse and efficiency",
    "ep.hall": "hall thruster: discharge current, magnetic field, thrust and efficiency from a lumped model",
    "ep.optics": "particle simulation of the ion optics: beamlet trajectories, grid impingement and divergence",
    "ep.hall_run": "time-dependent hall discharge: ionization and current oscillations",
    "ep.spiral": "low-thrust spiral transfer with solar-array degradation (edelbaum)",
    "nuc.perf": "nuclear thermal rocket performance: isp, flow, reactor power and core size against a chemical engine",
    "nuc.crit": "one-group reactor criticality: k_eff, critical size and mass, reflector savings and flux shape",
    "nuc.element": "thermal hydraulics of a fuel element: propellant, wall and peak fuel temperatures and pressure drop",
    "nuc.cycles": "hot bleed, cold bleed and expander cycles for a nuclear thermal rocket with real hydrogen",
    "nuc.poisons": "xenon and samarium poisoning over a burn, coast and restart, against the control-drum margin",
    "nuc.kinetics": "reactor point kinetics with feedback: startup under temperature control, reactivity steps and scram",
    "vehicle.staging": "staging budget: ideal velocity, masses and thrust-to-weight, with atmosphere, drag and wind",
    "vehicle.ascent": "two-dimensional point-mass ascent over a round earth with staging, wind, drag and imported thrust",
    "toy.springmass": "damped spring and mass, a check of the run and playback pipeline (not a propulsion model)",
}
PLANNED = "planned, not built yet"
