# Gaps

Values the library does not have yet, with where better data could come from. Ordered by impact on a preliminary
design run.

## Fluids (block A)

1. **Temperature-dependent properties of storables (N2O4, MMH, UDMH, N2H4, H2O2 70/90/95/98 %).** Only scalars
   (Huzel & Huang Table 1-4, NIST WebBook, CEA) are in the library. Runtime option: `rocketprops` (GPL-3.0, not
   bundled). Open primary sources for a bundled table: the Rocketdyne propellant handbooks and AFRL reports listed
   by RocketProps (e.g. the Rocketdyne N2O4 report), NTRS 19680009997, DTIC ADA482266.
2. **Isopropanol liquid properties** (density, viscosity, vapour pressure versus T): not in CoolProp or RocketProps.
   Candidates: DIPPR 801 (paywalled), NIST WebBook fluid pages (not available for isopropanol), Yaws handbook.
3. **RP-1 as a real fuel.** `RP1_SURR` is pure n-dodecane: density about 3 % low and critical point different from
   RP-1. Better: the NIST four-component RP-1 surrogate (Huber et al., Energy & Fuels 2009), available in REFPROP
   (paywalled) or rebuildable in CoolProp as a mixture if its binary parameters are published.
4. **N2O transport properties** (viscosity, conductivity): CoolProp 8.0 has no model, so `mu` and `k` are empty in
   `tables/N2O.csv`. Candidates: Lemmon & Huber transport correlations in REFPROP; Span-Wagner-based fits in
   N2O hybrid literature.
5. **Ethanol-water mixture transport** is CoolProp's estimation, and some high-pressure points are empty
   (`checks.md`, section 4). The mixture tables stop at 420 K because the mixture flash fails nearer the critical
   point; regenerative-jacket outlet temperatures above that need another source.
6. **H2O2 blend critical points and saturation curves**: only the extrapolated pure-H2O2 critical point is sourced
   (NIST, uncertainty 15 K / 2 MPa).
7. **Surface tension and speed of sound of storables**: not sourced yet.
8. **CEA species for hybrid fuels** (HTPB, paraffin, PE, PMMA, ABS): not in the NASA thermo database; they belong to
   block F with literature heats of formation.
9. **Liquid isopropanol heat of formation**: the CEA database has only the gas (`IPA_GAS`); a liquid card needs the
   enthalpy of vaporization (NIST lists 45 ± 3 kJ/mol average).

## Components (block B)

1. **Valve opening and closing time laws** and stroke times: user input for now. Candidates: SSME System Engineer
   Handbook (valve sequences, in the literature archive, scanned), NASA SP-8080 (liquid rocket pressure regulators,
   relief valves, check valves, burst disks) and SP-8097 (liquid rocket valve components) on NTRS.
2. **Pressure regulator, burst disc, filter, flexible joint and tee/bend-specific models**: only loss coefficients
   (Table 7-4) so far; no regulator dynamics record yet (H&H Table 7-6 has a dome-loaded regulator example).
3. **Cavitating venturi** discharge coefficient and pressure-recovery ratio.
4. **Inducer head-breakdown law and pump cavitation curves**: Stenning, MIT GTL Report 44 (archive) and NASA
   SP-8052 (turbopump inducers).
5. **Nondimensional pump H-Q curve family** versus specific speed: H&H Fig. 6-24 is one pump only; NASA SP-8109.
6. **Turbines**: partial admission, reaction and radial turbines, choked-flow relation (NASA SP-8110).
7. **Shaft inertia** of real turbopumps (SSME handbook, SP-8107).
8. **Injector per-type Cd, atomization and mixing efficiency** (NASA SP-8089); film and transpiration cooling;
   supercritical coolant corrections (block B follow-up).
9. **Nozzle**: method-of-characteristics validation cases, dual-bell and loss breakdown (NASA SP-8120).
10. **Joukowsky surge relation**: implemented in the code, primary citation not yet in `sources.csv`.
11. **Gas generator**: fuel-rich product properties are in `cycles/turbine_drive_gas.csv` (Table 6-4) for
    LOX/RP-1, N2O4/UDMH and LOX/LH2 only; LOX/CH4 and oxidizer-rich (ORSC) preburner gas have no source yet.

## Reference engines (block D)

1. **Flight engines have top-level data only** (thrust, Isp, MR, expansion ratio, mass, size). No flow rates,
   throat areas, turbopump or preburner data, so no closure check runs for them. Candidates: RS-25 from the SSME
   System Engineer Handbook in the literature archive (numbers only), J-2 and F-1 engine manuals on NTRS.
2. **Ambient condition** of RL10, F-1 and J-2 thrust and Isp is not printed in the sources used (`condition =
   unstated`). RS-25 values other than thrust carry no power level.
3. **J-2 chamber pressure**: the source says only "a little over 700 psia"; not entered.
4. **F-1 chamber pressure** is printed as 1,100 psi without a station (injector end or nozzle stagnation).
5. **RL10B-2 length and nozzle diameter**: the data sheet layout mixes the RL10B-2 and RL10C-2-1 rows; only the
   values the RL10 web page lists for that model are entered.
6. **Not yet entered**: Vulcain 2, Vinci, LE-7A, RD-180, Merlin, Raptor, BE-4 and other modern engines (primary
   sources: ESA, JAXA and manufacturer sheets; many commercial engines publish only rounded numbers).

## Cycles (block C)

1. **No expander reference engine with a sourced cycle label**: the RL10 sources used do not name the cycle.
2. **Tap-off cycle** has no PropSim implementation (`propsim_cycle = null`).
3. **Cycle-efficiency curve versus chamber pressure** (H&H Fig. 6-28) not digitized.
4. **Oxidizer-rich staged combustion and full-flow staged combustion**: no public source with numbers chosen yet.

## Graphs and dashboards (block H)

1. **SSME staged-combustion example graph** is not included yet: its station data (pressures, temperatures and flows)
   come from documents marked proprietary or private-use, and the repository is public. A public NASA source is needed.
2. **Estimated values in the examples** (counts in `checks.md` section 7): line lengths, manifold volumes, tank
   sizes, valve stroke times and sequence thresholds are not published for the Huzel & Huang engines. The pressure-fed
   kerolox and expander hydrolox graphs adapt published engines to other propellants or another cycle.
3. **Jacket and dome hydraulic losses** are lumped into equivalent orifices until the jacket model carries its own
   pressure drop.
4. **Suction lines and vehicle tanks** are not modelled; tank pressure is the published pump inlet pressure.
5. **Component ranges widened** to admit published engines: injector dp/P_c 0.1 to 0.3 (A-2 fuel 0.125, A-4 0.25
   against the 0.15 to 0.2 rule of thumb), nozzle C_F efficiency up to 1.05 (A-2 to A-4 list 1.01 to 1.02),
   turbine inlet temperature from 100 K and pressure ratio from 1.05 (expander turbines).

## Later blocks

Materials, hybrid and electric arrive in later PRs; their gaps will be listed here.
