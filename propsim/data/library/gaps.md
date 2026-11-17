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
11. **Gas generator**: H&H Table 6-4 (fuel-rich GG products) not transcribed yet; temperature limits come with
    the materials block.

## Later blocks

Cycles, reference engines, materials, hybrid, electric and the schemas arrive in later PRs; their gaps
will be listed here.
