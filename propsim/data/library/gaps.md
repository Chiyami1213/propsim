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

10. **N2O transport properties** (update to item 4): the NIST WebBook does have N2O viscosity and thermal conductivity
    (Huber, NISTIR 8209, 2018, https://doi.org/10.6028/NIST.IR.8209; thermal conductivity data of Richter and Sage 1963).
    Ten state points are recorded in `fluids/webbook_check_points.csv`, but `tables/N2O.csv` still has empty `mu` and `k`
    columns. To fill them: implement the NISTIR 8209 correlations (an open NIST report) or tabulate the WebBook grid,
    which would copy a large part of SRD 69 (see `sources.csv` for its terms).
11. **LCH4 viscosity** follows CoolProp's published 2006 correlation; the WebBook uses a newer correlation that is
    unpublished (Quinones-Cisneros, Huber and Deiters, 2011) and cannot be implemented independently
    (`conflicts.md`).
12. **NIST-JANAF species beyond the 12 transcribed** (for example NH3, N2O, C2H4, HCl, H2O2, HF, condensed species such as
    Al2O3, and the JANAF tables of metals for solid-propellant products) are not in the library; the script
    `tools/build_janaf_tables.py` takes more species from its `SPECIES` table. JANAF gives no fitted polynomials: heat
    capacities between the 100 K rows are interpolated (`propsim.data.janaf_cp`).
13. **JANAF and WebBook redistribution terms**: both are copyrighted NIST compilations (`sources.csv`). Whether recording
    the 12 tables and 40 state points here is covered by NIST's terms is for the repository owner to confirm.

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

1. **SSME example graph** uses public NASA data only (1981 Space Shuttle News Reference at rated power, NASA
   TP-3700 heat balance for the preburner split). Still missing from public sources: turbine efficiencies and exit
   pressures, injector and valve pressure drops, coolant channel geometry, shaft inertias, the 104.5 % power level
   station set and the controller start schedule. The main valve losses through the printed 2.5 in passages
   (oxidizer about 820 psi) look high; the passage size may refer to a different part of the valve.
2. **Estimated values in the examples** (counts in `checks.md` section 7): line lengths, manifold volumes, tank
   sizes, valve stroke times and sequence thresholds are not published for the Huzel & Huang engines. The pressure-fed
   kerolox and expander hydrolox graphs adapt published engines to other propellants or another cycle.
3. **Jacket and dome hydraulic losses** are lumped into equivalent orifices until the jacket model carries its own
   pressure drop.
4. **Suction lines and vehicle tanks** are not modelled; tank pressure is the published pump inlet pressure.
5. **Component ranges widened** to admit published engines: injector dp/P_c 0.1 to 0.3 (A-2 fuel 0.125, A-4 0.25
   against the 0.15 to 0.2 rule of thumb), nozzle C_F efficiency up to 1.05 (A-2 to A-4 list 1.01 to 1.02),
   turbine inlet temperature from 100 K and pressure ratio from 1.05 (expander turbines).

## Materials (block E)

1. **Above 300 K**: Inconel 718 is covered to 1366 K (Special Metals bulletin) and GRCop-84 has room temperature
   and 538 degC strength plus a conductivity range. Still missing: GRCop-84 conductivity, expansion and modulus
   versus temperature (NASA/TM-2005-213566 gives them only as plots), CuCrZr, OFHC copper, C-103 niobium and
   304/316/Ti-6Al-4V above 300 K. Candidate public source: MIL-HDBK-5J (public domain, figures need digitizing).
2. **Strength** versus temperature for the aluminum, stainless and titanium alloys; Inconel 718, GRCop-84 and
   Inconel X have points.
3. **Density, melting point and service temperature** are not in the NIST pages; to come from the same sources.
4. **Ablatives and graphite** (carbon-phenolic, silica-phenolic, carbon-carbon): density, conductivity and
   char/erosion data from NASA SP-8xxx monographs.
5. **Missing NIST properties**: Inconel 718 and Ti-6Al-4V specific heat and modulus, copper modulus.

6. **Above 300 K for the chamber-wall alloys**: no values were added in this pass because the candidate public sources
   give the data as plots or were not read in full. Candidates: NASA CR-134806 (Esposito and Zabora, 1975, NTRS
   19750021165: Amzirc, NARloy-Z, copper, silver, nickel, 27.6 to 810.9 K, graphs); NASA CR-134627 (NARloy-Z low-cycle
   fatigue, 1974, tensile data at several temperatures); NASA CR-198529 (Ellis and Michal, 1996, Cu-Cr-Nb against
   NARloy-Z, NTRS 19970002915); ANL-75-55 (evaluated enthalpy, specific heat, expansion and conductivity of 304L and 316
   stainless steels, 300 to 3000 K, tables); NASA CR-54151 (Battelle, 316 stainless conductivity 0 to 1800 degF, NTRS
   19650024830). The NIST cryogenic pages (`property_fits.csv`) stop at 300 K.

## Hybrid (block F)

1. **More regression laws**: HTPB/N2O, paraffin/N2O, HDPE, PMMA and ABS with N2O and LOX, including G^n x^m
   forms with a length exponent. Public candidates: NASA/NTRS reports, university theses and AIAA papers.
2. **SP-1a O/F-corrected law** (Karabeyoglu 2003 eq. 3): its layout could not be read from the PDF text layer.
3. **Hot-fire cases**: only SP-1a/GOX so far; per-test grain length and fuel mass are not tabulated, and the
   chamber pressure is not marked gauge or absolute. Test 4L-12 is left out: its initial flux disagrees with its
   flow rate and port by a factor of 2.3. N2O cases with thrust and pressure histories are still needed.
4. **Fuel properties**: density, heat of gasification and melt-layer properties of paraffin and HTPB blends.
5. **Graphite nozzle erosion rates** and post-combustion chamber guidance.

## Electric (block G)

1. **Only NEXT so far.** Throttle tables for NSTAR, Hall thrusters (SPT-100, BPT-4000, HiVHAc, HERMeS/AEPS),
   arcjets and resistojets are still needed from open NASA/NTRS or AIAA sources.
2. **NEXT end-of-life performance**: the tables are beginning of life; degradation with throughput is not tabulated.
3. **Thruster power on the nominal table**: Table 3 prints PPU input power only; thruster power is taken as
   P_PPU eta_PPU (`propsim.data.thruster_power`), which carries the two-figure rounding of eta_PPU.
4. **Mass and geometry**: thruster, PPU and feed-system masses and the gimbal envelope are not in the library.
5. **NASA/TM-2002-211892** (Oleson et al.) Table 2 has a further NEXT throttle set, but the power column's unit is
   not legible in the text layer, so it was not transcribed.
