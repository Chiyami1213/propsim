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

## Later blocks

Components, cycles, reference engines, materials, hybrid, electric and the schemas arrive in later PRs; their gaps
will be listed here.
