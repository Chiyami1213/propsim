# Conflicts

Where sources disagree, both values stay in the data (`preferred` marks the one used) and the choice is explained
here. Generated cross-checks are in `checks.md`, section 3.

## MMH normal boiling point

| source | value | location |
|---|---|---|
| Huzel & Huang (HH-SP125) | 359.3 K (187 degF) | Table 1-4, p. 22 |
| NIST WebBook | 364 K | Majer and Svoboda 1985 (no uncertainty given) |

Spread 1.3 %. **Used: Huzel & Huang**, because it describes propellant-grade MMH and the NIST entry has no stated
uncertainty. The 4.7 K difference matters only for vapour-pressure and cavitation margins near the boiling point. To
resolve it, use the MMH propellant specification (MIL-PRF-27404) or the Aston et al. vapour-pressure data cited by
NIST.

## LOX/RP-1 chamber temperature at O/F = 2.00

Huzel & Huang Table 1-9 gives 3455 K; the shipped CEA tables give 3352 K (-3.0 %), while c* and Isp agree within
1.1 %. Likely cause: the 1960s RP-1 heat of formation behind Huzel & Huang versus the CEA RP-1 card (Mehta et al.
1995). **Used: the CEA tables** (modern species data). No action needed for preliminary design.

## A-1 turbine shaft torque unit

Huzel & Huang Table 3-2 (p.65) prints the A-1 turbine torque as 20380 in-lb. Sample Calculation (6-5) (p.199) uses
20380 ft-lb, and only ft-lb agrees with the printed 27140 bhp at 7000 rpm (T omega = P within 0.1 %; in-lb would be
12 times too small). Both rows are kept in `engine_params.csv`; the sample-calculation row is preferred. The A-2
torques (Table 3-3) are in-lb and agree with their power and speed.

## A-1 thrust-chamber thrust

Table 3-2 gives 747000 lb; Sample Calculation (6-6) (p.201) uses the test value 747300 lb (0.04 %). The table
value is preferred.

## Table 6-4 gas constant at 1250 degF (LOX/RP-1)

Table 6-4 (p.197) prints R = 58.6 ft/degR at 1250 degF, between 47.1 at 1200 degF and 50.4 at 1300 degF; c_p (gamma -
1) / gamma gives 50.2. The printed value is kept as published (checked against the page image) and flagged in
`checks.md`; use c_p and gamma for that row, or interpolate R from the neighbours.

## Inconel 718 thermal conductivity near room temperature

NIST's cryogenic fit gives 9.7 W/(m K) at 294 K; the Special Metals bulletin (Table 5, annealed and aged) gives
79 Btu in/(ft2 h degF) = 11.4 W/(m K) at 70 degF, 17 % higher. The NIST data end at 275 K and the heat-treatment
states may differ. Both are kept: use the NIST fit below 294 K and the bulletin points above.

## SSME chamber pressure and area ratio

| source | value | location |
|---|---|---|
| Space Shuttle News Reference 1981 (NASA-SNR-1981) | 2970 psia (summary), about 3000 psi (text); area ratio 77.5 | pp. 2-4, 2-9 |
| NASA TP-3700 heat balance (NASA-TP-3700) | 3006 psia | Table A.6 p. 26 |
| L3Harris RS-25 spec sheet 2024 (L3HARRIS-RS25-2024) | 20.64 MPa (2994 psia); area ratio 69 | spec sheet |

The SSME example graph uses the 1981 summary value 2970 psia and area ratio 77.5, because its pump and turbine
station data come from the same document. The 2024 sheet describes the current RS-25 for SLS and stays in
`engines/engine_params.csv`.

## Small differences, no action

- LH2 and LOX critical and boiling points: CoolProp (modern equations of state) and Huzel & Huang Table 1-5 agree
  within 0.8 %. CoolProp is preferred.
- Hydrazine normal boiling point: Huzel & Huang 386.15 K vs NIST 386.65 K (0.13 %).
