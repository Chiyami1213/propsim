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

## Small differences, no action

- LH2 and LOX critical and boiling points: CoolProp (modern equations of state) and Huzel & Huang Table 1-5 agree
  within 0.8 %. CoolProp is preferred.
- Hydrazine normal boiling point: Huzel & Huang 386.15 K vs NIST 386.65 K (0.13 %).
