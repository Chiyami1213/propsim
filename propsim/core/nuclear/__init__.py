r"""Nuclear thermal propulsion (NTR): performance, reactor physics, fuel-element thermal hydraulics, engine cycles,
reactor kinetics with thermal feedback, and fission-product poisoning.

The equations follow Emrich, *Principles of Nuclear Rocket Propulsion*, 2nd ed. (2023); each function cites the chapter
and equation. Numbers that come from a source carry the source in a comment. Everything here is SI unless a name says
otherwise (Emrich works in cm, g and W·cm⁻³; the tests convert).

Submodules
----------
``performance``  Isp, thrust, reactor power and core size of a frozen-flow hydrogen (or other) NTR (Emrich Ch. 2).
``criticality``  one-group bare and reflected cores: buckling, :math:`k_{eff}`, critical size and mass (Ch. 8).
``thermal``      axial-flow fuel cell: chopped-cosine power, propellant / wall / peak fuel temperature, pressure drop
                 (Ch. 9).
``cycles``       hot bleed, cold bleed and expander cycle energy balances with real hydrogen (Ch. 3).
``kinetics``     six-group point kinetics with fuel, moderator and propellant-density feedback, decay heat and the
                 reactor transfer function (Ch. 11-12).
``poisons``      iodine-135 / xenon-135 and promethium-149 / samarium-149 over a burn and shutdown history (Ch. 13).
"""
from __future__ import annotations

R_U = 8314.462618            # (J·kmol⁻¹·K⁻¹) universal gas constant (2019 SI exact)
SIGMA_SB = 5.670374419e-8    # (W·m⁻²·K⁻⁴) Stefan-Boltzmann constant (CODATA 2018)
BARN = 1e-28                 # (m²)

# Six-group delayed-neutron data for thermal fission of U-235: decay constant (s⁻¹) and yield fraction beta_i.
# Source: Emrich 2e Table 11.1 (Keepin, Physics of Nuclear Kinetics, 1965). Total beta = 0.0065.
DELAYED_U235 = {
    "lambda": (0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01),
    "beta": (0.00022, 0.00142, 0.00127, 0.00257, 0.00075, 0.00027),
}
