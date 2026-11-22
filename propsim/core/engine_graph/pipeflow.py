r"""Pipe-flow losses for the engine-graph solver: friction in every flow regime, fitting loss coefficients, orifice
discharge coefficients from the edge geometry, isothermal compressible line flow and the water-hammer surge.

Sources (formulas only; cited by equation number):

* Churchill, S. W., "Friction-factor equation spans all fluid-flow regimes", Chem. Eng. 84 (1977) 91-92, as given in
  Rennels & Hudson, *Pipe Flow* (Wiley, 2012), eq. 8.13 (one smooth curve through laminar, critical and turbulent flow).
* Rennels & Hudson ch. 9-15: entrances (eqs. 9.1, 9.2, the sharp-edged value 0.57 from eq. 10.4 at beta = 0),
  sudden contraction (10.3, 10.4), sudden expansion (Borda-Carnot, 11.6), discharge into a large volume (ch. 12,
  K = 1), orifices (13.3, 13.4, 13.13 with the thickness factor in the form that reproduces Table 13.2), bends (15.1)
  and miter bends (15.5).
* Rennels & Hudson eq. 4.26b: isothermal flow of an ideal gas with friction.
* Rennels & Hudson eqs. 21.1-21.4 (SI form): wave speed, Joukowsky surge and round-trip time.

Every loss coefficient is on the velocity head of the pipe it is attached to unless stated.
"""
from __future__ import annotations

import math



# ------------------------------------------------------------------------------------------------------------ friction
def churchill(Re: float, eps_D: float = 0.0) -> float:
    r"""Darcy friction factor, all regimes (Churchill 1977):

    .. math:: f = 8\left[\left(\tfrac{8}{Re}\right)^{12} + (A+B)^{-3/2}\right]^{1/12},\;
              A = \left[2.457\ln\frac{1}{(7/Re)^{0.9}+0.27\,\varepsilon/D}\right]^{16},\; B = \left(\tfrac{37530}{Re}\right)^{16}

    It is 64/Re in laminar flow and the Colebrook value in fully turbulent flow, continuous in between."""
    Re = max(float(Re), 1e-3)
    A = (2.457 * math.log(1.0 / ((7.0 / Re) ** 0.9 + 0.27 * max(eps_D, 0.0)))) ** 16
    B = (37530.0 / Re) ** 16
    return 8.0 * ((8.0 / Re) ** 12 + (A + B) ** -1.5) ** (1.0 / 12.0)


# ------------------------------------------------------------------------------------------------------------ fittings
def _lam(beta: float) -> float:
    """Jet velocity ratio at the vena contracta of a sharp-edged contraction (eq. 10.3)."""
    return 1.0 + 0.622 * (1.0 - 0.215 * beta**2 - 0.785 * beta**5)


def contraction_sharp(beta: float) -> float:
    """Sudden contraction from a larger pipe, ``beta = d_small/d_large``, on the small pipe's velocity (eq. 10.4);
    0.57 for an entrance from a large volume."""
    lam = _lam(beta)
    return 0.0696 * (1.0 - beta**5) * lam**2 + (lam - 1.0) ** 2


def entrance_rounded(r_d: float) -> float:
    """Entrance from a large volume with an inlet radius ``r/d`` (eq. 9.2): 0.57 sharp, 0.03 at r/d = 1."""
    r_d = min(max(r_d, 0.0), 1.0)
    lam = 1.0 + 0.622 * (1.0 - 0.30 * math.sqrt(r_d) - 0.70 * r_d) ** 4
    return 0.0696 * (1.0 - 0.569 * r_d) * lam**2 + (lam - 1.0) ** 2


def entrance_reentrant(t_d: float) -> float:
    """Pipe projecting into a large volume with wall thickness ``t/d`` (eq. 9.1, t/d <= 0.05): 1.12 for a knife edge,
    0.57 at 0.05 and beyond."""
    t_d = min(max(t_d, 0.0), 0.05)
    return 1.12 - 22.0 * t_d + 216.0 * t_d**2 + 80.0 * t_d**3


def expansion_sudden(beta: float) -> float:
    """Sudden expansion into a larger pipe, ``beta = d_small/d_large``, on the small pipe's velocity (Borda-Carnot,
    eq. 11.6); 1 for discharge into a large volume."""
    return (1.0 - beta**2) ** 2


def bend(alpha_deg: float, r_d: float, f: float) -> float:
    """Smooth bend of angle ``alpha`` and radius ``r/d`` (eq. 15.1): friction along the bend, separation and the
    secondary flow."""
    a = math.radians(alpha_deg)
    s = math.sin(a / 2.0)
    r_d = max(r_d, 0.5)
    return f * a * r_d + (0.10 + 2.4 * f) * s + 6.6 * f * (math.sqrt(s) + s) / r_d ** (4.0 * a / math.pi)


def miter(alpha_deg: float) -> float:
    """Single miter bend of angle ``alpha`` (eq. 15.5)."""
    s = math.sin(math.radians(alpha_deg) / 2.0)
    return 0.42 * s + 2.56 * s**3


def orifice_K(beta: float = 0.0, t_do: float = 0.0, f: float = 0.02) -> float:
    """Orifice loss on the orifice's own velocity, ``beta = d_o/d_pipe`` (0: in a wall or manifold), ``t/d_o`` its
    length.  Sharp (t = 0, eq. 13.3) to thick (eq. 13.13, thickness factor C_th = [1 - (t/1.4d_o)^2.5]^3.45); past
    t = 1.4 d_o the jet has reattached and the bore adds its friction over the extra length (PROPSIM-EST form of
    eq. 13.15, continuous at 1.4 d_o).  2.81 sharp in a wall, 1.57 at t = 1.4 d_o."""
    lam = _lam(beta)
    acc = 0.0696 * (1.0 - beta**5) * lam**2
    x = min(max(t_do, 0.0) / 1.4, 1.0)
    cth = (1.0 - x**2.5) ** 3.45
    K = acc + cth * (lam - beta**2) ** 2 + (1.0 - cth) * ((lam - 1.0) ** 2 + (1.0 - beta**2) ** 2)
    if t_do > 1.4:
        K += f * (t_do - 1.4)
    return K


def cd_from_K(K: float) -> float:
    """Discharge coefficient of an element whose whole loss is ``K`` velocity heads at its own area
    (dp = K rho v^2/2 = mdot^2/(2 rho (Cd A)^2))."""
    return 1.0 / math.sqrt(K)


def fittings_K(opt: dict, f: float) -> tuple[float, dict]:
    """Loss coefficient (line velocity) of the fittings set on a line's options, with the itemised values."""
    def num(key, default):
        try:
            return float(opt.get(key, default))
        except (TypeError, ValueError):
            return float(default)
    parts = {}
    ent = opt.get("entrance", "none")
    if ent == "sharp":
        parts["entrance"] = contraction_sharp(0.0)
    elif ent == "rounded":
        parts["entrance"] = entrance_rounded(num("entrance_ratio", 0.1))
    elif ent == "re-entrant":
        parts["entrance"] = entrance_reentrant(num("entrance_ratio", 0.0))
    beta_c = num("contraction_beta", 1.0)
    if 0.0 < beta_c < 1.0:
        parts["contraction"] = contraction_sharp(beta_c)
    n = int(num("bends", 0))
    if n > 0:
        parts["bends"] = n * bend(num("bend_angle", 90.0), num("bend_r_over_d", 1.5), f)
    nm = int(num("miters", 0))
    if nm > 0:
        parts["miters"] = nm * miter(num("bend_angle", 90.0))
    beta_e = num("expansion_beta", 1.0)
    if 0.0 < beta_e < 1.0:
        parts["expansion"] = expansion_sudden(beta_e)
    if opt.get("exit", "none") == "free_discharge":
        parts["exit"] = 1.0
    return float(sum(parts.values())), parts


# ------------------------------------------------------------------------------------------------------- compressible
def isothermal_residual(p1: float, p2: float, G: float, RT: float, K: float) -> float:
    r"""Isothermal ideal-gas line with friction (eq. 4.26b), as a pressure residual (Pa):
    :math:`\frac{p_1^2-p_2^2}{p_1+p_2} - \frac{G|G|RT}{p_1+p_2}\left[K + 2\ln\frac{p_1}{p_2}\right]`."""
    p1, p2 = max(p1, 1.0), max(p2, 1.0)
    return (p1 * p1 - p2 * p2 - G * abs(G) * RT * (K + 2.0 * math.log(p1 / p2))) / (p1 + p2)


# ------------------------------------------------------------------------------------------------------- water hammer
def wave_speed(a_fluid: float, rho: float, D: float = 0.0, E: float = 0.0, t_wall: float = 0.0) -> float:
    """Pressure-wave speed in a liquid line (eq. 21.1, SI): the fluid's sound speed, slowed by the wall's stretch
    when its modulus ``E`` and thickness are known (rigid wall, the upper bound, otherwise)."""
    if E > 0 and t_wall > 0 and D > 0:
        B = rho * a_fluid**2
        return math.sqrt(B / rho / (1.0 + B * D / (E * t_wall)))
    return a_fluid


def surge(rho: float, a: float, dV: float, L: float, t_close: float | None = None) -> tuple[float, float]:
    """Joukowsky surge ``rho a dV`` (eqs. 21.2-21.3) and the round-trip time ``2L/a`` (21.4); a closure slower than
    the round trip cuts the surge in the ratio t_rt/t_close."""
    dp = rho * a * abs(dV)
    t_rt = 2.0 * L / max(a, 1e-9)
    if t_close is not None and t_close > t_rt:
        dp *= t_rt / t_close
    return dp, t_rt


__all__ = ["churchill", "contraction_sharp", "entrance_rounded", "entrance_reentrant", "expansion_sudden", "bend",
           "miter", "orifice_K", "cd_from_K", "fittings_K", "isothermal_residual", "wave_speed", "surge"]
