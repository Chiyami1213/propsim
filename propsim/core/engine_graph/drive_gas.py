r"""Gas-generator and preburner products: turbine drive gas temperature, ``cp`` and ``gamma`` from the mixture ratio.

Three models, tried in this order by ``model="auto"``:

``table``
    Huzel & Huang Table 6-4 (``cycles/turbine_drive_gas.csv``): fuel-rich LOX/RP-1, LOX/LH2 and N2O4/UDMH gas versus
    mixture ratio, interpolated inside the published range.
``frozen``
    Complete combustion with the excess reactant frozen: all carbon to CO2, hydrogen to H2O, nitrogen to N2, and the
    surplus left as O2 (oxidizer-rich) or H2 (fuel-rich hydrogen).  Adiabatic temperature from the enthalpy balance
    with reactant enthalpies of formation from ``fluids/cea_species.csv`` (CEA thermo data) plus their sensible
    enthalpy from CoolProp, and product enthalpies of formation from NIST-JANAF (Chase 1998, 4th ed.) with ideal-gas
    sensible enthalpies from CoolProp.  It is the usual preliminary-design approximation for oxidizer-rich gas and for
    hydrogen-rich gas at turbine temperatures; fuel-rich hydrocarbon gas (soot, CO, CH4) is outside it.
``user``
    The temperature, ``cp`` and ``gamma`` given on the node.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from ..constants import R_U
from .props import coolprop_name, props

# product gases: CoolProp name, molar mass (kg/mol), enthalpy of formation at 298.15 K (J/mol, NIST-JANAF)
PRODUCTS = {"H2O": ("Water", 0.01801528, -241826.0), "CO2": ("CarbonDioxide", 0.0440095, -393522.0),
            "N2": ("Nitrogen", 0.0280134, 0.0), "O2": ("Oxygen", 0.0319988, 0.0),
            "H2": ("Hydrogen", 0.00201588, 0.0)}
TABLE_PAIRS = {("LOX", "RP1"): "LOX/RP1", ("LOX", "LH2"): "LOX/LH2", ("N2O4", "UDMH"): "N2O4/UDMH"}


@lru_cache(maxsize=None)
def _table(pair: str) -> dict[str, np.ndarray]:
    from ...data import cycle_table
    rows = [r for r in cycle_table("turbine_drive_gas") if r["propellants"] == pair]
    rows.sort(key=lambda r: float(r["MR"]))
    return {k: np.array([float(r[k]) for r in rows]) for k in ("MR", "T_inlet", "cp", "gamma", "R")}


def table_gas(ox: str, fuel: str, mr: float):
    pair = TABLE_PAIRS.get((ox, fuel))
    if pair is None:
        return None
    t = _table(pair)
    if not (t["MR"][0] <= mr <= t["MR"][-1]):
        return None
    f = lambda k: float(np.interp(mr, t["MR"], t[k]))
    return dict(T=f("T_inlet"), cp=f("cp"), gamma=f("gamma"), model="Huzel & Huang Table 6-4")


@lru_cache(maxsize=None)
def _species(fid: str) -> list[tuple[str, float]]:
    """``(species_id, mass fraction)`` of a library fluid in ``cea_species.csv`` terms."""
    from ...data import cea_mixtures, cea_species
    sp, mx = cea_species(), cea_mixtures()
    if fid in sp:
        return [(fid, 1.0)]
    if fid in mx:
        out = []
        for part in mx[fid]["components"].split(";"):
            k, v = part.split(":")
            out.append((k, float(v)))
        return out
    alias = {"RP1_SURR": "RP1", "GHE": "GHE", "LN2": "GN2"}
    if fid in alias and alias[fid] in sp:
        return [(alias[fid], 1.0)]
    raise ValueError(f"no CEA species data for {fid!r}")


def _elements(text: str) -> dict[str, float]:
    tok = text.split()
    return {tok[i]: float(tok[i + 1]) for i in range(0, len(tok) - 1, 2)}


def _reactant(fid: str, T: float, p: float) -> tuple[dict[str, float], float]:
    """Moles of each element and enthalpy (J) per kg of fluid at ``(T, p)``."""
    from ...data import cea_species
    sp = cea_species()
    el: dict[str, float] = {}
    H = 0.0
    for sid, w in _species(fid):
        row = sp[sid]
        M = float(row["molar_mass"])
        n = w / M
        for e, k in _elements(row["elements"]).items():
            el[e] = el.get(e, 0.0) + n * k
        H += n * float(row["dHf"])
    name = coolprop_name(fid)
    if name is not None and len(_species(fid)) == 1:                 # sensible enthalpy from the CEA reference T
        row = sp[_species(fid)[0][0]]
        pr = props(fid)
        T_ref = float(row["T_ref"])
        try:
            h_ref = pr.h_pT(101325.0 if row["phase"] == "condensed" else 1e3, T_ref)
            if row["phase"] == "condensed" and T_ref < 200:          # cryogen: CEA value is the saturated liquid
                import CoolProp.CoolProp as CP
                st = props(fid)._st()
                st.update(CP.QT_INPUTS, 0.0, T_ref)
                h_ref = st.hmass()
            H += pr.h_pT(p, T) - h_ref
        except ValueError:
            pass
    return el, H


@lru_cache(maxsize=None)
def _h_sens(gas: str, T: float) -> float:
    """Ideal-gas sensible enthalpy (J/mol) above 298.15 K; beyond the CoolProp range cp is held at its last value."""
    import CoolProp.CoolProp as CP
    name, M, _ = PRODUCTS[gas]
    st = CP.AbstractState("HEOS", name)
    Tmax = st.Tmax() - 1.0
    Tc = min(T, Tmax)

    def h(t):
        st.update(CP.PT_INPUTS, 100.0, t)
        return st.hmass()
    out = h(Tc) - h(298.15)
    if T > Tmax:
        st.update(CP.PT_INPUTS, 100.0, Tmax)
        out += st.cp0mass() * (T - Tmax)
    return out * M


@lru_cache(maxsize=None)
def _cp(gas: str, T: float) -> float:
    import CoolProp.CoolProp as CP
    name, M, _ = PRODUCTS[gas]
    st = CP.AbstractState("HEOS", name)
    st.update(CP.PT_INPUTS, 100.0, min(max(T, 250.0), st.Tmax() - 1.0))
    return st.cp0mass() * M


def frozen_gas(ox: str, fuel: str, mr: float, T_ox: float, T_fuel: float, p: float):
    """Complete-combustion products of 1 kg of mixture at mixture ratio ``mr`` (oxidizer / fuel mass)."""
    try:
        eo, Ho = _reactant(ox, T_ox, p)
        ef, Hf = _reactant(fuel, T_fuel, p)
    except (ValueError, KeyError):
        return None
    yo, yf = mr / (1.0 + mr), 1.0 / (1.0 + mr)
    el = {e: yo * eo.get(e, 0.0) + yf * ef.get(e, 0.0) for e in set(eo) | set(ef)}
    H_in = yo * Ho + yf * Hf
    C, H, O, N = (el.get(k, 0.0) for k in ("C", "H", "O", "N"))
    if set(el) - {"C", "H", "O", "N"}:
        return None
    need = 2.0 * C + 0.5 * H
    n = {"CO2": C, "N2": 0.5 * N}
    if O >= need:
        n["H2O"], n["O2"] = 0.5 * H, 0.5 * (O - need)
    elif C == 0.0:
        n["H2O"], n["H2"] = O, 0.5 * (H - 2.0 * O)
    else:
        return None                                                  # fuel-rich hydrocarbon: not a frozen mixture
    n = {k: v for k, v in n.items() if v > 1e-12}

    def resid(T):
        return sum(v * (PRODUCTS[k][2] + _h_sens(k, round(T, 1))) for k, v in n.items()) - H_in
    try:
        T = brentq(resid, 298.15, 5000.0, xtol=0.5)
    except ValueError:
        return None
    mass = sum(v * PRODUCTS[k][1] for k, v in n.items())
    cp = sum(v * _cp(k, T) for k, v in n.items()) / mass
    R = R_U * sum(n.values()) / mass
    return dict(T=float(T), cp=float(cp), gamma=float(cp / (cp - R)), model="complete combustion, frozen excess")


def drive_gas(ox: str, fuel: str, mr: float, T_ox: float, T_fuel: float, p: float, model: str = "auto",
              user: tuple[float | None, float | None, float | None] = (None, None, None)) -> dict:
    """Drive-gas ``T, cp, gamma, model`` for a gas generator or preburner; raises ``ValueError`` if no model applies."""
    if model == "user" or (model == "auto" and all(v is not None for v in user)):
        T, cp, g = user
        if None in user:
            raise ValueError("user drive-gas model needs T_out, cp_gas and gamma_gas on the node")
        return dict(T=T, cp=cp, gamma=g, model="user values")
    if model in ("auto", "table"):
        r = table_gas(ox, fuel, mr)
        if r is not None:
            return r
        if model == "table":
            raise ValueError(f"Table 6-4 has no {ox}/{fuel} gas at O/F {mr:.3f}")
    r = frozen_gas(ox, fuel, mr, T_ox, T_fuel, p)
    if r is None:
        raise ValueError(f"no drive-gas model for {ox}/{fuel} at O/F {mr:.3f}: fuel-rich hydrocarbon gas needs the "
                         "user model (set T_out, cp_gas and gamma_gas)")
    return r
