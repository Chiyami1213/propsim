r"""Fluid states for the engine-graph solver: propellant liquids/real gases (CoolProp) and ideal combustion gases.

A :class:`State` is what flows out of a component port.  Pure propellants keep their own specific enthalpy ``h`` (the
CoolProp reference of that fluid) so pumping, heating in a cooling jacket and expansion in a turbine are real-fluid
calculations; combustion products are an ideal gas with ``cp``, ``gamma`` and ``R``.  Pressure is not part of the
state: it belongs to the station (the junction of ports) the state is in.

Fluids are the ``fluid_id`` values of ``propsim/data/library/fluids/fluids.csv``.  RP-1 has no CoolProp model: its
n-dodecane surrogate (``RP1_SURR``) supplies enthalpy, viscosity and conductivity and the density is scaled to the
published 293 K value.  Storables without an equation of state (N2O4, MMH, UDMH, N2H4, peroxides, IPA) are treated as
incompressible at their published density with temperature held at the tank value.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from ..constants import R_U

SURROGATE = {"RP1": "RP1_SURR"}


@dataclass(frozen=True)
class State:
    """Flowing state at a port. ``kind`` is ``"fluid"`` (a library propellant) or ``"gas"`` (ideal combustion gas)."""

    kind: str
    fluid: str | None
    T: float                  # (K)
    h: float                  # (J/kg) fluid reference for "fluid", cp*T for "gas"
    ox_frac: float            # mass fraction that came from the oxidizer feed (for the mixture ratio)
    cp: float = 0.0           # (J/(kg K)) gas only
    gamma: float = 0.0        # gas only
    R: float = 0.0            # (J/(kg K)) gas only

    def with_(self, **kw) -> "State":
        return replace(self, **kw)


@lru_cache(maxsize=None)
def _fluid_row(fid: str) -> dict[str, str]:
    from ...data import fluids
    rows = fluids()
    if fid not in rows:
        raise ValueError(f"unknown fluid {fid!r} (not in fluids.csv)")
    return rows[fid]


@lru_cache(maxsize=None)
def _ref_density(fid: str) -> float | None:
    """Published liquid density at 293 K (mean of min/max when the source gives a range)."""
    from ...data import fluid_scalars
    vals = {r["property"]: float(r["value"]) for r in fluid_scalars(fid) if r["value"]}
    if "rho_liq" in vals:
        return vals["rho_liq"]
    if "rho_liq_min" in vals and "rho_liq_max" in vals:
        return 0.5 * (vals["rho_liq_min"] + vals["rho_liq_max"])
    return None


def coolprop_name(fid: str) -> str | None:
    """CoolProp fluid string of a library fluid (the surrogate's for RP-1), or ``None``."""
    row = _fluid_row(SURROGATE.get(fid, fid))
    return row["coolprop_name"] if row["model"] == "coolprop" and row["coolprop_name"] else None


@lru_cache(maxsize=None)
def _state(name: str):
    import CoolProp.CoolProp as CP
    backend, _, fl = name.rpartition("::")
    return CP.AbstractState(backend or "HEOS", fl)


class Props:
    """Thermophysical properties of one library fluid at ``(p, h)`` or ``(p, T)``."""

    def __init__(self, fid: str) -> None:
        self.fid = fid
        self.cp_name = coolprop_name(fid)
        self.rho_scale = 1.0
        self.incompressible = self.cp_name is None
        self.rho_ref = _ref_density(fid)
        if self.incompressible and self.rho_ref is None:
            raise ValueError(f"fluid {fid!r} has neither an equation of state nor a published density")
        if fid in SURROGATE and self.rho_ref is not None:
            st = _state(self.cp_name)
            import CoolProp.CoolProp as CP
            st.update(CP.PT_INPUTS, 101325.0, 293.15)
            self.rho_scale = self.rho_ref / st.rhomass()

    # --- CoolProp helpers -------------------------------------------------------------------------------------
    def _st(self):
        return _state(self.cp_name)

    def h_pT(self, p: float, T: float) -> float:
        if self.incompressible:
            return p / self.rho_ref
        import CoolProp.CoolProp as CP
        st = self._st()
        st.update(CP.PT_INPUTS, max(p, 1e3), T)
        return st.hmass()

    def T_ph(self, p: float, h: float, T_hint: float) -> float:
        if self.incompressible:
            return T_hint
        import CoolProp.CoolProp as CP
        st = self._st()
        try:
            st.update(CP.HmassP_INPUTS, h, max(p, 1e3))
            return st.T()
        except ValueError:
            return T_hint

    def rho_at(self, p: float, h: float, T_hint: float) -> float:
        """Density alone at ``(p, h)``: the hot path of the solver, so no transport properties are evaluated."""
        if self.incompressible:
            return self.rho_ref
        import CoolProp.CoolProp as CP
        st = self._st()
        try:
            st.update(CP.HmassP_INPUTS, h, max(p, 1e3))
            return st.rhomass() * self.rho_scale
        except ValueError:
            return self.at(p, h, T_hint)["rho"]

    def sound_speed(self, p: float, h: float) -> float:
        """Speed of sound (m/s) at ``(p, h)``; 0 when the equation of state cannot give it."""
        if self.incompressible:
            return 0.0
        import CoolProp.CoolProp as CP
        st = self._st()
        try:
            st.update(CP.HmassP_INPUTS, h, max(p, 1e3))
            return float(st.speed_sound())
        except ValueError:
            return 0.0

    def at(self, p: float, h: float, T_hint: float) -> dict[str, float]:
        """``T, rho, mu, cp, k, phase`` (``phase``: liquid, gas or supercritical) and ``s`` at ``(p, h)``."""
        if self.incompressible:
            return dict(T=T_hint, rho=self.rho_ref, mu=1e-3, cp=2000.0, k=0.15, phase="liquid", s=0.0)
        import CoolProp.CoolProp as CP
        st = self._st()
        p = max(p, 1e3)
        try:
            st.update(CP.HmassP_INPUTS, h, p)
        except ValueError:
            try:
                st.update(CP.PT_INPUTS, p, float(np.clip(T_hint, st.Tmin() + 0.5, st.Tmax() - 1.0)))
            except ValueError:
                try:
                    st.update(CP.PT_INPUTS, max(p, 2e5), st.Tmin() + 2.0)
                except ValueError:                       # far outside the fluid's range (a diverging iterate)
                    st.update(CP.PT_INPUTS, float(np.clip(p, 2e5, 0.5 * st.pmax())), st.Tmin() + 2.0)
        T = st.T()
        try:
            mu, k = st.viscosity(), st.conductivity()
        except ValueError:
            mu, k = 1e-4, 0.1
        q = st.Q()
        if 0.0 <= q <= 1.0:
            phase = "two-phase"
        elif T > st.T_critical() and p > st.p_critical():
            phase = "supercritical"
        else:
            phase = "liquid" if st.rhomass() > st.rhomass_critical() else "gas"
        try:
            cp = st.cpmass()
        except ValueError:
            cp = 2000.0
        return dict(T=T, rho=st.rhomass() * self.rho_scale, mu=mu, cp=cp, k=k, phase=phase, s=st.smass())

    def h_ps(self, p: float, s: float) -> float:
        """Isentropic enthalpy at ``p`` (turbine expansion of a pure fluid)."""
        import CoolProp.CoolProp as CP
        st = self._st()
        st.update(CP.PSmass_INPUTS, max(p, 1e3), s)
        return st.hmass()

    def p_vap(self, T: float) -> float:
        """Saturation pressure at ``T`` (0 above the critical point or without an equation of state)."""
        if self.incompressible:
            from ...data import fluid_scalars
            for r in fluid_scalars(self.fid):
                if r["property"] == "p_vap" and r["value"]:
                    return float(r["value"])
            return 0.0
        import CoolProp.CoolProp as CP
        st = self._st()
        if T >= st.T_critical():
            return 0.0
        try:
            st.update(CP.QT_INPUTS, 0.0, T)
            return st.p()
        except ValueError:
            return 0.0

    def molar_mass(self) -> float:
        if self.incompressible:
            from ...data import cea_species
            sp = cea_species().get(self.fid)
            return float(sp["molar_mass"]) if sp else 0.03
        return self._st().molar_mass()


@lru_cache(maxsize=None)
def props(fid: str) -> Props:
    return Props(fid)


def liquid_state(fid: str, p: float, T: float, ox_frac: float) -> State:
    return State("fluid", fid, T, props(fid).h_pT(p, T), ox_frac)


def density(s: State, p: float) -> float:
    if s.kind == "gas":
        return max(p, 1e3) / (s.R * s.T)
    return props(s.fluid).rho_at(p, s.h, s.T)


def as_gas(s: State, p: float) -> tuple[float, float, float]:
    """``(cp, R, T)`` of a state seen as an ideal gas (for mixing a pure fluid into combustion gas)."""
    if s.kind == "gas":
        return s.cp, s.R, s.T
    pr = props(s.fluid)
    a = pr.at(p, s.h, s.T)
    return a["cp"], R_U / pr.molar_mass(), a["T"]


def mix(states: list[tuple[float, State]], p: float) -> State | None:
    """Adiabatic mixing of ``(mdot, state)`` streams at pressure ``p``.

    Streams of one pure fluid mix by enthalpy; anything else mixes as ideal gases (mass-weighted ``cp`` and ``R``,
    temperature from the ``cp T`` balance)."""
    states = [(m, s) for m, s in states if s is not None]
    if not states:
        return None
    w = [max(m, 0.0) for m, _ in states]
    tot = sum(w)
    if tot <= 1e-12:
        w = [1.0] * len(states)
        tot = float(len(states))
    oxf = sum(wi * s.ox_frac for wi, (_, s) in zip(w, states)) / tot
    if len(states) == 1:
        return states[0][1].with_(ox_frac=oxf)
    kinds = {(s.kind, s.fluid) for _, s in states}
    if len(kinds) == 1 and states[0][1].kind == "fluid":
        s0 = states[0][1]
        h = sum(wi * s.h for wi, (_, s) in zip(w, states)) / tot
        T = props(s0.fluid).T_ph(p, h, s0.T)
        return State("fluid", s0.fluid, T, h, oxf)
    gs = [as_gas(s, p) for _, s in states]
    cp = sum(wi * g[0] for wi, g in zip(w, gs)) / tot
    R = sum(wi * g[1] for wi, g in zip(w, gs)) / tot
    T = sum(wi * g[0] * g[2] for wi, g in zip(w, gs)) / (cp * tot)
    gamma = cp / (cp - R)
    return State("gas", None, T, cp * T, oxf, cp=cp, gamma=gamma, R=R)


def gas_state(T: float, cp: float, gamma: float, ox_frac: float) -> State:
    R = cp * (gamma - 1.0) / gamma
    return State("gas", None, T, cp * T, ox_frac, cp=cp, gamma=gamma, R=R)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))
