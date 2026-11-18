"""Closure checks of the reference engines (brief rule 5): published numbers must agree with each other.

Each check compares two ways of getting one quantity from published values of ``engines/engine_params.csv``:

- ``cstar``: mdot c* = P_c A_t (tolerance 5 %)
- ``thrust_isp``: F = mdot I_sp g0, thrust chamber and engine (3 %)
- ``thrust_cf``: F = C_F P_c A_t (3 %)
- ``mixture_ratio``: mdot_ox / mdot_fuel = MR (1 %)
- ``pump_power``: P = mdot g0 H / eta (3 %)
- ``shaft_torque``: P = T omega (2 %)
- ``power_balance``: turbine power = pump power + auxiliary drive (1 %)
- ``turbine_power``: Huzel & Huang eq. (6-19) with Table 6-4 gas properties (3 %)
- ``cycle_efficiency``: engine I_sp from eq. (6-22) (1 %)

Used by ``tools/check_library.py`` (writes ``checks.md``) and the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import cycle_table, engine_params, engines

G0 = 9.80665


@dataclass(frozen=True)
class Closure:
    engine_id: str
    check: str
    detail: str
    lhs: float
    rhs: float
    tol: float  # relative

    @property
    def diff(self) -> float:
        return (self.lhs - self.rhs) / self.rhs

    @property
    def ok(self) -> bool:
        return abs(self.diff) <= self.tol


def _values(engine_id: str) -> dict[tuple[str, str], float]:
    return {
        (r["subsystem"], r["property"]): float(r["value"])
        for r in engine_params(engine_id, preferred_only=True)
    }


def _turbine_gas(pair: str, T: float) -> dict[str, float] | None:
    for r in cycle_table("turbine_drive_gas"):
        if r["propellants"] == pair and abs(float(r["T_inlet"]) - T) < 0.5:
            return {k: float(r[k]) for k in ("cp", "gamma", "R", "MR")}
    return None


def engine_closures(engine_id: str) -> list[Closure]:
    v = _values(engine_id)
    out: list[Closure] = []

    def has(*keys: tuple[str, str]) -> bool:
        return all(k in v for k in keys)

    tc = "thrust_chamber"
    if has((tc, "mdot_ox"), (tc, "mdot_fuel")):
        md = v[tc, "mdot_ox"] + v[tc, "mdot_fuel"]
        if has((tc, "cstar"), (tc, "pc_ns"), (tc, "At")):
            out.append(
                Closure(
                    engine_id,
                    "cstar",
                    "thrust chamber mdot c* vs P_c A_t (N)",
                    md * v[tc, "cstar"],
                    v[tc, "pc_ns"] * v[tc, "At"],
                    0.05,
                )
            )
        if has((tc, "thrust"), (tc, "Isp")):
            out.append(
                Closure(
                    engine_id,
                    "thrust_isp",
                    "thrust chamber mdot Isp g0 vs F (N)",
                    md * v[tc, "Isp"] * G0,
                    v[tc, "thrust"],
                    0.03,
                )
            )
        if has(
            (tc, "MR"),
        ):
            out.append(
                Closure(
                    engine_id,
                    "mixture_ratio",
                    "thrust chamber mdot_ox/mdot_fuel vs MR",
                    v[tc, "mdot_ox"] / v[tc, "mdot_fuel"],
                    v[tc, "MR"],
                    0.01,
                )
            )
    if has((tc, "thrust"), (tc, "CF"), (tc, "pc_ns"), (tc, "At")):
        out.append(
            Closure(
                engine_id,
                "thrust_cf",
                "C_F P_c A_t vs thrust chamber F (N)",
                v[tc, "CF"] * v[tc, "pc_ns"] * v[tc, "At"],
                v[tc, "thrust"],
                0.03,
            )
        )
    if has(("engine", "mdot_ox"), ("engine", "mdot_fuel"), ("engine", "thrust"), ("engine", "Isp")):
        md = v["engine", "mdot_ox"] + v["engine", "mdot_fuel"]
        out.append(
            Closure(
                engine_id,
                "thrust_isp",
                "engine mdot Isp g0 vs F (N)",
                md * v["engine", "Isp"] * G0,
                v["engine", "thrust"],
                0.03,
            )
        )
        out.append(
            Closure(
                engine_id,
                "mixture_ratio",
                "engine mdot_ox/mdot_fuel vs MR",
                v["engine", "mdot_ox"] / v["engine", "mdot_fuel"],
                v["engine", "MR"],
                0.01,
            )
        )

    pumps = [s for s in ("ox_pump", "fuel_pump") if (s, "power") in v]
    for s in pumps:
        if has((s, "mdot"), (s, "head"), (s, "efficiency")):
            out.append(
                Closure(
                    engine_id,
                    "pump_power",
                    f"{s} mdot g0 H / eta vs P (W)",
                    v[s, "mdot"] * G0 * v[s, "head"] / v[s, "efficiency"],
                    v[s, "power"],
                    0.03,
                )
            )
    for s in ("turbine", "ox_turbine", "fuel_turbine"):
        if has((s, "torque"), (s, "speed"), (s, "power")):
            out.append(
                Closure(
                    engine_id,
                    "shaft_torque",
                    f"{s} T omega vs P (W)",
                    v[s, "torque"] * v[s, "speed"],
                    v[s, "power"],
                    0.02,
                )
            )
    if ("turbine", "power") in v and len(pumps) == 2:
        need = v["ox_pump", "power"] + v["fuel_pump", "power"] + v.get(("shaft", "aux_power"), 0.0)
        out.append(
            Closure(
                engine_id,
                "power_balance",
                "turbine P vs pumps + auxiliary drive (W)",
                v["turbine", "power"],
                need,
                0.01,
            )
        )
    for t, p in (("ox_turbine", "ox_pump"), ("fuel_turbine", "fuel_pump")):
        if has((t, "power"), (p, "power")):
            out.append(
                Closure(engine_id, "power_balance", f"{t} P vs {p} P (W)", v[t, "power"], v[p, "power"], 0.02)
            )

    e = engines()[engine_id]
    if has(
        ("turbine", "T_inlet"),
        ("turbine", "p_inlet"),
        ("turbine", "p_exhaust"),
        ("turbine", "mdot"),
        ("turbine", "efficiency"),
        ("turbine", "power"),
    ):
        gas = _turbine_gas(f"{e['oxidizer']}/{e['fuel']}", v["turbine", "T_inlet"])
        if gas:
            g = gas["gamma"]
            dh = (
                gas["cp"]
                * v["turbine", "T_inlet"]
                * (1 - (v["turbine", "p_exhaust"] / v["turbine", "p_inlet"]) ** ((g - 1) / g))
            )
            out.append(
                Closure(
                    engine_id,
                    "turbine_power",
                    "eq. (6-19) eta mdot dH vs P (W)",
                    v["turbine", "efficiency"] * v["turbine", "mdot"] * dh,
                    v["turbine", "power"],
                    0.03,
                )
            )
    if has(
        (tc, "thrust"),
        (tc, "Isp"),
        ("turbine_exhaust", "thrust"),
        ("turbine", "mdot"),
        ("engine", "Isp"),
        (tc, "mdot_ox"),
        (tc, "mdot_fuel"),
    ):
        mtc = v[tc, "mdot_ox"] + v[tc, "mdot_fuel"]
        eta = (v[tc, "thrust"] + v["turbine_exhaust", "thrust"]) / (
            (mtc + v["turbine", "mdot"]) * v[tc, "Isp"] * G0
        )
        out.append(
            Closure(
                engine_id,
                "cycle_efficiency",
                "eq. (6-22) eta_sc Isp_tc vs engine Isp (s)",
                eta * v[tc, "Isp"],
                v["engine", "Isp"],
                0.01,
            )
        )
    return out


def all_closures() -> list[Closure]:
    return [c for eid in engines() for c in engine_closures(eid)]


def turbine_gas_consistency(tol: float = 0.03) -> list[tuple[dict[str, str], float]]:
    """Rows of Table 6-4 whose R differs from c_p (gamma - 1) / gamma by more than ``tol`` (relative)."""
    bad = []
    for r in cycle_table("turbine_drive_gas"):
        cp, g, R = float(r["cp"]), float(r["gamma"]), float(r["R"])
        d = R / (cp * (g - 1) / g) - 1
        if abs(d) > tol:
            bad.append((r, d))
    return bad


# ---- loss-coefficient cross-check against Rennels & Hudson, Pipe Flow (2012), Ch. 10 and 11 ----


def sudden_expansion_K(d_small: float, d_large: float) -> float:
    """Borda-Carnot K on the inlet (small pipe) velocity, K1 = (1 - (d1/d2)^2)^2 (Pipe Flow eq. 11.6)."""
    beta = d_small / d_large
    return (1.0 - beta**2) ** 2


def sharp_contraction_K(d_small: float, d_large: float) -> float:
    """Sharp-edged contraction K on the downstream (small pipe) velocity (Pipe Flow eq. 10.3, 10.4).

    K2 = 0.0696 (1 - beta^5) lambda^2 + (lambda - 1)^2,
    lambda = 1 + 0.622 (1 - 0.215 beta^2 - 0.785 beta^5).
    """
    beta = d_small / d_large
    lam = 1.0 + 0.622 * (1.0 - 0.215 * beta**2 - 0.785 * beta**5)
    return 0.0696 * (1.0 - beta**5) * lam**2 + (lam - 1.0) ** 2
