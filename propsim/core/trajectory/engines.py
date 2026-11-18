"""Stage engine presets from the component library (``propsim/data/library/engines``): thrust, specific impulse and
nozzle exit area of reference engines.

Only what the library holds is used.  Where it lists a value with an *unstated* ambient condition (RL10 and J-2), the
engine's normal operating environment is assumed and the preset says so: vacuum for those upper-stage engines.  The exit
area is taken from the library's exit diameter, or derived from its sea-level and vacuum thrust
(:math:`A_e=(F_{vac}-F_{sl})/p_{sl}`), and is 0 (back-pressure loss ignored) when the library has neither.
"""
from __future__ import annotations

import math

from .. import constants
from ... import data

P_SL = constants.P_SEA_LEVEL

# engine_id -> (label, ambient condition assumed for values the library lists as unstated, or None)
PRESETS: dict[str, tuple[str, str | None]] = {
    "RS25": ("RS-25 (sea-level and vacuum data)", None),
    "J2": ("J-2 (vacuum assumed)", "vac"),
    "RL10A_4_2": ("RL10A-4-2 (vacuum assumed)", "vac"),
    "RL10B_2": ("RL10B-2 (vacuum assumed)", "vac"),
    "RL10C_1": ("RL10C-1 (vacuum assumed)", "vac"),
    "RL10C_3": ("RL10C-3 (vacuum assumed)", "vac"),
}
CUSTOM = "custom"


def _value(eid: str, sub: str, prop: str, cond: str | None):
    try:
        return data.engine_value(eid, sub, prop, cond)
    except KeyError:
        return None


def preset(engine_id: str) -> dict:
    """``dict(F_vac (N), isp (s), A_e (m²), note (str))`` of a library engine; raises ``KeyError`` for an unknown id."""
    label, assumed = PRESETS[engine_id]
    F_vac = _value(engine_id, "engine", "thrust", "vac")
    isp = _value(engine_id, "engine", "Isp", "vac")
    notes = []
    if F_vac is None:
        F_vac = _value(engine_id, "engine", "thrust", None)
        notes.append("thrust ambient unstated in the library, treated as vacuum")
    if isp is None:
        isp = _value(engine_id, "engine", "Isp", None)
        notes.append("Isp ambient unstated in the library, treated as vacuum")
    F_sl = _value(engine_id, "engine", "thrust", "sl")
    d_e = _value(engine_id, "thrust_chamber", "exit_diameter", None)
    if F_sl is not None and F_vac is not None:
        A_e, how = max(F_vac - F_sl, 0.0) / P_SL, "derived from the sea-level and vacuum thrust"
    elif d_e is not None:
        A_e, how = math.pi * d_e**2 / 4.0, "from the library exit diameter"
    else:
        A_e, how = 0.0, "unknown in the library: back-pressure loss ignored"
    notes.append(f"exit area {how}")
    return dict(F_vac=float(F_vac), isp=float(isp), A_e=float(A_e), label=label, note="; ".join(notes))


def choices() -> tuple[list[str], list[str]]:
    return [CUSTOM, *PRESETS], ["Custom (use the values below)", *[v[0] for v in PRESETS.values()]]
