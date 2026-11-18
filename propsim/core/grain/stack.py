r"""A motor as a stack of grains, each with its own section, propellant and end-face inhibition (JSON-friendly).

Stored as the text of a JSON list, one object per grain from the head end to the aft end::

    [{"L": 0.20, "burn_head": true, "burn_aft": true,
      "prop": {"a_mm": 6.0, "n": 0.40, "cstar": 1450.0},
      "section": {"R_out": 60, "shapes": [{"kind": "circle", "r": 14}]}},
     {"L": 0.20}]

``L`` is in m.  A grain without ``section`` uses the motor's section; a grain without ``prop`` uses the motor's
propellant, and a partial ``prop`` overrides only the fields it names (same names as the propellant parameters:
``rho_p``, ``a_mm``, ``n``, ``sigma_p``, ``T_i``, ``cstar``, ``Tc``, ``gamma``, ``alpha``, ``beta``).  End faces are
inhibited unless ``burn_head`` / ``burn_aft`` is true.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

PROP_KEYS = ("rho_p", "a_mm", "n", "sigma_p", "T_i", "cstar", "Tc", "gamma", "alpha", "beta")


@dataclass(frozen=True)
class GrainDef:
    L: float
    section: dict | None = None
    prop: tuple[tuple[str, float], ...] = ()
    burn_head: bool = False
    burn_aft: bool = False

    def to_dict(self) -> dict:
        d: dict = {"L": self.L}
        if self.section is not None:
            d["section"] = self.section
        if self.prop:
            d["prop"] = dict(self.prop)
        if self.burn_head:
            d["burn_head"] = True
        if self.burn_aft:
            d["burn_aft"] = True
        return d


def parse_stack(text: str) -> list[GrainDef]:
    """Parse and validate a stack; an empty string means 'no stack' (one grain from the motor's own parameters)."""
    if not text or not text.strip():
        return []
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e
    if not isinstance(raw, list) or not raw:
        raise ValueError("a grain stack is a non-empty JSON list")
    out = []
    for k, g in enumerate(raw, start=1):
        if not isinstance(g, dict) or "L" not in g:
            raise ValueError(f"grain {k}: needs a length 'L' (m)")
        L = float(g["L"])
        if not 0.005 <= L <= 5.0:
            raise ValueError(f"grain {k}: L must be between 0.005 and 5 (m)")
        prop = g.get("prop") or {}
        bad = [x for x in prop if x not in PROP_KEYS]
        if bad:
            raise ValueError(f"grain {k}: unknown propellant field(s) {bad}; use {list(PROP_KEYS)}")
        sec = g.get("section")
        if sec is not None and not isinstance(sec, dict):
            raise ValueError(f"grain {k}: 'section' must be an object")
        out.append(GrainDef(L, sec, tuple(sorted((a, float(b)) for a, b in prop.items())), bool(g.get("burn_head")),
                            bool(g.get("burn_aft"))))
    return out


def dump_stack(defs: list[GrainDef]) -> str:
    return json.dumps([d.to_dict() for d in defs], sort_keys=True) if defs else ""
