r"""A motor as a stack of grains, each with its own section, propellant and end-face inhibition (JSON-friendly).

Stored as the text of a JSON list, one object per grain from the head end to the aft end::

    [{"L": 0.20, "burn_head": true, "burn_aft": true,
      "prop": {"a_mm": 6.0, "n": 0.40, "cstar": 1450.0},
      "section": {"R_out": 60, "shapes": [{"kind": "circle", "r": 14}]}},
     {"L": 0.20}]

``L`` is in m.  A grain without ``section`` uses the motor's section; a grain without ``prop`` uses the motor's
propellant, and a partial ``prop`` overrides only the fields it names (same names as the propellant parameters:
``rho_p``, ``a_mm``, ``n``, ``sigma_p``, ``T_i``, ``cstar``, ``Tc``, ``gamma``, ``gamma_noz``, ``alpha``, ``beta``,
and ``law``: ``single`` for a and n, or a library pressure-band law such as ``KNSB``).  End faces are
inhibited unless ``burn_head`` / ``burn_aft`` is true.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

PROP_KEYS = ("rho_p", "a_mm", "n", "sigma_p", "T_i", "cstar", "Tc", "gamma", "alpha", "beta", "law", "gamma_noz")
# hybrid fuel grains: density, regression law and its constants (a in mm/s with G in kg/m²/s), and the chemistry:
# "pair" = another CEA pair, or "chem" = {"of": [...], "cstar": [...], "Tc": [...], "gamma": [...]}
FUEL_KEYS = ("rho_f", "law", "a", "n", "B", "pair", "chem")
_TEXT_KEYS = ("law", "pair", "chem")


@dataclass(frozen=True)
class GrainDef:
    L: float
    section: dict | None = None
    prop: tuple[tuple[str, object], ...] = ()
    burn_head: bool = False
    burn_aft: bool = False

    def to_dict(self) -> dict:
        d: dict = {"L": self.L}
        if self.section is not None:
            d["section"] = self.section
        if self.prop:
            d["prop"] = {k: _thaw(v) for k, v in self.prop}
        if self.burn_head:
            d["burn_head"] = True
        if self.burn_aft:
            d["burn_aft"] = True
        return d


def parse_stack(text: str, keys: tuple[str, ...] = PROP_KEYS) -> list[GrainDef]:
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
        bad = [x for x in prop if x not in keys]
        if bad:
            raise ValueError(f"grain {k}: unknown {'fuel' if keys is FUEL_KEYS else 'propellant'} field(s) {bad}; "
                             f"use {list(keys)}")
        sec = g.get("section")
        if sec is not None and not isinstance(sec, dict):
            raise ValueError(f"grain {k}: 'section' must be an object")
        try:
            vals = tuple(sorted((a, _freeze(b) if a in _TEXT_KEYS else float(b)) for a, b in prop.items()))
        except (TypeError, ValueError) as e:
            raise ValueError(f"grain {k}: a propellant value is not a number ({e})") from e
        out.append(GrainDef(L, sec, vals, bool(g.get("burn_head")), bool(g.get("burn_aft"))))
    return out


def _freeze(v):
    """Hashable form of a text or table value (dicts and lists become tuples)."""
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    if isinstance(v, list):
        return tuple(_freeze(x) for x in v)
    return v


def _thaw(v):
    if isinstance(v, tuple) and v and all(isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) for x in v):
        return {k: _thaw(x) for k, x in v}
    if isinstance(v, tuple):
        return [_thaw(x) for x in v]
    return v


def dump_stack(defs: list[GrainDef]) -> str:
    return json.dumps([d.to_dict() for d in defs], sort_keys=True) if defs else ""
