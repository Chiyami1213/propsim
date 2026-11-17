"""Parameter schema: one declaration drives sliders, tooltips, validation and docs.

Values in a Params model are in the *display units* given by ``units`` (e.g. mm, MPa); the model's
``to_si()`` converts at the boundary so all physics runs in SI.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


def P(default: float, *, symbol: str, desc: str, units: str, dims: str, lo: float, hi: float, step: float,
      group: str = "") -> Any:
    r"""Declare a parameter. ``symbol`` is LaTeX without ``$``; ``dims`` e.g. ``"M T⁻¹"``."""
    return Field(default, ge=lo, le=hi, description=desc,
                 json_schema_extra={"symbol": symbol, "units": units, "dims": dims, "lo": lo, "hi": hi,
                                    "step": step, "group": group})


class ParamModel(BaseModel):
    """Base class for parameter schemas."""

    model_config = {"validate_assignment": True}

    @classmethod
    def fields_meta(cls) -> list[dict]:
        out = []
        for name, f in cls.model_fields.items():
            extra = dict(f.json_schema_extra or {})
            extra.update(name=name, default=f.default, desc=f.description or "")
            out.append(extra)
        return out

    def label(self, name: str) -> str:
        """LaTeX label ``$symbol$ (units)`` for a field."""
        m = next(m for m in self.fields_meta() if m["name"] == name)
        return rf"${m['symbol']}$ ({m['units']})" if m["units"] else rf"${m['symbol']}$ (–)"
