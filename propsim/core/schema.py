"""Parameter schema: one declaration drives sliders, tooltips, validation and docs.

Values in a Params model are in the *display units* given by ``units`` (e.g. mm, MPa); the model's
``to_si()`` converts at the boundary so all physics runs in SI.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


def P(default: float, *, symbol: str, desc: str, units: str, dims: str, lo: float, hi: float, step: float,
      group: str = "") -> Any:
    r"""Declare a parameter. ``symbol`` is LaTeX without ``$``; ``dims`` e.g. ``"M T⁻¹"``."""
    return Field(default, ge=lo, le=hi, description=desc,
                 json_schema_extra={"symbol": symbol, "units": units, "dims": dims, "lo": lo, "hi": hi,
                                    "step": step, "group": group})


def C(default: str, *, choices: list[str], symbol: str, desc: str, labels: list[str] | None = None,
      group: str = "") -> Any:
    r"""Declare a discrete choice (combo box). ``choices`` are the stored values, ``labels`` the shown text."""
    return Field(default, description=desc,
                 json_schema_extra={"kind": "choice", "symbol": symbol, "units": "", "dims": "1", "group": group,
                                    "choices": list(choices), "labels": list(labels or choices)})


def H(default: Any = "") -> Any:
    """Hidden field: stored with the parameters (and sent to the Run worker) but never shown as a slider or listed in
    :meth:`ParamModel.fields_meta` (e.g. thrust profiles imported from other runs)."""
    return Field(default, json_schema_extra={"kind": "hidden"})


def T(default: str = "", *, label: str, desc: str, editor: str, group: str = "") -> Any:
    """Text value edited in a dialog (e.g. a custom grain drawn as a list of shapes): shown as a button plus a one-line
    summary and stored as text, so it is saved with the case and sent to the Run worker like any other parameter.
    ``editor`` names the dialog in :mod:`propsim.ui.grain_editor`."""
    return Field(default, description=desc,
                 json_schema_extra={"kind": "text", "symbol": rf"\mathrm{{{label.replace(' ', chr(92) + ' ')}}}",
                                    "units": "", "dims": "1", "group": group, "editor": editor})


class ParamModel(BaseModel):
    """Base class for parameter schemas."""

    model_config = {"validate_assignment": True}

    @model_validator(mode="after")
    def _check_choices(self):
        for name, f in type(self).model_fields.items():
            ch = (f.json_schema_extra or {}).get("choices")
            if ch is not None and getattr(self, name) not in ch:
                raise ValueError(f"{name}={getattr(self, name)!r} not in {ch}")
        return self

    @classmethod
    def fields_meta(cls) -> list[dict]:
        out = []
        for name, f in cls.model_fields.items():
            extra = dict(f.json_schema_extra or {})
            if extra.get("kind") == "hidden":
                continue
            extra.setdefault("kind", "slider")
            extra.update(name=name, default=f.default, desc=f.description or "")
            out.append(extra)
        return out

    def label(self, name: str) -> str:
        """LaTeX label ``$symbol$ (units)`` for a field."""
        m = next(m for m in self.fields_meta() if m["name"] == name)
        return rf"${m['symbol']}$ ({m['units']})" if m["units"] else rf"${m['symbol']}$ (–)"
