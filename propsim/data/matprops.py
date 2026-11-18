"""Evaluate the material property fits of ``materials/property_fits.csv`` (NIST cryogenic forms).

Forms, with ``c = (a..i)`` and T in K:

- ``log10_poly``: :math:`y = 10^{\\sum_k c_k (\\log_{10} T)^k}`
- ``poly_floor``: :math:`y = \\sum_k c_k T^k` for :math:`T > T_{low}`, else :math:`y = y_{low}`
- ``cu_rational`` (OFHC copper conductivity):
  :math:`\\log_{10} y = (a + c T^{1/2} + e T + g T^{3/2} + i T^2) / (1 + b T^{1/2} + d T + f T^{3/2} + h T^2)`

The printed value is multiplied by ``scale`` to give the SI unit of the row (GPa to Pa, 10^-5 strain,
10^-6 /K).
``thermal_strain`` is (L - L293) / L293.

    >>> from propsim.data.matprops import material_property
    >>> material_property("SS304", "k", 77.0)
"""

from __future__ import annotations

import math

from . import material_fits

LETTERS = "abcdefghi"


def _eval(r: dict[str, str], T: float) -> float:
    c = [float(r[k]) if r[k] != "" else 0.0 for k in LETTERS]
    if r["form"] == "log10_poly":
        x = math.log10(T)
        y = 10.0 ** sum(ck * x**k for k, ck in enumerate(c))
    elif r["form"] == "poly_floor":
        if r["T_low"] != "" and T < float(r["T_low"]):
            y = float(r["y_low"])
        else:
            y = sum(ck * T**k for k, ck in enumerate(c))
    elif r["form"] == "cu_rational":
        a, b, cc, d, e, f, g, h, i = c
        s = math.sqrt(T)
        y = 10.0 ** (
            (a + cc * s + e * T + g * T * s + i * T * T) / (1 + b * s + d * T + f * T * s + h * T * T)
        )
    else:
        raise ValueError(f"unknown form {r['form']!r}")
    return y * float(r["scale"])


def material_property(material_id: str, prop: str, T: float, segment: str | None = None) -> float:
    """SI value of ``prop`` (k, cp, E, thermal_strain, alpha) at ``T`` (K) from the library fits.

    ``segment`` selects a fit family (e.g. ``"RRR100"`` for copper conductivity); fits split by temperature
    range (``low``/``high``) are chosen by ``T``. Raises ``KeyError`` when no fit covers the request and
    ``ValueError`` outside the fit's equation range.
    """
    rows = [r for r in material_fits() if r["material_id"] == material_id and r["property"] == prop]
    if segment is not None:
        rows = [r for r in rows if r["segment"] == segment]
    if not rows:
        raise KeyError(f"{material_id}.{prop} has no fit (see gaps.md)")
    if len({r["segment"] for r in rows} - {"low", "high"}) > 1:
        raise KeyError(f"{material_id}.{prop}: choose segment from {sorted(r['segment'] for r in rows)}")
    for r in rows:
        if float(r["T_min"]) <= T <= float(r["T_max"]):
            return _eval(r, T)
    lo = min(float(r["T_min"]) for r in rows)
    hi = max(float(r["T_max"]) for r in rows)
    raise ValueError(f"{material_id}.{prop}: T = {T} K outside the fit range {lo} to {hi} K")
