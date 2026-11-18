r"""Plain-font rendering of short math labels.

Names of buttons, list and menu items and combo entries (``N_2O Blowdown``, ``Field: M (-)``, ``Cutaway 45 deg``) are
exempt from the "all math in Computer Modern" rule: they use the same font as every other piece of UI text but keep true
super- and subscripts.  :func:`plain_runs` turns the TeX of one ``$...$`` segment into ``(text, level)`` runs
(level 0 = text, -1 = subscript, +1 = superscript); anything that needs real typesetting (fractions, accents, roots)
raises :class:`NeedsMathtext` and the caller draws that segment with mathtext instead.
"""
from __future__ import annotations

import re

_SYMBOLS = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "varepsilon": "ε", "epsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ",
    "phi": "φ", "psi": "ψ", "omega": "ω", "Delta": "Δ", "Pi": "Π", "Sigma": "Σ", "Omega": "Ω", "Gamma": "Γ",
    "Phi": "Φ", "times": "×", "cdot": "·", "pm": "±", "approx": "≈", "geq": "≥", "leq": "≤", "rightarrow": "→",
    "infty": "∞", "circ": "°", "%": "%",
}
_SPACES = {",": " ", " ": " ", ";": " ", ":": " ", "!": ""}
_FONT_COMMANDS = {"mathrm", "text", "mathit", "mathbf"}
_COMMAND = re.compile(r"\\([A-Za-z]+|.)")


class NeedsMathtext(Exception):
    """The segment uses typesetting a plain run cannot express."""


def _group(src: str, j: int) -> tuple[str, int]:
    """Argument starting at ``src[j]`` (a ``{...}`` group, a command or one character) and the index after it."""
    if j >= len(src):
        raise NeedsMathtext
    if src[j] == "{":
        depth = 0
        for k in range(j, len(src)):
            depth += (src[k] == "{") - (src[k] == "}")
            if depth == 0:
                return src[j + 1:k], k + 1
        raise NeedsMathtext
    m = _COMMAND.match(src, j) if src[j] == "\\" else None
    if m:
        return m.group(0), m.end()
    return src[j], j + 1


def _scan(src: str, level: int) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    i = 0
    while i < len(src):
        ch = src[i]
        if ch in "_^":
            body, i = _group(src, i + 1)
            out.extend(_scan(body, -1 if ch == "_" else 1))
        elif ch == "{":
            body, i = _group(src, i)
            out.extend(_scan(body, level))
        elif ch == "\\":
            m = _COMMAND.match(src, i)
            if not m:
                raise NeedsMathtext
            name, i = m.group(1), m.end()
            if name in _SPACES:
                out.append((_SPACES[name], level))
            elif name in _SYMBOLS:
                out.append((_SYMBOLS[name], 0 if name == "circ" else level))      # the degree sign is already raised
            elif name in _FONT_COMMANDS:
                body, i = _group(src, i)
                out.extend(_scan(body, level))
            else:
                raise NeedsMathtext
        elif ch in "}$":
            raise NeedsMathtext
        else:
            out.append(("−" if ch == "-" and level else ch, level))
            i += 1
    return out


def plain_runs(tex: str) -> list[tuple[str, int]]:
    r"""Runs of one ``$...$`` segment (without the dollar signs), neighbours of equal level joined."""
    merged: list[tuple[str, int]] = []
    for text, level in _scan(tex, 0):
        if merged and merged[-1][1] == level:
            merged[-1] = (merged[-1][0] + text, level)
        else:
            merged.append((text, level))
    return merged
