r"""Text standards for everything shown in the UI.

* :func:`to_math` - all math is LaTeX (mathtext): units in parentheses, dimensions, chemical formulas, unicode super- and
  subscripts, Greek letters, dotted symbols and loose ``K_n`` / ``L*`` tokens outside ``$...$`` are rewritten as math.
  Idempotent: text already inside ``$...$`` is left alone.
* :func:`apa` - APA title case for panel headers, menus, titles, tabs and section names.
* :func:`hover_desc` - hover helper descriptions are all lower case (outside math).
"""
from __future__ import annotations

import re

_SUP = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺ⁿˣ", "0123456789-+nx"))
_SUB = dict(zip("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
_GREEK = {"α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\varepsilon", "ζ": r"\zeta",
          "η": r"\eta", "θ": r"\theta", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu", "π": r"\pi",
          "ρ": r"\rho", "σ": r"\sigma", "τ": r"\tau", "φ": r"\phi", "ψ": r"\psi", "ω": r"\omega", "Δ": r"\Delta",
          "Π": r"\Pi", "Σ": r"\Sigma", "Ω": r"\Omega", "Γ": r"\Gamma", "Φ": r"\Phi"}
_SYMS = {"−": "-", "×": r"\times", "≈": r"\approx", "≥": r"\geq", "≤": r"\leq", "→": r"\rightarrow", "±": r"\pm", "∞": r"\infty",
         "√": r"\sqrt{}"}
_DOTTED = {"ṁ": r"\dot{m}", "ṙ": r"\dot{r}", "ṗ": r"\dot{p}"}
UNITS = {"s", "ms", "µs", "min", "h", "K", "°C", "m", "mm", "cm", "km", "µm", "Pa", "kPa", "MPa", "GPa", "bar", "psi",
         "psia", "N", "kN", "MN", "kg", "g", "mg", "t", "L", "mL", "Hz", "kHz", "rpm", "W", "kW", "MW", "GW", "J", "kJ",
         "MJ", "V", "kV", "A", "mA", "eV", "T", "deg", "rad", "ft", "lbf", "lbm", "%", "kN·s", "N·s"}

_MATH_SPLIT = re.compile(r"(\$[^$]*\$)")
_UNIT_PAREN = re.compile(r"\(([^()\s$]+)\)")
_TOKEN_SUB = re.compile(r"(?<![A-Za-z0-9_$\\])([A-Za-z])_(\{[^}]+\}|[A-Za-z0-9]+)(?![A-Za-z0-9_])")
_TOKEN_STAR = re.compile(r"(?<![A-Za-z0-9\\$])([A-Za-z])\*(?![A-Za-z0-9*])")
_DEG = re.compile(r"(\d+(?:\.\d+)?)°")
_SPECIAL = re.compile("[" + "".join(_SUP) + "".join(_SUB) + "·" + "]")


def _unit_tex(u: str) -> str:
    """``kg·m⁻³`` -> ``\\mathrm{kg\\,m^{-3}}``; chemical subscripts ``N₂O₄`` -> ``\\mathrm{N_{2}O_{4}}``."""
    out, i = [], 0
    while i < len(u):
        ch = u[i]
        if ch in _SUP:
            j = i
            while j < len(u) and u[j] in _SUP:
                j += 1
            out.append("^{" + "".join(_SUP[c] for c in u[i:j]) + "}")
            i = j
        elif ch in _SUB:
            j = i
            while j < len(u) and u[j] in _SUB:
                j += 1
            out.append("_{" + "".join(_SUB[c] for c in u[i:j]) + "}")
            i = j
        elif ch == "·":
            out.append(r"{\cdot}")                                 # units keep their centre dots (Rule 2)
            i += 1
        elif ch == "%":
            out.append(r"\%")
            i += 1
        elif ch == "µ":
            out.append(r"\mu ")
            i += 1
        elif ch in _GREEK:
            out.append(_GREEK[ch] + " ")
            i += 1
        elif ch == "°":
            out.append(r"^{\circ}")
            i += 1
        elif ch == " ":
            out.append(r"\ ")
            i += 1
        else:
            out.append(ch)
            i += 1
    return r"\mathrm{" + "".join(out) + "}"


def dims_tex(d: str) -> str:
    """Dimension string ``M L⁻¹ T⁻²`` -> ``$[\\mathrm{M\\,L^{-1}\\,T^{-2}}]$`` (``1`` for dimensionless)."""
    d = (d or "1").strip()
    if d in ("1", "–", "-"):
        return r"$[1]$"
    body = _unit_tex(d.replace(" ", "·"))
    return "$[" + body + "]$"


def _is_unit(s: str) -> bool:
    return s in UNITS or bool(_SPECIAL.search(s)) and not any(c.isspace() for c in s)


def _plain_to_math(seg: str) -> str:
    """Rewrite one non-math text segment."""
    # parenthesised units: (kg·m⁻³), (MPa), (%)
    seg = _UNIT_PAREN.sub(lambda m: "($" + _unit_tex(m.group(1)) + "$)" if _is_unit(m.group(1)) else m.group(0), seg)
    # scale factors written as "x 1e-3" -> \times 10^{-3}
    seg = re.sub(r"(?<![\w$])[x×]\s?1e([+-]?\d+)", lambda m: r"$\times 10^{" + str(int(m.group(1))) + "}$", seg)
    # remaining tokens with super/subscripts or middle dots (chemical formulas, inline units)
    def tok(m: re.Match) -> str:
        w = m.group(0)
        if "$" in w or not _SPECIAL.search(w):
            return w
        lead = re.match(r"^[^\w]*", w).group(0)
        trail = re.search(r"[^\w⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺ⁿˣ₀₁₂₃₄₅₆₇₈₉%]*$", w).group(0)
        core = w[len(lead):len(w) - len(trail)] if trail else w[len(lead):]
        return lead + "$" + _unit_tex(core) + "$" + trail if core else w
    seg = _outside(seg, lambda s: re.sub(r"[^\s()$]+", tok, s))
    seg = _outside(seg, lambda s: _DEG.sub(lambda m: "$" + m.group(1) + r"^{\circ}$", s))
    for table in (_DOTTED, _GREEK, _SYMS):
        for k, v in table.items():
            seg = _outside(seg, lambda s, k=k, v=v: s.replace(k, "$" + v + "$"))
    seg = _outside(seg, lambda s: _TOKEN_SUB.sub(lambda m: "$" + m.group(1) + "_{" + m.group(2).strip("{}") + "}$", s))
    seg = _outside(seg, lambda s: _TOKEN_STAR.sub(lambda m: "$" + m.group(1) + "^{*}$", s))
    return seg


def _outside(text: str, fn) -> str:
    """Apply ``fn`` only to the parts of ``text`` that are not already ``$...$`` math (each pass sees new math)."""
    parts = _MATH_SPLIT.split(text)
    return "".join(p if (p.startswith("$") and p.endswith("$") and len(p) >= 2) else fn(p) for p in parts)


def to_math(text: str) -> str:
    """All math as LaTeX: rewrite plain-text math outside ``$...$`` (idempotent)."""
    if not text:
        return text
    parts = _MATH_SPLIT.split(text)
    out = [p if (p.startswith("$") and p.endswith("$") and len(p) >= 2) else _plain_to_math(p) for p in parts]
    return re.sub(r"\$\s*\$", " ", "".join(out)) if "$$" in "".join(out) else "".join(out)


def has_plain_math(text: str) -> bool:
    """True when math-looking characters remain outside ``$...$`` (used by the standards tests)."""
    for p in _MATH_SPLIT.split(text or ""):
        if p.startswith("$") and p.endswith("$") and len(p) >= 2:
            continue
        q = re.sub(r"(^|\s)·(\s|$)", " ", p)                    # a lone middle dot is a typographic separator
        if _SPECIAL.search(q) or any(c in q for c in _GREEK) or any(c in q for c in _DOTTED) or _TOKEN_SUB.search(q):
            return True
    return False


# ------------------------------------------------------------------------------------------- capitalization
MINOR = {"a", "an", "the", "and", "as", "but", "for", "if", "nor", "or", "so", "yet", "at", "by", "in", "of", "off",
         "on", "per", "to", "up", "via", "vs", "vs.", "v."}


def _cap_word(w: str, force: bool) -> str:
    m = re.match(r"^([^A-Za-z0-9]*)(.*?)([^A-Za-z0-9.]*)$", w)
    lead, core, trail = m.group(1), m.group(2), m.group(3)
    if not core:
        return w
    if core in UNITS or _SPECIAL.search(core) or (lead.endswith("(") and trail.startswith(")") and len(core) <= 4):
        return w                                                   # units are never title-cased: (kg·s⁻¹), (MPa), (s)
    for sep in ("-", "/"):                                         # compounds: capitalize every part
        if sep in core:
            return lead + sep.join(_cap_word(x, True) for x in core.split(sep)) + trail
    if any(c.isdigit() for c in core) or any(c.isupper() for c in core[1:]) or "_" in core:
        return w                                                   # acronyms, units, 3D, CO₂-like tokens: keep
    if not force and core.lower() in MINOR:
        return lead + core.lower() + trail
    return lead + core[0].upper() + core[1:] + trail


def apa(text: str) -> str:
    """APA title case: capitalize major words and words of 4+ letters, lowercase short minor words except first/last
    and after a colon or dash; math segments and acronyms are untouched."""
    if not text:
        return text
    parts = _MATH_SPLIT.split(text)
    words: list[tuple[int, int, str]] = []                        # (part index, word index, word)
    tokens: list[list[str]] = []
    for pi, p in enumerate(parts):
        if p.startswith("$") and p.endswith("$") and len(p) >= 2:
            tokens.append([p])
            words.append((pi, 0, p))
            continue
        toks = re.split(r"(\s+)", p)
        tokens.append(toks)
        for ti, t in enumerate(toks):
            if t and not t.isspace():
                words.append((pi, ti, t))
    n = len(words)
    force_next = True
    for k, (pi, ti, w) in enumerate(words):
        if w.startswith("$"):
            force_next = False
            continue
        is_last = k == n - 1
        tokens[pi][ti] = _cap_word(w, force_next or is_last)
        force_next = w.endswith(":") or w in ("—", "–")             # APA: capitalize after a colon or dash
    return "".join("".join(t) for t in tokens)


def hover_desc(text: str) -> str:
    """Hover helper descriptions are all lower case (math segments untouched)."""
    parts = _MATH_SPLIT.split(text or "")
    return "".join(p if (p.startswith("$") and p.endswith("$") and len(p) >= 2) else p.lower() for p in parts)


def hover_message(symbol: str, desc: str, dims: str) -> str:
    """``$sym$ — description  [dimensions]`` with the description lower case and everything math in LaTeX."""
    return to_math(rf"${symbol}$ — {hover_desc(desc)}  ") + dims_tex(dims)
