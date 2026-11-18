r"""Short names (buttons, list and combo items) keep the UI font and show true sub- and superscripts."""
from __future__ import annotations

import pytest

from propsim.ui.latex import latex_pixmap
from propsim.ui.plaintext import NeedsMathtext, plain_runs


def test_chemical_subscripts():
    assert plain_runs(r"\mathrm{N_{2}O_{4}}") == [("N", 0), ("2", -1), ("O", 0), ("4", -1)]


def test_degree_sign_is_not_raised():
    assert plain_runs(r"45^{\circ}") == [("45°", 0)]


def test_exponent_uses_minus_sign_and_superscript_level():
    assert plain_runs(r"\mathrm{kg\,m^{-3}}") == [("kg\u2009m", 0), ("\u22123", 1)]


def test_greek_and_single_character_scripts():
    assert plain_runs(r"\rho_c") == [("ρ", 0), ("c", -1)]


@pytest.mark.parametrize("tex", [r"\frac{A_e}{A_t}", r"\dot{m}", r"\sqrt{x}"])
def test_real_typesetting_falls_back_to_mathtext(tex):
    with pytest.raises(NeedsMathtext):
        plain_runs(tex)


def test_plain_pixmap_differs_from_mathtext_and_handles_fallback(qapp):
    a = latex_pixmap(r"N₂O Blowdown", "#ffffff", 9.5, 2.0, plain=True)
    b = latex_pixmap(r"N₂O Blowdown", "#ffffff", 9.5, 2.0)
    assert not a.isNull() and not b.isNull()
    assert a.toImage() != b.toImage()
    c = latex_pixmap(r"Ratio $\frac{A_e}{A_t}$ (-)", "#ffffff", 9.5, 2.0, plain=True)      # mathtext fallback segment
    assert not c.isNull()
