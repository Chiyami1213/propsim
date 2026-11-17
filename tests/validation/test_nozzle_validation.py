"""Nozzle validation: ideal C_F case from MATHKIT, Prandtl-Meyer, divergence loss, contour geometry."""
from __future__ import annotations

import numpy as np
import pytest

from propsim.core.nozzle import (CF, cf_curve, contour, divergence_factor, eps_from_pratio, exit_state,
                                 prandtl_meyer)

from .figs import new_fig, save


def test_ideal_cf_optimum_expansion():
    """gamma = 1.2, Pc/Pe = 34 at optimum expansion: eps = 5.305, Me = 2.828, C_F = 1.4977 (MathKit)."""
    g, pe_pc = 1.2, 1.0 / 34.0
    eps = float(eps_from_pratio(pe_pc, g)[0])
    Me, pe = exit_state(eps, g)
    cf = float(CF(g, pe_pc, pe_pc, eps))
    assert eps == pytest.approx(5.305, abs=5e-4)
    assert Me == pytest.approx(2.828, abs=5e-4)
    assert cf == pytest.approx(1.4977, abs=5e-5)
    eps_g, cf_a = cf_curve(g, pe_pc)
    fig, (ax,) = new_fig()
    ax.plot(eps_g, cf_a, color="#4ea1ff", label=r"$C_F$ at $P_a=P_c/34$")
    ax.plot([eps], [cf], "o", color="#ff5a3c", label=rf"MathKit: $\varepsilon$={eps:.3f}, $C_F$={cf:.4f}")
    ax.set_xlim(1, 20)
    ax.set_xlabel(r"$\varepsilon=\frac{A_e}{A_t}$ (–)")
    ax.set_ylabel(r"$C_F$ (–)")
    ax.legend()
    save(fig, "nozzle_ideal_cf")
    # the curve peaks at the optimum expansion
    assert eps_g[np.argmax(cf_a)] == pytest.approx(eps, rel=0.03)


def test_prandtl_meyer_reference_values():
    """Anderson Table A.5: nu(M=2, gamma=1.4) = 26.380 deg; limit nu(inf) = 90(sqrt(6)-1) = 130.454 deg."""
    assert np.degrees(prandtl_meyer(2.0, 1.4)) == pytest.approx(26.380, abs=1e-3)
    assert np.degrees(prandtl_meyer(1e8, 1.4)) == pytest.approx(130.454, abs=1e-3)
    assert float(prandtl_meyer(1.0, 1.4)) == 0.0


def test_divergence_factor():
    """Sutton Eq. 3-34: 15 deg cone gives lambda = 0.983."""
    assert float(divergence_factor(np.radians(15.0))) == pytest.approx(0.98296, abs=1e-5)
    assert float(divergence_factor(0.0)) == 1.0


@pytest.mark.parametrize("eps", [3.0, 6.0, 20.0, 60.0])
@pytest.mark.parametrize("kind", ["bell", "cone"])
def test_contour_geometry(eps, kind):
    Rt = 0.016
    x, r = contour(Rt, eps, kind=kind)
    i_t = np.argmin(np.abs(x))
    assert r.min() == pytest.approx(Rt, rel=2e-3)                 # throat radius
    assert abs(x[np.argmin(r)]) < 0.02 * Rt * 10                  # throat at x ~ 0
    assert r[-1] == pytest.approx(np.sqrt(eps) * Rt, rel=1e-9)    # exit radius from the area ratio
    assert np.all(np.diff(x) > 0)                                 # single-valued in x
    assert np.all(np.diff(r[i_t:]) >= -1e-12)                     # divergent section never converges
    # fixed point count so 3D meshes can be updated in place
    assert len(x) == len(contour(Rt, 2.0, kind=kind)[0])
    if kind == "cone":      # straight cone at the requested half angle
        s = np.diff(r[-20:]) / np.diff(x[-20:])
        assert np.degrees(np.arctan(s)) == pytest.approx(15.0, abs=1e-6)
