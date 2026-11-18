"""Solid grain design and internal ballistics, part 2: new sections, BATES segments, pressure-band burn laws, nozzle
losses, throat erosion, and validation against published motors.

* Kappa-SB static test KSB-002 (Nakka): a 4-segment KN/sorbitol BATES motor with published geometry, grain mass,
  throat, expansion ratio and delivered results.  Inputs are the published geometry, Nakka's 1999 KNSB pressure-band
  burn law, the density implied by the published grain mass, and the delivered c* implied by the published Isp and
  average C_F (the only calibrated input).  The model then predicts the burn time, the thrust coefficient and the
  total impulse, which are compared with the test.
* Space Shuttle RSRM (AIAA 2000-3803): the published burn-rate law reproduces the published aft-segment rate drop
  (0.401 to 0.392 in/s for 50 psi less pressure).  A sanity check of the burn law only; the RSRM grain is not modelled.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from propsim import data as D
from propsim.core.ballistics import solid as S
from propsim.core.grain import section2d as S2
from propsim.sims.solid_burn import SolidBallistics, SolidBallisticsParams, law_bands, parse_bands

from .figs import new_fig, save

PSI = 6894.757293168
IN = 0.0254


# ------------------------------------------------------------------------------------------- sections
def test_new_sections_match_closed_forms():
    R = 60.0
    d = S2.section("dgrain", R, offset=20.0)                       # flat at x = 20: port is a circular segment
    seg = R**2 * np.arccos(20.0 / R) - 20.0 * np.sqrt(R**2 - 20.0**2)
    assert d.area[0] == pytest.approx(seg, rel=5e-3)
    assert d.perim[0] == pytest.approx(2 * np.sqrt(R**2 - 20.0**2), rel=0.01)       # grid-limited at the wall
    assert d.web == pytest.approx(R + 20.0, rel=0.02)
    m = S2.section("moon", R, R_p=15.0, offset=20.0)               # offset bore: same start, longer web
    assert m.area[0] == pytest.approx(np.pi * 15.0**2, rel=5e-3)
    assert m.web == pytest.approx(R + 20.0 - 15.0, rel=0.02)
    c = S2.section("cslot", R, w=8.0, depth=0.55)                  # slot open at the wall: two sides plus the floor
    assert c.perim[0] == pytest.approx(2 * 0.55 * R + 8.0, rel=0.03)
    x = S2.section("xcore", R, n=4, w=8.0, depth=0.55)              # cross of 4 arms: area = 2 bars - centre square
    assert x.area[0] == pytest.approx(2 * (2 * 0.55 * R) * 8.0 - 8.0**2, rel=0.02)


@pytest.mark.parametrize("kind", ["moon", "cslot", "dgrain", "xcore"])
def test_new_sections_conserve(kind):
    b = S2.section(kind, 60.0, offset=20.0)
    assert np.all(np.diff(b.area) >= 0) and b.area[-1] == pytest.approx(b.A_case, rel=1e-9)
    assert b.perim[0] > 0 and b.perim[-1] == 0.0


def test_bates_segments_match_the_closed_form():
    """N segments with both ends burning: A_b = N [2 pi (r+w)(L-2w) + 2 pi (R^2-(r+w)^2)] (Sutton Ch. 12)."""
    R, r0, L = 0.0279, 0.0095, 0.10
    b = S2.section("circle", R * 1e3, R_p=r0 * 1e3)
    tab = S.SectionTable(b.d * 1e-3, b.area * 1e-6, b.perim * 1e-3, b.A_case * 1e-6)
    w = np.linspace(0.0, 0.9 * (R - r0), 20)
    exact = 4 * (2 * np.pi * (r0 + w) * (L - 2 * w) + 2 * np.pi * (R**2 - (r0 + w) ** 2))
    assert S.segment_burn_area(tab, L, w, 4, 2) == pytest.approx(exact, rel=6e-3)
    assert S.segment_web(tab, L, 2) == pytest.approx(R - r0, rel=0.01)
    assert S.segment_web(tab, 0.02, 2) == pytest.approx(0.01)            # short segment: the ends meet first


def test_end_burner_is_neutral_and_burns_its_length():
    """Port-less grain burning on its aft face: constant K_n = A_case / A_t, burn time = L / r(p_eq)."""
    p = SolidBallisticsParams(kind="endburner", N=2, t_end=40.0, L_g=0.2, D_t=10.5, alpha=0.0)
    r = SolidBallistics().run(p, lambda f: None, threading.Event())
    prop = p.prop(0.0)
    Kn = np.pi * (p.R_out * 1e-3) ** 2 / (np.pi * (p.D_t * 1e-3) ** 2 / 4)
    pe = S.p_equilibrium(prop, Kn)
    assert r.meta["Kn_0"] == pytest.approx(Kn, rel=1e-3)
    assert r.meta["t_burn"] == pytest.approx(0.2 / float(prop.rate(pe)), rel=0.02)
    assert r.meta["Pc_avg_MPa"] == pytest.approx(pe / 1e6, rel=0.02)
    assert r.meta["m_prop"] - p.m_ig * 1e-3 == pytest.approx(r.meta["m_loaded"], rel=0.01)


# ------------------------------------------------------------------------------------------- burn law
def test_pressure_bands_are_the_published_table():
    bands = D.solid_burn_bands("KNSB")
    prop = S.SolidProp(bands=bands, sigma_p=0.0)
    # Nakka 1999 Table 5, metric half: r = a p^n in each band (mm/s, MPa)
    for p_MPa, a, n in ((0.5, 10.71, 0.625), (1.0, 8.763, -0.314), (2.0, 7.852, -0.013), (5.0, 3.907, 0.535),
                        (8.0, 9.653, 0.064)):
        assert prop.rate(p_MPa * 1e6) * 1e3 == pytest.approx(a * p_MPa**n, rel=1e-4)
    # the published bands nearly join: the rate jumps by less than 5 % at every band edge
    for lo, hi, *_ in bands[1:]:
        assert prop.rate(lo * 1.0001) / prop.rate(lo * 0.9999) == pytest.approx(1.0, abs=0.05)
    # equilibrium is self-consistent (mass balance) and sits in the right band
    pe = S.p_equilibrium(prop, 300.0)
    assert prop.rho_p * prop.cstar * 300.0 * prop.rate(pe) == pytest.approx(pe, rel=1e-9)


def test_band_law_equals_single_law_when_bands_agree():
    single = S.SolidProp(sigma_p=0.002, Ti=310.0)
    two = S.SolidProp(sigma_p=0.002, Ti=310.0, bands=S.bands_from_mm_MPa([(0.1, 3.0, 5.0, 0.35), (3.0, 30.0, 5.0, 0.35)]))
    for Kn in (80.0, 200.0, 500.0):
        assert S.p_equilibrium(two, Kn) == pytest.approx(S.p_equilibrium(single, Kn), rel=1e-12)


def test_band_input_validation():
    with pytest.raises(ValueError):
        parse_bands("[[0.1, 2.0, 5.0, 0.3], [2.5, 5.0, 5.0, 0.3]]")      # gap between bands
    with pytest.raises(ValueError):
        parse_bands("")
    assert law_bands("single") == ()
    assert len(law_bands("KNDX")) == 5


# ------------------------------------------------------------------------------------------- nozzle and throat
def _bates(prop, **kw):
    b = S2.section("circle", 60.0, R_p=20.0)
    tab = S.SectionTable(b.d * 1e-3, b.area * 1e-6, b.perim * 1e-3, b.A_case * 1e-6)
    base = {"At": np.pi / 4 * 0.03**2, "eps": 8.0, "N": 4, "m_ig": 0.0, "t_ig": 1e-3, "v_fs": 1e9, "p_ign": 0.0,
            "p0": 2e5, "R_case": 0.06}
    base.update(kw)
    return S.SolidMotor.from_grains([S.Grain(prop, tab, 0.15, True, True)] * 3, **base)


def test_nozzle_losses_scale_thrust_and_cstar_efficiency_raises_pressure():
    prop = S.SolidProp(alpha=0.0)
    ideal = S.integrate(_bates(prop), 6.0, 301)
    lossy = S.integrate(_bates(prop, alpha_div=15.0, eta_F=0.97), 6.0, 301)
    lam = 0.5 * (1 + np.cos(np.radians(15.0)))
    i = 150
    assert ideal["Pc"][i] == pytest.approx(lossy["Pc"][i], rel=1e-9)        # losses after the throat leave Pc alone
    assert lossy["CF"][i] < ideal["CF"][i] * lam * 0.97 * 1.001
    # a lower delivered c* passes more mass per unit pressure: the equilibrium is lower by eta^(1/(1-n))
    assert S.p_equilibrium(prop, 300.0, 0.9) / S.p_equilibrium(prop, 300.0) == pytest.approx(0.9 ** (1 / (1 - prop.n)))
    eta = S.integrate(_bates(prop, eta_c=0.9), 6.0, 301)
    j = 20                                                                    # early: both near the initial K_n
    assert eta["Pc"][j] / ideal["Pc"][j] == pytest.approx(0.9 ** (1 / (1 - prop.n)), rel=0.02)
    vac = S.integrate(_bates(prop, p_amb=0.0), 6.0, 301)
    assert np.all(vac["F"][50:250] > ideal["F"][50:250])                      # no back-pressure term in vacuum


def test_throat_erosion_grows_the_throat_and_lowers_pressure():
    prop = S.SolidProp(alpha=0.0)
    m = _bates(prop, edot_t=2e-4, p_ref_t=5e6, k_t=0.0)                       # 0.2 mm/s, pressure-independent
    ch = S.integrate(m, 6.0, 301)
    t = np.linspace(0, 6.0, 301)
    burning = ch["mdot_gen"] > 0
    t_b = t[burning][-1] - t[burning][0]
    assert ch["Dt"][-1] - ch["Dt"][0] == pytest.approx(0.2 * t_b, rel=0.03)
    ref = S.integrate(_bates(prop), 6.0, 301)
    assert ch["Pc"][250] < ref["Pc"][250] and ch["eps"][-1] < ch["eps"][0]


def test_eng_export_round_trip():
    t = np.linspace(0.0, 3.0, 301)
    F = np.where((t > 0.1) & (t < 2.1), 1000.0, 0.0)
    txt = S.eng_text(t, F, "Test motor", 54.0, 300.0, 1.2)
    lines = [ln for ln in txt.splitlines() if not ln.startswith(";")]
    head = lines[0].split()
    assert head[0] == "Test_motor" and head[1:4] == ["54", "300", "0"] and float(head[4]) == pytest.approx(1.2)
    pts = np.array([[float(x) for x in ln.split()] for ln in lines[1:]])
    assert pts[-1, 1] == 0.0 and np.all(np.diff(pts[:, 0]) > 0) and pts[0, 0] > 0
    assert np.trapezoid(pts[:, 1], pts[:, 0]) == pytest.approx(2000.0, rel=0.03)      # a step is the worst case
    assert S.impulse_class(2000.0).startswith("K") and S.impulse_class(1.0).startswith("A")


# ------------------------------------------------------------------------------------------- Kappa-SB (KSB-002)
def kappa_sb_motor(N: int = 8) -> tuple[S.SolidMotor, dict]:
    T = D.static_tests("KSB-002")
    v = lambda k: float(T[k]["value"])
    Ro, rc, L, n = v("grain_od") / 2, v("core_d") / 2, v("segment_length"), int(v("n_segments"))
    rho = v("grain_mass") / (n * np.pi * (Ro**2 - rc**2) * L)                 # as-cast density from the grain mass
    cstar_d = v("isp_delivered") * 9.80665 / v("cf_avg")                       # delivered c* = Isp g0 / C_F (test)
    b = S2.section("circle", Ro * 1e3, R_p=rc * 1e3)
    tab = S.SectionTable(b.d * 1e-3, b.area * 1e-6, b.perim * 1e-3, b.A_case * 1e-6)
    # KNSB thermochemistry is not published in the cited sources; Nakka gives KNSU (Isp within 1 % of KNSB)
    prop = S.SolidProp(rho_p=rho, cstar=cstar_d, gamma=D.solid_value("KNSU", "gamma"),
                       gamma_noz=D.solid_value("KNSU", "gamma_2ph"), Tc=D.solid_value("KNSU", "Tc"),
                       bands=D.solid_burn_bands("KNSB"), alpha=0.0, sigma_p=0.0)
    m = S.SolidMotor.from_grains([S.Grain(prop, tab, L, True, True)] * n, At=np.pi / 4 * v("throat_d_pre") ** 2,
                                 eps=v("expansion_ratio"), N=N, m_ig=0.005, t_ig=0.05, v_fs=1e3, R_case=Ro,
                                 V_extra=1e-4, alpha_div=np.degrees(v("divergence_half_angle")))
    return m, {k: v(k) for k in ("total_impulse", "isp_delivered", "cf_avg", "burn_time", "grain_mass")}


@pytest.fixture(scope="module")
def kappa():
    m, ref = kappa_sb_motor()
    t = np.linspace(0.0, 3.0, 301)
    ch = S.integrate(m, 3.0, 301)
    return m, ref, t, ch, S.summary(m, t, ch)


def test_kappa_sb_geometry(kappa):
    m, ref, _, _, s = kappa
    assert s["m_loaded"] == pytest.approx(ref["grain_mass"], rel=0.005)       # density reproduces the loaded mass
    assert 1550.0 < m.prop.rho_p < 1841.0                                     # below the ideal KNSB density
    assert s["Kn_0"] == pytest.approx(336.0, rel=0.01)                         # 4 [pi d L + 2 pi/4 (D^2 - d^2)] / A_t
    assert s["port_throat_0"] > 2.0 and not s["port_choked"]


def test_kappa_sb_burn_time(kappa):
    """'Forceful thrusting continued for about two seconds' (KSB-002): the KNSB band law gives 1.6-2.4 s."""
    s = kappa[4]
    assert s["t_burn"] == pytest.approx(kappa[1]["burn_time"], abs=0.4)
    assert 3.79 < s["Pc_avg_MPa"] < 7.03                                       # runs in Nakka's 4th KNSB band


def test_kappa_sb_thrust_coefficient_and_impulse(kappa):
    """Ideal two-phase C_F with the 12 deg divergence loss is within 6 % of the measured 1.53; the total impulse and
    Isp within 6 % (the residual is the nozzle loss PropSim does not model unless eta_F is set)."""
    m, ref, t, ch, s = kappa
    sel = ch["Pc"] > 0.1 * ch["Pc"].max()
    cf = np.trapezoid(ch["F"][sel] * 1e3, t[sel]) / np.trapezoid(ch["Pc"][sel] * 1e6 * m.At, t[sel])
    assert cf == pytest.approx(ref["cf_avg"], rel=0.06) and cf > ref["cf_avg"]
    assert s["I_t_Ns"] == pytest.approx(ref["total_impulse"], rel=0.06)
    assert s["Isp_avg"] == pytest.approx(ref["isp_delivered"], rel=0.06)
    assert s["impulse_class"].startswith("K")                                  # 1821 N s is a K motor
    fig, ax = new_fig(2)
    ax[0].plot(t, ch["Pc"], label="PropSim")
    ax[0].set(xlabel=r"$t$ (s)", ylabel=r"$P_c$ (MPa)", title="Kappa-SB (KSB-002 inputs)")
    ax[1].plot(t, ch["F"] * 1e3, label=f"PropSim: {s['I_t_Ns']:.0f} N·s, Isp {s['Isp_avg']:.0f} s")
    ax[1].axhline(ref["total_impulse"] / ref["burn_time"], ls="--", c="k",
                  label=f"test mean: {ref['total_impulse']:.0f} N·s over ~{ref['burn_time']:.0f} s")
    ax[1].set(xlabel=r"$t$ (s)", ylabel=r"$F$ (N)")
    ax[1].legend(fontsize=7)
    save(fig, "solid_kappa_sb")


# ------------------------------------------------------------------------------------------- RSRM burn law
def test_rsrm_burn_law_reproduces_published_aft_rate_drop():
    row = D.solid_burn_laws()["RSRM_PBAN"][0]
    prop = S.SolidProp(a=float(row["a"]), n=float(row["n"]), sigma_p=0.0)
    assert prop.rate(625 * PSI) / IN == pytest.approx(0.368, rel=1e-4)        # the reference point
    T = D.static_tests("RSRM")
    hi, lo = float(T["aft_rate_high"]["value"]), float(T["aft_rate_low"]["value"])
    p_aft = (hi / prop.a) ** (1 / prop.n)                                     # pressure where r = 0.401 in/s
    assert prop.rate(p_aft - float(T["aft_pressure_drop"]["value"])) == pytest.approx(lo, rel=0.002)
    assert 700 * PSI < p_aft < float(T["head_end_pressure"]["value"])         # aft pressure below the 914 psi head end
