"""Component library: file integrity, schema, published-value cross-checks and table spot values."""

from __future__ import annotations

import json

import numpy as np
import pytest

from propsim import data

jsonschema = pytest.importorskip("jsonschema")


def test_library_integrity():
    """Every row has provenance, an SI unit string and known ids (propsim.data.validate)."""
    assert data.validate() == []


def test_sourced_value_schema():
    schema = json.loads((data.LIBRARY / "schemas" / "sourced_value.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    ok = {
        "value": 90.19,
        "unit": "K",
        "quality": "derived",
        "source_id": "COOLPROP-8.0.0",
        "location": "EOS Schmidt-FPE-1985",
        "formula": "PropsSI('T','P',101325,'Q',0,'Oxygen')",
    }
    jsonschema.validate(ok, schema)
    with pytest.raises(jsonschema.ValidationError):  # a derived value must say how it was derived
        jsonschema.validate({k: v for k, v in ok.items() if k != "formula"}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**ok, "quality": "guess"}, schema)


def test_every_coolprop_fluid_has_a_table():
    for fid, f in data.fluids().items():
        if f["model"] == "coolprop":
            t = data.fluid_table(fid)
            assert len(t["T"]) > 500, fid
            assert np.all(t["rho"] > 0), fid


@pytest.mark.parametrize(
    "fid, T, p, rho_ref, tol",
    [
        # Huzel & Huang Table 1-5: 1.142 and 0.071 g/cm3 at the normal boiling point
        ("LOX", 90.0, 0.2e6, 1141.0, 0.01),
        ("LH2", 20.0, 0.2e6, 71.0, 0.02),
        ("H2O", 298.15, 0.1e6, 997.0, 0.002),
    ],
)
def test_table_density_spot_values(fid, T, p, rho_ref, tol):
    """Nearest grid point (exact T, p not required) against a published liquid density."""
    t = data.fluid_table(fid)
    sel = t["p"] == p
    i = np.argmin(np.abs(t["T"][sel] - T))
    assert t["rho"][sel][i] == pytest.approx(rho_ref, rel=tol + 0.002 * abs(t["T"][sel][i] - T))


def test_preferred_scalar_lookup():
    assert data.scalar("LOX", "T_nbp") == pytest.approx(90.19, abs=0.05)
    with pytest.raises(KeyError):
        data.scalar("IPA", "rho_liq")  # a known gap: no sourced density yet


def test_mixtures_reference_cea_species():
    sp = data.cea_species()
    for mid, m in data.cea_mixtures().items():
        assert all(c.split(":")[0] in sp for c in m["components"].split(";")), mid


def test_shipped_cea_tables_match_huzel_huang():
    """Shipped CEA tables against Huzel & Huang Tables 1-8 to 1-10 (P_c = 1000 psia, optimum expansion)."""
    from propsim.core.thermo.cea import Thermo

    pairs = {("LOX", "RP1"): "LOX/RP-1", ("LOX", "LH2"): "LOX/LH2", ("N2O4", "MMH"): "N2O4/MMH"}
    n = 0
    for r in data.cea_validation():
        key = (r["oxidizer"], r["fuel"])
        if key not in pairs:
            continue
        th = Thermo(pairs[key], source="tables")
        mr, pc, pe = float(r["OF"]), float(r["pc"]), float(r["pe"])
        assert float(th.chamber(mr, pc).cstar) == pytest.approx(float(r["cstar"]), rel=0.01), r["case_id"]
        isp = float(np.max(th.isp_sl(mr, pc, np.linspace(1.5, 100.0, 2000), pa=pe)))
        assert isp == pytest.approx(float(r["Isp_opt"]), rel=0.015), r["case_id"]
        n += 1
    assert n == 5


def _component_validator():
    from referencing import Registry, Resource

    schemas = data.LIBRARY / "schemas"
    sv = json.loads((schemas / "sourced_value.schema.json").read_text(encoding="utf-8"))
    comp = json.loads((schemas / "component.schema.json").read_text(encoding="utf-8"))
    registry = Registry().with_resource(sv["$id"], Resource.from_contents(sv))
    return jsonschema.Draft202012Validator(comp, registry=registry)


def test_component_records_match_schema():
    v = _component_validator()
    recs = data.components()
    assert len(recs) >= 19
    for t, rec in recs.items():
        errors = sorted(v.iter_errors(rec), key=str)
        assert not errors, f"{t}: {errors[0].message}"


def test_valve_loss_coefficients_fall_as_valve_opens():
    """Huzel & Huang Table 7-4: K decreases monotonically with opening for butterfly and ball valves."""
    rows = data.component_table("loss_coefficients")
    for fitting in ("valve_butterfly", "valve_ball"):
        pts = sorted(
            (float(r["geometry_value"]), float(r["K_nominal"])) for r in rows if r["fitting"] == fitting
        )
        ks = [k for _, k in pts]
        assert ks == sorted(ks, reverse=True), fitting


def test_pump_efficiency_curves_are_ordered_by_capacity():
    """Fig. 6-23: at a given specific speed, larger pumps are more efficient (digitized peaks)."""
    rows = data.component_table("pump_efficiency_vs_specific_speed")
    peaks = {}
    for r in rows:
        peaks[r["capacity_band"]] = max(peaks.get(r["capacity_band"], 0.0), float(r["efficiency"]))
    order = [
        "below_100_gpm",
        "100_200_gpm",
        "200_500_gpm",
        "500_1000_gpm",
        "1000_3000_gpm",
        "3000_10000_gpm",
        "over_10000_gpm",
    ]
    assert [peaks[b] for b in order] == sorted(peaks[b] for b in order)


def test_a1_turbopump_power_balance():
    """Huzel & Huang A-1 engine (Table 3-2): pump powers + auxiliary drive = turbine power within 1 %."""
    bhp = 745.69987
    pumps = (14850 + 11790) * bhp
    aux = data.components()["shaft"]["parameters"][1]["default"]["value"]
    turbine = 27140 * bhp
    assert (pumps + aux) == pytest.approx(turbine, rel=0.01)


def test_engine_closures_hold():
    """Brief rule 5: c*, thrust, C_F, MR, pump and turbine power close for every engine with flows."""
    from propsim.data import closure

    cl = closure.all_closures()
    assert {c.engine_id for c in cl} == {"HH_A1", "HH_A2", "HH_A3", "HH_A4"}
    bad = [(c.engine_id, c.detail, f"{100 * c.diff:+.2f} %") for c in cl if not c.ok]
    assert not bad
    kinds = {c.check for c in cl if c.engine_id == "HH_A1"}
    assert {
        "cstar",
        "thrust_isp",
        "thrust_cf",
        "pump_power",
        "power_balance",
        "turbine_power",
        "cycle_efficiency",
    } <= kinds


def test_engine_value_units_are_si():
    assert data.engine_value("HH_A1", "thrust_chamber", "pc_ns") == pytest.approx(1000 * 6894.757, rel=1e-6)
    assert data.engine_value("RS25", "engine", "thrust", "vac") == pytest.approx(512300 * 4.4482216, rel=1e-6)
    with pytest.raises(KeyError):
        data.engine_value("J2", "thrust_chamber", "pc")  # a known gap: the source gives no exact number


def test_a1_torque_conflict_keeps_the_consistent_unit():
    """Table 3-2 prints 20380 in-lb; only ft-lb matches 27140 bhp at 7000 rpm (conflicts.md)."""
    rows = [r for r in data.engine_params("HH_A1") if r["property"] == "torque"]
    assert len(rows) == 2
    pref = next(r for r in rows if r["preferred"] == "1")
    assert "6-5" in pref["location"]


def test_cycle_records_match_schema():
    from referencing import Registry, Resource

    schemas = data.LIBRARY / "schemas"
    sv = json.loads((schemas / "sourced_value.schema.json").read_text(encoding="utf-8"))
    cyc = json.loads((schemas / "cycle.schema.json").read_text(encoding="utf-8"))
    registry = Registry().with_resource(sv["$id"], Resource.from_contents(sv))
    v = jsonschema.Draft202012Validator(cyc, registry=registry)
    doc = json.loads((data.CYCLES / "cycles.json").read_text(encoding="utf-8"))
    assert not list(v.iter_errors(doc))
    from propsim.core.feed.cycles import CYCLES

    for c in data.cycles().values():
        assert c["propsim_cycle"] in (*CYCLES, None)


def _validator(name: str):
    from referencing import Registry, Resource

    schemas = data.LIBRARY / "schemas"
    sv = json.loads((schemas / "sourced_value.schema.json").read_text(encoding="utf-8"))
    sch = json.loads((schemas / name).read_text(encoding="utf-8"))
    registry = Registry().with_resource(sv["$id"], Resource.from_contents(sv))
    return jsonschema.Draft202012Validator(sch, registry=registry)


def test_example_graphs_match_schema():
    v = _validator("graph.schema.json")
    gs = data.graphs()
    assert set(gs) == {"gg_kerolox", "pressure_fed_kerolox", "expander_hydrolox", "ssme_staged_combustion"}
    for gid, g in gs.items():
        errors = sorted(v.iter_errors(g), key=str)
        assert not errors, f"{gid}: {errors[0].message}"


def test_default_panels_match_schema():
    v = _validator("dashboard.schema.json")
    doc = json.loads((data.DASHBOARDS / "default_panels.json").read_text(encoding="utf-8"))
    assert not list(v.iter_errors(doc))
    assert set(data.default_panels()) == set(data.components())


def test_graph_check_catches_bad_wiring():
    """The integrity check rejects a reversed edge, a wrong unit and a missing user-input parameter."""
    import copy

    g = copy.deepcopy(data.graphs()["pressure_fed_kerolox"])
    e = g["edges"][0]
    e["from"], e["to"] = e["to"], e["from"]
    g["nodes"][0]["params"]["p_set"]["unit"] = "K"
    del g["nodes"][1]["params"]["length"]
    problems: list[str] = []
    data._check_graph(g, problems)
    text = "\n".join(problems)
    assert "against port directions" in text
    assert "expected 'Pa'" in text
    assert "user-input parameter 'length' missing" in text


def test_gg_kerolox_uses_published_a1_values():
    g = data.graphs()["gg_kerolox"]
    nodes = {n["id"]: n for n in g["nodes"]}
    assert nodes["chamber"]["params"]["A_t"]["value"] == pytest.approx(487 * 6.4516e-4, rel=1e-5)
    assert nodes["ox_pump"]["params"]["D_tip"]["quality"] == "published"
    targets = {(t["node"], t["output"]): t["value"]["value"] for t in g["validation_targets"]}
    assert targets["nozzle", "thrust"] == pytest.approx(747000 * 4.4482216, rel=1e-6)


def test_material_fits_evaluate_in_si():
    from propsim.data.matprops import material_property

    # Plausibility at room temperature (not a second source): conductivities, modulus, copper expansion
    assert material_property("SS304", "k", 300.0) == pytest.approx(15.3, rel=0.02)
    assert material_property("CU_OFHC", "k", 300.0, segment="RRR100") == pytest.approx(396.0, rel=0.02)
    assert material_property("SS304", "E", 293.0) == pytest.approx(199.5e9, rel=0.01)
    assert material_property("CU_OFHC", "alpha", 300.0) == pytest.approx(16.6e-6, rel=0.02)
    # thermal strain is zero at the 293 K reference and the floor below T_low
    assert abs(material_property("TI6AL4V", "thermal_strain", 293.0)) < 2e-6
    assert material_property("SS304", "thermal_strain", 10.0) == pytest.approx(-300.04e-5)
    with pytest.raises(KeyError):
        material_property("CU_OFHC", "k", 100.0)  # copper conductivity needs an RRR segment
    with pytest.raises(ValueError):
        material_property("IN718", "k", 900.0)  # NIST fits stop at 300 K (gaps.md)


def test_inconel_x_points_match_the_graph_wall():
    k = [r for r in data.material_points("INCONEL_X") if r["property"] == "k"]
    wall = {n["id"]: n for n in data.graphs()["gg_kerolox"]["nodes"]}["regen_jacket"]["params"]["k_wall"]
    assert any(float(r["value"]) == pytest.approx(wall["value"], rel=1e-4) for r in k)


def test_high_temperature_points():
    rows = data.material_points("IN718")
    e70 = next(r for r in rows if r["property"] == "E" and r["location"].startswith("Table 4 at 70 degF"))
    assert float(e70["value"]) == pytest.approx(29.0e6 * 6894.757, rel=1e-5)
    assert float(e70["T_min"]) == pytest.approx(294.26, abs=0.01)
    gr = [r for r in data.material_points("GRCOP84") if r["property"] == "yield_strength"]
    assert {round(float(r["T_max"])) for r in gr} == {294, 811}


def test_regression_laws_convert_to_si():
    # SP-1a/GOX: r = 0.488 G^0.62 in mm/s with G in g/(cm2 s); 30 g/(cm2 s) = 300 kg/(m2 s)
    assert data.regression_rate("SP1A_GOX", 300.0) == pytest.approx(0.488e-3 * 30.0**0.62, rel=1e-5)
    # Kibbey 18 % Al: in/s with lbm/(in2 s)
    G_us = 0.2
    G_si = G_us * 0.45359237 / 0.0254**2
    assert data.regression_rate("HTPB_AL18_GOX", G_si) == pytest.approx(
        0.1262 * G_us**0.7121 * 0.0254, rel=1e-5
    )


def test_hotfire_flux_columns_are_aligned():
    """Initial oxidizer flux equals mdot / initial port area within 10 % (throttled test excepted)."""
    import math

    cases = data.hotfire_cases()
    assert len(cases) >= 20
    for c in cases:
        if "Thr" in c["case_id"]:
            continue
        g = float(c["mdot_ox"]) / (math.pi * float(c["D_port0"]) ** 2 / 4)
        assert g == pytest.approx(float(c["G_ox0"]), rel=0.10), c["case_id"]


def test_next_throttle_table_converts_to_si():
    rows = data.throttle_table("NEXT")
    assert len(rows) == 56
    top = next(r for r in rows if r["point"] == "T3-01")  # 5.76 mg/s, 236 mN, 4190 s, 7.220 kW PPU input
    assert float(top["mdot"]) == pytest.approx(5.76e-6)
    assert float(top["thrust"]) == pytest.approx(0.236)
    assert float(top["P_in_ppu"]) == pytest.approx(7220.0)
    assert float(top["life"]) == pytest.approx(37.6 * 3.6e6)
    assert len(data.throttle_table("NEXT", "extended")) == 16


def test_next_throttle_points_close():
    """Thrust = mdot Isp g0 and eta_t = F^2 / (2 mdot P_thruster) hold on every printed point."""
    for r in data.throttle_table():
        m, F, isp = float(r["mdot"]), float(r["thrust"]), float(r["Isp"])
        assert m * isp * data.G0 == pytest.approx(F, rel=0.01), r["point"]
        eta = F**2 / (2 * m * data.thruster_power(r))
        assert eta == pytest.approx(float(r["eta_thruster"]), rel=0.03), r["point"]


def test_ssme_graph_uses_public_sources_and_closes():
    """SSME graph: only public sources, flows close on the printed pump flows, turbines exhaust above P_c."""
    g = data.graphs()["ssme_staged_combustion"]
    allowed = {"NASA-SNR-1981", "NASA-TP-3700", "PROPSIM-EST", "HH-SP125", "COOLPROP-8.0.0"}
    vals = [v for n in g["nodes"] for v in n["params"].values()] + list(g["design_point"].values())
    assert {v["source_id"] for v in vals} <= allowed
    dp = {k: v["value"] for k, v in g["design_point"].items()}
    assert dp["mdot_fuel"] == pytest.approx(147 * 0.45359237, rel=0.01)  # printed LPFTP flow
    assert dp["mdot_ox"] == pytest.approx(883 * 0.45359237, rel=0.01)  # printed LPOTP flow
    pc = next(t["value"]["value"] for t in g["validation_targets"] if t["output"] == "p_c")
    assert min(dp["hpft_exit_pressure"], dp["hpot_exit_pressure"]) > pc
