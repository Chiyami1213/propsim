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
