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
