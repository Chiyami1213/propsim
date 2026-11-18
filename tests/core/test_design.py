"""Designs: file round trip and what the analysis modules take from them."""
from __future__ import annotations

import pytest

from propsim import data
from propsim import design as D


def test_design_round_trip(tmp_path):
    d = D.Design("solid", "my grain", {"R_out": 55.0, "R_p": 12.0}, "user", "note")
    p = d.save(tmp_path / "g")
    assert p.name.endswith(D.SUFFIX)
    assert D.Design.load(p) == d
    with pytest.raises(ValueError):
        D.Design.from_dict({"format": "other"})


def test_solid_design_feeds_the_solid_modules():
    from propsim.modules.solid_models import SolidBallisticsParams
    d = D.Design("solid", "s", {"R_out": 55.0, "L_g": 0.9, "n": 0.4, "burn": 50.0})
    up, clamped = D.fit(D.updates(d, "solid.ballistics"), SolidBallisticsParams)
    assert up == {"R_out": 55.0, "L_g": 0.9, "n": 0.4} and not clamped
    assert D.updates(d, "hybrid.burn0d") == {}                   # a solid design does not apply to a hybrid module
    assert SolidBallisticsParams(**up).R_out == 55.0


def test_hybrid_design_maps_port_and_web():
    d = D.Design("hybrid", "h", {"R_out": 60.0, "R_p": 20.0, "L_g": 0.8, "mdot_ox": 2.2})
    up = D.updates(d, "hybrid.burn1d")
    assert up["R_p0"] == 20.0 and up["web"] == 40.0 and up["L_g"] == 0.8 and up["mdot_ox"] == 2.2


def test_liquid_design_gives_pressure_flow_and_throat():
    from propsim.sims.liquid_run import CycleParams
    d = D.Design("liquid", "a1", {"graph": data.graphs()["gg_kerolox"]}, "gg_kerolox")
    raw = D.updates(d, "liquid.cycle")
    assert 5.0 < raw["Pc"] < 9.0 and 1.5 < raw["MR"] < 3.5 and raw["D_t"] > 300.0
    up, clamped = D.fit(raw, CycleParams)
    assert "D_t" in clamped and up["D_t"] == 200.0                  # larger than the module's throat range
    CycleParams(**up)                                              # the values are valid parameters
