"""Nuclear thermal designer window: one window for the five design studies of a nuclear thermal rocket (performance,
reactor criticality, fuel element, engine cycle, xenon and samarium poisons).  Each study keeps its own parameters while
you move between them; the design is the set of all of them, saved as ``*.psimdesign.json`` and used by the nuclear
analysis modules (the kinetics and startup run takes the core temperature, pressure and inlet temperature from the
performance study).  Built on the same window as the solid and hybrid motor designers."""
from __future__ import annotations

from ..modules.nuclear_models import (CriticalityModel, CyclesModel, FuelElementModel, NTRPerformanceModel,
                                      PoisonsModel)
from .grain_designer import GrainDesigner

STUDIES = {
    NTRPerformanceModel.key: ("Performance", NTRPerformanceModel),
    CriticalityModel.key: ("Reactor criticality", CriticalityModel),
    FuelElementModel.key: ("Fuel element", FuelElementModel),
    CyclesModel.key: ("Engine cycle", CyclesModel),
    PoisonsModel.key: ("Xenon and samarium", PoisonsModel),
}


class NuclearDesigner(GrainDesigner):
    MODES = STUDIES
    MODE_LABEL = " Study "
    TITLE = "Nuclear Thermal Designer"
    USE_TIP = "make this the active design: the nuclear analysis modules start from it"

    def __init__(self, main=None, study: str = "nuc.perf", parent=None) -> None:
        self.saved: dict[str, dict] = {}                    # study key -> its parameters, kept while switching studies
        super().__init__(main, study, parent)

    # ---- the design is every study's parameters ---------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        if getattr(self, "form", None) is not None and self.ctrl is not None:
            self.saved[self.ctrl.key] = self.form.values().model_dump()      # remember the study being left
        super().set_mode(mode)
        if mode in self.saved:
            keep = set(self.ctrl.Params.model_fields)
            self.form.set_values({k: v for k, v in self.saved[mode].items() if k in keep})
            self.refresh()

    def make_design(self):
        from .. import design as D
        if self.ctrl is not None and self.form is not None:
            self.saved[self.ctrl.key] = self.form.values().model_dump()
        return D.Design("nuclear", "Nuclear thermal design", dict(self.saved), "user")

    def load_design(self, d) -> None:
        self.saved = {k: dict(v) for k, v in d.data.items() if k in self.MODES}
        key = self.ctrl.key if self.ctrl is not None else next(iter(self.MODES))
        self.set_mode(key if key in self.MODES else next(iter(self.MODES)))
        keep = set(self.ctrl.Params.model_fields)
        if self.ctrl.key in self.saved:
            self.form.set_values({k: v for k, v in self.saved[self.ctrl.key].items() if k in keep})
            self.refresh()

    def open_in_main(self) -> None:
        if self.main is None:
            return
        self.saved[self.ctrl.key] = self.form.values().model_dump()
        self.main.select_model(self.ctrl.key, self.form.values().model_dump())
        self.main.raise_()
        self.main.activateWindow()
