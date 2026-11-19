"""Electric propulsion designer window: the four design studies of an electric propulsion system (mission and power
trades, electrothermal thruster, gridded ion thruster, Hall thruster) in one window, on the same base as the nuclear and
motor designers.  The design is every study's parameters; the electric analysis modules start from it (the ion optics run
from the ion thruster geometry, the Hall discharge run from the Hall thruster, the low-thrust spiral from the mission
trade)."""
from __future__ import annotations

from ..modules.ep_models import HallLiveModel, IonLiveModel, ThermalLiveModel, TradesLiveModel
from .nuclear_designer import NuclearDesigner

STUDIES = {
    TradesLiveModel.key: ("Mission and power trades", TradesLiveModel),
    ThermalLiveModel.key: ("Electrothermal thruster", ThermalLiveModel),
    IonLiveModel.key: ("Gridded ion thruster", IonLiveModel),
    HallLiveModel.key: ("Hall thruster", HallLiveModel),
}


class ElectricDesigner(NuclearDesigner):
    MODES = STUDIES
    TITLE = "Electric Propulsion Designer"
    USE_TIP = "make this the active design: the electric analysis modules start from it"

    def __init__(self, main=None, study: str = "ep.trades", parent=None) -> None:
        super().__init__(main, study, parent)

    def make_design(self):
        from .. import design as D
        if self.ctrl is not None and self.form is not None:
            self.saved[self.ctrl.key] = self.form.values().model_dump()
        return D.Design("electric", "Electric propulsion design", dict(self.saved), "user")
