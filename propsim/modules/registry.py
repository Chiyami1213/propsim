"""Model registry: the module tree shown in the left dock. Planned (unimplemented) entries are greyed out."""
from __future__ import annotations

from .base import ModelController
from .engine_live import EngineLiveModel
from .ep_models import (HallDischargeModel, HallLiveModel, IonLiveModel, IonOpticsModel, SpiralModel, ThermalLiveModel,
                        TradesLiveModel)
from .hybrid_burn import HybridBurnModel
from .hybrid_live import PortLiveModel, RegressionLiveModel
from .hybrid_live_ext import FlowLiveModel, FuelLiveModel, StabilityLiveModel
from .liquid_models import (ChuggingModel, CycleModel, FeedTransientModel, InjectorLiveModel,
                            RegenLiveModel)
from .cp_models import CPMotorModel, CPMotorUSModel
from .solid_models import (Burnback3DModel, BurnRateLiveModel, SolidBallisticsModel, SolidGrainLiveModel)
from .hybrid_run import HybridBurn1DModel, N2OBlowdownModel
from .nozzle_live import NozzleLiveModel
from .nuclear_models import (CriticalityModel, CyclesModel, FuelElementModel, KineticsModel, NTRPerformanceModel,
                             PoisonsModel)
from .toy import SpringMassModel
from .vehicle_models import AscentModel, VehicleLiveModel

# module -> list of (label, mode, controller class or None when planned). Labels are plain text (Qt tree).
TREE: dict[str, list[tuple[str, str, type[ModelController] | None]]] = {
    "Common": [("Thermochemistry & nozzle", "live", EngineLiveModel)],
    "Liquid": [("Injector & chamber", "live", InjectorLiveModel), ("Regenerative cooling", "live", RegenLiveModel),
               ("Feed network transient", "run", FeedTransientModel), ("Pump-fed cycles", "run", CycleModel),
               ("Chugging & L* stability", "run", ChuggingModel)],
    "Hybrid": [("Nozzle thrust coefficient", "live", NozzleLiveModel), ("Regression laws", "live", RegressionLiveModel),
               ("Port & grain", "live", PortLiveModel),
               ("Core flow fields", "live", FlowLiveModel), ("Fuel & oxidizer", "live", FuelLiveModel),
               ("Instability & scaling", "live", StabilityLiveModel), ("0-D burn", "run", HybridBurnModel),
               ("Quasi-1D burn", "run", HybridBurn1DModel), ("N₂O blowdown", "run", N2OBlowdownModel), ("ILFI stability", "run", None)],
    "Solid": [("Grain designer", "live", SolidGrainLiveModel), ("Burn rate & K_n", "live", BurnRateLiveModel),
              ("3D burnback", "run", Burnback3DModel), ("Internal ballistics", "run", SolidBallisticsModel),
              ("CP motor (quasi-steady)", "run", CPMotorModel), ("CP motor (US units)", "run", CPMotorUSModel)],
    "Electric": [("Mission & power trades", "live", TradesLiveModel), ("Electrothermal thrusters", "live", ThermalLiveModel),
                 ("Gridded ion thruster", "live", IonLiveModel), ("Hall thruster", "live", HallLiveModel),
                 ("Ion optics", "run", IonOpticsModel), ("Hall discharge", "run", HallDischargeModel),
                 ("Low-thrust spiral", "run", SpiralModel)],
    "Nuclear": [("NTR performance", "live", NTRPerformanceModel), ("Reactor criticality", "live", CriticalityModel),
                ("Fuel element thermal hydraulics", "live", FuelElementModel), ("NTR engine cycles", "live", CyclesModel),
                ("Xenon & samarium", "live", PoisonsModel), ("Reactor kinetics & startup", "run", KineticsModel)],
    "Vehicle": [("Vehicle & staging", "live", VehicleLiveModel), ("Point-mass ascent", "run", AscentModel)],
    "Toy": [("Spring-mass", "run", SpringMassModel)],
}

_CACHE: dict[str, ModelController] = {}


def get(key: str) -> ModelController:
    """Controller instance by key (one per model, created lazily)."""
    if key not in _CACHE:
        for entries in TREE.values():
            for _, _, cls in entries:
                if cls is not None and cls.key == key:
                    _CACHE[key] = cls()
                    return _CACHE[key]
        raise KeyError(key)
    return _CACHE[key]
