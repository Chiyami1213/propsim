"""Model registry: the module tree shown in the left dock. Planned (unimplemented) entries are greyed out."""
from __future__ import annotations

from .base import ModelController
from .hybrid_burn import HybridBurnModel
from .nozzle_live import NozzleLiveModel
from .toy import SpringMassModel

# module -> list of (label, mode, controller class or None when planned). Labels are plain text (Qt tree).
TREE: dict[str, list[tuple[str, str, type[ModelController] | None]]] = {
    "Liquid": [("Injector & chamber", "live", None), ("Feed network transient", "run", None)],
    "Hybrid": [("Nozzle thrust coefficient", "live", NozzleLiveModel), ("Regression laws", "live", None),
               ("Port flow field", "live", None), ("0-D burn", "run", HybridBurnModel),
               ("N₂O blowdown", "run", None), ("ILFI stability", "run", None)],
    "Solid": [("Grain designer", "live", None), ("3D burnback", "run", None)],
    "Electric": [("Ion optics", "live", None), ("Hall breathing mode", "run", None)],
    "Vehicle": [("Point-mass ascent", "run", None)],
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
