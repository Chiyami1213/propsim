"""Run-mode driver interface: ``run(params, progress, cancel) -> Result`` (no Qt)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from threading import Event
from typing import Callable

from ..core.result import Result


class Simulation(ABC):
    """Builds the state, integrates, and emits a ``Result``. Must call ``progress(0..1)`` regularly and
    raise ``InterruptedError`` when ``cancel`` is set."""

    @abstractmethod
    def run(self, params, progress: Callable[[float], None], cancel: Event) -> Result: ...
