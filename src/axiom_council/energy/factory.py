"""Energy-meter factory: a node's `energy_meter` field picks how its power is measured.

A meter whose source cannot be read on this machine (tool missing, laptop on external power)
degrades to the configured-watts estimate, and the reading says why. A new power source is
one `register_meter` call.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Protocol

from axiom_council.energy.meters import (
    EstimateMeter,
    Measurement,
    MeterUnavailable,
    SampledMeter,
    read_battery,
    read_macmon,
    read_nvidia,
)
from axiom_council.registry import NodeSpec


class EnergyMeter(Protocol):
    def check(self) -> None: ...

    def measure(self) -> AbstractContextManager[Measurement]: ...


@dataclass(frozen=True)
class PowerSource:
    read_watts: Callable[[], float]
    scope: str


_SOURCES: dict[str, PowerSource] = {}


def register_meter(name: str, read_watts: Callable[[], float], scope: str) -> None:
    if name in _SOURCES or name == "estimate":
        raise ValueError(f"energy meter {name!r} is already registered")
    _SOURCES[name] = PowerSource(read_watts, scope)


def meter_names() -> list[str]:
    return sorted([*_SOURCES, "estimate"])


def meter_for(node: NodeSpec, interval: float = 0.5) -> EnergyMeter:
    """The node's configured meter, or an estimate that records why the meter is not used."""
    name = node.energy_meter
    if name == "estimate":
        return EstimateMeter(node.watts_under_load)
    source = _SOURCES.get(name)
    if source is None:
        raise ValueError(
            f"node {node.node_id} uses unknown energy meter {name!r}; known: {meter_names()}"
        )
    meter = SampledMeter(source.read_watts, name, source.scope, node.watts_under_load, interval)
    try:
        meter.check()
    except MeterUnavailable as exc:
        return EstimateMeter(node.watts_under_load, note=f"{name} unavailable: {exc}")
    return meter


register_meter("macmon", read_macmon, "soc")
register_meter("battery", read_battery, "system")
register_meter("nvidia-smi", read_nvidia, "gpu")
