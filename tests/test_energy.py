"""Energy meters: parsing each power source, sampling, fallbacks and the factory."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from axiom_council.energy import (
    EstimateMeter,
    MeterUnavailable,
    meter_for,
    meter_names,
    register_meter,
)
from axiom_council.energy import meters as m
from axiom_council.energy.meters import SampledMeter
from axiom_council.evals.bakeoff import Bakeoff, Contestant
from axiom_council.evals.fixtures import load_fixtures
from axiom_council.ledger import CapabilityLedger
from axiom_council.registry import Registry

from .fake_provider import FakeProvider
from .test_bakeoff import GOOD_PLAN, REPO

UNPLUGGED = (
    '"ExternalConnected" = No\n"Voltage" = 13000\n"InstantAmperage" = 18446744073709550616\n'
)


def test_battery_reads_system_draw_only_when_unplugged() -> None:
    assert m.battery_watts(UNPLUGGED) == pytest.approx(13.0)  # -1000 mA x 13.0 V
    with pytest.raises(MeterUnavailable, match="external power"):
        m.battery_watts(UNPLUGGED.replace("= No", "= Yes"))
    with pytest.raises(MeterUnavailable, match="Voltage"):
        m.battery_watts('"InstantAmperage" = 5\n')
    charging = UNPLUGGED.replace("18446744073709550616", "1500")
    assert m.battery_watts(charging) == 0.0  # current flowing in is not consumption


def test_nvidia_and_macmon_parsing() -> None:
    assert m.nvidia_watts("41.25\n12.50\n") == pytest.approx(53.75)
    with pytest.raises(MeterUnavailable):
        m.nvidia_watts("[N/A]\n")
    assert m.macmon_watts(json.dumps({"all_power": 7.5, "sys_power": 13.6})) == 7.5
    with pytest.raises(MeterUnavailable, match="unreadable"):
        m.macmon_watts("{}")


def test_sampled_meter_averages_and_stops_even_when_work_fails() -> None:
    values = iter([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    meter = SampledMeter(lambda: next(values), "fake", "soc", fallback_watts=99, interval=0.01)

    def work() -> None:
        with meter.measure() as inner:
            holder.append(inner)
            time.sleep(0.05)
            raise RuntimeError("boom")

    holder: list[Any] = []
    with pytest.raises(RuntimeError, match="boom"):
        work()
    measured = holder[0]
    reading = measured.reading
    assert reading is not None and not reading.estimated and reading.samples >= 2
    assert 10 <= reading.mean_watts <= 80 and reading.watt_hours > 0


def test_sampled_meter_without_samples_falls_back_to_configured_watts() -> None:
    def broken() -> float:
        raise MeterUnavailable("gone")

    meter = SampledMeter(broken, "fake", "soc", fallback_watts=30, interval=0.01)
    with meter.measure() as measured:
        pass
    reading = measured.reading
    assert reading is not None and reading.estimated and reading.mean_watts == 30
    assert "no fake samples" in reading.note
    with pytest.raises(MeterUnavailable):
        meter.check()


def test_estimate_meter_is_labelled() -> None:
    with EstimateMeter(30).measure() as measured:
        pass
    assert measured.reading is not None and measured.reading.estimated
    assert measured.reading.scope == "configured"


def test_command_runner_reports_missing_and_failing_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(MeterUnavailable, match="not found"):
        m.read_battery()
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.CalledProcessError(1, "x")

    monkeypatch.setattr("subprocess.run", fail)
    with pytest.raises(MeterUnavailable, match="failed"):
        m.read_nvidia()

    def ok(*args: Any, **kwargs: Any) -> Any:
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps({"all_power": 4.0}) + "\n")

    monkeypatch.setattr("subprocess.run", ok)
    assert m.read_macmon() == 4.0


def test_factory_picks_meter_by_node_config(
    registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = registry.nodes["mac-m5"]
    assert {"macmon", "battery", "nvidia-smi", "estimate"} <= set(meter_names())
    estimate = meter_for(node.model_copy(update={"energy_meter": "estimate"}))
    assert isinstance(estimate, EstimateMeter)

    monkeypatch.setattr("shutil.which", lambda name: None)  # tool missing on this machine
    degraded = meter_for(node)  # configured as macmon
    assert isinstance(degraded, EstimateMeter) and "macmon unavailable" in degraded.note

    register_meter("fake-meter", lambda: 5.0, "gpu")
    live = meter_for(node.model_copy(update={"energy_meter": "fake-meter"}), interval=0.01)
    assert isinstance(live, SampledMeter) and live.scope == "gpu"
    with pytest.raises(ValueError, match="already registered"):
        register_meter("fake-meter", lambda: 5.0, "gpu")
    with pytest.raises(ValueError, match="unknown energy meter"):
        meter_for(node.model_copy(update={"energy_meter": "rapl"}))


def test_bakeoff_records_energy_and_carbon(registry: Registry, tmp_path: Path) -> None:
    chat = FakeProvider({"plan": json.dumps(GOOD_PLAN), "review": "{}"})
    ledger = CapabilityLedger(tmp_path / "ledger.sqlite3")
    bake = Bakeoff(
        registry,
        load_fixtures(REPO / "evals/fixtures"),
        chat,
        ledger,
        tmp_path / "r.jsonl",
        "mac-m5",
        EstimateMeter(3600.0),  # 1 Wh per second keeps the arithmetic visible
    )
    record = bake.run([Contestant("qwen3:4b")])[0]
    ledger.close()
    assert record.energy_method == "estimate" and record.energy_estimated
    factor = registry.settings.energy.grid_kg_co2e_per_kwh
    assert factor > 0 and record.co2e_g == pytest.approx(record.energy_wh * factor, abs=1e-4)
