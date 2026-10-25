"""Energy meters: measure watts while a block of work runs and report watt-hours.

Each meter samples one power source in a background thread (or streams it from a child
process) and always stops cleanly, even when the measured work raises. When a source gives
no samples, the reading falls back to the node's configured watts and says so.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnergyReading:
    method: str  # macmon | battery | nvidia-smi | estimate
    scope: str  # soc | system | gpu | configured
    seconds: float
    samples: int
    mean_watts: float
    estimated: bool
    note: str = ""

    @property
    def watt_hours(self) -> float:
        return self.mean_watts * self.seconds / 3600


@dataclass
class Measurement:
    """Filled in when the `measure()` block exits."""

    reading: EnergyReading | None = None


class MeterUnavailable(RuntimeError):
    pass


@dataclass
class EstimateMeter:
    """Configured watts x wall time. Always available; always labelled as an estimate."""

    watts: float
    method: str = "estimate"
    note: str = ""

    def check(self) -> None:
        return None

    @contextmanager
    def measure(self) -> Iterator[Measurement]:
        result = Measurement()
        start = time.monotonic()
        try:
            yield result
        finally:
            result.reading = self._reading(time.monotonic() - start)

    def _reading(self, seconds: float) -> EnergyReading:
        return EnergyReading(self.method, "configured", seconds, 0, self.watts, True, self.note)


@dataclass
class SampledMeter:
    """Polls `read_watts` every `interval` seconds while the measured block runs."""

    read_watts: Callable[[], float]
    method: str
    scope: str
    fallback_watts: float
    interval: float = 0.5
    _samples: list[float] = field(default_factory=list, init=False)

    def check(self) -> None:
        self.read_watts()  # raises MeterUnavailable when the source cannot be read

    def _poll(self, stop: threading.Event) -> None:
        while not stop.wait(self.interval):
            try:
                self._samples.append(self.read_watts())
            except MeterUnavailable:
                return

    @contextmanager
    def measure(self) -> Iterator[Measurement]:
        self._samples = []
        result, stop = Measurement(), threading.Event()
        worker = threading.Thread(target=self._poll, args=(stop,), daemon=True)
        start = time.monotonic()
        worker.start()
        try:
            yield result
        finally:
            stop.set()
            worker.join(timeout=5)
            result.reading = self._reading(time.monotonic() - start)

    def _reading(self, seconds: float) -> EnergyReading:
        if not self._samples:
            note = f"no {self.method} samples; configured watts used"
            return EnergyReading(
                self.method, "configured", seconds, 0, self.fallback_watts, True, note
            )
        mean = sum(self._samples) / len(self._samples)
        return EnergyReading(self.method, self.scope, seconds, len(self._samples), mean, False)


# --- power sources -------------------------------------------------------------------


def _run(cmd: list[str], timeout: float = 5) -> str:
    exe = shutil.which(cmd[0])
    if exe is None:
        raise MeterUnavailable(f"{cmd[0]} not found")
    try:
        done = subprocess.run(  # noqa: S603 -- fixed argv from this module, resolved via which
            [exe, *cmd[1:]], capture_output=True, text=True, timeout=timeout, check=True
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise MeterUnavailable(f"{cmd[0]} failed: {exc}") from exc
    return done.stdout


_IOREG_INT = r'"{key}" = (\d+)'


def battery_watts(ioreg_output: str) -> float:
    """System draw from the battery gauge: -current x voltage. Valid only when unplugged."""

    def field_value(key: str) -> int:
        match = re.search(_IOREG_INT.format(key=key), ioreg_output)
        if match is None:
            raise MeterUnavailable(f"battery gauge has no {key}")
        return int(match.group(1))

    if re.search(r'"ExternalConnected" = Yes', ioreg_output):
        raise MeterUnavailable("on external power; battery draw does not measure the system")
    millivolts = field_value("Voltage")
    raw = field_value("InstantAmperage")
    milliamps = raw - (1 << 64) if raw >= (1 << 63) else raw  # unsigned 64-bit in ioreg
    return max(0.0, -milliamps * millivolts / 1_000_000)


def read_battery() -> float:
    return battery_watts(_run(["ioreg", "-rn", "AppleSmartBattery"]))


def nvidia_watts(csv_output: str) -> float:
    values = [float(v) for v in csv_output.split() if v.replace(".", "", 1).isdigit()]
    if not values:
        raise MeterUnavailable("nvidia-smi reported no power draw")
    return sum(values)  # all GPUs on the node


def read_nvidia() -> float:
    return nvidia_watts(
        _run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"])
    )


def macmon_watts(line: str) -> float:
    """SoC power (CPU + GPU + ANE + DRAM) from one `macmon pipe` JSON line."""
    try:
        return float(json.loads(line)["all_power"])
    except (ValueError, KeyError, TypeError) as exc:
        raise MeterUnavailable(f"unreadable macmon sample: {exc}") from exc


def read_macmon() -> float:
    return macmon_watts(_run(["macmon", "pipe", "-s", "1", "-i", "200"]).splitlines()[0])
