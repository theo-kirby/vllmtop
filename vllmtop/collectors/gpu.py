"""Collect a :class:`GpuSnapshot` from NVML (pynvml)."""

from __future__ import annotations

from typing import Optional

from ..state import GpuSnapshot

try:
    import pynvml

    _PYNVML_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - import guard
    pynvml = None  # type: ignore[assignment]
    _PYNVML_IMPORT_ERROR = str(exc)


def _decode(name) -> Optional[str]:
    if isinstance(name, bytes):
        return name.decode("utf-8", errors="replace")
    return name


class GpuCollector:
    """Initialises NVML once and returns a GPU snapshot per poll.

    Any NVML failure degrades gracefully: the snapshot's ``available`` flag is
    False and ``error`` carries a message so the UI can show the panel as
    unavailable rather than crashing.
    """

    def __init__(self, gpu_index: int) -> None:
        self.gpu_index = gpu_index
        self._handle = None
        self._name: Optional[str] = None
        self._init_error: Optional[str] = None
        self._init()

    def _init(self) -> None:
        if pynvml is None:
            self._init_error = f"pynvml unavailable: {_PYNVML_IMPORT_ERROR}"
            return
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._name = _decode(pynvml.nvmlDeviceGetName(self._handle))
        except Exception as exc:
            self._init_error = f"NVML init failed: {exc}"
            self._handle = None

    def poll(self) -> GpuSnapshot:
        if self._handle is None:
            return GpuSnapshot(available=False, error=self._init_error)

        h = self._handle
        snap = GpuSnapshot(available=True, name=self._name)
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            snap.util_gpu = float(util.gpu)
            snap.util_mem = float(util.memory)

            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            snap.mem_used = float(mem.used)
            snap.mem_total = float(mem.total)

            snap.temperature = float(
                pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            )
            # Power values come back in milliwatts.
            snap.power_usage = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            try:
                snap.power_limit = (
                    pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
                )
            except Exception:
                snap.power_limit = 0.0
            snap.sm_clock = float(
                pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            )
            try:
                snap.fan_speed = float(pynvml.nvmlDeviceGetFanSpeed(h))
            except Exception:
                snap.fan_speed = 0.0
        except Exception as exc:
            return GpuSnapshot(available=False, error=f"NVML poll failed: {exc}",
                               name=self._name)
        return snap

    def close(self) -> None:
        if pynvml is not None and self._handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._handle = None
