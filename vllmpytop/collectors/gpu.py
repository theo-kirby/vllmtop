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


def _optional(fn):
    """Call an NVML getter, mapping "not supported" to ``None``.

    Devices legitimately lack individual metrics — e.g. GB10 (Grace Blackwell)
    has unified memory and no fan, so NVML reports NOT_SUPPORTED for framebuffer
    memory, fans and power limits. Those gaps must not sink the whole snapshot;
    real errors still propagate to the caller.
    """
    try:
        return fn()
    except pynvml.NVMLError_NotSupported:
        return None


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
        # Utilization drives the panel's primary chart, so losing it makes the
        # snapshot meaningless; every other metric is optional and degrading
        # independently (see ``_optional``).
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
        except Exception as exc:
            return GpuSnapshot(available=False, error=f"NVML poll failed: {exc}",
                               name=self._name)

        snap = GpuSnapshot(available=True, name=self._name,
                           util_gpu=float(util.gpu),
                           util_mem=float(util.memory))
        try:
            mem = _optional(lambda: pynvml.nvmlDeviceGetMemoryInfo(h))
            if mem is not None:
                snap.mem_used = float(mem.used)
                snap.mem_total = float(mem.total)

            temp = _optional(
                lambda: pynvml.nvmlDeviceGetTemperature(
                    h, pynvml.NVML_TEMPERATURE_GPU)
            )
            if temp is not None:
                snap.temperature = float(temp)

            # Power values come back in milliwatts.
            power = _optional(lambda: pynvml.nvmlDeviceGetPowerUsage(h))
            if power is not None:
                snap.power_usage = power / 1000.0

            limit = _optional(
                lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(h)
            )
            if limit is not None:
                snap.power_limit = limit / 1000.0

            clock = _optional(
                lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
            )
            if clock is not None:
                snap.sm_clock = float(clock)

            fan = _optional(lambda: pynvml.nvmlDeviceGetFanSpeed(h))
            if fan is not None:
                snap.fan_speed = float(fan)
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
