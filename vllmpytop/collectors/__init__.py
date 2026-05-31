"""Data collectors: one per source, each returning a snapshot dataclass.

Submodules:
- :mod:`collectors.vllm` — scrape vLLM /metrics (Prometheus exposition text)
- :mod:`collectors.gpu` — poll NVIDIA NVML for GPU utilisation, VRAM, etc.
- :mod:`collectors.access_log` — tail a log file or ``docker logs`` for the
  request feed (vLLM request-log lines from ``--enable-log-requests``)
"""
