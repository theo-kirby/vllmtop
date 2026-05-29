from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_URL = "http://localhost:8000"
DEFAULT_INTERVAL = 1.0
DEFAULT_GPU_INDEX = 0

# How many samples to keep per series. Generous so charts have history to
# resample from; resampled down to panel width at draw time.
HISTORY_LEN = 512

# Network timeout for a single /metrics GET (seconds). Kept short so a stalled
# server can't wedge the background poller.
HTTP_TIMEOUT = 2.0


@dataclass
class AppConfig:
    url: str = DEFAULT_URL
    interval: float = DEFAULT_INTERVAL
    gpu_index: int = DEFAULT_GPU_INDEX
    no_gpu: bool = False
    history_len: int = HISTORY_LEN
    http_timeout: float = HTTP_TIMEOUT

    @property
    def metrics_url(self) -> str:
        return self.url.rstrip("/") + "/metrics"

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(url=os.environ.get("VLLMTOP_URL", DEFAULT_URL))
