"""Snapshot dataclasses, ring-buffer series, and rate/histogram math.

The collectors produce *raw* snapshots (current counter/gauge values). The UI
thread feeds successive snapshots into a :class:`History`, which turns counters
into rates and histograms into recent-average latencies, appending the derived
values to per-series ring buffers for charting.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


class Series:
    """Fixed-length ring buffer of floats for chart history."""

    def __init__(self, maxlen: int) -> None:
        self._buf: Deque[float] = deque(maxlen=maxlen)

    def append(self, value: float) -> None:
        self._buf.append(float(value))

    def values(self) -> List[float]:
        return list(self._buf)

    @property
    def last(self) -> float:
        return self._buf[-1] if self._buf else 0.0

    def __len__(self) -> int:
        return len(self._buf)


@dataclass
class Histogram:
    """A Prometheus histogram's cumulative state at one point in time."""

    count: float = 0.0
    sum: float = 0.0
    # Cumulative bucket counts keyed by upper bound (`le`). +Inf stored as math.inf.
    buckets: Dict[float, float] = field(default_factory=dict)


@dataclass
class VllmSnapshot:
    """Raw current values pulled from one /metrics scrape."""

    reachable: bool = False
    error: Optional[str] = None
    model_name: Optional[str] = None

    # Engine / config info (from labels on info-style metrics).
    process_start_time: Optional[float] = None  # unix epoch; for uptime
    cache_dtype: Optional[str] = None  # KV-cache precision, e.g. "fp8"
    block_size: Optional[str] = None
    gpu_memory_utilization: Optional[str] = None  # configured target, e.g. "0.88"
    num_gpu_blocks: Optional[str] = None
    enable_prefix_caching: Optional[bool] = None
    engine_awake: Optional[bool] = None

    # Counters
    generation_tokens_total: float = 0.0
    prompt_tokens_total: float = 0.0
    prompt_tokens_cached_total: float = 0.0
    num_preemptions_total: float = 0.0
    prefix_cache_hits_total: float = 0.0
    prefix_cache_queries_total: float = 0.0
    request_success_total: float = 0.0  # summed across finish reasons

    # Gauges
    num_requests_running: float = 0.0
    num_requests_waiting: float = 0.0
    kv_cache_usage_perc: float = 0.0

    # Latency histograms
    ttft: Histogram = field(default_factory=Histogram)
    inter_token: Histogram = field(default_factory=Histogram)
    e2e: Histogram = field(default_factory=Histogram)
    queue_time: Histogram = field(default_factory=Histogram)

    # Per-request size / phase-timing histograms (observed once per completed
    # request). Used for the requests panel's per-request averages list.
    req_prompt_tokens: Histogram = field(default_factory=Histogram)
    req_gen_tokens: Histogram = field(default_factory=Histogram)
    prefill_time: Histogram = field(default_factory=Histogram)
    decode_time: Histogram = field(default_factory=Histogram)

    @property
    def prefix_cache_hit_rate(self) -> float:
        """Cumulative prefix-cache hit rate as a fraction in [0, 1]."""
        if self.prefix_cache_queries_total <= 0:
            return 0.0
        return self.prefix_cache_hits_total / self.prefix_cache_queries_total


@dataclass
class GpuSnapshot:
    """Raw current GPU values from NVML."""

    available: bool = False
    error: Optional[str] = None
    name: Optional[str] = None
    util_gpu: float = 0.0  # percent
    util_mem: float = 0.0  # percent (memory-controller utilisation)
    mem_used: float = 0.0  # bytes
    mem_total: float = 0.0  # bytes
    temperature: float = 0.0  # deg C
    power_usage: float = 0.0  # watts
    power_limit: float = 0.0  # watts
    sm_clock: float = 0.0  # MHz
    fan_speed: float = 0.0  # percent

    @property
    def mem_used_perc(self) -> float:
        if self.mem_total <= 0:
            return 0.0
        return 100.0 * self.mem_used / self.mem_total


@dataclass
class AccessLogEntry:
    """One parsed uvicorn access-log line (an HTTP call vLLM served).

    Built from the server's logs, not its metrics — so it carries no prompt or
    response text, only the request envelope: who called, which endpoint, the
    status, and when we observed it.
    """

    t: float  # wall-clock time we read the line (time.time())
    client: str  # "ip:port"
    method: str  # "POST"
    path: str  # "/v1/chat/completions"
    status: int  # 200

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 400


@dataclass
class Snapshot:
    """A combined vLLM + GPU sample taken at one monotonic instant."""

    monotonic: float
    vllm: VllmSnapshot
    gpu: GpuSnapshot
    access_log: List[AccessLogEntry] = field(default_factory=list)
    access_error: Optional[str] = None


def compute_rate(
    prev_value: float, prev_t: float, cur_value: float, cur_t: float
) -> float:
    """Δvalue/Δt for a monotonic counter.

    Guards against non-positive Δt (returns 0) and counter resets — if the
    value decreased (server restart), treat the rate as 0 rather than negative.
    """
    dt = cur_t - prev_t
    if dt <= 0:
        return 0.0
    dv = cur_value - prev_value
    if dv < 0:
        return 0.0
    return dv / dt


def histogram_recent_avg(prev: Histogram, cur: Histogram) -> float:
    """Recent mean = Δsum / Δcount between two scrapes.

    Far more useful live than the cumulative average. Returns 0 when no new
    observations landed in the interval (or on a counter reset).
    """
    dcount = cur.count - prev.count
    dsum = cur.sum - prev.sum
    if dcount <= 0 or dsum < 0:
        return 0.0
    return dsum / dcount


def histogram_avg(prev: Histogram, cur: Histogram) -> float:
    """Mean observation, recent if possible, else cumulative.

    Like :func:`histogram_recent_avg` but falls back to the all-time mean
    (sum/count) when no new observations landed in the window. Per-request
    histograms only update when a request *completes*, so a pure recent
    average would flicker to 0 between completions; this keeps it stable.
    """
    dcount = cur.count - prev.count
    dsum = cur.sum - prev.sum
    if dcount > 0 and dsum >= 0:
        return dsum / dcount
    if cur.count > 0:
        return cur.sum / cur.count
    return 0.0


def histogram_quantile(prev: Histogram, cur: Histogram, q: float) -> float:
    """Approximate quantile q (0..1) of observations in the (prev, cur) window.

    Uses the per-bucket count deltas and linear interpolation within the
    containing bucket. Returns 0 if no observations occurred. Stretch feature;
    recent-average is the v1 must-have.
    """
    les = sorted(cur.buckets)
    if not les:
        return 0.0
    # Per-bucket (non-cumulative) deltas over the window.
    delta_total = cur.count - prev.count
    if delta_total <= 0:
        return 0.0

    target = q * delta_total
    prev_cum = 0.0
    cur_cum = 0.0
    lower_bound = 0.0
    for le in les:
        d = cur.buckets.get(le, 0.0) - prev.buckets.get(le, 0.0)
        if d < 0:
            d = 0.0
        cur_cum += d
        if cur_cum >= target:
            if math.isinf(le):
                return lower_bound
            # Linear interpolation within [lower_bound, le].
            span = cur_cum - prev_cum
            if span <= 0:
                return le
            frac = (target - prev_cum) / span
            return lower_bound + frac * (le - lower_bound)
        prev_cum = cur_cum
        if not math.isinf(le):
            lower_bound = le
    return lower_bound


class History:
    """Holds per-series ring buffers and derives rates from raw snapshots.

    Call :meth:`update` with each new :class:`Snapshot`; it computes derived
    quantities against the previous snapshot and appends them to the series.
    """

    # Names of the series we keep, for iteration in tests/UI.
    SERIES_NAMES = (
        "gen_tok_s",
        "prompt_tok_s",
        "gpu_util",
        "gpu_mem_perc",
        "gpu_temp",
        "gpu_power",
        "running",
        "waiting",
        "kv_cache",
        "ttft",
        "tpot",
        "e2e",
        "queue_time",
        "req_prompt_tok",
        "req_gen_tok",
        "req_prefill",
        "req_decode",
    )

    def __init__(self, maxlen: int) -> None:
        self.maxlen = maxlen
        self.series: Dict[str, Series] = {
            name: Series(maxlen) for name in self.SERIES_NAMES
        }
        self._prev: Optional[Snapshot] = None
        # Latest derived scalars for big-number / hbar display.
        self.derived: Dict[str, float] = {name: 0.0 for name in self.SERIES_NAMES}

    def update(self, snap: Snapshot) -> None:
        prev = self._prev
        v, g = snap.vllm, snap.gpu

        # --- vLLM rates (only meaningful with a previous reachable sample) ---
        if prev is not None and prev.vllm.reachable and v.reachable:
            pt, ct = prev.monotonic, snap.monotonic
            gen = compute_rate(
                prev.vllm.generation_tokens_total, pt,
                v.generation_tokens_total, ct,
            )
            prompt = compute_rate(
                prev.vllm.prompt_tokens_total, pt, v.prompt_tokens_total, ct,
            )
            ttft = histogram_recent_avg(prev.vllm.ttft, v.ttft)
            tpot = histogram_recent_avg(prev.vllm.inter_token, v.inter_token)
            e2e = histogram_recent_avg(prev.vllm.e2e, v.e2e)
            queue = histogram_recent_avg(prev.vllm.queue_time, v.queue_time)
            req_prompt = histogram_avg(prev.vllm.req_prompt_tokens,
                                       v.req_prompt_tokens)
            req_gen = histogram_avg(prev.vllm.req_gen_tokens, v.req_gen_tokens)
            req_prefill = histogram_avg(prev.vllm.prefill_time, v.prefill_time)
            req_decode = histogram_avg(prev.vllm.decode_time, v.decode_time)
        else:
            gen = prompt = ttft = tpot = e2e = queue = 0.0
            req_prompt = req_gen = req_prefill = req_decode = 0.0

        self._set("gen_tok_s", gen)
        self._set("prompt_tok_s", prompt)
        self._set("ttft", ttft)
        self._set("tpot", tpot)
        self._set("e2e", e2e)
        self._set("queue_time", queue)
        self._set("req_prompt_tok", req_prompt)
        self._set("req_gen_tok", req_gen)
        self._set("req_prefill", req_prefill)
        self._set("req_decode", req_decode)

        # --- vLLM gauges (instantaneous) ---
        self._set("running", v.num_requests_running)
        self._set("waiting", v.num_requests_waiting)
        self._set("kv_cache", v.kv_cache_usage_perc * 100.0)

        # --- GPU gauges ---
        self._set("gpu_util", g.util_gpu)
        self._set("gpu_mem_perc", g.mem_used_perc)
        self._set("gpu_temp", g.temperature)
        self._set("gpu_power", g.power_usage)

        self._prev = snap

    def _set(self, name: str, value: float) -> None:
        self.series[name].append(value)
        self.derived[name] = value
