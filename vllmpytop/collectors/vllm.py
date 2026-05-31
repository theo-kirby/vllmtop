"""Collect a :class:`VllmSnapshot` from a vLLM /metrics endpoint."""

from __future__ import annotations

import math
import urllib.error
import urllib.request
from typing import Dict, Optional

from prometheus_client.parser import text_string_to_metric_families

from ..state import Histogram, VllmSnapshot

# Scalar series we read directly by sample name (summed across engine labels).
_SCALAR_NAMES = (
    "vllm:generation_tokens_total",
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:request_success_total",
)

# Cap on a single /metrics body. The endpoint is plain text and normally well
# under 1 MB; this bounds memory if a compromised/misconfigured server (or a
# MITM on plain HTTP) streams an unbounded body.
MAX_METRICS_BYTES = 16 * 1024 * 1024

# Histogram base names -> attribute on VllmSnapshot.
_HISTOGRAMS = {
    "vllm:time_to_first_token_seconds": "ttft",
    "vllm:inter_token_latency_seconds": "inter_token",
    "vllm:e2e_request_latency_seconds": "e2e",
    "vllm:request_queue_time_seconds": "queue_time",
    "vllm:request_prompt_tokens": "req_prompt_tokens",
    "vllm:request_generation_tokens": "req_gen_tokens",
    "vllm:request_prefill_time_seconds": "prefill_time",
    "vllm:request_decode_time_seconds": "decode_time",
}


def _le_to_float(le: str) -> float:
    if le in ("+Inf", "Inf", "inf"):
        return math.inf
    try:
        return float(le)
    except ValueError:
        return math.inf


def parse_metrics(text: str) -> VllmSnapshot:
    """Parse Prometheus exposition text into a :class:`VllmSnapshot`.

    Pure function (no I/O) so it can be unit-tested against a fixture. Values
    are summed across `engine` labels in case of multiple engines.
    """
    snap = VllmSnapshot(reachable=True)
    scalars: Dict[str, float] = {name: 0.0 for name in _SCALAR_NAMES}
    hists: Dict[str, Histogram] = {base: Histogram() for base in _HISTOGRAMS}

    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            name = sample.name
            value = sample.value
            labels = sample.labels
            if snap.model_name is None:
                mn = labels.get("model_name")
                if mn:
                    snap.model_name = mn

            # Process start time (stdlib process collector) -> uptime.
            if name == "process_start_time_seconds":
                snap.process_start_time = value
                continue

            # CacheConfig is exposed as a single info gauge whose labels carry
            # the interesting config (KV-cache dtype, block size, etc.).
            if name == "vllm:cache_config_info":
                snap.cache_dtype = labels.get("cache_dtype") or snap.cache_dtype
                snap.block_size = labels.get("block_size") or snap.block_size
                snap.num_gpu_blocks = (
                    labels.get("num_gpu_blocks") or snap.num_gpu_blocks
                )
                snap.gpu_memory_utilization = (
                    labels.get("gpu_memory_utilization")
                    or snap.gpu_memory_utilization
                )
                epc = labels.get("enable_prefix_caching")
                if epc is not None:
                    snap.enable_prefix_caching = epc == "True"
                continue

            # Engine sleep state: awake=1 means the engine is serving.
            if name == "vllm:engine_sleep_state" and labels.get(
                "sleep_state"
            ) == "awake":
                snap.engine_awake = value == 1.0
                continue

            if name in scalars:
                scalars[name] += value
                continue

            for base, _attr in _HISTOGRAMS.items():
                if not name.startswith(base):
                    continue
                hist = hists[base]
                if name == base + "_sum":
                    hist.sum += value
                elif name == base + "_count":
                    hist.count += value
                elif name == base + "_bucket":
                    le = _le_to_float(sample.labels.get("le", "+Inf"))
                    hist.buckets[le] = hist.buckets.get(le, 0.0) + value
                break

    snap.generation_tokens_total = scalars["vllm:generation_tokens_total"]
    snap.prompt_tokens_total = scalars["vllm:prompt_tokens_total"]
    snap.prompt_tokens_cached_total = scalars["vllm:prompt_tokens_cached_total"]
    snap.num_preemptions_total = scalars["vllm:num_preemptions_total"]
    snap.prefix_cache_hits_total = scalars["vllm:prefix_cache_hits_total"]
    snap.prefix_cache_queries_total = scalars["vllm:prefix_cache_queries_total"]
    snap.num_requests_running = scalars["vllm:num_requests_running"]
    snap.num_requests_waiting = scalars["vllm:num_requests_waiting"]
    snap.kv_cache_usage_perc = scalars["vllm:kv_cache_usage_perc"]
    snap.request_success_total = scalars["vllm:request_success_total"]

    for base, attr in _HISTOGRAMS.items():
        setattr(snap, attr, hists[base])

    return snap


class VllmCollector:
    """Fetches and parses /metrics, returning a snapshot each poll."""

    def __init__(self, metrics_url: str, timeout: float) -> None:
        self.metrics_url = metrics_url
        self.timeout = timeout

    def poll(self) -> VllmSnapshot:
        try:
            req = urllib.request.Request(
                self.metrics_url, headers={"Accept": "text/plain"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                raw = resp.read(MAX_METRICS_BYTES + 1)
                if len(raw) > MAX_METRICS_BYTES:
                    return VllmSnapshot(
                        reachable=False,
                        error=f"/metrics body exceeded {MAX_METRICS_BYTES} bytes",
                    )
                text = raw.decode(charset, errors="replace")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return VllmSnapshot(reachable=False, error=_short_error(exc))

        try:
            return parse_metrics(text)
        except Exception as exc:  # parser shouldn't fail, but never crash the UI
            return VllmSnapshot(reachable=False, error=f"parse error: {exc}")


def _short_error(exc: Exception) -> str:
    reason = getattr(exc, "reason", None)
    return str(reason) if reason is not None else str(exc)
