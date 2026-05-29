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
)

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
            if snap.model_name is None:
                mn = sample.labels.get("model_name")
                if mn:
                    snap.model_name = mn

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
                text = resp.read().decode(charset, errors="replace")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return VllmSnapshot(reachable=False, error=_short_error(exc))

        try:
            return parse_metrics(text)
        except Exception as exc:  # parser shouldn't fail, but never crash the UI
            return VllmSnapshot(reachable=False, error=f"parse error: {exc}")


def _short_error(exc: Exception) -> str:
    reason = getattr(exc, "reason", None)
    return str(reason) if reason is not None else str(exc)
