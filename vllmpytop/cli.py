"""Command-line entry point: argparse, then either --dump-json or the TUI."""

from __future__ import annotations

import argparse
import json
import sys
import time

from .collectors.gpu import GpuCollector
from .collectors.vllm import VllmCollector
from .config import (
    AppConfig,
    DEFAULT_GPU_INDEX,
    DEFAULT_INTERVAL,
    DEFAULT_URL,
)
from .state import History, Snapshot


def _build_parser() -> argparse.ArgumentParser:
    import os

    p = argparse.ArgumentParser(
        description="A btop-style TUI for monitoring a vLLM instance and its GPU.",
    )
    p.add_argument(
        "--url",
        default=os.environ.get("VLLMTOP_URL", DEFAULT_URL),
        help="vLLM base URL (default: %(default)s, env VLLMTOP_URL)",
    )
    p.add_argument(
        "--interval", type=float, default=DEFAULT_INTERVAL,
        help="poll interval in seconds (default: %(default)s)",
    )
    p.add_argument(
        "--gpu-index", type=int, default=DEFAULT_GPU_INDEX,
        help="NVML GPU index (default: %(default)s)",
    )
    p.add_argument("--no-gpu", action="store_true", help="disable the GPU panel")
    p.add_argument(
        "--dump-json", action="store_true",
        help="collect a snapshot, print it as JSON, and exit (no TTY needed)",
    )
    return p


def _config_from_args(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        url=args.url,
        interval=max(0.1, args.interval),
        gpu_index=args.gpu_index,
        no_gpu=args.no_gpu,
    )


def collect_pair(config: AppConfig) -> tuple[Snapshot, Snapshot]:
    """Take two snapshots `interval` apart so rates are populated."""
    vllm = VllmCollector(config.metrics_url, config.http_timeout)
    gpu = None if config.no_gpu else GpuCollector(config.gpu_index)

    def once() -> Snapshot:
        from .state import GpuSnapshot

        gsnap = gpu.poll() if gpu is not None else GpuSnapshot(available=False)
        return Snapshot(monotonic=time.monotonic(), vllm=vllm.poll(), gpu=gsnap)

    first = once()
    time.sleep(min(config.interval, 2.0))
    second = once()
    if gpu is not None:
        gpu.close()
    return first, second


def dump_json(config: AppConfig) -> int:
    first, second = collect_pair(config)
    history = History(config.history_len)
    history.update(first)
    history.update(second)

    out = {
        "url": config.url,
        "reachable": second.vllm.reachable,
        "error": second.vllm.error,
        "model_name": second.vllm.model_name,
        "gpu_available": second.gpu.available,
        "gpu_error": second.gpu.error,
        "derived": history.derived,
        "raw_vllm": {
            "num_requests_running": second.vllm.num_requests_running,
            "num_requests_waiting": second.vllm.num_requests_waiting,
            "kv_cache_usage_perc": second.vllm.kv_cache_usage_perc,
            "generation_tokens_total": second.vllm.generation_tokens_total,
            "prompt_tokens_total": second.vllm.prompt_tokens_total,
            "num_preemptions_total": second.vllm.num_preemptions_total,
            "prefix_cache_hit_rate": second.vllm.prefix_cache_hit_rate,
        },
        "raw_gpu": {
            "name": second.gpu.name,
            "util_gpu": second.gpu.util_gpu,
            "mem_used": second.gpu.mem_used,
            "mem_total": second.gpu.mem_total,
            "mem_used_perc": second.gpu.mem_used_perc,
            "temperature": second.gpu.temperature,
            "power_usage": second.gpu.power_usage,
            "power_limit": second.gpu.power_limit,
            "sm_clock": second.gpu.sm_clock,
            "fan_speed": second.gpu.fan_speed,
        },
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    config = _config_from_args(args)

    if args.dump_json:
        return dump_json(config)

    # Import lazily so --dump-json works in environments without curses.
    from .ui.app import App

    return App(config).run()


if __name__ == "__main__":
    raise SystemExit(main())
