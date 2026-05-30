import math
from pathlib import Path

from vllmpytop.collectors.vllm import parse_metrics

FIXTURE = Path(__file__).parent / "metrics_fixture.txt"


def _snap():
    return parse_metrics(FIXTURE.read_text())


def test_reachable_and_model():
    snap = _snap()
    assert snap.reachable is True
    assert snap.model_name == "Qwen/Qwen3.6-35B-A3B"


def test_scalar_values():
    snap = _snap()
    assert snap.num_requests_running == 0.0
    assert snap.num_requests_waiting == 0.0
    assert snap.kv_cache_usage_perc == 0.0
    # 5.7351109e+07 from the fixture.
    assert snap.prompt_tokens_total == 57351109.0
    assert snap.generation_tokens_total == 1799939.0
    assert snap.num_preemptions_total == 0.0


def test_histograms_parsed():
    snap = _snap()
    # Histograms exist with a +Inf bucket and consistent count.
    for hist in (snap.ttft, snap.inter_token, snap.e2e, snap.queue_time):
        assert math.inf in hist.buckets
        # The +Inf bucket count equals the total observation count.
        assert hist.buckets[math.inf] == hist.count


def test_prefix_cache_hit_rate_guarded():
    snap = _snap()
    # queries_total is 0 in the fixture -> guarded to 0, no ZeroDivisionError.
    assert snap.prefix_cache_hit_rate == 0.0


def test_engine_info_parsed():
    snap = _snap()
    # Pulled from the labels on vllm:cache_config_info.
    assert snap.cache_dtype == "fp8"
    assert snap.block_size == "16"
    assert snap.num_gpu_blocks == "144"
    assert snap.gpu_memory_utilization == "0.88"
    assert snap.enable_prefix_caching is False
    # process_start_time_seconds gauge from the stdlib process collector.
    assert snap.process_start_time == 1.77995408251e09
    # engine_sleep_state{sleep_state="awake"} == 1.0
    assert snap.engine_awake is True
    # request_success_total summed across finished_reason labels (12218 + 0s).
    assert snap.request_success_total == 12218.0
