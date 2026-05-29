import math

from vllmtop.config import HISTORY_LEN
from vllmtop.state import (
    GpuSnapshot,
    Histogram,
    History,
    Series,
    Snapshot,
    VllmSnapshot,
    compute_rate,
    histogram_quantile,
    histogram_recent_avg,
)
from vllmtop.ui.widgets import braille_chart


def test_compute_rate_basic():
    assert compute_rate(100.0, 10.0, 200.0, 12.0) == 50.0


def test_compute_rate_guards():
    # Non-positive dt -> 0.
    assert compute_rate(100.0, 10.0, 200.0, 10.0) == 0.0
    # Counter reset (value decreased) -> 0, not negative.
    assert compute_rate(200.0, 10.0, 50.0, 12.0) == 0.0


def test_histogram_recent_avg():
    prev = Histogram(count=10.0, sum=5.0)
    cur = Histogram(count=14.0, sum=13.0)
    # (13-5)/(14-10) = 2.0
    assert histogram_recent_avg(prev, cur) == 2.0
    # No new observations -> 0.
    assert histogram_recent_avg(cur, cur) == 0.0


def test_histogram_quantile():
    # Buckets le=1 -> 0, le=2 -> 0, le=5 -> 10 new obs in window.
    prev = Histogram(count=0.0, buckets={1.0: 0.0, 2.0: 0.0, 5.0: 0.0, math.inf: 0.0})
    cur = Histogram(count=10.0, buckets={1.0: 0.0, 2.0: 0.0, 5.0: 10.0, math.inf: 10.0})
    q = histogram_quantile(prev, cur, 0.5)
    # All mass in (2, 5]; median interpolates inside that bucket.
    assert 2.0 <= q <= 5.0


def test_series_ring_buffer():
    s = Series(maxlen=3)
    for i in range(5):
        s.append(i)
    assert s.values() == [2, 3, 4]
    assert s.last == 4


def _snap(t, gen_total, running, gpu_util):
    return Snapshot(
        monotonic=t,
        vllm=VllmSnapshot(
            reachable=True,
            generation_tokens_total=gen_total,
            num_requests_running=running,
        ),
        gpu=GpuSnapshot(available=True, util_gpu=gpu_util),
    )


def test_history_rate_from_two_samples():
    h = History(HISTORY_LEN)
    h.update(_snap(0.0, 1000.0, 1, 10.0))
    h.update(_snap(2.0, 1200.0, 2, 20.0))
    # (1200-1000)/2 = 100 tok/s
    assert h.derived["gen_tok_s"] == 100.0
    assert h.derived["running"] == 2.0
    assert h.derived["gpu_util"] == 20.0
    assert h.series["gen_tok_s"].values() == [0.0, 100.0]


def test_braille_chart_shape():
    series = [math.sin(i / 5.0) for i in range(100)]
    w, height = 20, 4
    rows = braille_chart(series, w, height, 0.0, 1.0)
    assert len(rows) == height
    for row in rows:
        assert len(row) == w
        # Every cell is a braille glyph in the U+2800 block.
        assert all(0x2800 <= ord(ch) <= 0x28FF for ch in row)
