import math

import pytest

from vllmpytop.config import HISTORY_LEN
from vllmpytop.state import (
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
from vllmpytop.ui.widgets import braille_chart, fmt_duration, stacked_chart_down


def test_braille_chart_flip():
    # A quarter-height value lights one dot-row: the bottom one normally, the
    # top one when flipped (for the lower half of a mirrored chart).
    kw = dict(width=1, height=1, vmin=0.0, vmax=1.0, baseline=False)
    normal = braille_chart([0.25], **kw)[0]
    flipped = braille_chart([0.25], flip=True, **kw)[0]
    assert normal != flipped
    assert ord(normal) - 0x2800 == 0x80  # bottom dot of the right column
    assert ord(flipped) - 0x2800 == 0x08  # top dot of the right column


def test_stacked_chart_down_bands():
    # near dominates the cell -> band 0; far dominates -> band 1; empty -> -1.
    g = stacked_chart_down([1.0], [0.0], width=1, height=1, vmax=1.0)
    assert g[0][0][1] == 0
    g = stacked_chart_down([0.0], [1.0], width=1, height=1, vmax=1.0)
    assert g[0][0][1] == 1
    g = stacked_chart_down([], [], width=1, height=1, vmax=1.0)
    assert g[0][0] == (" ", -1)


def test_fmt_duration():
    assert fmt_duration(45) == "45s"
    assert fmt_duration(12 * 60) == "12m"
    assert fmt_duration(3 * 3600 + 20 * 60) == "3h 20m"
    assert fmt_duration(2 * 86400 + 4 * 3600) == "2d 4h"
    assert fmt_duration(-1) == "—"
    assert fmt_duration(float("inf")) == "—"


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


def test_window_avg_seeds_then_converges():
    # The first sample seeds all three windows to the current value (no cold
    # ramp from zero), then each EMA decays toward the new value at its own
    # rate: 1m converges fastest, 15m slowest.
    h = History(HISTORY_LEN)
    h.update(_snap(0.0, 0.0, 1, 50.0))
    assert h.avg["gpu_util"] == [50.0, 50.0, 50.0]
    for i in range(1, 400):
        h.update(_snap(float(i), 0.0, 1, 80.0))
    one, five, fifteen = h.avg["gpu_util"]
    # Headed from 50 toward 80; shorter windows are further along.
    assert one > five > fifteen > 50.0
    assert one == pytest.approx(80.0, abs=0.5)


def test_braille_chart_shape():
    series = [math.sin(i / 5.0) for i in range(100)]
    w, height = 20, 4
    rows = braille_chart(series, w, height, 0.0, 1.0)
    assert len(rows) == height
    for row in rows:
        assert len(row) == w
        # Every cell is a braille glyph in the U+2800 block.
        assert all(0x2800 <= ord(ch) <= 0x28FF for ch in row)
