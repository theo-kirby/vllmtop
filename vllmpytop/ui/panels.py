"""The four bordered panels, btop-style. Each draws into a Rect on screen."""

from __future__ import annotations

import curses
import time
from typing import List, Sequence, Tuple

from ..state import History, MergedLogEntry, Snapshot, VllmSnapshot
from .layout import Rect
from .theme import (
    PAIR_BOX_CACHE,
    PAIR_BOX_GPU,
    PAIR_BOX_LAT,
    PAIR_BOX_REQ,
    PAIR_BOX_THRU,
    PAIR_CYAN,
    PAIR_DIM,
    PAIR_DIV,
    PAIR_GREEN,
    PAIR_HI,
    PAIR_INACTIVE,
    PAIR_MAGENTA,
    PAIR_PINK,
    PAIR_PURPLE,
    PAIR_RED,
    PAIR_TITLE,
    PAIR_YELLOW,
    Theme,
)
from .widgets import (
    big_number,
    braille_chart,
    fmt_bytes,
    fmt_duration,
    fmt_seconds,
    hbar,
    log_headroom_scale,
    stacked_chart_down,
)

# btop box-drawing symbols.
_ROUND = {"lu": "╭", "ru": "╮", "ld": "╰", "rd": "╯"}
_H, _V = "─", "│"
_TITLE_L, _TITLE_R = "┐", "┌"  # frame the title on the top edge
_TITLE_LD, _TITLE_RD = "┘", "└"  # frame title2 on the bottom edge
_SUPERSCRIPT = ("⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹")

# Floor for the request-stack axis. Without a full-scale capacity (vLLM doesn't
# export max_num_seqs as a metric) the chart autoscales to the windowed max — so
# a steady "1 running" collapses the axis to 1 and fills the whole chart. Anchor
# the axis here so a low, steady count reads as a low line with room to climb;
# real bursts past this still autoscale the axis up. Tune to taste.
_REQ_STACK_FLOOR = 16.0


class Painter:
    """Thin wrapper over a curses window with edge-safe writes."""

    def __init__(self, win, theme: Theme) -> None:
        self.win = win
        self.theme = theme

    def text(self, y: int, x: int, s: str, attr: int = 0) -> None:
        if y < 0 or x < 0:
            return
        try:
            self.win.addstr(y, x, s, attr)
        except curses.error:
            # Writing to the last cell raises; ignore clipped output.
            pass

    def box(self, rect: Rect, title: str, num: int = 0,
            title_pair: int = PAIR_TITLE, title2: str = "",
            border_pair: int = PAIR_DIV) -> Rect:
        """Draw a btop-style rounded box and return the inner content Rect.

        ``num`` (1-9) is drawn as a superscript before the title, marking the
        key that toggles the panel. ``title2`` is shown as a tab on the bottom
        edge. ``border_pair`` colors the outline and title dividers (btop tints
        each box differently). Returns the inner content Rect.
        """
        t = self.theme
        y, x, h, w = rect.y, rect.x, rect.h, rect.w
        if h < 2 or w < 2:
            return Rect(y, x, 0, 0)
        border = t.attr(border_pair)

        # Edges and rounded corners.
        line = _H * (w - 2)
        self.text(y, x, _ROUND["lu"] + line + _ROUND["ru"], border)
        self.text(y + h - 1, x, _ROUND["ld"] + line + _ROUND["rd"], border)
        for i in range(1, h - 1):
            self.text(y + i, x, _V, border)
            self.text(y + i, x + w - 1, _V, border)

        self._title_tab(y, x, num, title, title_pair, border,
                        _TITLE_L, _TITLE_R)
        if title2:
            self._title_tab(y + h - 1, x, 0, title2, title_pair, border,
                            _TITLE_LD, _TITLE_RD)
        return Rect(y + 1, x + 1, h - 2, w - 2)

    def _title_tab(self, y: int, x: int, num: int, title: str, title_pair: int,
                   border: int, left: str, right: str) -> None:
        t = self.theme
        cx = x + 2
        self.text(y, cx, left, border)
        cx += 1
        if num:
            sup = _SUPERSCRIPT[max(0, min(9, num))]
            self.text(y, cx, sup, t.attr(PAIR_HI, bold=True))
            cx += 1
        label = f" {title} "
        self.text(y, cx, label, t.attr(title_pair, bold=True))
        cx += len(label)
        self.text(y, cx, right, border)


def _draw_chart(p: Painter, inner: Rect, series: Sequence[float],
                vmin: float, vmax: float, pair: int,
                gradient: bool = False) -> None:
    if inner.h <= 0 or inner.w <= 0:
        return
    rows = braille_chart(series, inner.w, inner.h, vmin, vmax)
    n = len(rows)
    flat = p.theme.attr(pair)
    for i, row in enumerate(rows):
        # btop colors a graph by vertical position: bottom row ~green, top ~red,
        # independent of the data. Row 0 is the top, so its fraction is highest.
        attr = p.theme.grad_attr((n - i) / n) if gradient and n else flat
        p.text(inner.y + i, inner.x, row, attr)


def _draw_dual_chart(p: Painter, rect: Rect, util: Sequence[float],
                     running: Sequence[float], waiting: Sequence[float],
                     util_now: float, run_now: float, wait_now: float) -> None:
    """A btop-style mirrored chart split at a shared centre line.

    GPU utilisation grows up from the centre (positional green->red gradient,
    like the other graphs). Below it, the request count grows down as a stacked
    two-band chart: ``running`` (green) nearest the centre, ``waiting``
    (magenta) stacked beyond. Corner labels name each half.
    """
    if rect.h <= 0 or rect.w <= 0:
        return
    t = p.theme
    top_h = max(1, rect.h // 2)
    bot_h = rect.h - top_h

    # Log-headroom scale: GPU util sits at 0 (idle) or 85-95% (busy), so a
    # linear axis crushes the busy variation into a thin strip at the top.
    # Scaling the headroom spreads 85-95% across the upper chart while idle
    # still reads as a thin floor.
    top_rows = braille_chart(
        log_headroom_scale(util, 100.0), rect.w, top_h, 0.0, 1.0
    )
    n = len(top_rows)
    for i, row in enumerate(top_rows):
        attr = t.grad_attr((n - i) / n) if n else t.attr(0)
        p.text(rect.y + i, rect.x, row, attr)

    if bot_h > 0:
        stack_vmax = max(
            (r + w for r, w in zip(running, waiting)), default=0.0
        )
        grid = stacked_chart_down(running, waiting, rect.w, bot_h,
                                  max(stack_vmax, _REQ_STACK_FLOOR), log=True)
        band_pair = (PAIR_GREEN, PAIR_MAGENTA)
        for r, cols in enumerate(grid):
            for c, (glyph, band) in enumerate(cols):
                if band < 0:
                    continue
                p.text(rect.y + top_h + r, rect.x + c, glyph,
                       t.attr(band_pair[band]))

    # Corner labels so each half is self-explanatory (colour-coded by series).
    p.text(rect.y, rect.x, f" gpu {util_now:.0f}% ", t.attr(PAIR_DIM, dim=True))
    if bot_h > 0:
        y = rect.y + rect.h - 1
        run_lbl = f" run {run_now:.0f} "
        p.text(y, rect.x, run_lbl, t.attr(PAIR_GREEN))
        p.text(y, rect.x + len(run_lbl), f"wait {wait_now:.0f} ",
               t.attr(PAIR_MAGENTA))


def _draw_meter(p: Painter, y: int, x: int, w: int, pct: float) -> None:
    """A btop-style horizontal meter: cells fade green->yellow->red along the
    bar, with each cell colored by its absolute position (0-100%). Only the
    filled portion is colored; the remainder is a dim track."""
    if w <= 0:
        return
    t = p.theme
    pct = max(0.0, min(100.0, pct))
    for i in range(1, w + 1):
        ypct = round(i * 100.0 / w)
        if pct >= ypct:
            p.text(y, x + i - 1, "█", t.grad_attr(ypct / 100.0))
        else:
            p.text(y, x + i - 1, "░" * (w - i + 1), t.attr(PAIR_INACTIVE, dim=True))
            break


def _as_pct(value: str) -> str:
    """Render a fractional config value (e.g. "0.88") as a percent."""
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return value


def _rule(label: str, width: int) -> str:
    """A faint labelled separator, e.g. ``┄ model ┄┄┄┄┄``."""
    if width <= 0:
        return ""
    return f"┄ {label} ".ljust(width, "┄")[:width]


def _vllm_lines(v: VllmSnapshot) -> List[Tuple[str, int]]:
    """Compact ``(text, color-pair)`` lines summarising the vLLM engine.

    Folded into the GPU panel's stats column beneath a divider, so each line is
    a single pre-joined string sized to be readable in a narrow column.
    """
    if not v.reachable:
        return [("vLLM " + (v.error or "unreachable"), PAIR_YELLOW)]

    out: List[Tuple[str, int]] = []
    model = v.model_name or "unknown model"
    if v.engine_awake is True:
        model = "● " + model  # awake
    elif v.engine_awake is False:
        model = "○ " + model  # sleeping
    out.append((model, PAIR_TITLE))

    line1 = []
    if v.process_start_time:
        line1.append("up " + fmt_duration(time.time() - v.process_start_time))
    if v.cache_dtype:
        line1.append(v.cache_dtype)
    if v.request_success_total:
        line1.append(big_number(v.request_success_total) + " srv")
    if line1:
        out.append(("  ".join(line1), PAIR_DIM))

    line2 = []
    if v.enable_prefix_caching is not None:
        line2.append("prefix " + ("on" if v.enable_prefix_caching else "off"))
    if v.num_gpu_blocks:
        line2.append(v.num_gpu_blocks + " blk")
    if v.gpu_memory_utilization:
        line2.append(_as_pct(v.gpu_memory_utilization) + " tgt")
    if line2:
        out.append(("  ".join(line2), PAIR_DIM))
    return out


def _draw_vllm_text_rows(p: Painter, y0: int, h: int, x: int, w: int,
                         v: VllmSnapshot, theme) -> None:
    """Render compact vLLM model/engine info rows (full width, no chart).

    Used as a fallback when the GPU side is unavailable, filling the bottom
    half of the gpu panel with vLLM details alone.
    """
    if h <= 0 or w <= 0:
        return
    lines = _vllm_lines(v)
    for i, (text, pair) in enumerate(lines):
        if i >= h:
            break
        p.text(y0 + i, x, text[:w],
               theme.attr(pair, bold=pair == PAIR_TITLE))


def draw_gpu(p: Painter, rect: Rect, snap: Snapshot, hist: History,
             num: int = 0) -> None:
    """GPU panel — chart on the left, info column on the right.

    The info column is split horizontally: GPU bars on top, vLLM bars on
    bottom, separated by a ``┄ vllm ┄`` rule (perf-style split).
    """
    t = p.theme
    inner = p.box(rect, "gpu", num, title2="vllm", border_pair=PAIR_BOX_GPU)
    g = snap.gpu
    v = snap.vllm
    if inner.h <= 0 or inner.w <= 0:
        return

    if not g.available:
        msg = "GPU unavailable" + (f": {g.error}" if g.error else "")
        p.text(inner.y, inner.x, msg[: inner.w], t.attr(PAIR_DIM, dim=True))
        # Still show vLLM info in the info column if there's room.
        info_w = min(42, max(28, inner.w * 2 // 5))
        chart_w = inner.w - info_w - 1
        if info_w >= 16 and v.reachable:
            ix = inner.x + chart_w + 3
            iw = inner.w - chart_w - 2
            # Vertical divider between chart area and info.
            for i in range(inner.h):
                p.text(inner.y + i, inner.x + chart_w, _V, t.attr(PAIR_DIV))
            mid_y = inner.y + inner.h // 2
            p.text(mid_y - 1, ix, _rule("vllm", iw), t.attr(PAIR_DIV))
            _draw_vllm_text_rows(p, mid_y, inner.h - mid_y,
                                  ix, iw, v, t)
        return

    util_series = hist.series["gpu_util"].values()
    running_series = hist.series["running"].values()
    waiting_series = hist.series["waiting"].values()

    # Right column width (info), same as the original layout.
    info_w = min(42, max(28, inner.w * 2 // 5))
    chart_w = inner.w - info_w - 1  # 1 for the vertical divider
    if chart_w < 8:  # too narrow; chart takes the whole panel
        _draw_chart(p, inner, log_headroom_scale(util_series, 100.0),
                    0.0, 1.0, 0, gradient=True)
        return

    # Left: full-height mirrored dual chart.
    _draw_dual_chart(
        p, Rect(inner.y, inner.x, inner.h, chart_w),
        util=util_series, running=running_series, waiting=waiting_series,
        util_now=g.util_gpu, run_now=v.num_requests_running,
        wait_now=v.num_requests_waiting,
    )

    # Vertical divider between chart and info column.
    div_x = inner.x + chart_w
    for i in range(inner.h):
        p.text(inner.y + i, div_x, _V, t.attr(PAIR_DIV))
    rx = div_x + 2
    rw = inner.w - chart_w - 2

    # --- Info column (right side), split horizontally ---
    # Reserve 2 rows at the top for GPU name + bar, 2 at the bottom for
    # vLLM, and one row for the divider in between.
    mid = inner.h // 2  # divider row

    def bar_row(y: int, label: str, pct: float) -> None:
        p.text(y, rx, label, t.attr(PAIR_DIM))
        bw = rw - 10  # 5 for label, 5 for trailing " NN%"
        if bw <= 0:
            return
        bx = rx + 5
        _draw_meter(p, y, bx, bw, pct)
        p.text(y, bx + bw + 1, f"{pct:3.0f}%",
               t.grad_attr(pct / 100.0, bold=True))

    def text_row(y: int, value: str, pair: int = PAIR_DIM) -> None:
        p.text(y, rx, value[:rw], t.attr(pair))

    # GPU rows (top half, rows 0 .. mid-1).
    temp_pair = t.threshold(g.temperature, 70, 85)
    pwr_frac = (100.0 * g.power_usage / g.power_limit
                if g.power_limit > 0 else 0.0)

    gpu_rows = [
        lambda y: text_row(y, g.name or "GPU", PAIR_TITLE),
        lambda y: bar_row(y, "util ", g.util_gpu),
        lambda y: bar_row(y, "vram ", g.mem_used_perc),
        lambda y: bar_row(y, "pwr  ", pwr_frac),
        lambda y: text_row(y,
                           f"{g.temperature:.0f}°C  fan {g.fan_speed:.0f}%",
                           temp_pair),
    ]

    for i, draw in enumerate(gpu_rows):
        y = inner.y + i
        if y >= inner.y + mid:
            break
        draw(y)

    # Divider row: ``┄ vllm ┄┄┄`` spanning the info column.
    p.text(inner.y + mid, rx, _rule("vllm", rw), t.attr(PAIR_DIV))

    # vLLM rows (bottom half, rows mid+1 .. end).
    model = v.model_name or "unknown model"
    if v.engine_awake is True:
        model = "● " + model
    elif v.engine_awake is False:
        model = "○ " + model

    running_max = max(max(running_series, default=0.0),
                      max(waiting_series, default=0.0), 1.0)
    run_pct = 100.0 * v.num_requests_running / running_max
    wait_pct = 100.0 * v.num_requests_waiting / running_max
    kv_pct = v.kv_cache_usage_perc * 100.0

    def draw_wbar(y: int) -> None:
        p.text(y, rx, "wait ", t.attr(PAIR_DIM))
        _draw_meter(p, y, rx + 5, max(1, rw - 10), wait_pct)
        wpair = PAIR_MAGENTA if v.num_requests_waiting > 0 else PAIR_DIM
        p.text(y, rx + rw - 5, f"{wait_pct:3.0f}%",
               t.attr(wpair, bold=True))

    vllm_rows = [
        lambda y: text_row(y, model, PAIR_TITLE),
        lambda y: bar_row(y, "run  ", run_pct),
        draw_wbar,
        lambda y: bar_row(y, "kv   ", kv_pct),
    ]

    for i, draw in enumerate(vllm_rows):
        y = inner.y + mid + 1 + i
        if y >= inner.y + inner.h:
            break
        draw(y)


def _draw_mirror_chart(p: Painter, rect: Rect, top: Sequence[float],
                       top_label: str, bottom: Sequence[float],
                       bottom_label: str) -> None:
    """A btop-style mirrored chart of two single series sharing a centre line.

    ``top`` grows up from the centre, ``bottom`` grows down from it; each half
    scales to its own max (the series can differ wildly in magnitude). Each half
    is coloured like a btop network graph — ``top`` with the download gradient,
    ``bottom`` with the upload gradient — fading dark at the centre baseline to
    bright at the peak. The current value of each is a bold white corner label.
    """
    if rect.h <= 0 or rect.w <= 0:
        return
    t = p.theme
    top_h = max(1, rect.h // 2)
    bot_h = rect.h - top_h

    top_vmax = max(max(top, default=0.0), 1e-9)
    top_rows = braille_chart(top, rect.w, top_h, 0.0, top_vmax)
    n = len(top_rows)
    for i, row in enumerate(top_rows):
        # Top row (i=0) is the peak; the centre baseline is dimmest.
        attr = t.net_attr((n - i) / n, up=False) if n else t.attr(0)
        p.text(rect.y + i, rect.x, row, attr)

    if bot_h > 0:
        bot_vmax = max(max(bottom, default=0.0), 1e-9)
        bot_rows = braille_chart(bottom, rect.w, bot_h, 0.0, bot_vmax, flip=True)
        m = len(bot_rows)
        for i, row in enumerate(bot_rows):
            # Grows downward, so the peak is the last (bottom) row.
            attr = t.net_attr((i + 1) / m, up=True) if m else t.attr(0)
            p.text(rect.y + top_h + i, rect.x, row, attr)

    p.text(rect.y, rect.x, f" {top_label} ", t.attr(PAIR_PURPLE, bold=True))
    if bot_h > 0:
        p.text(rect.y + rect.h - 1, rect.x, f" {bottom_label} ",
               t.attr(PAIR_PINK, bold=True))


def _draw_stat_column(p: Painter, y0: int, h: int, rx: int, rw: int,
                      rows: Sequence[Tuple[str, int, str]]) -> None:
    """Render a btop mem/disks-style stats column: each row is a white label on
    the left and a colour-coded value right-aligned within ``rw``. Rows past the
    available height ``h`` are skipped."""
    t = p.theme
    for i, (label, vpair, value) in enumerate(rows):
        if i >= h:
            break
        y = y0 + i
        p.text(y, rx, label[:rw], t.attr(PAIR_DIM))
        vs = value[: max(0, rw)]
        p.text(y, rx + max(0, rw - len(vs)), vs, t.attr(vpair, bold=True))


def draw_throughput(p: Painter, rect: Rect, snap: Snapshot, hist: History,
                    num: int = 0) -> None:
    t = p.theme
    inner = p.box(rect, "throughput", num, title2="tok/s",
                  border_pair=PAIR_BOX_THRU)
    if inner.h <= 0:
        return

    gen = hist.series["gen_tok_s"].values()
    prompt = hist.series["prompt_tok_s"].values()
    gen_now = hist.derived["gen_tok_s"]
    prompt_now = hist.derived["prompt_tok_s"]

    # btop net-style split: the mirrored gradient chart on the left, a stats
    # column (white labels) on the right. Generation tok/s grows up from the
    # centre line, prompt (prefill) tok/s grows down from it.
    right_w = min(24, max(18, inner.w // 3))
    chart_w = inner.w - right_w - 1
    if chart_w < 8:  # too narrow to split; chart takes the whole panel
        _draw_mirror_chart(
            p, inner,
            top=gen, top_label="gen " + big_number(gen_now, " tok/s"),
            bottom=prompt,
            bottom_label="prompt " + big_number(prompt_now, " tok/s"),
        )
        return

    _draw_mirror_chart(
        p, Rect(inner.y, inner.x, inner.h, chart_w),
        top=gen, top_label="", bottom=prompt, bottom_label="",
    )

    div_x = inner.x + chart_w
    for i in range(inner.h):
        p.text(inner.y + i, div_x, _V, t.attr(PAIR_DIV))
    rx = div_x + 2
    rw = inner.w - chart_w - 2

    _draw_stat_column(p, inner.y, inner.h, rx, rw, [
        ("gen", PAIR_DIM, big_number(gen_now, " tok/s")),
        ("gen pk", PAIR_DIM, big_number(max(gen, default=0.0), " tok/s")),
        ("prompt", PAIR_DIM, big_number(prompt_now, " tok/s")),
        ("prm pk", PAIR_DIM, big_number(max(prompt, default=0.0), " tok/s")),
    ])


def draw_requests(p: Painter, rect: Rect, snap: Snapshot, hist: History,
                  num: int = 0) -> None:
    """Request feed panel — live list of vLLM inference requests.

    Each row shows request age, prompt text (truncated), request ID, and
    max_tokens. Prompt text requires vLLM ≥ 0.11.3 with ``--enable-log-requests``.
    Without a log source, a hint tells you how to enable the feed. Newest first.
    """
    t = p.theme
    inner = p.box(rect, "requests", num, title2="activity",
                  border_pair=PAIR_BOX_REQ)
    if inner.h <= 0:
        return

    p.text(inner.y, inner.x, _H * inner.w, t.attr(PAIR_DIV))
    feed_rect = Rect(inner.y + 1, inner.x, inner.h - 1, inner.w)
    _draw_request_feed(p, feed_rect, snap.merged_log, snap.access_error)


def _draw_perf_stacked(p: Painter, inner: Rect, hist: History,
                       metrics: Sequence[Tuple[str, str, int]],
                       kv: float, hit: float) -> None:
    """Narrow-terminal fallback: latency rows with inline sparklines, then the
    KV-cache chart below — the pre-split single-column layout."""
    t = p.theme
    for i, (label, key, pair) in enumerate(metrics):
        if i >= inner.h:
            break
        val = hist.derived[key]
        p.text(inner.y + i, inner.x, f"{label:<5}", t.attr(PAIR_DIM))
        p.text(inner.y + i, inner.x + 6, f"{fmt_seconds(val):>7}",
               t.attr(pair, bold=True))
        spark_x = inner.x + 14
        spark_w = inner.w - 14
        if spark_w > 2:
            series = hist.series[key].values()
            vmax = max(max(series, default=0.0), 1e-6)
            rowsb = braille_chart(series, spark_w, 1, 0.0, vmax)
            if rowsb:
                p.text(inner.y + i, spark_x, rowsb[0], t.attr(pair))

    dy = inner.y + len(metrics)
    if dy >= inner.y + inner.h:
        return
    p.text(dy, inner.x, _rule(f"kv {kv:.0f}%  prefix {hit:.0f}%", inner.w),
           t.attr(PAIR_DIV))
    chart_y = dy + 1
    if chart_y < inner.y + inner.h:
        _draw_chart(p, Rect(chart_y, inner.x, inner.y + inner.h - chart_y,
                            inner.w),
                    hist.series["kv_cache"].values(), 0.0, 100.0, 0,
                    gradient=True)


def draw_perf(p: Painter, rect: Rect, snap: Snapshot, hist: History,
              num: int = 0) -> None:
    """Latency metrics (recent avg) beside the KV-cache usage chart,
    with per-request phase-timing sparklines below a divider."""
    t = p.theme
    inner = p.box(rect, "perf", num, title2="recent avg",
                  border_pair=PAIR_BOX_LAT)
    if inner.h <= 0:
        return

    metrics = [
        ("TTFT", "ttft", PAIR_CYAN),
        ("TPOT", "tpot", PAIR_GREEN),
        ("e2e", "e2e", PAIR_MAGENTA),
        ("queue", "queue_time", PAIR_DIM),
    ]

    # btop mem/disks-style split: ALL the graph lines fill the left side, ALL
    # the text labels + values sit in a narrow column on the right.
    right_w = min(18, max(12, inner.w // 4))
    chart_w = inner.w - right_w - 1
    if chart_w < 8:  # too narrow to split; fall back to the stacked layout
        _draw_perf_stacked(p, inner, hist, metrics,
                           hist.derived["kv_cache"],
                           snap.vllm.prefix_cache_hit_rate * 100.0)
        return

    div_x = inner.x + chart_w
    for i in range(inner.h):
        p.text(inner.y + i, div_x, _V, t.attr(PAIR_DIV))
    rx = div_x + 2
    rw = inner.w - chart_w - 2

    # Left: one colour-coded sparkline per latency metric (one row each), then
    # the kv-cache gradient chart fills whatever height remains.
    for i, (label, key, pair) in enumerate(metrics):
        if i >= inner.h:
            break
        series = hist.series[key].values()
        vmax = max(max(series, default=0.0), 1e-6)
        rowsb = braille_chart(series, chart_w, 1, 0.0, vmax)
        if rowsb:
            p.text(inner.y + i, inner.x, rowsb[0], t.attr(pair))

    # Per-request section: separator after latency metrics, then sparklines.
    pr_y = inner.y + len(metrics) + 1  # separator row after latency sparklines
    has_pr = pr_y < inner.y + inner.h

    if has_pr:
        p.text(pr_y, inner.x, _rule("per-request", chart_w), t.attr(PAIR_DIV))

    pr_metrics = [
        ("p-tok", "req_prompt_tok", PAIR_CYAN),
        ("g-tok", "req_gen_tok", PAIR_GREEN),
        ("prefill", "req_prefill", PAIR_YELLOW),
        ("decode", "req_decode", PAIR_PURPLE),
    ]
    pr_text_rows = []
    pr_start = pr_y + 1 if has_pr else inner.y + inner.h  # no-op if no room
    for i, (label, key, pair) in enumerate(pr_metrics):
        row_y = pr_start + i
        if row_y >= inner.y + inner.h:
            break
        series = hist.series[key].values()
        vmax = max(max(series, default=0.0), 1e-6)
        rowsb = braille_chart(series, chart_w, 1, 0.0, vmax)
        if rowsb:
            p.text(row_y, inner.x, rowsb[0], t.attr(pair))
        val = hist.derived[key]
        if "tok" in key:
            pr_text_rows.append((label, PAIR_DIM, f"{val:.0f}"))
        else:
            pr_text_rows.append((label, PAIR_DIM, fmt_seconds(val)))

    # Right: all text rows, aligned with the lines on the left.
    rows = [(label, PAIR_DIM, fmt_seconds(hist.derived[key]))
            for label, key, _ in metrics]
    rows.extend(pr_text_rows)
    _draw_stat_column(p, inner.y, inner.h, rx, rw, rows)


# Metrics shown in the 1·5·15 windowed-average panel:
# (label, series key, is_latency). Latencies render with fmt_seconds; the rest
# with big_number.
_LOADAVG_ROWS = [
    ("gen tok/s", "gen_tok_s", False),
    ("prompt tok/s", "prompt_tok_s", False),
    ("running", "running", False),
    ("waiting", "waiting", False),
    ("gpu util", "gpu_util", False),
    ("gpu mem", "gpu_mem_perc", False),
    ("kv cache", "kv_cache", False),
    ("ttft", "ttft", True),
    ("tpot", "tpot", True),
    ("e2e", "e2e", True),
    ("queue", "queue_time", True),
]


def draw_loadavg(p: Painter, rect: Rect, snap: Snapshot, hist: History,
                 num: int = 0) -> None:
    """1·5·15 panel — load-average style windowed means of the key metrics.

    Each row shows the current value beside its 1-, 5- and 15-minute
    exponential moving averages (``History.avg``), so a glance tells you whether
    load is rising or falling across each horizon. The "now" column is bold; the
    window columns are dim, matching btop's "current vs. history" emphasis.
    """
    t = p.theme
    inner = p.box(rect, "1·5·15", num, title2="windowed avg",
                  border_pair=PAIR_BOX_CACHE)
    if inner.h <= 0 or inner.w <= 0:
        return

    # Five columns: a flexible label, then now / 1m / 5m / 15m right-aligned.
    num_w = max(6, min(9, inner.w // 6))
    label_w = inner.w - num_w * 4
    if label_w < 8:
        num_w = max(5, (inner.w - 8) // 4)
        label_w = inner.w - num_w * 4
    if label_w < 6 or num_w < 5:
        return  # too narrow for the table

    def cell(y: int, col_i: int, s: str, pair: int, bold: bool) -> None:
        s = s[:num_w]
        cx = inner.x + label_w + col_i * num_w
        p.text(y, cx + num_w - len(s), s, t.attr(pair, bold=bold))

    # Header row.
    p.text(inner.y, inner.x, "metric"[:label_w], t.attr(PAIR_DIM, dim=True))
    for i, h in enumerate(("now", "1m", "5m", "15m")):
        cell(inner.y, i, h, PAIR_DIM, False)

    for r, (label, key, is_lat) in enumerate(_LOADAVG_ROWS):
        y = inner.y + 1 + r
        if y >= inner.y + inner.h:
            break
        p.text(y, inner.x, label[:label_w], t.attr(PAIR_DIM))
        emas = hist.avg.get(key) or [0.0, 0.0, 0.0]
        values = (hist.derived.get(key, 0.0), emas[0], emas[1], emas[2])
        for i, v in enumerate(values):
            s = fmt_seconds(v) if is_lat else big_number(v)
            cell(y, i, s, PAIR_TITLE if i == 0 else PAIR_DIM, i == 0)


def _truncate_prompt(prompt: str, maxlen: int = 30) -> str:
    """Truncate a prompt for display, normalizing whitespace and adding … when needed.

    Prompts are end-user-supplied (they arrive via the vLLM request log), so we
    also strip control characters — ESC, BEL, C1, etc. — that ``str.split()``
    leaves behind and that could otherwise carry terminal escape sequences into
    the operator's terminal. curses neutralises these too, but stripping makes
    the safety explicit and protects any non-curses rendering path.
    """
    if not prompt:
        return ""
    # Normalize whitespace, then drop C0/C1 control chars (e.g. ESC 0x1b) so no
    # escape sequence can reach the terminal.
    text = " ".join(prompt.split())
    text = "".join(c for c in text if ord(c) >= 0x20 and not 0x7f <= ord(c) < 0xa0)
    if len(text) > maxlen:
        return text[:maxlen - 1] + "…"
    return text


def _fmt_chars(n: int) -> str:
    """Compact character count for the feed's size column: 42, 1.2k, 440k, 1.3M."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1000:.0f}k"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def _draw_request_feed(p: Painter, rect: Rect, entries, error=None) -> None:
    """Render the vLLM inference request feed into ``rect`` — no box.

    Each row is one request parsed from vLLM's request log
    (``--enable-log-requests``): its request id, logged prompt size (chars) and
    max_tokens. Newest first.

    Columns (left to right, right columns drop out on narrow panels):

        age  req id (flex)  size  max_tok

    ``age`` is how long ago the request was observed. ``req id`` flexes to fill
    the available width. ``size`` is the logged prompt length in characters
    (exact, not a token count), and ``max_tok`` is the output cap; both are
    fixed-width on the right and drop off as the panel narrows.
    """
    t = p.theme
    if rect.h <= 0 or rect.w <= 0:
        return
    if error:
        p.text(rect.y, rect.x, ("⚠ " + error)[: rect.w],
               t.attr(PAIR_YELLOW, dim=True))
        return
    if not entries:
        hint = "run vLLM with --enable-log-requests to show inference requests"
        p.text(rect.y, rect.x, hint[: rect.w], t.attr(PAIR_DIM, dim=True))
        return

    now = time.time()

    # Columns: age | req id (flex) | size | max_tok. The request id flexes to
    # fill the row; size and max_tok are fixed-width on the right and drop out
    # as the panel narrows.
    age_w, size_w, tok_w = 4, 6, 6
    x_age = rect.x
    x_req = x_age + age_w + 1
    show_tok = rect.w >= 20
    show_size = rect.w >= 28

    trailing = (size_w + 1 if show_size else 0) + (tok_w + 1 if show_tok else 0)
    req_w = max(4, rect.x + rect.w - x_req - trailing)
    x_size = x_req + req_w + 1
    x_tok = (x_size + size_w + 1) if show_size else (x_req + req_w + 1)

    hdr = t.attr(PAIR_DIM, dim=True)
    p.text(rect.y, x_age, "age", hdr)
    p.text(rect.y, x_req, "req id"[:req_w], hdr)
    if show_size:
        p.text(rect.y, x_size, "size".rjust(size_w)[:size_w], hdr)
    if show_tok:
        p.text(rect.y, x_tok, "max_tok"[:tok_w], hdr)

    for i, e in enumerate(entries):
        y = rect.y + 1 + i
        if y >= rect.y + rect.h:
            break
        p.text(y, x_age, fmt_duration(max(0.0, now - e.t)).rjust(age_w)[:age_w],
               t.attr(PAIR_DIM))
        rid = e.request_id[:req_w] if e.request_id else "—"
        p.text(y, x_req, rid.ljust(req_w)[:req_w],
               t.attr(PAIR_CYAN) if e.request_id else t.attr(PAIR_DIM, dim=True))
        if show_size:
            has_size = e.prompt_chars is not None
            size_val = _fmt_chars(e.prompt_chars) if has_size else "—"
            p.text(y, x_size, size_val.rjust(size_w)[:size_w],
                   t.attr(PAIR_GREEN) if has_size else t.attr(PAIR_DIM, dim=True))
        if show_tok:
            tok_val = (f"{e.max_tokens:>{tok_w}}" if e.max_tokens is not None
                       else "-".rjust(tok_w))
            p.text(y, x_tok, tok_val[:tok_w],
                   t.attr(PAIR_CYAN, bold=True) if e.max_tokens is not None
                   else t.attr(PAIR_DIM, dim=True))
