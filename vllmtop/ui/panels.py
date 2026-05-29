"""The five bordered panels, btop-style. Each draws into a Rect on screen."""

from __future__ import annotations

import curses
from typing import Sequence

from ..state import History, Snapshot
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
    PAIR_TITLE,
    Theme,
)
from .widgets import (
    big_number,
    braille_chart,
    fmt_bytes,
    fmt_seconds,
    hbar,
)

# btop box-drawing symbols.
_ROUND = {"lu": "╭", "ru": "╮", "ld": "╰", "rd": "╯"}
_H, _V = "─", "│"
_TITLE_L, _TITLE_R = "┐", "┌"  # frame the title on the top edge
_TITLE_LD, _TITLE_RD = "┘", "└"  # frame title2 on the bottom edge
_SUPERSCRIPT = ("⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹")


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


def draw_gpu(p: Painter, rect: Rect, snap: Snapshot, hist: History,
             num: int = 0) -> None:
    t = p.theme
    inner = p.box(rect, "gpu", num, border_pair=PAIR_BOX_GPU)
    g = snap.gpu
    if inner.h <= 0:
        return
    if not g.available:
        msg = "GPU unavailable" + (f": {g.error}" if g.error else "")
        p.text(inner.y, inner.x, msg[: inner.w], t.attr(PAIR_DIM, dim=True))
        return

    util_series = hist.series["gpu_util"].values()

    # btop GPU box: utilisation chart on the left, stats column on the right.
    right_w = min(34, max(22, inner.w // 3))
    chart_w = inner.w - right_w - 1
    if chart_w < 8:  # too narrow to split; chart takes the whole panel
        _draw_chart(p, inner, util_series, 0.0, 100.0, 0, gradient=True)
        return

    _draw_chart(p, Rect(inner.y, inner.x, inner.h, chart_w),
                util_series, 0.0, 100.0, 0, gradient=True)

    # Vertical divider, then the stats column to its right.
    div_x = inner.x + chart_w
    for i in range(inner.h):
        p.text(inner.y + i, div_x, _V, t.attr(PAIR_DIV))
    rx = div_x + 2
    rw = inner.w - chart_w - 2  # content width inside the stats column

    def bar_row(y: int, label: str, pct: float) -> None:
        p.text(y, rx, label, t.attr(PAIR_DIM))
        bw = rw - 10  # 5 for label, 5 for the trailing "  nn%"
        if bw <= 0:
            return
        bx = rx + 5
        _draw_meter(p, y, bx, bw, pct)
        p.text(y, bx + bw + 1, f"{pct:3.0f}%", t.grad_attr(pct / 100.0, bold=True))

    def text_row(y: int, label: str, value: str, pair: int = PAIR_DIM) -> None:
        p.text(y, rx, label, t.attr(PAIR_DIM))
        p.text(y, rx + 5, value[: max(0, rw - 5)], t.attr(pair))

    temp_pair = t.threshold(g.temperature, 70, 85)
    pwr_frac = 100.0 * g.power_usage / g.power_limit if g.power_limit > 0 else 0.0

    # Each entry draws one row; we render as many as the panel height allows.
    rows = [
        lambda y: p.text(y, rx, (g.name or "GPU")[:rw],
                         t.attr(PAIR_TITLE, bold=True)),
        lambda y: bar_row(y, "util ", g.util_gpu),
        lambda y: bar_row(y, "vram ", g.mem_used_perc),
        lambda y: bar_row(y, "pwr  ", pwr_frac),
        lambda y: text_row(y, "temp ",
                           f"{g.temperature:.0f}°C  fan {g.fan_speed:.0f}%",
                           temp_pair),
        lambda y: text_row(y, "watt ",
                           f"{g.power_usage:.0f}/{g.power_limit:.0f}W"),
        lambda y: text_row(y, "clk  ", f"SM {g.sm_clock:.0f}MHz"),
        lambda y: text_row(y, "mem  ",
                           f"{fmt_bytes(g.mem_used)}/{fmt_bytes(g.mem_total)}"),
    ]
    for i, draw in enumerate(rows):
        if i >= inner.h:
            break
        draw(inner.y + i)


def _panel_number(p: Painter, inner: Rect, label: str, value_text: str,
                  series: Sequence[float], vmax: float, pair: int) -> None:
    """A sub-panel: label + big number on one line, chart below."""
    t = p.theme
    p.text(inner.y, inner.x, label, t.attr(PAIR_DIM))
    p.text(inner.y, inner.x + len(label) + 1, value_text, t.attr(pair, bold=True))
    chart_h = max(1, inner.h - 1)
    vmax = max(vmax, 1e-9)
    _draw_chart(p, Rect(inner.y + 1, inner.x, chart_h, inner.w),
                series, 0.0, vmax, pair)


def draw_throughput(p: Painter, rect: Rect, snap: Snapshot, hist: History,
                    num: int = 0) -> None:
    inner = p.box(rect, "throughput", num, title2="tok/s",
                  border_pair=PAIR_BOX_THRU)
    if inner.h <= 0:
        return
    gen = hist.series["gen_tok_s"]
    prompt = hist.series["prompt_tok_s"]
    half = max(1, inner.h // 2)

    gmax = max(max(gen.values(), default=0.0), 1.0)
    _panel_number(
        p, Rect(inner.y, inner.x, half, inner.w),
        "gen", big_number(hist.derived["gen_tok_s"], " tok/s"),
        gen.values(), gmax, PAIR_GREEN,
    )
    pmax = max(max(prompt.values(), default=0.0), 1.0)
    _panel_number(
        p, Rect(inner.y + half, inner.x, inner.h - half, inner.w),
        "prompt", big_number(hist.derived["prompt_tok_s"], " tok/s"),
        prompt.values(), pmax, PAIR_CYAN,
    )


def draw_requests(p: Painter, rect: Rect, snap: Snapshot, hist: History,
                  num: int = 0) -> None:
    t = p.theme
    inner = p.box(rect, "requests", num, title2="per-req",
                  border_pair=PAIR_BOX_REQ)
    if inner.h <= 0:
        return
    v = snap.vllm
    running = hist.series["running"].values()
    waiting = hist.series["waiting"].values()
    scale = max(max(running, default=0.0), max(waiting, default=0.0), 1.0)

    bw = max(1, inner.w - 18)
    p.text(inner.y, inner.x, "running ", t.attr(PAIR_DIM))
    p.text(inner.y, inner.x + 8, hbar(v.num_requests_running, scale, bw),
           t.attr(PAIR_GREEN))
    p.text(inner.y, inner.x + 9 + bw, f"{v.num_requests_running:.0f}",
           t.attr(PAIR_GREEN, bold=True))

    if inner.h > 1:
        wpair = PAIR_MAGENTA if v.num_requests_waiting > 0 else PAIR_DIM
        p.text(inner.y + 1, inner.x, "waiting ", t.attr(PAIR_DIM))
        p.text(inner.y + 1, inner.x + 8, hbar(v.num_requests_waiting, scale, bw),
               t.attr(wpair))
        p.text(inner.y + 1, inner.x + 9 + bw, f"{v.num_requests_waiting:.0f}",
               t.attr(wpair, bold=True))

    # Per-request averages list (replaces the running chart). vLLM's /metrics
    # is aggregate-only, so these are recent-average sizes/timings per completed
    # request, not individual in-flight requests.
    list_y = inner.y + 2
    if list_y >= inner.y + inner.h:
        return
    p.text(list_y, inner.x, _H * inner.w, t.attr(PAIR_DIV))

    d = hist.derived
    rows = [
        ("prompt tok", f"~{d['req_prompt_tok']:.0f}", PAIR_CYAN),
        ("gen tok", f"~{d['req_gen_tok']:.0f}", PAIR_GREEN),
        ("prefill", fmt_seconds(d["req_prefill"]), PAIR_CYAN),
        ("decode", fmt_seconds(d["req_decode"]), PAIR_GREEN),
        ("preempt", f"{v.num_preemptions_total:.0f}", PAIR_MAGENTA
         if v.num_preemptions_total > 0 else PAIR_DIM),
    ]
    bottom = inner.y + inner.h
    for i, (label, value, pair) in enumerate(rows):
        ry = list_y + 1 + i
        if ry >= bottom:
            break
        p.text(ry, inner.x, label, t.attr(PAIR_DIM))
        p.text(ry, inner.x + inner.w - len(value), value, t.attr(pair, bold=True))


def draw_latency(p: Painter, rect: Rect, snap: Snapshot, hist: History,
                 num: int = 0) -> None:
    t = p.theme
    inner = p.box(rect, "latency", num, title2="recent avg",
                  border_pair=PAIR_BOX_LAT)
    if inner.h <= 0:
        return
    rows = [
        ("TTFT ", "ttft", PAIR_CYAN),
        ("TPOT ", "tpot", PAIR_GREEN),
        ("e2e  ", "e2e", PAIR_MAGENTA),
        ("queue", "queue_time", PAIR_DIM),
    ]
    for i, (label, key, pair) in enumerate(rows):
        if i >= inner.h:
            break
        val = hist.derived[key]
        p.text(inner.y + i, inner.x, label, t.attr(PAIR_DIM))
        p.text(inner.y + i, inner.x + 6, f"{fmt_seconds(val):>7}",
               t.attr(pair, bold=True))
        # Sparkline of the series to the right.
        spark_x = inner.x + 14
        spark_w = inner.w - 14
        if spark_w > 2:
            series = hist.series[key].values()
            vmax = max(max(series, default=0.0), 1e-6)
            rowsb = braille_chart(series, spark_w, 1, 0.0, vmax)
            if rowsb:
                p.text(inner.y + i, spark_x, rowsb[0], t.attr(pair))


def draw_cache(p: Painter, rect: Rect, snap: Snapshot, hist: History,
               num: int = 0) -> None:
    t = p.theme
    inner = p.box(rect, "cache", num, border_pair=PAIR_BOX_CACHE)
    if inner.h <= 0:
        return
    v = snap.vllm
    kv = hist.derived["kv_cache"]  # already a percent
    label = "KV   "
    p.text(inner.y, inner.x, label, t.attr(PAIR_DIM))
    bw = inner.w - len(label) - 6
    if bw > 0:
        _draw_meter(p, inner.y, inner.x + len(label), bw, kv)
        p.text(inner.y, inner.x + len(label) + bw + 1, f"{kv:3.0f}%",
               t.grad_attr(kv / 100.0, bold=True))

    if inner.h > 1:
        hit = v.prefix_cache_hit_rate * 100.0
        p.text(inner.y + 1, inner.x,
               f"prefix hit {hit:5.1f}%  "
               f"({v.prefix_cache_hits_total:.0f}/{v.prefix_cache_queries_total:.0f})",
               t.attr(PAIR_GREEN if hit > 0 else PAIR_DIM))

    if inner.h > 3:
        _draw_chart(p, Rect(inner.y + 2, inner.x, inner.h - 2, inner.w),
                    hist.series["kv_cache"].values(), 0.0, 100.0, 0,
                    gradient=True)
