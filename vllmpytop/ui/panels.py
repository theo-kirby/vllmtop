"""The five bordered panels, btop-style. Each draws into a Rect on screen."""

from __future__ import annotations

import curses
import time
from typing import List, Sequence, Tuple

from ..state import History, Snapshot, VllmSnapshot
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
    stacked_chart_down,
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

    top_rows = braille_chart(util, rect.w, top_h, 0.0, 100.0)
    n = len(top_rows)
    for i, row in enumerate(top_rows):
        attr = t.grad_attr((n - i) / n) if n else t.attr(0)
        p.text(rect.y + i, rect.x, row, attr)

    if bot_h > 0:
        stack_vmax = max(
            (r + w for r, w in zip(running, waiting)), default=0.0
        )
        grid = stacked_chart_down(running, waiting, rect.w, bot_h,
                                  max(stack_vmax, 1.0))
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


def draw_gpu(p: Painter, rect: Rect, snap: Snapshot, hist: History,
             num: int = 0) -> None:
    """GPU panel with a compact vLLM model summary folded into its stats column.

    Utilisation chart on the left; the right column carries the live GPU bars
    and stats, then a divider and a compact view of the served model + engine.
    """
    t = p.theme
    inner = p.box(rect, "gpu", num, title2="vllm", border_pair=PAIR_BOX_GPU)
    g = snap.gpu
    v = snap.vllm
    if inner.h <= 0:
        return
    if not g.available:
        # No GPU: still surface the vLLM summary so the panel stays useful.
        msg = "GPU unavailable" + (f": {g.error}" if g.error else "")
        p.text(inner.y, inner.x, msg[: inner.w], t.attr(PAIR_DIM, dim=True))
        y = inner.y + 2
        for text, pair in _vllm_lines(v):
            if y >= inner.y + inner.h:
                break
            p.text(y, inner.x, text[: inner.w],
                   t.attr(pair, bold=pair == PAIR_TITLE))
            y += 1
        return

    util_series = hist.series["gpu_util"].values()
    running_series = hist.series["running"].values()
    waiting_series = hist.series["waiting"].values()

    # The right column is a touch wider now that it also carries the vLLM
    # summary; the utilisation chart is pushed over to make room.
    right_w = min(42, max(26, inner.w * 2 // 5))
    chart_w = inner.w - right_w - 1
    if chart_w < 8:  # too narrow to split; chart takes the whole panel
        _draw_chart(p, inner, util_series, 0.0, 100.0, 0, gradient=True)
        return

    # btop-style mirrored chart: GPU utilisation grows up from the centre line,
    # the request count (running + waiting) grows down from it.
    _draw_dual_chart(
        p, Rect(inner.y, inner.x, inner.h, chart_w),
        util=util_series, running=running_series, waiting=waiting_series,
        util_now=g.util_gpu, run_now=v.num_requests_running,
        wait_now=v.num_requests_waiting,
    )

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
    # GPU stats first, then a divider and the compact vLLM summary.
    rows = [
        lambda y: p.text(y, rx, (g.name or "GPU")[:rw],
                         t.attr(PAIR_TITLE, bold=True)),
        lambda y: bar_row(y, "util ", g.util_gpu),
        lambda y: bar_row(y, "vram ", g.mem_used_perc),
        lambda y: bar_row(y, "pwr  ", pwr_frac),
        lambda y: text_row(y, "temp ",
                           f"{g.temperature:.0f}°C  fan {g.fan_speed:.0f}%",
                           temp_pair),
        lambda y: text_row(y, "core ",
                           f"{g.power_usage:.0f}/{g.power_limit:.0f}W  "
                           f"{g.sm_clock:.0f}MHz  "
                           f"{fmt_bytes(g.mem_used)}/{fmt_bytes(g.mem_total)}"),
        lambda y: p.text(y, rx, _rule("model", rw), t.attr(PAIR_DIV)),
    ]
    for text, pair in _vllm_lines(v):
        rows.append(
            lambda y, s=text, pr=pair: p.text(
                y, rx, s[:rw], t.attr(pr, bold=pr == PAIR_TITLE)
            )
        )

    for i, draw in enumerate(rows):
        if i >= inner.h:
            break
        draw(inner.y + i)


def _draw_mirror_chart(p: Painter, rect: Rect, top: Sequence[float],
                       top_pair: int, top_label: str, bottom: Sequence[float],
                       bottom_pair: int, bottom_label: str) -> None:
    """A btop-style mirrored chart of two single series sharing a centre line.

    ``top`` grows up from the centre, ``bottom`` grows down from it; each half
    scales to its own max (the series can differ wildly in magnitude). The
    current value of each is shown as a bold, colour-matched corner label.
    """
    if rect.h <= 0 or rect.w <= 0:
        return
    t = p.theme
    top_h = max(1, rect.h // 2)
    bot_h = rect.h - top_h

    top_vmax = max(max(top, default=0.0), 1e-9)
    for i, row in enumerate(braille_chart(top, rect.w, top_h, 0.0, top_vmax)):
        p.text(rect.y + i, rect.x, row, t.attr(top_pair))

    if bot_h > 0:
        bot_vmax = max(max(bottom, default=0.0), 1e-9)
        bot_rows = braille_chart(bottom, rect.w, bot_h, 0.0, bot_vmax, flip=True)
        for i, row in enumerate(bot_rows):
            p.text(rect.y + top_h + i, rect.x, row, t.attr(bottom_pair))

    p.text(rect.y, rect.x, f" {top_label} ", t.attr(top_pair, bold=True))
    if bot_h > 0:
        p.text(rect.y + rect.h - 1, rect.x, f" {bottom_label} ",
               t.attr(bottom_pair, bold=True))


def draw_throughput(p: Painter, rect: Rect, snap: Snapshot, hist: History,
                    num: int = 0) -> None:
    inner = p.box(rect, "throughput", num, title2="tok/s",
                  border_pair=PAIR_BOX_THRU)
    if inner.h <= 0:
        return
    # Mirrored chart: generation tok/s grows up from the centre line, prompt
    # (prefill) tok/s grows down from it.
    _draw_mirror_chart(
        p, inner,
        top=hist.series["gen_tok_s"].values(), top_pair=PAIR_GREEN,
        top_label="gen " + big_number(hist.derived["gen_tok_s"], " tok/s"),
        bottom=hist.series["prompt_tok_s"].values(), bottom_pair=PAIR_CYAN,
        bottom_label="prompt " + big_number(hist.derived["prompt_tok_s"], " tok/s"),
    )


def draw_requests(p: Painter, rect: Rect, snap: Snapshot, hist: History,
                  num: int = 0) -> None:
    t = p.theme
    inner = p.box(rect, "requests", num, title2="activity",
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

    # Under the bars: a live feed of the HTTP calls vLLM served, parsed from its
    # access log (request envelope only — no prompt/response text). Newest first.
    list_y = inner.y + 2
    if list_y >= inner.y + inner.h:
        return
    p.text(list_y, inner.x, _H * inner.w, t.attr(PAIR_DIV))
    _draw_access_feed(
        p, Rect(list_y + 1, inner.x, inner.y + inner.h - list_y - 1, inner.w),
        snap.access_log, snap.access_error,
    )


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


def _draw_access_feed(p: Painter, rect: Rect, entries, error=None) -> None:
    """Render the access-log feed (header + rows) into ``rect`` — no box.

    Shown beneath the requests panel's bars. The request envelope only — age,
    status, method, endpoint, client (vLLM doesn't log prompt/response text
    unless started with ``--enable-log-requests``). Newest first.
    """
    t = p.theme
    if rect.h <= 0 or rect.w <= 0:
        return
    if error:
        p.text(rect.y, rect.x, ("⚠ " + error)[: rect.w],
               t.attr(PAIR_YELLOW, dim=True))
        return
    if not entries:
        hint = "no recent calls — pass --docker <name> or --log-file <path>"
        p.text(rect.y, rect.x, hint[: rect.w], t.attr(PAIR_DIM, dim=True))
        return

    # Columns: age | code | verb | endpoint (flex) | client (right-aligned).
    age_w, code_w, meth_w = 4, 4, 5
    client_w = 21 if rect.w >= 62 else 0
    x_age = rect.x
    x_code = x_age + age_w + 1
    x_meth = x_code + code_w + 1
    x_path = x_meth + meth_w + 1
    x_client = rect.x + rect.w - client_w
    path_w = max(6, (x_client - 1 if client_w else rect.x + rect.w) - x_path)

    hdr = t.attr(PAIR_DIM, dim=True)
    p.text(rect.y, x_age, "age", hdr)
    p.text(rect.y, x_code, "code", hdr)
    p.text(rect.y, x_meth, "verb", hdr)
    p.text(rect.y, x_path, "endpoint"[:path_w], hdr)
    if client_w:
        p.text(rect.y, x_client, "client", hdr)

    now = time.time()
    for i, e in enumerate(entries):
        y = rect.y + 1 + i
        if y >= rect.y + rect.h:
            break
        p.text(y, x_age, fmt_duration(max(0.0, now - e.t)).rjust(age_w)[:age_w],
               t.attr(PAIR_DIM))
        code_pair = (PAIR_RED if e.status >= 500
                     else PAIR_YELLOW if e.status >= 400 else PAIR_GREEN)
        p.text(y, x_code, str(e.status), t.attr(code_pair, bold=True))
        p.text(y, x_meth, e.method[:meth_w], t.attr(PAIR_DIM))
        p.text(y, x_path, e.path[:path_w], t.attr(PAIR_TITLE))
        if client_w:
            p.text(y, x_client, e.client[:client_w], t.attr(PAIR_DIM, dim=True))
