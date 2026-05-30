"""Hand-rolled drawing primitives for the btop look.

The headline widget is :func:`braille_chart`, which packs a series into a
2x4-dot Unicode braille matrix per cell for `2w x 4h`-dot resolution.
"""

from __future__ import annotations

import math
from typing import List, Sequence

# Braille dot bit values, indexed [column][row] with row 0 at the top.
#   1 4        0x01 0x08
#   2 5   ->   0x02 0x10
#   3 6        0x04 0x20
#   7 8        0x40 0x80
_DOT_BITS = (
    (0x01, 0x02, 0x04, 0x40),  # left column, top->bottom
    (0x08, 0x10, 0x20, 0x80),  # right column, top->bottom
)
_BRAILLE_BASE = 0x2800


def _resample_tail(series: Sequence[float], n: int) -> List[float]:
    """Return the most recent `n` samples (newest on the right).

    Fewer than `n` samples are returned as-is; the caller right-aligns them so
    the chart fills from the right and scrolls left as history accumulates.
    """
    vals = list(series)
    if len(vals) >= n:
        return vals[-n:]
    return vals


def braille_chart(
    series: Sequence[float],
    width: int,
    height: int,
    vmin: float,
    vmax: float,
    baseline: bool = True,
    flip: bool = False,
) -> List[str]:
    """Render `series` as `height` rows of `width` braille glyphs.

    Each cell is a 2x4 dot matrix, so the effective resolution is
    ``2*width`` columns by ``4*height`` dots. Values are clipped to
    [vmin, vmax]; the most recent samples are shown right-aligned.

    With ``baseline`` (btop's ``no_zero`` behavior) every sampled column lights
    at least its bottom dot, so the graph keeps a continuous floor instead of
    leaving gaps where the value is zero.

    With ``flip`` the bars grow downward from the top row instead of upward
    from the bottom — used for the lower half of a btop-style mirrored chart,
    so its floor sits on the shared centre line.
    """
    if width <= 0 or height <= 0:
        return []

    dot_cols = 2 * width
    dot_rows = 4 * height

    vals = _resample_tail(series, dot_cols)
    # Right-align: figure out which dot-columns have data.
    pad = dot_cols - len(vals)

    span = vmax - vmin
    if span <= 0:
        span = 1.0

    # cells[row][col] accumulates braille bits.
    cells = [[0] * width for _ in range(height)]

    for i, value in enumerate(vals):
        gc = pad + i  # global dot column
        norm = (value - vmin) / span
        if norm < 0:
            norm = 0.0
        elif norm > 1:
            norm = 1.0
        filled = int(round(norm * dot_rows))
        if baseline:
            filled = max(1, filled)  # always keep a one-dot floor
        elif filled <= 0:
            continue
        cell_col = gc // 2
        sub_col = gc % 2
        # Fill from the bottom dot-row upward, or the top downward when flipped.
        for k in range(filled):
            gr = k if flip else dot_rows - 1 - k
            cell_row = gr // 4
            sub_row = gr % 4
            cells[cell_row][cell_col] |= _DOT_BITS[sub_col][sub_row]

    return [
        "".join(chr(_BRAILLE_BASE + bits) for bits in row) for row in cells
    ]


def stacked_chart_down(
    near: Sequence[float],
    far: Sequence[float],
    width: int,
    height: int,
    vmax: float,
) -> List[List[tuple]]:
    """Two stacked series growing *downward* from the top row.

    Used for the lower half of a btop-style mirrored chart: ``near`` grows from
    the centre line outward (its floor keeps the axis continuous), and ``far``
    is stacked beyond it. Returns a ``height x width`` grid of ``(glyph, band)``
    where ``band`` is 0 for ``near``, 1 for ``far``, or -1 for an empty cell.

    Because a braille cell packs 2x4 dots but can carry only one colour, a
    cell's band is whichever series owns more of its lit dots (ties -> near).
    """
    grid: List[List[tuple]] = [[(" ", -1)] * width for _ in range(height)]
    if width <= 0 or height <= 0:
        return grid

    dot_cols = 2 * width
    dot_rows = 4 * height
    span = vmax if vmax > 0 else 1.0
    near_v, far_v = list(near), list(far)
    near_pad = dot_cols - len(near_v)
    far_pad = dot_cols - len(far_v)

    near_cells = [[0] * width for _ in range(height)]
    far_cells = [[0] * width for _ in range(height)]

    for gc in range(dot_cols):
        ni, fi = gc - near_pad, gc - far_pad
        n_has = 0 <= ni < len(near_v)
        f_has = 0 <= fi < len(far_v)
        if not (n_has or f_has):
            continue
        nval = near_v[ni] if n_has else 0.0
        fval = far_v[fi] if f_has else 0.0
        n_dots = int(round(min(1.0, max(0.0, nval / span)) * dot_rows))
        n_dots = max(1, n_dots)  # floor: keep the centre axis continuous
        f_dots = int(round(min(1.0, max(0.0, fval / span)) * dot_rows))
        if n_dots + f_dots > dot_rows:
            f_dots = dot_rows - n_dots
        cell_col, sub_col = gc // 2, gc % 2
        for k in range(n_dots):  # k=0 is the top (centre) dot-row
            near_cells[k // 4][cell_col] |= _DOT_BITS[sub_col][k % 4]
        for k in range(n_dots, n_dots + f_dots):
            far_cells[k // 4][cell_col] |= _DOT_BITS[sub_col][k % 4]

    for row in range(height):
        for col in range(width):
            nb, fb = near_cells[row][col], far_cells[row][col]
            if not nb and not fb:
                continue
            band = 0 if bin(nb).count("1") >= bin(fb).count("1") else 1
            grid[row][col] = (chr(_BRAILLE_BASE + (nb | fb)), band)
    return grid


def hbar(value: float, vmax: float, width: int) -> str:
    """A horizontal bar of `width` chars filled proportional to value/vmax."""
    if width <= 0:
        return ""
    if vmax <= 0:
        frac = 0.0
    else:
        frac = max(0.0, min(1.0, value / vmax))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def big_number(value: float, unit: str = "") -> str:
    """Compact human-readable number with a unit suffix."""
    if value >= 1_000_000:
        text = f"{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        text = f"{value / 1_000:.1f}k"
    elif value >= 100:
        text = f"{value:.0f}"
    elif value >= 10:
        text = f"{value:.1f}"
    else:
        text = f"{value:.2f}"
    return f"{text}{unit}" if unit else text


def histogram_bars(values: Sequence[float], vmax: float) -> str:
    """Render a small inline sparkline using block elements (8 levels)."""
    blocks = " ▁▂▃▄▅▆▇█"
    if vmax <= 0:
        vmax = 1.0
    out = []
    for v in values:
        frac = max(0.0, min(1.0, v / vmax))
        idx = int(round(frac * (len(blocks) - 1)))
        out.append(blocks[idx])
    return "".join(out)


def fmt_seconds(value: float) -> str:
    """Human latency: ms below 1s, otherwise seconds."""
    if value <= 0:
        return "—"
    if value < 1.0:
        return f"{value * 1000:.0f}ms"
    return f"{value:.2f}s"


def fmt_duration(seconds: float) -> str:
    """Compact uptime: '45s', '12m', '3h 20m', '2d 4h'."""
    if not math.isfinite(seconds) or seconds < 0:
        return "—"
    s = int(seconds)
    days, rem = divmod(s, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {mins}m"
    if mins:
        return f"{mins}m"
    return f"{secs}s"


def fmt_bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}PB"
