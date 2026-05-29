"""Hand-rolled drawing primitives for the btop look.

The headline widget is :func:`braille_chart`, which packs a series into a
2x4-dot Unicode braille matrix per cell for `2w x 4h`-dot resolution.
"""

from __future__ import annotations

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
) -> List[str]:
    """Render `series` as `height` rows of `width` braille glyphs.

    Each cell is a 2x4 dot matrix, so the effective resolution is
    ``2*width`` columns by ``4*height`` dots. Values are clipped to
    [vmin, vmax]; the most recent samples are shown right-aligned.

    With ``baseline`` (btop's ``no_zero`` behavior) every sampled column lights
    at least its bottom dot, so the graph keeps a continuous floor instead of
    leaving gaps where the value is zero.
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
        # Fill from the bottom dot-row upward.
        for k in range(filled):
            gr = dot_rows - 1 - k
            cell_row = gr // 4
            sub_row = gr % 4
            cells[cell_row][cell_col] |= _DOT_BITS[sub_col][sub_row]

    return [
        "".join(chr(_BRAILLE_BASE + bits) for bits in row) for row in cells
    ]


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


def fmt_bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}PB"
