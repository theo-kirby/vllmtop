"""Compute panel rectangles from the terminal size; reflow on resize."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

MIN_COLS = 62
MIN_LINES = 20


@dataclass(frozen=True)
class Rect:
    y: int
    x: int
    h: int
    w: int


@dataclass
class Layout:
    panels: Dict[str, Rect]
    too_small: bool = False


def _split(start: int, total: int, n: int) -> list[tuple[int, int]]:
    """Split `total` into n segments, distributing the remainder to the front."""
    base = total // n
    rem = total % n
    out = []
    pos = start
    for i in range(n):
        seg = base + (1 if i < rem else 0)
        out.append((pos, seg))
        pos += seg
    return out


# The grid below the GPU panel is two columns. `requests` gets a column to
# itself so it's tall enough for its per-request list; the rest stack opposite.
GRID_LEFT = ("throughput", "latency", "cache")
GRID_RIGHT = ("requests",)
GRID_ORDER = GRID_LEFT + GRID_RIGHT


def _place_column(panels: Dict[str, "Rect"], names, x: int, w: int,
                  y: int, h: int) -> None:
    """Stack `names` vertically within the column at (x, w), spanning (y, h)."""
    for name, (ry, rh) in zip(names, _split(y, h, len(names))):
        panels[name] = Rect(ry, x, rh, w)


def compute_layout(lines: int, cols: int, enabled) -> Layout:
    """Lay out the footer and the enabled panels for the terminal size.

    Panels start at the top of the screen (no header or footer bar). GPU (if
    enabled) spans the full width; the remaining enabled panels reflow into a
    2-column grid below. Hiding panels gives the rest more room, matching btop.
    """
    if cols < MIN_COLS or lines < MIN_LINES:
        return Layout(panels={}, too_small=True)

    body_y = 0
    body_h = lines  # full height; no footer bar
    body_w = cols

    gpu_on = "gpu" in enabled
    grid = [name for name in GRID_ORDER if name in enabled]

    panels: Dict[str, Rect] = {}

    # The GPU panel is taller than a single grid cell because its stats column
    # also carries the compact vLLM model summary.
    if gpu_on and grid:
        gpu_h = max(9, int(body_h * 0.5))
    elif gpu_on:
        gpu_h = body_h
    else:
        gpu_h = 0

    if gpu_on:
        panels["gpu"] = Rect(body_y, 0, gpu_h, body_w)

    grid_y = body_y + gpu_h
    grid_h = body_h - gpu_h

    if grid and grid_h > 0:
        left = [n for n in GRID_LEFT if n in enabled]
        right = [n for n in GRID_RIGHT if n in enabled]
        if left and right:
            (lx, lw), (rx, rw) = _split(0, body_w, 2)
            _place_column(panels, left, lx, lw, grid_y, grid_h)
            _place_column(panels, right, rx, rw, grid_y, grid_h)
        else:  # only one column populated -> it spans the full width
            _place_column(panels, left or right, 0, body_w, grid_y, grid_h)

    return Layout(panels=panels, too_small=False)
