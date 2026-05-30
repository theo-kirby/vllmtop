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


# Order the grid panels appear in; GPU is laid out separately (full width).
GRID_ORDER = ("throughput", "requests", "latency", "cache")


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

    if gpu_on and grid:
        gpu_h = max(7, int(body_h * 0.38))
    elif gpu_on:
        gpu_h = body_h
    else:
        gpu_h = 0

    if gpu_on:
        panels["gpu"] = Rect(body_y, 0, gpu_h, body_w)

    grid_y = body_y + gpu_h
    grid_h = body_h - gpu_h

    if grid and grid_h > 0:
        nrows = (len(grid) + 1) // 2
        idx = 0
        for ry, rh in _split(grid_y, grid_h, nrows):
            remaining = len(grid) - idx
            ncols = 2 if remaining >= 2 else 1
            for cx, cw in _split(0, body_w, ncols):
                panels[grid[idx]] = Rect(ry, cx, rh, cw)
                idx += 1

    return Layout(panels=panels, too_small=False)
