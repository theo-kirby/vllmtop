"""Layout primitives: a recursive split tree placed into terminal rects.

A *view* is a tree of nested splits referencing panels by id. Each :class:`Node`
is either a leaf (a single panel) or a split arranging its children in a row
(side by side, splitting width) or a column (stacked, splitting height), sized
proportional to per-child weights.

:func:`compute_layout` prunes panels that aren't available (e.g. the gpu panel
on a CPU-only host), reflows the remaining tree, and returns the screen
:class:`Rect` for each visible panel. This replaces the old hand-wired
2-column grid: the original "overview" arrangement is now just one tree literal
in ``views.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Set, Tuple

MIN_COLS = 62
MIN_LINES = 20


@dataclass(frozen=True)
class Rect:
    y: int
    x: int
    h: int
    w: int


@dataclass(frozen=True)
class Node:
    """A node in a view's layout tree.

    A leaf carries ``panel`` (its panel id) and no children. A split carries
    ``children`` and ``vertical`` (True = stack into a column, splitting
    height; False = lay out side by side in a row, splitting width).
    ``weight`` sizes this node against its siblings.
    """

    weight: int = 1
    panel: Optional[str] = None
    vertical: bool = False
    children: Tuple["Node", ...] = ()


def leaf(panel: str, weight: int = 1) -> Node:
    return Node(weight=weight, panel=panel)


def row(*children: Node, weight: int = 1) -> Node:
    """Arrange children side by side (split the available width)."""
    return Node(weight=weight, vertical=False, children=tuple(children))


def col(*children: Node, weight: int = 1) -> Node:
    """Stack children top to bottom (split the available height)."""
    return Node(weight=weight, vertical=True, children=tuple(children))


@dataclass
class Layout:
    panels: Dict[str, Rect]
    too_small: bool = False


def _split_weighted(start: int, total: int,
                    weights: List[int]) -> List[Tuple[int, int]]:
    """Split ``total`` into segments proportional to ``weights``.

    The last segment absorbs the rounding remainder so the segments exactly
    tile ``[start, start + total)``.
    """
    tw = sum(weights) or 1
    out: List[Tuple[int, int]] = []
    pos, used = start, 0
    for i, wt in enumerate(weights):
        seg = (total - used) if i == len(weights) - 1 else (total * wt) // tw
        out.append((pos, seg))
        pos += seg
        used += seg
    return out


def prune(node: Optional[Node], available: Set[str]) -> Optional[Node]:
    """Drop leaves whose panel isn't available and collapse empty splits.

    A split with a single surviving child collapses to that child (keeping the
    parent's weight, so the survivor inherits the parent's slot). Returns
    ``None`` when nothing in the subtree is available.
    """
    if node is None:
        return None
    if node.panel is not None:
        return node if node.panel in available else None
    kids = [k for k in (prune(c, available) for c in node.children) if k]
    if not kids:
        return None
    if len(kids) == 1:
        return replace(kids[0], weight=node.weight)
    return replace(node, children=tuple(kids))


def _place(node: Node, rect: Rect, out: Dict[str, Rect]) -> None:
    if node.panel is not None:
        out[node.panel] = rect
        return
    weights = [c.weight for c in node.children]
    if node.vertical:
        for child, (ry, rh) in zip(
            node.children, _split_weighted(rect.y, rect.h, weights)
        ):
            _place(child, Rect(ry, rect.x, rh, rect.w), out)
    else:
        for child, (rx, rw) in zip(
            node.children, _split_weighted(rect.x, rect.w, weights)
        ):
            _place(child, Rect(rect.y, rx, rect.h, rw), out)


def compute_layout(lines: int, cols: int, view: Node,
                   available: Set[str]) -> Layout:
    """Place ``view``'s available panels into the terminal rect.

    Panels start at the top of the screen (no header/footer bar). Panels not in
    ``available`` are pruned and the rest reflow to fill the freed space.
    """
    if cols < MIN_COLS or lines < MIN_LINES:
        return Layout(panels={}, too_small=True)
    pruned = prune(view, available)
    if pruned is None:
        return Layout(panels={}, too_small=False)
    out: Dict[str, Rect] = {}
    _place(pruned, Rect(0, 0, lines, cols), out)
    return Layout(panels=out, too_small=False)
