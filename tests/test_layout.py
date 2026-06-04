from vllmpytop.ui.layout import (
    MIN_COLS,
    MIN_LINES,
    Rect,
    col,
    compute_layout,
    leaf,
    prune,
    row,
)
from vllmpytop.ui.registry import REGISTRY
from vllmpytop.ui.views import VIEWS

ALL = set(REGISTRY)


def test_row_splits_width_col_splits_height():
    tree = col(leaf("a"), row(leaf("b"), leaf("c")))
    lay = compute_layout(40, 100, tree, {"a", "b", "c"})
    a, b, c = lay.panels["a"], lay.panels["b"], lay.panels["c"]
    # col: a on top, the row beneath, each ~half the height.
    assert a == Rect(0, 0, 20, 100)
    assert b.y == c.y == 20 and b.h == c.h == 20
    # row: b and c side by side, splitting the width, tiling it exactly.
    assert b.x == 0 and c.x == b.w and b.w + c.w == 100


def test_weights_are_proportional():
    lay = compute_layout(50, 80, col(leaf("a", weight=2), leaf("b", weight=3)),
                         {"a", "b"})
    assert lay.panels["a"].h == 20  # 2/5 of 50
    assert lay.panels["b"].h == 30  # 3/5 of 50


def test_prune_drops_unavailable_and_collapses():
    tree = row(leaf("gpu"), leaf("requests"))
    pruned = prune(tree, {"requests"})
    # The surviving sibling collapses up and inherits the row's slot.
    assert pruned.panel == "requests"


def test_unavailable_panel_reflows_to_fill():
    # With gpu absent the remaining panels reflow to use the whole screen.
    lay = compute_layout(40, 120, VIEWS[0].root, ALL - {"gpu"})
    assert "gpu" not in lay.panels
    ys = min(r.y for r in lay.panels.values())
    assert ys == 0  # something now occupies the freed top region


def test_too_small_flags():
    lay = compute_layout(MIN_LINES - 1, MIN_COLS - 1, VIEWS[0].root, ALL)
    assert lay.too_small and not lay.panels


def test_views_only_reference_registered_panels():
    def panels_in(node):
        if node.panel is not None:
            return {node.panel}
        out = set()
        for child in node.children:
            out |= panels_in(child)
        return out

    for view in VIEWS:
        assert panels_in(view.root) <= ALL, view.name
