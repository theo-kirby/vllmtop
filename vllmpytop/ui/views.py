"""Built-in views: named layout trees over the panel registry.

Each view is a fixed arrangement the user cycles through with the number keys
(``1``-``N``) or ``Tab``. Panels unavailable in the current environment are
pruned at layout time (see :mod:`.layout`), so a view referencing the gpu panel
degrades gracefully on a CPU-only host.

To add a view: build a tree from :func:`~.layout.leaf`/:func:`~.layout.row`/
:func:`~.layout.col` and append a :class:`View` to :data:`VIEWS`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .layout import Node, col, leaf, row


@dataclass(frozen=True)
class View:
    name: str
    root: Node


# The default arrangement: gpu spans the top ~40% (weight 2 of 5), with
# throughput over the 1·5·15 averages, beside the request feed below
# (weight 3 of 5).
_OVERVIEW = col(
    leaf("gpu", weight=2),
    row(
        col(leaf("throughput", weight=3), leaf("loadavg", weight=2)),
        leaf("requests"),
        weight=3,
    ),
)

# Load-average style 1/5/15-minute windows over the key metrics, with the live
# throughput and latency charts beneath for context.
_RATES = col(
    leaf("loadavg", weight=3),
    row(leaf("throughput"), leaf("perf"), weight=2),
)

# Request-feed focus: the live feed large, with a compact gpu/engine strip.
_REQUESTS = col(
    leaf("gpu", weight=2),
    leaf("requests", weight=3),
)

# GPU / engine focus.
_GPU = col(
    leaf("gpu", weight=3),
    leaf("throughput", weight=2),
)

VIEWS = (
    View("overview", _OVERVIEW),
    View("1·5·15", _RATES),
    View("requests", _REQUESTS),
    View("gpu", _GPU),
)
