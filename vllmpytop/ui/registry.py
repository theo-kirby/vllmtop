"""Panel registry: metadata + draw fn for every panel a view can reference.

Decouples panel identity from any particular view. A :class:`Panel` names its
draw function and what it ``requires`` — a capability that must be present for
the panel to be shown (e.g. ``"gpu"`` for a GPU collector). The app computes
the available capability set each frame and the layout prunes panels whose
requirement is unmet, so a view referencing the gpu panel still works on a
CPU-only host (that panel just drops out and its siblings reflow).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from . import panels

# A panel draw fn: (Painter, Rect, Snapshot, History, num) -> None.
DrawFn = Callable[..., None]


@dataclass(frozen=True)
class Panel:
    id: str
    title: str
    draw: DrawFn
    requires: Tuple[str, ...] = ()  # capabilities that must be available


REGISTRY: Dict[str, Panel] = {
    p.id: p
    for p in (
        Panel("gpu", "gpu", panels.draw_gpu, requires=("gpu",)),
        Panel("throughput", "throughput", panels.draw_throughput),
        Panel("requests", "requests", panels.draw_requests),
        Panel("perf", "perf", panels.draw_perf),
        Panel("loadavg", "1·5·15", panels.draw_loadavg),
    )
}
