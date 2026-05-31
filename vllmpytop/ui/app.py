"""The curses application: background poller thread + render loop.

:class:`Poller` runs in a daemon thread, scraping vLLM /metrics and polling
NVML at the configured interval. The main thread loops at a faster tick
(250 ms), reading the latest snapshot, deriving rates from the :class:`History`,
and redrawing the panels. Panel visibility is toggled with ``1``-``4``.
"""

from __future__ import annotations

import curses
import threading
import time
from typing import Optional

from ..collectors.access_log import AccessLogTailer
from ..collectors.gpu import GpuCollector
from ..collectors.vllm import VllmCollector
from ..config import AppConfig
from ..state import GpuSnapshot, History, MergedLogEntry, Snapshot, VllmSnapshot
from . import panels
from .layout import compute_layout
from .panels import Painter
from .theme import PAIR_DIM, PAIR_TITLE, PAIR_YELLOW, Theme

# Render tick: how often the UI wakes to handle input / redraw (seconds).
RENDER_TICK = 0.25
MIN_INTERVAL = 0.2
MAX_INTERVAL = 10.0

# Panels in order; the index+1 is the superscript number / toggle key.
PANELS = (
    ("gpu", panels.draw_gpu),
    ("throughput", panels.draw_throughput),
    ("requests", panels.draw_requests),
    ("perf", panels.draw_perf),
)


class Poller(threading.Thread):
    """Background thread that takes snapshots without blocking the UI."""

    def __init__(self, config: AppConfig,
                 tailer: Optional[AccessLogTailer] = None) -> None:
        super().__init__(daemon=True)
        self.config = config
        self.vllm = VllmCollector(config.metrics_url, config.http_timeout)
        self.gpu = None if config.no_gpu else GpuCollector(config.gpu_index)
        self.tailer = tailer
        self._lock = threading.Lock()
        self._latest: Optional[Snapshot] = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self.interval = config.interval

    def _collect(self) -> Snapshot:
        gsnap = self.gpu.poll() if self.gpu is not None else GpuSnapshot(
            available=False
        )
        merged = self.tailer.merged_log() if self.tailer is not None else []
        err = self.tailer.error if self.tailer is not None else None
        return Snapshot(monotonic=time.monotonic(), vllm=self.vllm.poll(),
                        gpu=gsnap, merged_log=merged, access_error=err)

    def run(self) -> None:
        while not self._stop.is_set():
            if not self._paused.is_set():
                snap = self._collect()
                with self._lock:
                    self._latest = snap
            # Sleep in small slices so interval/pause changes take effect fast.
            slept = 0.0
            while slept < self.interval and not self._stop.is_set():
                step = min(0.1, self.interval - slept)
                time.sleep(step)
                slept += step

    def take(self) -> Optional[Snapshot]:
        """Pop the latest snapshot if a new one is available."""
        with self._lock:
            snap = self._latest
            self._latest = None
            return snap

    def stop(self) -> None:
        self._stop.set()

    def toggle_pause(self) -> bool:
        if self._paused.is_set():
            self._paused.clear()
        else:
            self._paused.set()
        return self._paused.is_set()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def close(self) -> None:
        if self.gpu is not None:
            self.gpu.close()


class App:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.theme = Theme()
        self.history = History(config.history_len)
        self.tailer = (
            AccessLogTailer(file=config.log_file, docker=config.docker_container)
            if config.has_log_source else None
        )
        self.poller = Poller(config, self.tailer)
        self.show_help = False
        self.last: Optional[Snapshot] = None
        self.last_update_t = 0.0
        # Force a full repaint next frame (after a layout change leaves stale
        # cells that curses' diff-based refresh would otherwise keep).
        self._force_clear = False
        self.enabled = {name for name, _ in PANELS}
        if config.no_gpu:
            self.enabled.discard("gpu")

    def run(self) -> int:
        if self.tailer is not None:
            self.tailer.start()
        self.poller.start()
        try:
            return curses.wrapper(self._loop)
        finally:
            self.poller.stop()
            self.poller.close()
            if self.tailer is not None:
                self.tailer.stop()

    def _loop(self, stdscr) -> int:
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(int(RENDER_TICK * 1000))
        self.theme.init()

        while True:
            snap = self.poller.take()
            if snap is not None:
                self.history.update(snap)
                self.last = snap
                self.last_update_t = time.monotonic()

            self._draw(stdscr)

            ch = stdscr.getch()
            if ch == -1:
                continue
            if self.show_help:
                self.show_help = False
                continue
            if ch in (ord("q"), 27):  # q or Esc
                return 0
            elif ch in (ord("h"), ord("?")):
                self.show_help = True
            elif ch == ord("p"):
                self.poller.toggle_pause()
            elif ch in (ord("+"), ord("=")):
                self.poller.interval = max(MIN_INTERVAL, self.poller.interval / 2)
            elif ch == ord("-"):
                self.poller.interval = min(MAX_INTERVAL, self.poller.interval * 2)
            elif ord("1") <= ch <= ord("4"):
                self._toggle_panel(ch - ord("1"))
            elif ch == curses.KEY_RESIZE:
                self._force_clear = True  # layout recomputed each draw

    def _toggle_panel(self, idx: int) -> None:
        name = PANELS[idx][0]
        # Don't re-enable the GPU panel when there's no GPU collector behind it.
        if name == "gpu" and self.poller.gpu is None:
            return
        if name in self.enabled:
            self.enabled.discard(name)
        else:
            self.enabled.add(name)
        self._force_clear = True  # panels reflow; repaint to drop stale cells

    # ---- drawing ----------------------------------------------------------

    def _draw(self, stdscr) -> None:
        if self._force_clear:
            stdscr.clear()  # full physical repaint, no diff against stale cells
            self._force_clear = False
        else:
            stdscr.erase()
        lines, cols = stdscr.getmaxyx()
        layout = compute_layout(lines, cols, self.enabled)
        p = Painter(stdscr, self.theme)

        if layout.too_small:
            msg = "terminal too small — resize to at least 62x20"
            p.text(lines // 2, max(0, (cols - len(msg)) // 2), msg,
                   self.theme.attr(PAIR_YELLOW, bold=True))
            stdscr.noutrefresh()
            curses.doupdate()
            return

        snap = self.last or Snapshot(time.monotonic(), VllmSnapshot(),
                                     GpuSnapshot(), merged_log=[])
        for i, (name, fn) in enumerate(PANELS):
            rect = layout.panels.get(name)
            if rect is not None:
                fn(p, rect, snap, self.history, i + 1)

        if not layout.panels:
            msg = "all panels hidden — press 1-5 to show one"
            p.text(lines // 2, max(0, (cols - len(msg)) // 2), msg,
                   self.theme.attr(PAIR_DIM, dim=True))

        if self.show_help:
            self._draw_help(p, lines, cols)

        stdscr.noutrefresh()
        curses.doupdate()

    def _draw_help(self, p: Painter, lines: int, cols: int) -> None:
        t = self.theme
        body = [
            "vllmpytop — keybindings",
            "",
            "  q / Esc    quit",
            "  + / -      faster / slower refresh",
            "  p          pause / resume polling",
            "  1 - 4      toggle a panel on / off",
            "  h / ?      toggle this help",
            "",
            "Panels: ¹gpu  ²throughput  ³requests  ⁴perf",
            "  (gpu shows model + engine info; requests shows the call feed)",
            "",
            "press any key to close",
        ]
        bw = max(len(s) for s in body) + 4
        bh = len(body) + 2
        y0 = max(0, (lines - bh) // 2)
        x0 = max(0, (cols - bw) // 2)
        from .layout import Rect

        # Clear the area under the overlay so panel content doesn't bleed through.
        blank = " " * bw
        for row in range(bh):
            p.text(y0 + row, x0, blank, t.attr(0))
        inner = p.box(Rect(y0, x0, bh, bw), "help")
        for i, s in enumerate(body):
            if i < inner.h:
                pair = PAIR_TITLE if i == 0 else PAIR_DIM
                p.text(inner.y + i, inner.x + 1, s, t.attr(pair))
