"""Tail the vLLM server's uvicorn access log into a feed of HTTP calls.

vLLM's ``/metrics`` is aggregate-only and its request prompts are not logged
unless the server runs with ``--enable-log-requests``. The one per-request
signal available by default is the uvicorn access line, e.g.::

    (APIServer pid=1) INFO:  192.168.32.2:41854 - "POST /v1/chat/completions HTTP/1.1" 200 OK

So this collector parses those lines (client, method, path, status) into a
rolling :class:`AccessLogEntry` buffer. It is a live feed — no prompt/response
text, since none is present in the log.

The source is either a file (``--log-file``) or the stdout of a streaming
command such as ``docker logs -f`` (``--docker``). A background thread follows
it; the UI reads :meth:`AccessLogTailer.snapshot` each frame.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

from ..state import AccessLogEntry

# client - "METHOD path HTTP/x.y" status   (anywhere in the line, after any
# logger prefix like "(APIServer pid=1) INFO:  ").
_ACCESS_RE = re.compile(r'(\S+) - "([A-Z]+) (\S+) HTTP/[\d.]+" (\d{3})')


def parse_access_line(line: str) -> Optional[Tuple[str, str, str, int]]:
    """Parse a uvicorn access line into ``(client, method, path, status)``.

    Returns None for any line that is not an access log entry. Pure function so
    it can be unit-tested against sample log lines.
    """
    m = _ACCESS_RE.search(line)
    if not m:
        return None
    client, method, path, status = m.groups()
    try:
        return client, method, path, int(status)
    except ValueError:
        return None


class AccessLogTailer(threading.Thread):
    """Follows a log source and parses access lines into a rolling buffer.

    Starts at the *end* of the source (like ``tail -f``) so only calls observed
    while running are shown — that keeps each entry's age meaningful. Any
    failure (missing file, no ``docker``, container gone) is surfaced via
    :attr:`error` rather than crashing the UI.
    """

    # Infra endpoints we never surface (vllmtop's own polling, health checks).
    IGNORE_PATHS = frozenset({"/metrics", "/health", "/ping", "/version"})

    def __init__(self, *, file: Optional[str] = None,
                 docker: Optional[str] = None, maxlen: int = 200) -> None:
        super().__init__(daemon=True)
        self._file = file
        self._docker = docker
        self._buf: Deque[AccessLogEntry] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self.error: Optional[str] = None

    @property
    def source_label(self) -> str:
        if self._docker:
            return f"docker {self._docker}"
        if self._file:
            return os.path.basename(self._file)
        return "—"

    def run(self) -> None:
        try:
            if self._docker:
                self._follow_command(
                    ["docker", "logs", "-f", "--tail", "0", self._docker]
                )
            elif self._file:
                self._follow_file(self._file)
        except Exception as exc:  # never take down the UI thread
            self.error = str(exc)

    def _ingest(self, line: str) -> None:
        parsed = parse_access_line(line)
        if parsed is None:
            return
        client, method, path, status = parsed
        if path.split("?", 1)[0] in self.IGNORE_PATHS:
            return
        entry = AccessLogEntry(t=time.time(), client=client, method=method,
                               path=path, status=status)
        with self._lock:
            self._buf.append(entry)

    def _follow_command(self, cmd: List[str]) -> None:
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
        except FileNotFoundError:
            self.error = f"command not found: {cmd[0]}"
            return
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            self._ingest(line.rstrip("\n"))
        rc = self._proc.poll()
        if not self._stop.is_set() and rc not in (None, 0):
            self.error = f"`{' '.join(cmd)}` exited ({rc})"

    def _follow_file(self, path: str) -> None:
        while not self._stop.is_set():
            try:
                f = open(path, "r", errors="replace")
            except OSError as exc:
                self.error = str(exc)
                time.sleep(1.0)
                continue
            self.error = None
            with f:
                f.seek(0, os.SEEK_END)  # start at the end, like tail -f
                while not self._stop.is_set():
                    line = f.readline()
                    if line:
                        self._ingest(line.rstrip("\n"))
                        continue
                    time.sleep(0.2)
                    try:  # detect truncation / rotation -> reopen
                        if os.stat(path).st_size < f.tell():
                            break
                    except OSError:
                        break

    def snapshot(self, n: Optional[int] = None) -> List[AccessLogEntry]:
        """Return recent entries, newest first (at most ``n``)."""
        with self._lock:
            items = list(self._buf)
        items.reverse()
        return items if n is None else items[:n]

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
