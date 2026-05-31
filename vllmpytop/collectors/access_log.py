"""Tail the vLLM server's uvicorn access log into a feed of HTTP calls.

vLLM's ``/metrics`` is aggregate-only and its request prompts are not logged
unless the server runs with ``--enable-log-requests``. The one per-request
signal available by default is the uvicorn access line, e.g.::

    (APIServer pid=1) INFO:  192.168.32.2:41854 - "POST /v1/chat/completions HTTP/1.1" 200 OK

So this collector parses those lines (client, method, path, status) into a
rolling :class:`AccessLogEntry` buffer. When vLLM ≥ 0.11.3 runs with
``--enable-log-requests`` the request-log lines also carry the prompt text
(truncated by vLLM's ``max_log_len``), which we parse and merge in.

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

from ..state import MergedLogEntry

# client - "METHOD path HTTP/x.y" status   (anywhere in the line, after any
# logger prefix like "(APIServer pid=1) INFO:  ").
_ACCESS_RE = re.compile(r'(\S+) - "([A-Z]+) (\S+) HTTP/[\d.]+" (\d{3})')

# vLLM --enable-log-requests log lines.
# On vLLM ≥ 0.11.3 (PR #29227) the prompt is present at INFO level.
# vLLM formats with `prompt: %r` which adds quotes: prompt: 'text',
#   "Received request <id>: prompt: '<text>', params: SamplingParams(... max_tokens=<n>, ...)"
# On older vLLM the prompt is omitted (only at DEBUG level):
#   "Received request <id>: params: SamplingParams(... max_tokens=<n>, ...)"
# Two regexes: try the new format first (with prompt), then the old (without).
# re.DOTALL is needed because vLLM prompts (especially chat-template'd ones)
# contain actual newline characters that . wouldn't match otherwise.
_NEW_REQUEST_RE = re.compile(
    r"Received request (?P<id>\S+): prompt: '(?P<prompt>.*?)', "
    r"params: SamplingParams\(.*?max_tokens=(?P<tok>\d+)",
    re.DOTALL,
)
_OLD_REQUEST_RE = re.compile(
    r"Received request (?P<id>\S+): "
    r"params: SamplingParams\(.*?max_tokens=(?P<tok>\d+)"
)

# Max prompt length we display in the feed (truncated with …).
# vLLM itself truncates at max_log_len (default 1000); we truncate again
# to keep the terminal column manageable.
MAX_PROMPT_DISPLAY = 30

# vLLM request ids are "<prefix>-<hex>", and the prefix identifies the endpoint
# that produced them. The uvicorn access line carries the real path but no id;
# the request-log line carries the id but no path — so we recover the endpoint
# from the prefix to give request-log-driven rows a meaningful endpoint column.
_ID_PREFIX_ENDPOINT = {
    "chatcmpl": "/v1/chat/completions",
    "cmpl": "/v1/completions",
    "embd": "/v1/embeddings",
    "pool": "/pooling",
    "score": "/score",
    "rerank": "/rerank",
    "classify": "/classify",
}


def endpoint_for_request_id(request_id: str) -> str:
    """Best-effort map a vLLM request id to the endpoint that produced it."""
    prefix = request_id.split("-", 1)[0]
    return _ID_PREFIX_ENDPOINT.get(prefix, f"/{prefix}" if prefix else "?")


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
        self._merged: Deque[MergedLogEntry] = deque(maxlen=maxlen)
        self._req_logging_seen = False  # True once a request-log line is parsed
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
        now = time.time()

        # A request-log line (--enable-log-requests) is self-describing: it
        # carries the request id, max_tokens and (vLLM ≥ 0.11.3) the prompt. We
        # surface it directly as a feed row rather than trying to correlate it
        # with the access line — the access line has no request id and is logged
        # at a different point in the request's life, so the two can't be
        # matched reliably under concurrency.
        m = _NEW_REQUEST_RE.search(line) or _OLD_REQUEST_RE.search(line)
        if m:
            gd = m.groupdict()
            prompt = gd.get('prompt')  # None for old format (no 'prompt' group)
            entry = MergedLogEntry(
                t=now, method="POST",
                path=endpoint_for_request_id(gd['id']),
                request_id=gd['id'], max_tokens=int(gd['tok']),
                prompt=prompt if prompt else None,
            )
            with self._lock:
                self._req_logging_seen = True
                self._merged.append(entry)
            return

        # A uvicorn access line carries the client and HTTP status but no prompt.
        # When request logging is on, every inference request already produces a
        # request-log row, so the access line is redundant and we skip it. When
        # request logging is off, access lines are the only per-request signal,
        # so they become the feed.
        parsed = parse_access_line(line)
        if parsed is not None:
            client, method, path, status = parsed
            if path.split("?", 1)[0] in self.IGNORE_PATHS:
                return
            with self._lock:
                if self._req_logging_seen:
                    return
                self._merged.append(MergedLogEntry(
                    t=now, client=client, method=method,
                    path=path, status=status))

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

    def merged_log(self, n: Optional[int] = None) -> List[MergedLogEntry]:
        """Return merged entries, newest first (at most ``n``).

        Each entry carries access-log fields (method, path, status, client)
        optionally enriched with request-log fields (request_id, max_tokens, prompt)
        when vLLM runs with --enable-log-requests. Prompt text is available on
        vLLM ≥ 0.11.3 (PR #29227).
        """
        with self._lock:
            items = list(self._merged)
        items.reverse()
        return items if n is None else items[:n]

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
