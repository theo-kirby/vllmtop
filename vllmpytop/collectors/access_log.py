"""Tail the vLLM server's log into a feed of inference requests.

vLLM's ``/metrics`` is aggregate-only and its request prompts are not logged
unless the server runs with ``--enable-log-requests``. With that flag vLLM
emits a request-log line per inference call, e.g.::

    Received request chatcmpl-abc: prompt: 'Hello', params: SamplingParams(... max_tokens=100 ...)

So this collector parses those lines (request id, max_tokens and, on vLLM ≥
0.11.3 via PR #29227, the prompt text truncated by ``max_log_len``) into a
rolling :class:`MergedLogEntry` buffer. Uvicorn access lines (the HTTP
envelope and status) are ignored — only the actual inference requests are
shown.

The source is either a file (``--log-file``) or the stdout of a streaming
command such as ``docker logs -f`` (``--docker``). A background thread follows
it; the UI reads :meth:`AccessLogTailer.merged_log` each frame.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from collections import deque
from typing import Deque, List, Optional

from ..state import MergedLogEntry

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


class AccessLogTailer(threading.Thread):
    """Follows a log source and parses vLLM request-log lines into a buffer.

    Starts at the *end* of the source (like ``tail -f``) so only requests
    observed while running are shown — that keeps each entry's age meaningful.
    Any failure (missing file, no ``docker``, container gone) is surfaced via
    :attr:`error` rather than crashing the UI.
    """

    def __init__(self, *, file: Optional[str] = None,
                 docker: Optional[str] = None, maxlen: int = 200) -> None:
        super().__init__(daemon=True)
        self._file = file
        self._docker = docker
        self._merged: Deque[MergedLogEntry] = deque(maxlen=maxlen)
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
        # We only surface vLLM request-log lines (--enable-log-requests): they
        # are self-describing, carrying the request id, max_tokens and (vLLM ≥
        # 0.11.3) the prompt — the actual inference requests. Uvicorn access
        # lines (the HTTP envelope and status) are deliberately ignored: they
        # have no request id and are logged at a different point in the
        # request's life, so they can't be correlated, and the prompt/tokens
        # are what's interesting anyway.
        m = _NEW_REQUEST_RE.search(line) or _OLD_REQUEST_RE.search(line)
        if not m:
            return
        gd = m.groupdict()
        prompt = gd.get('prompt')  # None for old format (no 'prompt' group)
        with self._lock:
            self._merged.append(MergedLogEntry(
                t=time.time(), method="POST",
                path=endpoint_for_request_id(gd['id']),
                request_id=gd['id'], max_tokens=int(gd['tok']),
                prompt=prompt if prompt else None,
                prompt_chars=len(prompt) if prompt else None,
            ))

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
