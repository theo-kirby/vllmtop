from vllmpytop.collectors.access_log import (
    AccessLogTailer, MAX_PROMPT_DISPLAY, parse_access_line)
from vllmpytop.ui.panels import _truncate_prompt


def test_parse_real_vllm_line():
    line = ('(APIServer pid=1) INFO:     192.168.32.2:41854 - '
            '"POST /v1/chat/completions HTTP/1.1" 200 OK')
    assert parse_access_line(line) == (
        "192.168.32.2:41854", "POST", "/v1/chat/completions", 200)


def test_parse_status_codes_and_methods():
    assert parse_access_line('1.2.3.4:5 - "GET /metrics HTTP/1.1" 200 OK') == (
        "1.2.3.4:5", "GET", "/metrics", 200)
    assert parse_access_line('1.2.3.4:5 - "POST /v1/x HTTP/1.1" 404 Not Found')[3] == 404


def test_parse_non_access_lines_return_none():
    assert parse_access_line("Successfully import tool parser Qwen3XMLToolParser !") is None
    assert parse_access_line(
        "INFO 05-30 [loggers.py:271] Engine 000: Avg prompt throughput: 1754") is None
    assert parse_access_line("") is None


def test_tailer_filters_infra_endpoints():
    tailer = AccessLogTailer(file="/nonexistent")  # not started
    tailer._ingest('1.1.1.1:2 - "GET /metrics HTTP/1.1" 200 OK')  # ignored
    tailer._ingest('1.1.1.1:2 - "GET /health HTTP/1.1" 200 OK')  # ignored
    tailer._ingest('9.9.9.9:8 - "POST /v1/chat/completions HTTP/1.1" 200 OK')
    snap = tailer.merged_log()
    assert len(snap) == 1
    assert snap[0].path == "/v1/chat/completions"
    assert snap[0].client == "9.9.9.9:8"
    assert snap[0].ok is True


def test_prompt_parse_vllm_new_format():
    """vLLM ≥ 0.11.3 (PR #29227) includes prompt in the INFO log line.

    vLLM formats with `prompt: %r` which wraps text in quotes:
    prompt: 'Hello, how are you?',
    """
    tailer = AccessLogTailer(file="/nonexistent")
    tailer._ingest(
        "Received request chatcmpl-abc: prompt: 'Hello, how are you?', "
        "params: SamplingParams(n=1, max_tokens=100), lora_request: None."
    )
    # Prompt goes into pending; no access line yet so merged_log is empty
    assert len(tailer.merged_log()) == 0
    # Now ingest the access line to trigger the merge
    tailer._ingest(
        '1.2.3.4:5678 - "POST /v1/chat/completions HTTP/1.1" 200 OK'
    )
    entries = tailer.merged_log()
    assert len(entries) == 1
    e = entries[0]
    assert e.request_id == "chatcmpl-abc"
    assert e.max_tokens == 100
    # Prompt is unquoted (regex strips the surrounding quotes from %r format)
    assert e.prompt == "Hello, how are you?"


def test_prompt_parse_vllm_old_format():
    """Older vLLM (< 0.11.3) omits the prompt from INFO log lines."""
    tailer = AccessLogTailer(file="/nonexistent")
    tailer._ingest(
        "Received request chatcmpl-xyz: "
        "params: SamplingParams(n=1, max_tokens=50), lora_request: None."
    )
    tailer._ingest(
        '1.2.3.4:9999 - "POST /v1/completions HTTP/1.1" 200 OK'
    )
    entries = tailer.merged_log()
    assert len(entries) == 1
    e = entries[0]
    assert e.request_id == "chatcmpl-xyz"
    assert e.max_tokens == 50
    assert e.prompt is None


def test_truncate_prompt():
    assert _truncate_prompt("short") == "short"
    assert _truncate_prompt("a" * 30) == "a" * 30
    assert _truncate_prompt("a" * 31) == "a" * 29 + "…"
    assert _truncate_prompt(None) == ""
    assert _truncate_prompt("") == ""


def test_max_prompt_display_constant():
    assert 10 <= MAX_PROMPT_DISPLAY <= 60  # reasonable terminal column width
