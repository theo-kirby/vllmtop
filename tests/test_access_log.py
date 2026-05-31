from vllmpytop.collectors.access_log import (
    AccessLogTailer, MAX_PROMPT_DISPLAY)
from vllmpytop.ui.panels import _truncate_prompt


def test_access_lines_are_ignored():
    """Uvicorn access lines (HTTP envelope/status) no longer produce rows."""
    tailer = AccessLogTailer(file="/nonexistent")  # not started
    tailer._ingest('9.9.9.9:8 - "POST /v1/chat/completions HTTP/1.1" 200 OK')
    tailer._ingest('1.1.1.1:2 - "GET /metrics HTTP/1.1" 200 OK')
    assert tailer.merged_log() == []


def test_prompt_parse_vllm_new_format():
    """vLLM ≥ 0.11.3 (PR #29227) includes prompt in the INFO log line.

    vLLM formats with `prompt: %r` which wraps text in quotes:
    prompt: 'Hello, how are you?',

    The request-log line is self-describing, so it becomes a feed row directly
    (endpoint inferred from the request-id prefix); status is None because the
    line is logged at arrival, before completion.
    """
    tailer = AccessLogTailer(file="/nonexistent")
    tailer._ingest(
        "Received request chatcmpl-abc: prompt: 'Hello, how are you?', "
        "params: SamplingParams(n=1, max_tokens=100), lora_request: None."
    )
    entries = tailer.merged_log()
    assert len(entries) == 1
    e = entries[0]
    assert e.request_id == "chatcmpl-abc"
    assert e.max_tokens == 100
    assert e.path == "/v1/chat/completions"  # inferred from "chatcmpl-" prefix
    assert e.status is None
    # Prompt is unquoted (regex strips the surrounding quotes from %r format)
    assert e.prompt == "Hello, how are you?"


def test_prompt_parse_vllm_old_format():
    """Older vLLM (< 0.11.3) omits the prompt from INFO log lines."""
    tailer = AccessLogTailer(file="/nonexistent")
    tailer._ingest(
        "Received request chatcmpl-xyz: "
        "params: SamplingParams(n=1, max_tokens=50), lora_request: None."
    )
    entries = tailer.merged_log()
    assert len(entries) == 1
    e = entries[0]
    assert e.request_id == "chatcmpl-xyz"
    assert e.max_tokens == 50
    assert e.path == "/v1/chat/completions"
    assert e.prompt is None


def test_endpoint_for_request_id():
    from vllmpytop.collectors.access_log import endpoint_for_request_id
    assert endpoint_for_request_id("chatcmpl-abc") == "/v1/chat/completions"
    assert endpoint_for_request_id("cmpl-abc") == "/v1/completions"
    assert endpoint_for_request_id("embd-abc") == "/v1/embeddings"
    assert endpoint_for_request_id("weird-abc") == "/weird"


def test_truncate_prompt():
    assert _truncate_prompt("short") == "short"
    assert _truncate_prompt("a" * 30) == "a" * 30
    assert _truncate_prompt("a" * 31) == "a" * 29 + "…"
    assert _truncate_prompt(None) == ""
    assert _truncate_prompt("") == ""


def test_max_prompt_display_constant():
    assert 10 <= MAX_PROMPT_DISPLAY <= 60  # reasonable terminal column width
