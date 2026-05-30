from vllmpytop.collectors.access_log import AccessLogTailer, parse_access_line


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
    snap = tailer.snapshot()
    assert len(snap) == 1
    assert snap[0].path == "/v1/chat/completions"
    assert snap[0].client == "9.9.9.9:8"
    assert snap[0].ok is True
