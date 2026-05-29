# vllmtop

A **btop-style terminal UI** for monitoring a running [vLLM](https://github.com/vllm-project/vllm)
instance and its GPU in real time. Hand-rolled braille charts, a responsive
curses layout, and a non-blocking background poller so the UI never stalls on
network or NVML latency.

![tui](docs/ui.png)

Rounded corners, superscript panel numbers in the title tabs, and a secondary
label on the bottom edge — matching [btop](https://github.com/aristocratos/btop)'s box style.

## What it shows

- **GPU** (via NVML / `pynvml`): utilisation %, VRAM used/total, temperature,
  power draw vs. limit, SM clock, fan — with green/yellow/red thresholds.
- **Throughput**: generation tok/s and prompt tok/s (rates derived from vLLM
  counters), as big numbers + braille charts.
- **Requests / Queue**: running vs. waiting requests and preemptions.
- **Latency** (recent average over the last poll interval — far more useful live
  than the cumulative average): TTFT, inter-token (TPOT), end-to-end, queue time.
- **Cache**: KV-cache usage % and prefix-cache hit rate.

Data comes from vLLM's Prometheus `/metrics` endpoint plus in-process NVML
polling. If vLLM goes away (e.g. a container restart) the UI shows a disconnect
banner and keeps the GPU panel live, then reconnects automatically.

## Install

Requires Python 3.10+ on Linux (curses is stdlib). A working NVIDIA driver is
needed for the GPU panel.

```bash
pip install .
# or, for development:
pip install -e ".[dev]"
```

Dependencies: `nvidia-ml-py` (NVML bindings) and `prometheus-client` (exposition
parser). The `/metrics` fetch uses stdlib `urllib`.

## Usage

```bash
vllmtop                              # monitor http://localhost:8000
vllmtop --url http://host:8000       # a remote vLLM server
vllmtop --interval 0.5               # poll twice a second
vllmtop --no-gpu                     # skip the GPU panel
python -m vllmtop                    # same thing, without the entry point
```

The server URL can also be set via the `VLLMTOP_URL` environment variable.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--url` | `http://localhost:8000` | vLLM base URL (env `VLLMTOP_URL`) |
| `--interval` | `1.0` | poll interval in seconds |
| `--gpu-index` | `0` | NVML GPU index |
| `--no-gpu` | off | disable the GPU panel |
| `--dump-json` | off | collect one snapshot, print JSON, exit (no TTY) |

### Keybindings

| Key | Action |
|-----|--------|
| `q` / `Esc` | quit |
| `+` / `-` | faster / slower refresh |
| `p` | pause / resume polling |
| `1`–`5` | toggle a panel on/off (¹gpu ²throughput ³requests ⁴latency ⁵cache) |
| `h` / `?` | toggle help overlay |

Each panel's title carries a superscript number (btop-style) showing the key
that toggles it. Hiding panels reflows the rest to fill the freed space.

### Headless smoke test

`--dump-json` collects two snapshots an interval apart (so rates are populated),
prints the result as JSON, and exits. Works without a TTY — handy for CI or
verifying connectivity:

```bash
python -m vllmtop --dump-json --url http://localhost:8000
```

## How it works

- A **background poller thread** scrapes `/metrics` and polls NVML every
  `interval` seconds, storing the latest combined snapshot under a lock. This
  keeps all I/O latency off the render path.
- The **UI loop** wakes on a short tick (250 ms), reads the latest snapshot,
  appends derived values (rates, recent-average latencies) to per-series ring
  buffers, and redraws — so render cadence is independent of poll cadence.
- **Counters → rates**: `Δvalue / Δt`, guarded against `Δt ≤ 0` and counter
  resets. **Histograms → recent average**: `Δsum / Δcount` between polls.
- **Braille charts**: each cell is a 2×4 Unicode braille dot matrix, giving
  `2w × 4h`-dot resolution for the smooth btop look.

## Development

```bash
pytest        # parser-against-fixture, rate math, braille rendering
```

## License

MIT — see [LICENSE](LICENSE).
