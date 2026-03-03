# Container Checkpoint/Restore Benchmark

This repository benchmarks checkpoint/restore latency for a stateful Python workload using:

- Docker checkpoint/restore
- `runc` + CRIU checkpoint/restore

The workload (`agent_workload.py`) increments a counter, mutates an in-memory buffer, and persists progress to `/work/state.json` so restore continuity can be verified.

## Files

- `bench_cr.py`: Benchmark runner for Docker and `runc`+CRIU paths.
- `agent_workload.py`: Stateful test workload executed inside the container.
- `Dockerfile`: Image definition for the workload container.
- `results.csv`: Example output.

## Prerequisites

- Linux host with checkpoint/restore support
- Python 3.11+
- Docker with checkpoint support enabled
- `runc` and CRIU installed on host (`runc --version`, `criu --version`)
- Privileges for `runc` checkpoint/restore (typically root/sudo)

## Quick Start

Build benchmark image:

```bash
docker build -t agent-sandbox-bench:latest .
```

Run both benchmark modes (default):

```bash
python3 bench_cr.py --iters 5 --out results.csv
```

Run Docker-only mode:

```bash
python3 bench_cr.py --run-docker --iters 3 --out results.csv
```

Run `runc`+CRIU-only mode:

```bash
python3 bench_cr.py --run-runc --iters 3 --out results.csv
```

## Important Notes

- `runc` benchmark artifacts are written under `./bench_out`.
- The runner uses `sudo` for `runc run/checkpoint/restore/delete`.
- Docker checkpoint size is read from `--docker-root` (default `/var/lib/docker`).

## Output

`results.csv` contains:

- `iter`
- `method`
- `checkpoint_ms`
- `restore_ms`
- `ckpt_size_bytes`
- `counter_before`
- `counter_after`
- `counter_continues`

`counter_continues=True` indicates the workload state advanced across restore.
