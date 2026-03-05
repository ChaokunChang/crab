# Container Checkpoint/Restore Benchmark

This repository benchmarks checkpoint/restore latency for a stateful Python workload using:

- Docker checkpoint/restore
- `runc` + CRIU checkpoint/restore

The workload (`agent_workload.py`) now does all of the following:

- Increments a counter and writes `/work/state.json`
- Mutates an in-memory buffer (to emulate agent runtime memory)
- Runs an HTTP server (`/healthz`) with in-memory request sequence + runtime ID

This lets the benchmark verify both disk-backed state continuity and in-memory HTTP runtime continuity across restore.

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
python3 bench_cr.py --iters 3 --out results.csv
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
- Docker HTTP workload port is `--http-port-base + iter` (default base `18080`).
- `runc` HTTP workload port is `--runc-http-port-base + iter` (default base `19080`).
- Some Docker runtimes can create checkpoints in `--checkpoint-dir` but cannot restore from it.
  Use `--docker-custom-ckpt-restore daemon` to bridge custom checkpoint files into daemon-managed storage before restore.
  Use `--docker-custom-ckpt-bridge copy|symlink|hardlink` to pick bridge behavior (default `copy`).

Example for tmpfs-backed checkpoint writes with daemon restore:

```bash
python3 bench_cr.py \
  --run-docker \
  --iters 3 \
  --out results_tmpfs_1G.csv \
  --mem-mb 1024 \
  --work-root /mnt/mytmpfs/bench_out \
  --use-custom-checkpoint-dir \
  --docker-custom-ckpt-restore daemon \
  --docker-custom-ckpt-bridge symlink
```

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
- `http_port`
- `http_before_ok`
- `http_after_ok`
- `http_runtime_same`
- `http_seq_before`
- `http_seq_after`
- `http_seq_continues`

`counter_continues=True` indicates the workload state advanced across restore.
`http_runtime_same=True` + `http_seq_continues=True` indicates the restored process kept serving with the same in-memory runtime identity and continued HTTP request sequence.
