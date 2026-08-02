# Container Checkpoint/Restore Benchmark

> Archived prototype: these scripts predate the current Crab daemon, SDK, and
> installer. They are not part of the supported v0 workflow. Start with the
> [current documentation](../docs/README.md); see [archived documentation](docs/README.md)
> for other historical design and experiment notes.

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
- `bench_runc_mem_sweep.py`: Sweep/plot runc latency vs `mem_mb`.
- `bench_runc_concurrency_sweep.py`: Sweep/plot runc latency vs `concurrency`.
- `bench_runc_mem_concurrency_sweep.py`: Sweep/plot runc latency vs (`mem_mb`, `concurrency`).
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

# concurrent checkpoint/restore of 4 containers per iteration
python3 bench_cr.py --iters 3 --concurrency 4 --out results_conc4.csv
```

Run Docker-only mode:

```bash
python3 bench_cr.py --run-docker --iters 3 --out results.csv
```

Run `runc`+CRIU-only mode:

```bash
python3 bench_cr.py --run-runc --iters 3 --out results.csv
```

Run `runc` latency sweep vs `concurrency`:

```bash
python3 bench_runc_concurrency_sweep.py --iters 3 --concurrency-values 1,2,4,8
```

Run `runc` latency sweep vs (`mem_mb`, `concurrency`):

```bash
python3 bench_runc_mem_concurrency_sweep.py --iters 3 --mem-values 128,512,1024 --concurrency-values 1,2,4
```

## Important Notes

- `runc` benchmark artifacts are written under `./bench_out`.
- The runner uses `sudo` for `runc run/checkpoint/restore/delete`.
- Docker checkpoint size is read from `--docker-root` (default `/var/lib/docker`).
- Docker HTTP workload port is `--http-port-base + (iter * concurrency + slot)` (default base `18080`).
- `runc` HTTP workload port is `--runc-http-port-base + (iter * concurrency + slot)` (default base `19080`).
- Some Docker runtimes can create checkpoints in `--checkpoint-dir` but cannot restore from it.
  Use `--docker-custom-ckpt-restore daemon` to bridge custom checkpoint files into daemon-managed storage before restore.
  Use `--docker-custom-ckpt-bridge copy|symlink|hardlink` to pick bridge behavior (default `copy`).
- `--concurrency` defaults to `1`. With `--concurrency > 1`, checkpoint/restore is executed in parallel batches per iteration.
- Use `--per-container-rows` to emit one row per container in addition to aggregate per-iteration rows.
- `--use-custom-checkpoint-dir` currently only supports `--concurrency 1`.

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
- `concurrency`
- `containers_total`
- `containers_ok`
- `row_kind` (`aggregate` or `container`)
- `container_slot`
- `container_name`

`counter_continues=True` indicates the workload state advanced across restore.
`http_runtime_same=True` + `http_seq_continues=True` indicates the restored process kept serving with the same in-memory runtime identity and continued HTTP request sequence.
