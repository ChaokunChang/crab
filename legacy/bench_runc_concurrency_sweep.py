#!/usr/bin/env python3
"""Benchmark runc checkpoint/restore performance across concurrency values and plot results."""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--concurrency-values",
        default="1,2,4,8",
        help="Comma-separated concurrency values.",
    )
    p.add_argument("--iters", type=int, default=3, help="Iterations per concurrency value.")
    p.add_argument("--mem-mb", type=int, default=128, help="MEM_MB passed to bench_cr.py.")
    p.add_argument("--image", default="agent-sandbox-bench:latest", help="Benchmark image.")
    p.add_argument("--work-root", default="./bench_out", help="Work directory for bench_cr.py.")
    p.add_argument("--bench-script", default="./bench_cr.py", help="Path to bench_cr.py.")
    p.add_argument(
        "--out-dir",
        default="./bench_out/concurrency_sweep",
        help="Directory for per-concurrency CSVs, merged CSVs, and plots.",
    )
    p.add_argument(
        "--plot-file",
        default="runc_concurrency_sweep.png",
        help="Output plot filename (inside --out-dir).",
    )
    p.add_argument(
        "--summary-csv",
        default="runc_concurrency_sweep_summary.csv",
        help="Summary CSV filename (inside --out-dir).",
    )
    p.add_argument(
        "--raw-csv",
        default="runc_concurrency_sweep_raw.csv",
        help="Merged raw CSV filename (inside --out-dir).",
    )
    p.add_argument(
        "--runc-http-port-base",
        type=int,
        default=19080,
        help="Base HTTP port passed to bench_cr.py.",
    )
    p.add_argument(
        "--http-port-stride",
        type=int,
        default=1000,
        help="Per-sweep offset to avoid port reuse between runs.",
    )
    p.add_argument("--python", default=sys.executable, help="Python executable used to invoke bench_cr.py.")
    p.add_argument("--warmup-s", type=float, default=2.0)
    return p.parse_args()


def parse_int_values(raw: str, field_name: str) -> list[int]:
    values = []
    for part in raw.split(","):
        v = int(part.strip())
        if v <= 0:
            raise ValueError(f"{field_name} must be > 0, got {v}")
        values.append(v)
    if not values:
        raise ValueError(f"no {field_name} values provided")
    return sorted(set(values))


def run_one(
    bench_script: Path,
    python_exe: str,
    image: str,
    work_root: Path,
    mem_mb: int,
    concurrency: int,
    iters: int,
    out_csv: Path,
    runc_http_port_base: int,
    warmup_s: float,
) -> None:
    cmd = [
        python_exe,
        str(bench_script),
        "--run-runc",
        "--iters",
        str(iters),
        "--mem-mb",
        str(mem_mb),
        "--concurrency",
        str(concurrency),
        "--image",
        image,
        "--work-root",
        str(work_root),
        "--runc-http-port-base",
        str(runc_http_port_base),
        "--out",
        str(out_csv),
        "--warmup-s",
        str(warmup_s),
    ]
    print(f"\n### concurrency={concurrency}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def load_rows(csv_path: Path, mem_mb: int, concurrency: int) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    loaded: list[dict[str, str]] = []
    for row in rows:
        if row.get("method") != "runc-criu":
            continue
        if row.get("row_kind", "aggregate") != "aggregate":
            continue
        row["mem_mb"] = str(mem_mb)
        row["concurrency"] = str(concurrency)
        loaded.append(row)
    return loaded


def safe_float(row: dict[str, str], key: str) -> float | None:
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def safe_bool(row: dict[str, str], key: str) -> bool | None:
    v = row.get(key)
    if v is None or v == "":
        return None
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    return None


def summarize(rows: list[dict[str, str]], conc_values: list[int]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for conc in conc_values:
        conc_rows = [r for r in rows if int(r["concurrency"]) == conc]
        ckpt_ms = [x for x in (safe_float(r, "checkpoint_ms") for r in conc_rows) if x is not None]
        restore_ms = [x for x in (safe_float(r, "restore_ms") for r in conc_rows) if x is not None]
        ckpt_size_mb = [
            x / (1024 * 1024)
            for x in (safe_float(r, "ckpt_size_bytes") for r in conc_rows)
            if x is not None
        ]
        http_after_ok = [x for x in (safe_bool(r, "http_after_ok") for r in conc_rows) if x is not None]
        summary.append(
            {
                "concurrency": conc,
                "n": len(conc_rows),
                "checkpoint_ms_mean": statistics.fmean(ckpt_ms) if ckpt_ms else None,
                "checkpoint_ms_median": statistics.median(ckpt_ms) if ckpt_ms else None,
                "restore_ms_mean": statistics.fmean(restore_ms) if restore_ms else None,
                "restore_ms_median": statistics.median(restore_ms) if restore_ms else None,
                "ckpt_size_mb_mean": statistics.fmean(ckpt_size_mb) if ckpt_size_mb else None,
                "http_after_ok_ratio": (
                    sum(1 for x in http_after_ok if x) / len(http_after_ok)
                    if http_after_ok
                    else None
                ),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def make_plot(summary_rows: list[dict[str, object]], out_path: Path, mem_mb: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from e

    x = [int(r["concurrency"]) for r in summary_rows]
    ckpt = [r["checkpoint_ms_mean"] for r in summary_rows]
    restore = [r["restore_ms_mean"] for r in summary_rows]
    size = [r["ckpt_size_mb_mean"] for r in summary_rows]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(x, ckpt, marker="o", label="checkpoint_ms_mean")
    axes[0].plot(x, restore, marker="o", label="restore_ms_mean")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_title(f"runc+CRIU latency vs concurrency (MEM_MB={mem_mb})")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, size, marker="o", color="tab:green", label="ckpt_size_mb_mean")
    axes[1].set_ylabel("Checkpoint Size (MB)")
    axes[1].set_xlabel("Concurrency")
    axes[1].set_title("Checkpoint size vs concurrency")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    conc_values = parse_int_values(args.concurrency_values, "concurrency")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    bench_script = Path(args.bench_script)

    if not bench_script.exists():
        raise FileNotFoundError(f"bench script not found: {bench_script}")

    all_rows: list[dict[str, str]] = []
    for idx, conc in enumerate(conc_values):
        run_csv = out_dir / f"results_runc_conc_{conc}.csv"
        run_one(
            bench_script=bench_script,
            python_exe=args.python,
            image=args.image,
            work_root=work_root,
            mem_mb=args.mem_mb,
            concurrency=conc,
            iters=args.iters,
            out_csv=run_csv,
            runc_http_port_base=args.runc_http_port_base + idx * args.http_port_stride,
            warmup_s=args.warmup_s,
        )
        all_rows.extend(load_rows(run_csv, mem_mb=args.mem_mb, concurrency=conc))

    raw_csv = out_dir / args.raw_csv
    if all_rows:
        raw_fields = [
            "mem_mb",
            "concurrency",
            "iter",
            "method",
            "checkpoint_ms",
            "restore_ms",
            "restore_ms_success_only",
            "restore_retries",
            "ckpt_size_bytes",
            "counter_before",
            "counter_after",
            "counter_continues",
            "http_port",
            "http_before_ok",
            "http_after_ok",
            "http_runtime_same",
            "http_seq_before",
            "http_seq_after",
            "http_seq_continues",
            "containers_total",
            "containers_ok",
            "row_kind",
            "container_slot",
            "container_name",
        ]
        write_csv(raw_csv, all_rows, raw_fields)

    summary_rows = summarize(all_rows, conc_values)
    summary_csv = out_dir / args.summary_csv
    summary_fields = [
        "concurrency",
        "n",
        "checkpoint_ms_mean",
        "checkpoint_ms_median",
        "restore_ms_mean",
        "restore_ms_median",
        "ckpt_size_mb_mean",
        "http_after_ok_ratio",
    ]
    write_csv(summary_csv, summary_rows, summary_fields)

    plot_path = out_dir / args.plot_file
    make_plot(summary_rows, plot_path, mem_mb=args.mem_mb)

    print("\nDone.")
    print(f"Raw results:     {raw_csv}")
    print(f"Summary results: {summary_csv}")
    print(f"Plot:            {plot_path}")


if __name__ == "__main__":
    main()
