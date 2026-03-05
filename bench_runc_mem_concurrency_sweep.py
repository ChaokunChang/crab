#!/usr/bin/env python3
"""Benchmark runc latency across (mem_mb, concurrency) grid and plot heatmaps."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mem-values",
        default="64,128,256,512,1024,2048",
        help="Comma-separated memory sizes in MB.",
    )
    p.add_argument(
        "--concurrency-values",
        default="1,2,4,8",
        help="Comma-separated concurrency values.",
    )
    p.add_argument("--iters", type=int, default=3, help="Iterations per (mem, concurrency) point.")
    p.add_argument("--image", default="agent-sandbox-bench:latest", help="Benchmark image.")
    p.add_argument("--work-root", default="./bench_out", help="Work directory for bench_cr.py.")
    p.add_argument("--bench-script", default="./bench_cr.py", help="Path to bench_cr.py.")
    p.add_argument(
        "--out-dir",
        default="./bench_out/mem_concurrency_sweep",
        help="Directory for per-point CSVs, merged CSVs, and plots.",
    )
    p.add_argument(
        "--plot-file",
        default="runc_mem_concurrency_sweep.png",
        help="Output plot filename (inside --out-dir).",
    )
    p.add_argument(
        "--summary-csv",
        default="runc_mem_concurrency_sweep_summary.csv",
        help="Summary CSV filename (inside --out-dir).",
    )
    p.add_argument(
        "--raw-csv",
        default="runc_mem_concurrency_sweep_raw.csv",
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
        help="Per-grid-point offset to avoid port reuse between sweeps.",
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
    print(f"\n### mem={mem_mb}MB, conc={concurrency}: {' '.join(cmd)}", flush=True)
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


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def summarize(rows: list[dict[str, str]], mem_values: list[int], conc_values: list[int]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for mem in mem_values:
        for conc in conc_values:
            point_rows = [
                r for r in rows if int(r["mem_mb"]) == mem and int(r["concurrency"]) == conc
            ]
            ckpt = [x for x in (safe_float(r, "checkpoint_ms") for r in point_rows) if x is not None]
            restore = [x for x in (safe_float(r, "restore_ms") for r in point_rows) if x is not None]
            size_mb = [
                x / (1024 * 1024)
                for x in (safe_float(r, "ckpt_size_bytes") for r in point_rows)
                if x is not None
            ]
            http_after_ok = [x for x in (safe_bool(r, "http_after_ok") for r in point_rows) if x is not None]
            summary.append(
                {
                    "mem_mb": mem,
                    "concurrency": conc,
                    "n": len(point_rows),
                    "checkpoint_ms_mean": sum(ckpt) / len(ckpt) if ckpt else None,
                    "restore_ms_mean": sum(restore) / len(restore) if restore else None,
                    "ckpt_size_mb_mean": sum(size_mb) / len(size_mb) if size_mb else None,
                    "http_after_ok_ratio": (
                        sum(1 for x in http_after_ok if x) / len(http_after_ok)
                        if http_after_ok
                        else None
                    ),
                }
            )
    return summary


def _matrix(
    summary_rows: list[dict[str, object]],
    mem_values: list[int],
    conc_values: list[int],
    key: str,
) -> list[list[float]]:
    lookup = {(int(r["mem_mb"]), int(r["concurrency"])): r.get(key) for r in summary_rows}
    data = []
    for mem in mem_values:
        row = []
        for conc in conc_values:
            v = lookup.get((mem, conc))
            row.append(float(v) if v is not None else float("nan"))
        data.append(row)
    return data


def make_plot(
    summary_rows: list[dict[str, object]],
    mem_values: list[int],
    conc_values: list[int],
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        raise RuntimeError(
            "matplotlib and numpy are required for plotting. Install with: pip install matplotlib numpy"
        ) from e

    ckpt_mat = np.array(_matrix(summary_rows, mem_values, conc_values, "checkpoint_ms_mean"))
    restore_mat = np.array(_matrix(summary_rows, mem_values, conc_values, "restore_ms_mean"))
    size_mat = np.array(_matrix(summary_rows, mem_values, conc_values, "ckpt_size_mb_mean"))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)

    im0 = axes[0].imshow(ckpt_mat, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_title("checkpoint_ms_mean")
    axes[0].set_xlabel("Concurrency")
    axes[0].set_ylabel("MEM_MB")
    axes[0].set_xticks(range(len(conc_values)), labels=[str(x) for x in conc_values])
    axes[0].set_yticks(range(len(mem_values)), labels=[str(x) for x in mem_values])
    fig.colorbar(im0, ax=axes[0], label="ms")

    im1 = axes[1].imshow(restore_mat, aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("restore_ms_mean")
    axes[1].set_xlabel("Concurrency")
    axes[1].set_ylabel("MEM_MB")
    axes[1].set_xticks(range(len(conc_values)), labels=[str(x) for x in conc_values])
    axes[1].set_yticks(range(len(mem_values)), labels=[str(x) for x in mem_values])
    fig.colorbar(im1, ax=axes[1], label="ms")

    im2 = axes[2].imshow(size_mat, aspect="auto", origin="lower", cmap="cividis")
    axes[2].set_title("ckpt_size_mb_mean")
    axes[2].set_xlabel("Concurrency")
    axes[2].set_ylabel("MEM_MB")
    axes[2].set_xticks(range(len(conc_values)), labels=[str(x) for x in conc_values])
    axes[2].set_yticks(range(len(mem_values)), labels=[str(x) for x in mem_values])
    fig.colorbar(im2, ax=axes[2], label="MB")

    fig.suptitle("runc+CRIU latency vs (MEM_MB, concurrency)")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    mem_values = parse_int_values(args.mem_values, "mem")
    conc_values = parse_int_values(args.concurrency_values, "concurrency")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    bench_script = Path(args.bench_script)

    if not bench_script.exists():
        raise FileNotFoundError(f"bench script not found: {bench_script}")

    all_rows: list[dict[str, str]] = []
    point_idx = 0
    for mem in mem_values:
        for conc in conc_values:
            run_csv = out_dir / f"results_runc_mem_{mem}_conc_{conc}.csv"
            run_one(
                bench_script=bench_script,
                python_exe=args.python,
                image=args.image,
                work_root=work_root,
                mem_mb=mem,
                concurrency=conc,
                iters=args.iters,
                out_csv=run_csv,
                runc_http_port_base=args.runc_http_port_base + point_idx * args.http_port_stride,
                warmup_s=args.warmup_s,
            )
            all_rows.extend(load_rows(run_csv, mem_mb=mem, concurrency=conc))
            point_idx += 1

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

    summary_rows = summarize(all_rows, mem_values, conc_values)
    summary_csv = out_dir / args.summary_csv
    summary_fields = [
        "mem_mb",
        "concurrency",
        "n",
        "checkpoint_ms_mean",
        "restore_ms_mean",
        "ckpt_size_mb_mean",
        "http_after_ok_ratio",
    ]
    write_csv(summary_csv, summary_rows, summary_fields)

    plot_path = out_dir / args.plot_file
    make_plot(summary_rows, mem_values, conc_values, plot_path)

    print("\nDone.")
    print(f"Raw results:     {raw_csv}")
    print(f"Summary results: {summary_csv}")
    print(f"Plot:            {plot_path}")


if __name__ == "__main__":
    main()
