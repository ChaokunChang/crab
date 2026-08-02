"""Compare benchmark.spec.* metrics across the incremental_demo variants.

Reads each variant's telemetry JSONL plus its result CSV and prints a side-by-
side table of fork_restore_ms (p50/p95/p99/mean), spec accept rate, spec net
gain, total wall time, and the new chain-sharing telemetry counters. No I/O
into the benchmark itself; safe to re-run after each variant completes.

Usage:
    python3 benchmarks/examples/terminus/incremental_demo_compare.py
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_ROOT = REPO_ROOT / "logs" / "terminus"

VARIANTS = [
    ("baseline", "spec.auto.incremental_demo.baseline"),
    ("chain_share", "spec.auto.incremental_demo.chain_sharing"),
    ("prefork", "spec.auto.incremental_demo.prefork"),
    ("lazy", "spec.auto.incremental_demo.lazy"),
    ("b_plus_d", "spec.auto.incremental_demo.b_plus_d"),
    ("all_opts", "spec.auto.incremental_demo.all_opts"),
]


def _load_metric_values(telemetry_path: Path, metric_name: str) -> list[float]:
    if not telemetry_path.exists():
        return []
    values: list[float] = []
    with telemetry_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("kind") != "metric":
                continue
            if record.get("name") != metric_name:
                continue
            try:
                values.append(float(record.get("value", 0.0)))
            except (TypeError, ValueError):
                continue
    return values


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def _summarize_metric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "p50": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _aggregate_csv(csv_path: Path) -> dict[str, float]:
    if not csv_path.exists():
        return {}
    total_runtime = 0.0
    accept_total = 0
    reject_total = 0
    saved_total = 0.0
    penalty_total = 0.0
    fork_create_total = 0
    fork_reuse_total = 0
    success_count = 0
    rows = 0
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            try:
                total_runtime += float(row.get("task_completion_ms") or 0.0)
            except ValueError:
                pass
            try:
                if float(row.get("success_ratio") or 0.0) >= 1.0:
                    success_count += 1
            except ValueError:
                pass
            accept_total += int(row.get("spec_accept_count") or 0)
            reject_total += int(row.get("spec_reject_count") or 0)
            try:
                saved_total += float(row.get("spec_saved_ms") or 0.0)
                penalty_total += float(row.get("spec_penalty_ms") or 0.0)
            except ValueError:
                pass
            fork_create_total += int(row.get("spec_fork_create_count") or 0)
            fork_reuse_total += int(row.get("spec_fork_reuse_count") or 0)
    accept_rate = (
        accept_total / (accept_total + reject_total)
        if (accept_total + reject_total) > 0
        else 0.0
    )
    return {
        "rows": rows,
        "success_count": success_count,
        "task_completion_ms_sum": total_runtime,
        "spec_accept_count": accept_total,
        "spec_reject_count": reject_total,
        "spec_accept_rate": accept_rate,
        "spec_saved_ms_sum": saved_total,
        "spec_penalty_ms_sum": penalty_total,
        "spec_net_gain_ms_sum": saved_total - penalty_total,
        "spec_fork_create_count": fork_create_total,
        "spec_fork_reuse_count": fork_reuse_total,
    }


def _format_row(values: list[float], precision: int = 1) -> str:
    fmt = "{:>10." + str(precision) + "f}"
    return " ".join(fmt.format(v) for v in values)


def main() -> None:
    print(f"{'metric':<32}", end="")
    for label, _ in VARIANTS:
        print(f"{label:>14}", end="")
    print()
    print("=" * (32 + 14 * len(VARIANTS)))

    rows_by_variant: dict[str, dict[str, object]] = {}
    for label, stem in VARIANTS:
        tele = LOG_ROOT / f"{stem}.telemetry.jsonl"
        csv_path = LOG_ROOT / f"{stem}.csv"
        fork_restore = _summarize_metric(_load_metric_values(tele, "benchmark.spec.fork_restore_ms"))
        net_gain = _summarize_metric(_load_metric_values(tele, "benchmark.spec.net_gain_ms"))
        chain_links = sum(_load_metric_values(tele, "benchmark.fork.chain_sharing_links"))
        chain_bytes = sum(_load_metric_values(tele, "benchmark.fork.chain_sharing_bytes_saved"))
        agg = _aggregate_csv(csv_path)
        rows_by_variant[label] = {
            "fork_restore_n": fork_restore["n"],
            "fork_restore_mean": fork_restore["mean"],
            "fork_restore_p50": fork_restore["p50"],
            "fork_restore_p95": fork_restore["p95"],
            "fork_restore_p99": fork_restore["p99"],
            "fork_restore_max": fork_restore["max"],
            "net_gain_mean": net_gain["mean"],
            "chain_links_total": chain_links,
            "chain_bytes_saved_total": chain_bytes,
            **agg,
        }

    metric_lines = [
        ("fork_restore_ms n", "fork_restore_n", 0),
        ("fork_restore_ms mean", "fork_restore_mean", 1),
        ("fork_restore_ms p50", "fork_restore_p50", 1),
        ("fork_restore_ms p95", "fork_restore_p95", 1),
        ("fork_restore_ms p99", "fork_restore_p99", 1),
        ("fork_restore_ms max", "fork_restore_max", 1),
        ("spec_accept_rate", "spec_accept_rate", 3),
        ("spec_net_gain_ms (sum)", "spec_net_gain_ms_sum", 1),
        ("spec_fork_create_count", "spec_fork_create_count", 0),
        ("spec_fork_reuse_count", "spec_fork_reuse_count", 0),
        ("task_completion_ms (sum)", "task_completion_ms_sum", 1),
        ("success_count", "success_count", 0),
        ("chain_sharing_links (total)", "chain_links_total", 0),
        ("chain_sharing_bytes_saved", "chain_bytes_saved_total", 0),
    ]

    for label, key, precision in metric_lines:
        print(f"{label:<32}", end="")
        for variant_label, _ in VARIANTS:
            value = rows_by_variant[variant_label].get(key, 0)
            try:
                fmt = "{:>14." + str(precision) + "f}"
                print(fmt.format(float(value)), end="")
            except (TypeError, ValueError):
                print(f"{str(value):>14}", end="")
        print()


if __name__ == "__main__":
    main()
