"""Compare N benchmark runs across many dimensions.

Use case: you ran the same dataset under several configs (e.g. nofault vs
spec, baseline vs incremental, the 5-way prefork/lazy/chain_sharing
ablation, throttled vs unthrottled) and want a side-by-side view of
where wall-clock, checkpoint cost, restore cost, fork-restore latency,
spec accept rate, replay cadence, and LLM overhead actually moved.

The script is opinionated: it leans on the `<prefix>.report/summary.json`
that the report pipeline already emits, so percentiles, per-task
rollups, and per-metric statistics come pre-computed. It falls back to
the `.telemetry.jsonl` + `.csv` pair only when the report dir is
missing.

Inputs are run "prefixes" — the shared stem of `.csv`, `.log`,
`.telemetry.jsonl`, and `.report/`:

    python3 -m scripts.compare_benchmark_runs \\
        logs/terminus/nofault.auto.spec_friendly.10tasks \\
        logs/terminus/spec.auto.spec_friendly.10tasks

You can also pass a `.report/` directory directly; the script strips the
`.report` suffix to find sibling artifacts. N>=2 runs are supported; the
first is treated as the baseline and subsequent runs render with deltas.

Sections rendered (auto-skipped when no run has the corresponding data):

    run_summary          wall-clock, success ratio, distinct counts
    verification         per-status counts from the main CSV
    checkpoint_cost      counts, scope mix, total/mean/p95 sizes, IO
    checkpoint_latency   flow / process / filesystem mean & p95 & p99
    restore              counts + flow/process/filesystem latency
    spec                 fork_restore_ms percentiles, accept rate,
                         spec_saved/penalty/net_gain, fork reuse
    cadence              fast-forward skips, pre/post/non-spec drains
    llm_overhead         interceptor delay & gate-wait percentiles
    per_task             per-sandbox wall-clock + verify (+ spec accepts)

Pass `--csv-out <path>` to also dump per-task data as a structured CSV
(one row per (run, task)).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# Benchmark CSVs embed verification stdout/stderr and the swebench JSON
# report in single columns, which routinely exceed Python's 128 KiB default.
csv.field_size_limit(sys.maxsize)


# ── Paths ───────────────────────────────────────────────────────────────────


@dataclass
class RunPaths:
    label: str
    prefix: Path
    report_dir: Path | None
    main_csv: Path | None
    telemetry: Path | None


_KNOWN_FILE_SUFFIXES = (".telemetry.jsonl", ".jsonl", ".csv", ".log", ".runner.log")


def _strip_known_file_suffix(name: str) -> str:
    for suffix in _KNOWN_FILE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _resolve_paths(arg: str) -> RunPaths:
    """Accept any of: bare prefix, file with known suffix, .report/ dir, or
    a report dir whose name ends `.report.<custom_suffix>` (the report
    pipeline writes `<stem>.report.<custom>` for ablation outputs)."""
    p = Path(arg)
    prefix: Path
    report_dir: Path | None = None

    if p.is_dir():
        if (p / "summary.json").is_file():
            report_dir = p
            name = p.name
            if name.endswith(".report"):
                prefix = p.with_name(name[: -len(".report")])
            elif ".report." in name:
                # `<stem>.report.<custom>` → `<stem>.<custom>`.
                idx = name.find(".report.")
                stem = name[:idx]
                custom = name[idx + len(".report") :]  # leading dot retained
                prefix = p.with_name(stem + custom)
            else:
                prefix = p
        else:
            prefix = p
    else:
        stripped = _strip_known_file_suffix(p.name)
        prefix = p.with_name(stripped)
        candidate = prefix.with_name(prefix.name + ".report")
        if candidate.is_dir() and (candidate / "summary.json").is_file():
            report_dir = candidate

    if report_dir is None:
        candidate = prefix.with_name(prefix.name + ".report")
        if candidate.is_dir() and (candidate / "summary.json").is_file():
            report_dir = candidate

    main_csv = prefix.with_name(prefix.name + ".csv")
    telemetry = prefix.with_name(prefix.name + ".telemetry.jsonl")

    return RunPaths(
        label=prefix.name,
        prefix=prefix,
        report_dir=report_dir,
        main_csv=main_csv if main_csv.is_file() else None,
        telemetry=telemetry if telemetry.is_file() else None,
    )


# ── Per-task ────────────────────────────────────────────────────────────────


@dataclass
class TaskRunStats:
    """Per-task rollup, keyed by leading task_run_id (e.g. `spec-3`)."""

    task_run_id: str
    task_id: str = ""
    sandbox_id: str = ""
    runtime_s: float = 0.0
    verification_status: str = "—"
    verification_ms: float = 0.0
    success_ratio: float | None = None

    # Spec (zero on nofault).
    spec_total_turns: int = 0
    spec_accept_count: int = 0
    spec_reject_count: int = 0
    spec_saved_ms: float = 0.0
    spec_penalty_ms: float = 0.0
    spec_hidden_penalty_ms: float = 0.0
    spec_net_gain_ms: float = 0.0
    spec_fork_create_count: int = 0
    spec_fork_reuse_count: int = 0

    # Replay cadence (terminus).
    fast_forward_skip_count: int = 0
    fast_forward_saved_ms: float = 0.0
    pre_fork_drain_count: int = 0
    pre_fork_drain_ms: float = 0.0
    post_match_drain_count: int = 0
    post_match_drain_ms: float = 0.0
    non_spec_drain_count: int = 0
    non_spec_drain_ms: float = 0.0

    @property
    def total_drain_count(self) -> int:
        return (
            self.pre_fork_drain_count
            + self.post_match_drain_count
            + self.non_spec_drain_count
        )

    @property
    def total_drain_s(self) -> float:
        return (
            self.pre_fork_drain_ms
            + self.post_match_drain_ms
            + self.non_spec_drain_ms
        ) / 1000.0

    @property
    def spec_accept_rate(self) -> float:
        if self.spec_total_turns <= 0:
            return 0.0
        return self.spec_accept_count / self.spec_total_turns


# ── Run ─────────────────────────────────────────────────────────────────────


@dataclass
class RunStats:
    label: str
    paths: RunPaths

    # Pulled from summary.json when available.
    summary: dict[str, Any] = field(default_factory=dict)

    # Per-task rollups keyed by task_run_id (the original sandbox id).
    tasks: dict[str, TaskRunStats] = field(default_factory=dict)

    # Run-level wall clock, derived from summary.json `started_at`/`finished_at`
    # (preferred) or telemetry timestamp range (fallback).
    run_wall_clock_s: float | None = None

    # Verification status counts for run-level reporting.
    verification_counts: dict[str, int] = field(default_factory=dict)

    @property
    def is_spec(self) -> bool:
        if self.summary.get("spec_fork_reuse_stats"):
            return True
        return any(t.spec_total_turns > 0 for t in self.tasks.values())

    @property
    def has_cadence(self) -> bool:
        cad = self.summary.get("replay_cadence_stats") or {}
        if any(cad.get(k, 0) for k in (
            "fast_forward_skip_count",
            "pre_fork_drain_count",
            "post_match_drain_count",
            "non_spec_drain_count",
        )):
            return True
        return any(
            t.fast_forward_skip_count or t.total_drain_count
            for t in self.tasks.values()
        )

    def by_task_index(self) -> dict[int, TaskRunStats]:
        out: dict[int, TaskRunStats] = {}
        for trid, t in self.tasks.items():
            try:
                idx = int(trid.split("-")[1])
            except (IndexError, ValueError):
                continue
            out[idx] = t
        return out

    def by_task_id(self) -> dict[str, list[TaskRunStats]]:
        """Group tasks by task_id. A run can have multiple sandboxes for the
        same task_id (e.g. a curated dataset with duplicate replicas), so
        the value is a list ordered by sandbox idx."""
        groups: dict[str, list[TaskRunStats]] = {}
        for t in self.tasks.values():
            if not t.task_id:
                continue
            groups.setdefault(t.task_id, []).append(t)
        for v in groups.values():
            v.sort(key=lambda t: t.sandbox_id)
        return groups

    def op(self, metric_name: str) -> dict[str, float] | None:
        """Get an `operation_summaries` row by metric_name (count/mean/p50/p90/p95/p99/max)."""
        for op in self.summary.get("operation_summaries", []) or []:
            if op.get("metric_name") == metric_name:
                return op
        return None

    def turn_metric(self, name: str, request_kind: str = "") -> dict[str, float] | None:
        """Get a `turn_analysis.metrics` row (e.g. llm_response_time, turn_time)."""
        for tm in self.summary.get("turn_analysis", {}).get("metrics", []) or []:
            if tm.get("metric_name") == name and tm.get("request_kind", "") == request_kind:
                return tm
        return None


# ── Loaders ─────────────────────────────────────────────────────────────────


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def _to_float(v: Any) -> float:
    if v is None or v == "":
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _to_int(v: Any) -> int:
    if v is None or v == "":
        return 0
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _trid_from_sandbox_id(sid: str) -> str:
    """Strip any promotion suffix to get the original task_run_id.

    Examples:
        spec-3              → spec-3 (already canonical)
        spec-3-spec-7       → spec-3 (spec-on-spec promotion)
        spec-0-prefork-19   → spec-0 (prefork promotion in spec mode)
        fault-2-recovered-1 → fault-2 (recovery promotion)
        nofault-0           → nofault-0

    The rule: if the id is `<scenario>-<int>-<rest>`, return
    `<scenario>-<int>`. Otherwise return as-is.
    """
    if not sid:
        return sid
    parts = sid.split("-")
    if len(parts) >= 3:
        try:
            int(parts[1])
        except ValueError:
            return sid
        return f"{parts[0]}-{parts[1]}"
    return sid


def _load_main_csv_into_tasks(
    csv_path: Path, tasks: dict[str, TaskRunStats]
) -> dict[str, int]:
    """Populate verification + spec metrics for each task, keyed by trid.

    Returns a histogram of verification_status values for run-level reporting."""
    verif_counts: dict[str, int] = {}
    if not csv_path.is_file():
        return verif_counts
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row.get("sandbox_id", "")
            if not sid:
                continue
            trid = _trid_from_sandbox_id(sid)
            t = tasks.setdefault(trid, TaskRunStats(task_run_id=trid))
            if not t.task_id:
                t.task_id = row.get("task_id", "") or t.task_id
            t.sandbox_id = sid
            t.verification_status = row.get("verification_status", "") or "—"
            t.verification_ms = _to_float(row.get("verification_ms"))
            sr = row.get("success_ratio", "")
            if sr != "":
                t.success_ratio = _to_float(sr)
            t.runtime_s = max(t.runtime_s, _to_float(row.get("task_completion_ms")) / 1000.0)
            t.spec_total_turns = _to_int(row.get("spec_total_turns"))
            t.spec_accept_count = _to_int(row.get("spec_accept_count"))
            t.spec_reject_count = _to_int(row.get("spec_reject_count"))
            t.spec_saved_ms = _to_float(row.get("spec_saved_ms"))
            t.spec_penalty_ms = _to_float(row.get("spec_penalty_ms"))
            t.spec_hidden_penalty_ms = _to_float(row.get("spec_hidden_penalty_ms"))
            t.spec_net_gain_ms = _to_float(row.get("spec_net_gain_ms"))
            t.spec_fork_create_count = _to_int(row.get("spec_fork_create_count"))
            t.spec_fork_reuse_count = _to_int(row.get("spec_fork_reuse_count"))
            verif_counts[t.verification_status] = verif_counts.get(t.verification_status, 0) + 1
    return verif_counts


def _load_replay_cadence_per_sandbox(
    csv_path: Path, tasks: dict[str, TaskRunStats]
) -> None:
    if not csv_path.is_file():
        return
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row.get("sandbox_id", "")
            trid = _trid_from_sandbox_id(sid)
            t = tasks.setdefault(trid, TaskRunStats(task_run_id=trid))
            t.fast_forward_skip_count += _to_int(row.get("fast_forward_skip_count"))
            t.fast_forward_saved_ms += _to_float(row.get("fast_forward_saved_ms"))
            t.pre_fork_drain_count += _to_int(row.get("pre_fork_drain_count"))
            t.pre_fork_drain_ms += _to_float(row.get("pre_fork_drain_ms"))
            t.post_match_drain_count += _to_int(row.get("post_match_drain_count"))
            t.post_match_drain_ms += _to_float(row.get("post_match_drain_ms"))
            t.non_spec_drain_count += _to_int(row.get("non_spec_drain_count"))
            t.non_spec_drain_ms += _to_float(row.get("non_spec_drain_ms"))


def _load_telemetry_fallback(telemetry: Path, tasks: dict[str, TaskRunStats]) -> tuple[float | None, str | None, str | None]:
    """Used only when summary.json is missing — reconstructs the minimum
    needed for the run-summary section: per-task wall-clock and
    fast-forward / drain counters."""
    if not telemetry.is_file():
        return None, None, None
    earliest: str | None = None
    latest: str | None = None
    starts: dict[str, tuple[str, str]] = {}
    finishes: dict[str, str] = {}
    sandbox_to_trid: dict[str, str] = {}

    for rec in _iter_jsonl(telemetry):
        if rec.get("kind") != "event":
            continue
        name = rec.get("name", "")
        attrs = rec.get("attributes", {}) or {}
        ts = rec.get("timestamp", "")
        if ts:
            if earliest is None or ts < earliest:
                earliest = ts
            if latest is None or ts > latest:
                latest = ts

        sid = attrs.get("sandbox_id", "")
        trid = attrs.get("task_run_id", "")
        if sid and trid:
            sandbox_to_trid[sid] = trid

        if name == "benchmark.task.run.start" and trid:
            starts[trid] = (ts, str(attrs.get("task_id", "")))
        elif name == "benchmark.task.run.finish" and trid:
            finishes[trid] = ts

        if name == "terminus.fast_forward.skip":
            cad_trid = trid or sandbox_to_trid.get(sid, "")
            if not cad_trid:
                continue
            t = tasks.setdefault(cad_trid, TaskRunStats(task_run_id=cad_trid))
            t.fast_forward_skip_count += 1
            t.fast_forward_saved_ms += _to_float(attrs.get("saved_ms"))

        if name in (
            "terminus.guard.pre_fork_drain",
            "terminus.guard.post_match_drain",
            "terminus.guard.non_spec_drain",
        ):
            cad_trid = trid or sandbox_to_trid.get(sid, "")
            if not cad_trid:
                continue
            t = tasks.setdefault(cad_trid, TaskRunStats(task_run_id=cad_trid))
            wait_ms = _to_float(attrs.get("drain_wait_ms"))
            if name.endswith("pre_fork_drain"):
                t.pre_fork_drain_count += 1
                t.pre_fork_drain_ms += wait_ms
            elif name.endswith("post_match_drain"):
                t.post_match_drain_count += 1
                t.post_match_drain_ms += wait_ms
            else:
                t.non_spec_drain_count += 1
                t.non_spec_drain_ms += wait_ms

    for trid, (ts, task_id) in starts.items():
        t = tasks.setdefault(trid, TaskRunStats(task_run_id=trid))
        if task_id and not t.task_id:
            t.task_id = task_id
        # task_run.duration covers the whole task and is a better source than CSV-only fallbacks.
        if ts and trid in finishes:
            d_start = _parse_iso(ts)
            d_end = _parse_iso(finishes[trid])
            if d_start and d_end:
                t.runtime_s = max(t.runtime_s, (d_end - d_start).total_seconds())

    wall = None
    if earliest and latest:
        d_start = _parse_iso(earliest)
        d_end = _parse_iso(latest)
        if d_start and d_end:
            wall = (d_end - d_start).total_seconds()
    return wall, earliest, latest


_SCENARIO_PREFIXES = ("spec-", "fault-", "nofault-", "tree-", "spot-")


def _is_fork_pseudo_task(task_id: str) -> bool:
    """task_summaries includes one entry per intermediate fork sandbox in
    spec mode; their task_id matches the sandbox id (e.g. `spec-0-spec-7`)
    rather than a real task name. Filter them out so per-task pairing
    only sees the real top-level tasks."""
    if not task_id:
        return True
    return task_id.startswith(_SCENARIO_PREFIXES)


def _load_task_summaries_into_tasks(
    summary: dict[str, Any], tasks: dict[str, TaskRunStats]
) -> None:
    """Pull per-task runtime / spec metrics from summary.json's
    `task_summaries` block.  Fault/nofault CSVs lack `task_completion_ms`,
    so this block is the only source of per-task wall-clock for those
    scenarios."""
    for ts in summary.get("task_summaries", []) or []:
        if _is_fork_pseudo_task(ts.get("task_id", "")):
            continue
        trid = ts.get("task_run_id") or ts.get("sandbox_id") or ""
        if not trid:
            continue
        t = tasks.setdefault(trid, TaskRunStats(task_run_id=trid))
        if ts.get("task_id") and not t.task_id:
            t.task_id = ts["task_id"]
        if ts.get("sandbox_id") and not t.sandbox_id:
            t.sandbox_id = ts["sandbox_id"]
        metrics = ts.get("metrics") or {}
        run_ms = metrics.get("benchmark.task.run.duration_ms")
        if run_ms is not None:
            t.runtime_s = max(t.runtime_s, _to_float(run_ms) / 1000.0)
        # Spec metrics may be absent (nofault) or zero (no accepts) — we
        # only fill from task_summaries if the CSV didn't.
        if t.spec_total_turns == 0:
            t.spec_saved_ms = max(t.spec_saved_ms, _to_float(metrics.get("benchmark.spec.saved_ms")))
            t.spec_penalty_ms = max(t.spec_penalty_ms, _to_float(metrics.get("benchmark.spec.penalty_ms")))
            t.spec_hidden_penalty_ms = max(
                t.spec_hidden_penalty_ms, _to_float(metrics.get("benchmark.spec.hidden_penalty_ms"))
            )
            t.spec_net_gain_ms = max(t.spec_net_gain_ms, _to_float(metrics.get("benchmark.spec.net_gain_ms")))


def _load_run(arg: str, label: str | None = None) -> RunStats:
    paths = _resolve_paths(arg)
    if label:
        paths.label = label

    run = RunStats(label=paths.label, paths=paths)

    if paths.report_dir is not None:
        with (paths.report_dir / "summary.json").open(encoding="utf-8") as f:
            run.summary = json.load(f)
        ts_start = _parse_iso(run.summary.get("started_at", ""))
        ts_end = _parse_iso(run.summary.get("finished_at", ""))
        if ts_start and ts_end:
            run.run_wall_clock_s = (ts_end - ts_start).total_seconds()
        _load_task_summaries_into_tasks(run.summary, run.tasks)

    if paths.main_csv is not None:
        run.verification_counts = _load_main_csv_into_tasks(paths.main_csv, run.tasks)

    if paths.report_dir is not None:
        _load_replay_cadence_per_sandbox(
            paths.report_dir / "replay_cadence_per_sandbox.csv", run.tasks
        )

    if not run.summary and paths.telemetry is not None:
        wall, _, _ = _load_telemetry_fallback(paths.telemetry, run.tasks)
        if wall is not None:
            run.run_wall_clock_s = wall

    return run


# ── Formatters ──────────────────────────────────────────────────────────────


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "    —   "
    return f"{value:8.1f}s"


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "    —   "
    return f"{value:8.1f}ms"


def _fmt_int(value: int | float | None) -> str:
    if value is None:
        return "    —   "
    return f"{int(value):>9d}"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "    —   "
    return f"{value * 100:8.1f}%"


def _fmt_bytes(value: float | None) -> str:
    if value is None:
        return "    —   "
    v = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(v) < 1024 or unit == "TB":
            return f"{v:7.1f}{unit:>2}"
        v /= 1024
    return f"{v:7.1f}TB"


def _fmt_delta_seconds(base: float | None, val: float | None) -> str:
    if base is None or val is None:
        return "    —   "
    delta = val - base
    if base != 0:
        return f"{delta:+8.1f}s ({delta / base * 100:+5.1f}%)"
    return f"{delta:+8.1f}s"


def _fmt_delta_ms(base: float | None, val: float | None) -> str:
    if base is None or val is None:
        return "    —   "
    delta = val - base
    if base != 0:
        return f"{delta:+8.1f}ms ({delta / base * 100:+5.1f}%)"
    return f"{delta:+8.1f}ms"


def _fmt_delta_int(base: int | float | None, val: int | float | None) -> str:
    if base is None or val is None:
        return "    —   "
    delta = (val or 0) - (base or 0)
    if base:
        return f"{int(delta):+9d} ({delta / base * 100:+5.1f}%)"
    return f"{int(delta):+9d}"


def _fmt_delta_bytes(base: float | None, val: float | None) -> str:
    if base is None or val is None:
        return "    —   "
    delta = (val or 0) - (base or 0)
    pct = f"({delta / base * 100:+5.1f}%)" if base else ""
    sign = "+" if delta >= 0 else "-"
    abs_v = abs(delta)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs_v < 1024 or unit == "TB":
            return f"{sign}{abs_v:6.1f}{unit:>2} {pct}"
        abs_v /= 1024
    return f"{sign}{abs_v:6.1f}TB {pct}"


def _fmt_delta_pct(base: float | None, val: float | None) -> str:
    if base is None or val is None:
        return "    —   "
    delta = (val or 0) - (base or 0)
    return f"{delta * 100:+8.1f}pp"


# ── Section rendering ───────────────────────────────────────────────────────


SECTION_DIVIDER = "─" * 100


def _render_section(
    title: str,
    runs: list[RunStats],
    rows: list[tuple[str, list[Any], str]],
    label_col_width: int = 28,
) -> list[str]:
    """Render a section: one row per metric, one column per run, plus
    deltas relative to the first run.

    `rows` is a list of (label, [val_run0, val_run1, ...], fmt_kind) where
    fmt_kind is one of: 's' seconds, 'ms', 'int', 'pct' (0..1 fraction),
    'bytes', 'raw' (already a string), 'count_total' (int with totals).
    """
    if not rows:
        return []

    val_w = 18
    delta_w = 24
    out: list[str] = []
    out.append("")
    out.append(f"=== {title} ===")
    header = f"{'metric':<{label_col_width}}"
    for run in runs:
        header += f" {run.label[:val_w]:>{val_w}}"
    for run in runs[1:]:
        header += f" {(run.label[:6] + ' Δ'):>{delta_w}}"
    out.append(header)
    out.append("─" * len(header))

    fmt_value = {
        "s": _fmt_seconds,
        "ms": _fmt_ms,
        "int": _fmt_int,
        "pct": _fmt_pct,
        "bytes": _fmt_bytes,
        "raw": lambda v: f"{v:>{val_w}}" if v is not None else f"{'—':>{val_w}}",
    }
    fmt_delta = {
        "s": _fmt_delta_seconds,
        "ms": _fmt_delta_ms,
        "int": _fmt_delta_int,
        "pct": _fmt_delta_pct,
        "bytes": _fmt_delta_bytes,
        "raw": lambda b, v: f"{'—':>{delta_w}}",
    }

    for label, values, kind in rows:
        line = f"{label:<{label_col_width}}"
        for v in values:
            cell = fmt_value[kind](v)
            line += f" {cell:>{val_w}}"
        for v in values[1:]:
            cell = fmt_delta[kind](values[0], v)
            line += f" {cell:>{delta_w}}"
        out.append(line)
    return out


def render_run_summary(runs: list[RunStats]) -> list[str]:
    rows: list[tuple[str, list[Any], str]] = []
    rows.append(("run wall-clock", [r.run_wall_clock_s for r in runs], "s"))
    sums = [
        sum(t.runtime_s for t in r.tasks.values()) if r.tasks else None
        for r in runs
    ]
    rows.append(("Σ task_completion", sums, "s"))
    # distinct_tasks straight from summary.json over-counts in spec mode
    # (each fork sandbox is summarized as its own row); use the filtered
    # task list instead.
    rows.append((
        "distinct_tasks",
        [len({t.task_id for t in r.tasks.values() if t.task_id}) or None for r in runs],
        "int",
    ))
    rows.append((
        "distinct_sandboxes",
        [r.summary.get("distinct_sandboxes") for r in runs],
        "int",
    ))
    rows.append((
        "distinct_checkpoints",
        [r.summary.get("distinct_checkpoints") for r in runs],
        "int",
    ))
    sr_means: list[float | None] = []
    for r in runs:
        ratios = [t.success_ratio for t in r.tasks.values() if t.success_ratio is not None]
        sr_means.append(sum(ratios) / len(ratios) if ratios else None)
    rows.append(("success_ratio (mean)", sr_means, "pct"))
    return _render_section("run summary", runs, rows)


def render_verification(runs: list[RunStats]) -> list[str]:
    statuses = sorted(
        {s for r in runs for s in r.verification_counts.keys() if s and s != "—"}
    )
    if not statuses and not any(r.verification_counts for r in runs):
        return []
    rows: list[tuple[str, list[Any], str]] = []
    rows.append((
        "verify_total (rows)",
        [sum(r.verification_counts.values()) or None for r in runs],
        "int",
    ))
    for s in statuses:
        rows.append((
            f"verify={s}",
            [r.verification_counts.get(s, 0) for r in runs],
            "int",
        ))
    pass_counts = []
    for r in runs:
        n = r.verification_counts.get("passed", 0)
        pass_counts.append(n)
    rows.append(("verify_passed", pass_counts, "int"))
    return _render_section("verification", runs, rows)


def render_checkpoint_cost(runs: list[RunStats]) -> list[str]:
    have_data = [bool(r.summary.get("checkpoint_analysis")) for r in runs]
    if not any(have_data):
        return []

    def ca(r: RunStats, k: str) -> Any:
        return (r.summary.get("checkpoint_analysis") or {}).get(k)

    def scope(r: RunStats, k: str) -> Any:
        return ((r.summary.get("checkpoint_analysis") or {}).get("scope_counts") or {}).get(k)

    rows: list[tuple[str, list[Any], str]] = [
        ("total_count", [ca(r, "total_count") for r in runs], "int"),
        ("success_count", [ca(r, "success_count") for r in runs], "int"),
        ("skip_count", [ca(r, "skip_count") for r in runs], "int"),
        ("fail_count", [ca(r, "fail_count") for r in runs], "int"),
        ("scope: full", [scope(r, "full") for r in runs], "int"),
        ("scope: filesystem_only", [scope(r, "filesystem_only") for r in runs], "int"),
    ]
    rows.extend([
        (
            "process freq /min",
            [
                f"{_to_float(ca(r, 'process_frequency_per_minute')):8.2f}/m"
                if ca(r, "process_frequency_per_minute") is not None
                else None
                for r in runs
            ],
            "raw",
        ),
        (
            "filesystem freq /min",
            [
                f"{_to_float(ca(r, 'filesystem_frequency_per_minute')):8.2f}/m"
                if ca(r, "filesystem_frequency_per_minute") is not None
                else None
                for r in runs
            ],
            "raw",
        ),
        (
            "fs_only / full ratio",
            [
                f"{_to_float(ca(r, 'filesystem_only_to_full_ratio')):8.2f}x"
                if ca(r, "filesystem_only_to_full_ratio") is not None
                else None
                for r in runs
            ],
            "raw",
        ),
        ("Σ proc dump bytes", [ca(r, "total_process_size_bytes") for r in runs], "bytes"),
        ("Σ fs written bytes", [ca(r, "total_filesystem_written_bytes") for r in runs], "bytes"),
        ("Σ checkpoint IO", [ca(r, "total_estimated_io_bytes") for r in runs], "bytes"),
        ("mean proc dump bytes", [ca(r, "mean_process_size_bytes") for r in runs], "bytes"),
        ("p95 proc dump bytes", [ca(r, "p95_process_size_bytes") for r in runs], "bytes"),
        ("mean fs written bytes", [ca(r, "mean_filesystem_written_bytes") for r in runs], "bytes"),
        ("p95 fs written bytes", [ca(r, "p95_filesystem_written_bytes") for r in runs], "bytes"),
        ("mean est IO / chkpt", [ca(r, "mean_estimated_io_bytes") for r in runs], "bytes"),
    ])
    # Per-dump size distributions (max + p99 only available via operation_summaries).
    for metric, label in (
        ("checkpoint.process.size_bytes", "proc dump size"),
        ("checkpoint.filesystem.written_bytes", "fs written bytes"),
        ("checkpoint.estimated_io_bytes", "est IO / chkpt"),
    ):
        seg = _percentile_rows(runs, metric, label, value_kind="bytes")
        if seg:
            rows.append((f"── {label} dist ──", [None] * len(runs), "raw"))
            rows.extend(seg)
    return _render_section("checkpoint cost", runs, rows)


def _percentile_rows(
    runs: list[RunStats],
    metric_name: str,
    label_prefix: str,
    value_kind: str = "ms",
) -> list[tuple[str, list[Any], str]]:
    """Build percentile rows for a metric in `operation_summaries`.

    `value_kind` controls how mean/p50/p90/p95/p99/max are formatted —
    use 'ms' for durations, 'bytes' for byte-valued metrics, etc."""
    rows: list[tuple[str, list[Any], str]] = []
    cells = [r.op(metric_name) for r in runs]
    if not any(cells):
        return rows
    for stat in ("count", "mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms"):
        vals = [c.get(stat) if c else None for c in cells]
        if all(v is None for v in vals):
            continue
        if stat == "count":
            rows.append((f"{label_prefix} count", vals, "int"))
        else:
            rows.append((f"{label_prefix} {stat[:-3]}", vals, value_kind))
    return rows


def render_checkpoint_latency(runs: list[RunStats]) -> list[str]:
    rows: list[tuple[str, list[Any], str]] = []
    for metric, label in (
        ("checkpoint.flow.duration_ms", "flow"),
        ("checkpoint.process.duration_ms", "process (final-dump)"),
        ("checkpoint.filesystem.duration_ms", "filesystem"),
        ("sandbox.checkpoint_process.duration_ms", "sandbox.checkpoint_proc"),
        ("sandbox.checkpoint_filesystem.duration_ms", "sandbox.checkpoint_fs"),
        ("sandbox.runtime_pause.duration_ms", "sandbox pause"),
    ):
        seg = _percentile_rows(runs, metric, label)
        if seg:
            rows.append((f"── {label} ──", [None] * len(runs), "raw"))
            rows.extend(seg)
    if not rows:
        return []
    return _render_section("checkpoint latency", runs, rows)


def render_restore(runs: list[RunStats]) -> list[str]:
    have_data = [bool(r.summary.get("restore_analysis")) for r in runs]
    if not any(have_data):
        return []

    def ra(r: RunStats, k: str) -> Any:
        return (r.summary.get("restore_analysis") or {}).get(k)

    rows: list[tuple[str, list[Any], str]] = [
        ("restore total_count", [ra(r, "total_count") for r in runs], "int"),
        ("restore success_count", [ra(r, "success_count") for r in runs], "int"),
        ("restore fail_count", [ra(r, "fail_count") for r in runs], "int"),
        ("restore mean est IO", [ra(r, "mean_estimated_io_bytes") for r in runs], "bytes"),
        ("source_gap mean (ms)", [ra(r, "mean_source_gap_ms") for r in runs], "ms"),
        ("source_gap p95 (ms)", [ra(r, "p95_source_gap_ms") for r in runs], "ms"),
        ("source_gap mean (turns)", [ra(r, "mean_source_gap_turns") for r in runs], "ms"),
    ]
    for metric, label in (
        ("restore.flow.duration_ms", "flow"),
        ("restore.process.duration_ms", "process"),
        ("restore.filesystem.duration_ms", "filesystem"),
    ):
        seg = _percentile_rows(runs, metric, f"restore-{label}")
        if seg:
            rows.append((f"── restore-{label} ──", [None] * len(runs), "raw"))
            rows.extend(seg)
    return _render_section("restore", runs, rows)


def render_spec(runs: list[RunStats]) -> list[str]:
    if not any(r.is_spec for r in runs):
        return []

    def fr(r: RunStats, k: str) -> Any:
        return (r.summary.get("spec_fork_reuse_stats") or {}).get(k)

    def tb(r: RunStats, k: str) -> Any:
        return (r.summary.get("spec_turn_breakdown") or {}).get(k)

    rows: list[tuple[str, list[Any], str]] = [
        ("total_turns", [fr(r, "total_turns") for r in runs], "int"),
        ("finalized_turns", [fr(r, "finalized_turns") for r in runs], "int"),
        ("turn: accepted", [tb(r, "accepted") for r in runs], "int"),
        ("turn: rejected_command_mismatch", [tb(r, "rejected_command_mismatch") for r in runs], "int"),
        ("turn: rejected_no_fork", [tb(r, "rejected_no_fork") for r in runs], "int"),
        ("turn: rejected_oracle_first", [tb(r, "rejected_oracle_first") for r in runs], "int"),
        ("forks_created", [fr(r, "forks_created") for r in runs], "int"),
        ("forks_reused", [fr(r, "forks_reused") for r in runs], "int"),
        ("cache_eligible", [fr(r, "cache_eligible") for r in runs], "int"),
    ]
    accept_rates: list[float | None] = []
    for r in runs:
        n = fr(r, "total_turns")
        a = tb(r, "accepted")
        if n and a is not None:
            accept_rates.append(a / n)
        else:
            accept_rates.append(None)
    rows.append(("accept rate (accepted/total)", accept_rates, "pct"))

    saved_sums = [
        sum(t.spec_saved_ms for t in r.tasks.values()) / 1000.0 if r.is_spec else None
        for r in runs
    ]
    rows.append(("Σ spec_saved (s)", saved_sums, "s"))
    penalty_sums = [
        sum(t.spec_penalty_ms for t in r.tasks.values()) / 1000.0 if r.is_spec else None
        for r in runs
    ]
    rows.append(("Σ spec_penalty (s)", penalty_sums, "s"))
    hidden_sums = [
        sum(t.spec_hidden_penalty_ms for t in r.tasks.values()) / 1000.0 if r.is_spec else None
        for r in runs
    ]
    rows.append(("Σ spec_hidden_penalty (s)", hidden_sums, "s"))
    netgain_sums = [
        sum(t.spec_net_gain_ms for t in r.tasks.values()) / 1000.0 if r.is_spec else None
        for r in runs
    ]
    rows.append(("Σ spec_net_gain (s)", netgain_sums, "s"))

    for metric, label in (
        ("benchmark.spec.fork_restore_ms", "fork_restore_ms"),
        ("benchmark.spec.speculative_exec_ms", "speculative_exec_ms"),
        ("benchmark.spec.saved_ms", "saved_ms"),
        ("benchmark.spec.penalty_ms", "penalty_ms"),
        ("benchmark.spec.hidden_penalty_ms", "hidden_penalty_ms"),
        ("benchmark.spec.net_gain_ms", "net_gain_ms"),
    ):
        seg = _percentile_rows(runs, metric, label)
        if seg:
            rows.append((f"── {label} ──", [None] * len(runs), "raw"))
            rows.extend(seg)
    return _render_section("spec / fork-restore", runs, rows)


def render_cadence(runs: list[RunStats]) -> list[str]:
    if not any(r.has_cadence for r in runs):
        return []

    def cad(r: RunStats, k: str) -> Any:
        return (r.summary.get("replay_cadence_stats") or {}).get(k)

    rows: list[tuple[str, list[Any], str]] = [
        ("fast_forward skips", [cad(r, "fast_forward_skip_count") for r in runs], "int"),
        ("fast_forward saved (s)", [
            (cad(r, "fast_forward_saved_ms") / 1000.0) if cad(r, "fast_forward_saved_ms") is not None else None
            for r in runs
        ], "s"),
        ("fast_forward intended sleep (s)", [
            (cad(r, "fast_forward_intended_sleep_ms") / 1000.0) if cad(r, "fast_forward_intended_sleep_ms") is not None else None
            for r in runs
        ], "s"),
        ("pre_fork drains", [cad(r, "pre_fork_drain_count") for r in runs], "int"),
        ("pre_fork drain (s)", [
            (cad(r, "pre_fork_drain_ms") / 1000.0) if cad(r, "pre_fork_drain_ms") is not None else None
            for r in runs
        ], "s"),
        ("pre_fork drain timeouts", [cad(r, "pre_fork_drain_timeouts") for r in runs], "int"),
        ("post_match drains", [cad(r, "post_match_drain_count") for r in runs], "int"),
        ("post_match drain (s)", [
            (cad(r, "post_match_drain_ms") / 1000.0) if cad(r, "post_match_drain_ms") is not None else None
            for r in runs
        ], "s"),
        ("non_spec drains", [cad(r, "non_spec_drain_count") for r in runs], "int"),
        ("non_spec drain (s)", [
            (cad(r, "non_spec_drain_ms") / 1000.0) if cad(r, "non_spec_drain_ms") is not None else None
            for r in runs
        ], "s"),
    ]
    return _render_section("replay cadence", runs, rows)


def render_llm_overhead(runs: list[RunStats]) -> list[str]:
    rows: list[tuple[str, list[Any], str]] = []
    for metric, label in (
        ("llm.gate_wait_ms", "gate_wait"),
        ("llm.crab_delay_ms", "crab_delay"),
        ("llm.interceptor_total_ms", "interceptor_total"),
        ("llm.upstream_latency_ms", "upstream_latency"),
        ("interceptor.request.forward.duration_ms", "interceptor.forward"),
        ("interceptor.response_gate.wait.duration_ms", "interceptor.gate_wait"),
    ):
        seg = _percentile_rows(runs, metric, label)
        if seg:
            rows.append((f"── {label} ──", [None] * len(runs), "raw"))
            rows.extend(seg)
    if not rows:
        return []
    return _render_section("LLM-path overhead", runs, rows)


def render_turn_analysis(runs: list[RunStats]) -> list[str]:
    rows: list[tuple[str, list[Any], str]] = []
    for metric in ("llm_response_time", "pure_llm_time", "action_time", "turn_time"):
        cells = [r.turn_metric(metric) for r in runs]
        if not any(cells):
            continue
        rows.append((f"── {metric} ──", [None] * len(runs), "raw"))
        for stat, label, kind in (
            ("count", "count", "int"),
            ("mean_ms", "mean", "ms"),
            ("p50_ms", "p50", "ms"),
            ("p95_ms", "p95", "ms"),
            ("p99_ms", "p99", "ms"),
            ("max_ms", "max", "ms"),
        ):
            vals = [c.get(stat) if c else None for c in cells]
            if all(v is None for v in vals):
                continue
            rows.append((f"{metric} {label}", vals, kind))
    if not rows:
        return []
    return _render_section("turn timing", runs, rows)


def render_per_task(runs: list[RunStats]) -> list[str]:
    """Per-task wall-clock + verify + spec accepts (compact for any N).

    Pairs by task_id so that runs with different sandbox-id assignments
    still line up. If a task_id has multiple replicas within a run, the
    extras are emitted as `<task_id>#2`, `<task_id>#3`, etc."""
    by_id = [r.by_task_id() for r in runs]
    keys: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for groups in by_id:
        for tid, lst in groups.items():
            for k in range(len(lst)):
                key = (tid, k)
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
    keys.sort()
    if not keys:
        return []

    any_spec = any(r.is_spec for r in runs)
    out: list[str] = []
    out.append("")
    out.append("=== per-task ===")

    val_w = 9
    hdr = f"{'task':<28}"
    for r in runs:
        hdr += f" {r.label[:val_w]:>{val_w}}"
    for r in runs:
        hdr += f" {(r.label[:5] + ' v'):>{val_w}}"
    if any_spec:
        for r in runs:
            if r.is_spec:
                hdr += f" {(r.label[:6] + ' acc'):>{val_w + 2}}"
    out.append(hdr)
    out.append("─" * len(hdr))

    sums = [0.0] * len(runs)

    for tid, k in keys:
        per_run: list[TaskRunStats | None] = []
        for groups in by_id:
            lst = groups.get(tid, [])
            per_run.append(lst[k] if k < len(lst) else None)
        label = tid if k == 0 else f"{tid}#{k + 1}"
        line = f"{label[:28]:<28}"
        for i, t in enumerate(per_run):
            if t is None or t.runtime_s == 0:
                line += f" {'—':>{val_w}}"
            else:
                line += f" {t.runtime_s:>{val_w - 1}.0f}s"
                sums[i] += t.runtime_s
        for t in per_run:
            v = (t.verification_status if t else "—") or "—"
            line += f" {v[:val_w]:>{val_w}}"
        if any_spec:
            for i, t in enumerate(per_run):
                if not runs[i].is_spec:
                    continue
                if t is None or t.spec_total_turns <= 0:
                    line += f" {'—':>{val_w + 2}}"
                else:
                    line += f" {t.spec_accept_count}/{t.spec_total_turns:<5}"
        out.append(line)

    out.append("─" * len(hdr))
    line = f"{'TOTAL':<28}"
    for s in sums:
        line += f" {s:>{val_w - 1}.0f}s"
    out.append(line)
    return out


def render_per_sandbox_io(runs: list[RunStats]) -> list[str]:
    """Per-task `mean_estimated_io_bytes` per checkpoint, lined up across runs.

    Mirrors the table that PR #25 used to show memory-heavy vs FS-heavy
    workloads.  Only emitted when at least 2 runs have checkpoint per-task
    rollups and they share at least one task_id.
    """
    per_run_by_task: list[dict[str, dict[str, Any]]] = []
    for r in runs:
        rows = (r.summary.get("checkpoint_analysis") or {}).get("per_task") or []
        per_run_by_task.append({
            row.get("task_id", ""): row
            for row in rows
            if row.get("task_id") and not _is_fork_pseudo_task(row["task_id"])
        })

    nonempty = [d for d in per_run_by_task if d]
    if len(nonempty) < 2:
        return []
    shared = set.intersection(*[set(d.keys()) for d in nonempty])
    if not shared:
        return []

    val_w = 14
    out: list[str] = []
    out.append("")
    out.append("=== per-task checkpoint IO (mean estimated bytes/checkpoint) ===")
    hdr = f"{'task_id':<32}"
    for r in runs:
        hdr += f" {r.label[:val_w]:>{val_w}}"
    out.append(hdr)
    out.append("─" * len(hdr))

    for task_id in sorted(shared):
        line = f"{task_id[:32]:<32}"
        for d in per_run_by_task:
            v = d.get(task_id, {}).get("mean_estimated_io_bytes")
            line += f" {_fmt_bytes(v):>{val_w}}"
        out.append(line)
    return out


# ── CSV output ──────────────────────────────────────────────────────────────


def write_paired_csv(path: Path, runs: list[RunStats]) -> None:
    """Dump per-task data with one row per (run, task) — easy to pivot."""
    fieldnames = [
        "label",
        "task_index",
        "task_id",
        "sandbox_id",
        "task_run_id",
        "runtime_s",
        "verification_status",
        "verification_ms",
        "success_ratio",
        "spec_total_turns",
        "spec_accept_count",
        "spec_reject_count",
        "spec_accept_rate",
        "spec_saved_ms",
        "spec_penalty_ms",
        "spec_hidden_penalty_ms",
        "spec_net_gain_ms",
        "spec_fork_create_count",
        "spec_fork_reuse_count",
        "fast_forward_skip_count",
        "fast_forward_saved_ms",
        "pre_fork_drain_count",
        "pre_fork_drain_ms",
        "post_match_drain_count",
        "post_match_drain_ms",
        "non_spec_drain_count",
        "non_spec_drain_ms",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            for trid, t in sorted(run.tasks.items()):
                try:
                    idx = int(trid.split("-")[1])
                except (IndexError, ValueError):
                    idx = -1
                writer.writerow({
                    "label": run.label,
                    "task_index": idx,
                    "task_id": t.task_id,
                    "sandbox_id": t.sandbox_id,
                    "task_run_id": trid,
                    "runtime_s": f"{t.runtime_s:.3f}",
                    "verification_status": t.verification_status,
                    "verification_ms": f"{t.verification_ms:.3f}",
                    "success_ratio": (
                        f"{t.success_ratio:.6f}" if t.success_ratio is not None else ""
                    ),
                    "spec_total_turns": t.spec_total_turns,
                    "spec_accept_count": t.spec_accept_count,
                    "spec_reject_count": t.spec_reject_count,
                    "spec_accept_rate": f"{t.spec_accept_rate:.6f}",
                    "spec_saved_ms": f"{t.spec_saved_ms:.3f}",
                    "spec_penalty_ms": f"{t.spec_penalty_ms:.3f}",
                    "spec_hidden_penalty_ms": f"{t.spec_hidden_penalty_ms:.3f}",
                    "spec_net_gain_ms": f"{t.spec_net_gain_ms:.3f}",
                    "spec_fork_create_count": t.spec_fork_create_count,
                    "spec_fork_reuse_count": t.spec_fork_reuse_count,
                    "fast_forward_skip_count": t.fast_forward_skip_count,
                    "fast_forward_saved_ms": f"{t.fast_forward_saved_ms:.3f}",
                    "pre_fork_drain_count": t.pre_fork_drain_count,
                    "pre_fork_drain_ms": f"{t.pre_fork_drain_ms:.3f}",
                    "post_match_drain_count": t.post_match_drain_count,
                    "post_match_drain_ms": f"{t.post_match_drain_ms:.3f}",
                    "non_spec_drain_count": t.non_spec_drain_count,
                    "non_spec_drain_ms": f"{t.non_spec_drain_ms:.3f}",
                })


# ── CLI ─────────────────────────────────────────────────────────────────────


def _default_label(run_arg: str) -> str:
    return _resolve_paths(run_arg).label


def _section_lookup() -> dict[str, callable]:
    return {
        "summary": render_run_summary,
        "verification": render_verification,
        "checkpoint_cost": render_checkpoint_cost,
        "checkpoint_latency": render_checkpoint_latency,
        "restore": render_restore,
        "spec": render_spec,
        "cadence": render_cadence,
        "llm_overhead": render_llm_overhead,
        "turn": render_turn_analysis,
        "per_task": render_per_task,
        "per_sandbox_io": render_per_sandbox_io,
    }


DEFAULT_SECTIONS = (
    "summary",
    "verification",
    "checkpoint_cost",
    "checkpoint_latency",
    "restore",
    "spec",
    "cadence",
    "llm_overhead",
    "turn",
    "per_task",
    "per_sandbox_io",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare N benchmark runs at run-level and per-task level. "
            "First run is treated as the baseline; deltas are shown relative to it."
        ),
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="Run prefixes (or .report dirs / .telemetry.jsonl / .csv paths). N>=2.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Short label for a run (repeatable, in order). Defaults to the prefix's filename.",
    )
    # Back-compat with the old 2-way invocation.
    parser.add_argument("--label-a", help=argparse.SUPPRESS)
    parser.add_argument("--label-b", help=argparse.SUPPRESS)
    parser.add_argument("--csv", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--csv-out",
        type=Path,
        help="Optional path to dump per-task data as CSV (one row per (run, task)).",
    )
    parser.add_argument(
        "--sections",
        help=(
            "Comma-separated subset of sections to render. "
            f"Default: all. Available: {', '.join(DEFAULT_SECTIONS)}"
        ),
    )
    args = parser.parse_args(argv)

    if len(args.runs) < 2:
        parser.error("need at least 2 runs to compare")

    labels: list[str | None] = list(args.label) + [None] * len(args.runs)
    # Back-compat: if --label-a/--label-b were used and --label wasn't, splice in.
    if args.label_a and not labels[0]:
        labels[0] = args.label_a
    if args.label_b and len(args.runs) >= 2 and not labels[1]:
        labels[1] = args.label_b
    runs = [_load_run(arg, labels[i] if i < len(labels) else None) for i, arg in enumerate(args.runs)]

    if args.sections:
        wanted = [s.strip() for s in args.sections.split(",") if s.strip()]
    else:
        wanted = list(DEFAULT_SECTIONS)
    renderers = _section_lookup()
    for section in wanted:
        fn = renderers.get(section)
        if fn is None:
            print(f"warning: unknown section {section!r}", file=sys.stderr)
            continue
        for line in fn(runs):
            print(line)

    csv_target = args.csv_out or args.csv
    if csv_target is not None:
        write_paired_csv(csv_target, runs)
        print(f"\nPer-task CSV written to {csv_target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
