"""Compare two benchmark runs side-by-side at the per-task level.

Use case: you ran the same dataset under two different configs (e.g.
nofault vs spec, or with vs without a knob) and want a per-task table
showing where wall-clock and the replay-cadence mechanisms moved.

Pairs tasks by the trailing numeric index in `task_run_id` (which is
stable across fork promotions in spec mode), so it works with the
sandbox-id rebinding pattern the spec controller uses
(e.g. `spec-3` → `spec-3-spec-7` after promotion).

Inputs are run "prefixes" — the shared stem of `.csv`, `.log`,
`.telemetry.jsonl`, and `.report/`:

    python3 -m scripts.compare_benchmark_runs \\
        logs/terminus/nofault.auto.spec_friendly.10tasks \\
        logs/terminus/spec.auto.spec_friendly.10tasks

Per-task columns shown:

    runtime          benchmark.task.run.{start,finish} delta
    FF skips/saved   terminus.fast_forward.skip events + saved_ms
    drain count/ms   terminus.guard.{pre_fork,post_match}_drain events
    spec turns/acc   spec.turn.finish events with accepted=1
    verify           verification_status from the .csv

Pass `--csv <path>` to also dump the paired data as structured CSV for
further analysis (one row per task index).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

# Benchmark CSVs embed verification stdout/stderr and the swebench JSON report
# in single columns, which routinely exceed Python's 128 KiB default.
csv.field_size_limit(sys.maxsize)


@dataclass
class TaskRunStats:
    task_run_id: str
    task_id: str = ""
    sandbox_id_start: str = ""
    sandbox_id_finish: str = ""
    started_at: str = ""
    finished_at: str = ""
    fast_forward_skip_count: int = 0
    fast_forward_saved_ms: float = 0.0
    fast_forward_intended_sleep_ms: float = 0.0
    pre_fork_drain_count: int = 0
    pre_fork_drain_ms: float = 0.0
    pre_fork_drain_timeouts: int = 0
    post_match_drain_count: int = 0
    post_match_drain_ms: float = 0.0
    post_match_drain_timeouts: int = 0
    non_spec_drain_count: int = 0
    non_spec_drain_ms: float = 0.0
    non_spec_drain_timeouts: int = 0
    spec_total_turns: int = 0
    spec_accept_count: int = 0
    verification_status: str = "—"

    @property
    def runtime_s(self) -> float:
        if not self.started_at or not self.finished_at:
            return 0.0
        return (
            datetime.fromisoformat(self.finished_at)
            - datetime.fromisoformat(self.started_at)
        ).total_seconds()

    @property
    def fast_forward_saved_s(self) -> float:
        return self.fast_forward_saved_ms / 1000.0

    @property
    def total_drain_s(self) -> float:
        return (
            self.pre_fork_drain_ms
            + self.post_match_drain_ms
            + self.non_spec_drain_ms
        ) / 1000.0

    @property
    def total_drain_count(self) -> int:
        return (
            self.pre_fork_drain_count
            + self.post_match_drain_count
            + self.non_spec_drain_count
        )

    @property
    def spec_accept_rate(self) -> float:
        if self.spec_total_turns <= 0:
            return 0.0
        return self.spec_accept_count / self.spec_total_turns


@dataclass
class RunStats:
    label: str
    prefix: Path
    tasks: dict[str, TaskRunStats] = field(default_factory=dict)
    run_wall_clock_s: float | None = None

    def by_task_index(self) -> dict[int, TaskRunStats]:
        out: dict[int, TaskRunStats] = {}
        for trid, t in self.tasks.items():
            try:
                idx = int(trid.split("-")[1])
            except (IndexError, ValueError):
                continue
            out[idx] = t
        return out


# ── Loading ─────────────────────────────────────────────────────────────────


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


def _resolve_run(prefix_arg: str) -> Path:
    """Accept a path with or without trailing extensions and return the
    prefix without suffix (so prefix.with_suffix('.X') gives the right
    artifact)."""
    p = Path(prefix_arg)
    # If user passed the .telemetry.jsonl or .csv, strip back to prefix.
    name = p.name
    for suffix in (".telemetry.jsonl", ".jsonl", ".csv", ".log"):
        if name.endswith(suffix):
            return p.with_name(name[: -len(suffix)])
    # If user passed the .report dir, drop ".report".
    if name.endswith(".report"):
        return p.with_name(name[: -len(".report")])
    return p


def _load_run(prefix_arg: str, label: str) -> RunStats:
    prefix = _resolve_run(prefix_arg)
    telemetry = prefix.with_name(prefix.name + ".telemetry.jsonl")
    main_csv = prefix.with_name(prefix.name + ".csv")
    if not telemetry.is_file():
        sys.exit(f"error: telemetry file not found: {telemetry}")

    run = RunStats(label=label, prefix=prefix)

    # Pass 1: telemetry events.
    starts: dict[str, tuple[str, str, str]] = {}
    finishes: dict[str, tuple[str, str]] = {}
    cadence_by_trid: dict[str, TaskRunStats] = {}
    spec_turn_by_trid: dict[str, list[int]] = {}
    earliest: str | None = None
    latest: str | None = None

    # Cadence events emit by sandbox_id (not task_run_id), but they fire
    # before promotion (still on the original sandbox). To stitch them onto
    # task_run_id, we maintain a sandbox→task_run_id index from
    # benchmark.task.run.start (and .finish for post-promotion sids).
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
            starts[trid] = (
                ts,
                str(attrs.get("task_id", "")),
                str(attrs.get("sandbox_id", "")),
            )
        elif name == "benchmark.task.run.finish" and trid:
            finishes[trid] = (ts, str(attrs.get("sandbox_id", "")))

        if name == "terminus.fast_forward.skip":
            cad_trid = trid or sandbox_to_trid.get(sid, "")
            if not cad_trid:
                continue
            t = cadence_by_trid.setdefault(
                cad_trid, TaskRunStats(task_run_id=cad_trid)
            )
            t.fast_forward_skip_count += 1
            t.fast_forward_saved_ms += float(attrs.get("saved_ms", 0) or 0)
            t.fast_forward_intended_sleep_ms += float(
                attrs.get("intended_sleep_ms", 0) or 0
            )

        if name in (
            "terminus.guard.pre_fork_drain",
            "terminus.guard.post_match_drain",
            "terminus.guard.non_spec_drain",
        ):
            cad_trid = trid or sandbox_to_trid.get(sid, "")
            if not cad_trid:
                continue
            t = cadence_by_trid.setdefault(
                cad_trid, TaskRunStats(task_run_id=cad_trid)
            )
            wait_ms = float(attrs.get("drain_wait_ms", 0) or 0)
            resolved = bool(int(attrs.get("drain_resolved", 0) or 0))
            if name.endswith("pre_fork_drain"):
                t.pre_fork_drain_count += 1
                t.pre_fork_drain_ms += wait_ms
                if not resolved:
                    t.pre_fork_drain_timeouts += 1
            elif name.endswith("post_match_drain"):
                t.post_match_drain_count += 1
                t.post_match_drain_ms += wait_ms
                if not resolved:
                    t.post_match_drain_timeouts += 1
            else:
                t.non_spec_drain_count += 1
                t.non_spec_drain_ms += wait_ms
                if not resolved:
                    t.non_spec_drain_timeouts += 1

        if name == "spec.turn.finish":
            cad_trid = trid or sandbox_to_trid.get(sid, "")
            if not cad_trid:
                continue
            counts = spec_turn_by_trid.setdefault(cad_trid, [0, 0])
            counts[0] += 1
            if int(attrs.get("accepted", 0) or 0) == 1:
                counts[1] += 1

    # Pass 2: stitch per-task records.
    for trid, (ts, task_id, sid) in starts.items():
        t = cadence_by_trid.setdefault(trid, TaskRunStats(task_run_id=trid))
        t.started_at = ts
        t.task_id = task_id
        t.sandbox_id_start = sid
    for trid, (ts, sid) in finishes.items():
        t = cadence_by_trid.setdefault(trid, TaskRunStats(task_run_id=trid))
        t.finished_at = ts
        t.sandbox_id_finish = sid
    for trid, (total, accepted) in spec_turn_by_trid.items():
        t = cadence_by_trid.setdefault(trid, TaskRunStats(task_run_id=trid))
        t.spec_total_turns = total
        t.spec_accept_count = accepted

    # Pass 3: verification status from main CSV. The CSV's sandbox_id is
    # the post-promotion form (e.g. `spec-3-spec-7`), while telemetry's
    # benchmark.task.run.finish emits the original sandbox_id (e.g.
    # `spec-3`). Match the CSV row to a task_run_id by stripping the
    # promotion suffix off the front: `spec-3-spec-7` → leading
    # `spec-3` matches the task_run_id directly.
    if main_csv.is_file():
        with main_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sid = row.get("sandbox_id", "")
                vstatus = row.get("verification_status", "")
                if not sid:
                    continue
                # Try direct match first (no promotion).
                target_trid = ""
                if sid in cadence_by_trid:
                    target_trid = sid
                else:
                    # Strip promotion suffix: `spec-3-spec-7` → `spec-3`.
                    parts = sid.split("-")
                    if len(parts) >= 2:
                        candidate = "-".join(parts[:2])
                        if candidate in cadence_by_trid:
                            target_trid = candidate
                if target_trid:
                    cadence_by_trid[target_trid].verification_status = (
                        vstatus or "—"
                    )

    run.tasks = cadence_by_trid
    if earliest and latest:
        run.run_wall_clock_s = (
            datetime.fromisoformat(latest) - datetime.fromisoformat(earliest)
        ).total_seconds()
    return run


# ── Rendering ───────────────────────────────────────────────────────────────


def _fmt_seconds(value: float) -> str:
    return f"{value:7.1f}s"


def _fmt_delta_seconds(value: float) -> str:
    return f"{value:+7.1f}s"


def _fmt_accept(t: TaskRunStats) -> str:
    if t.spec_total_turns <= 0:
        return "—"
    return f"{t.spec_accept_count}/{t.spec_total_turns} ({t.spec_accept_rate * 100:.1f}%)"


def _fmt_ff(t: TaskRunStats) -> str:
    if t.fast_forward_skip_count <= 0:
        return f"  0    0.0s"
    return f"{t.fast_forward_skip_count:>3} {t.fast_forward_saved_s:>6.1f}s"


def _fmt_drain(t: TaskRunStats) -> str:
    if t.total_drain_count <= 0:
        return f"  0    0.0s"
    return f"{t.total_drain_count:>3} {t.total_drain_s:>6.1f}s"


def render_table(run_a: RunStats, run_b: RunStats) -> str:
    """Render an aligned per-task comparison table to a string."""
    a_idx = run_a.by_task_index()
    b_idx = run_b.by_task_index()
    indices = sorted(set(a_idx) | set(b_idx))

    rows = []
    rows.append(
        f"{'idx':>3}  {'task':22}  "
        f"{run_a.label[:9]:>9}  {run_b.label[:9]:>9}    {'Δ':>9}"
        f"  | {run_a.label[:6] + ' FF':>9} {'saved':>7}"
        f" | {run_b.label[:6] + ' FF':>9} {'saved':>7}"
        f" | {run_a.label[:6] + ' drn':>9} {'wait':>7}"
        f" | {run_b.label[:6] + ' drn':>9} {'wait':>7}"
        f" | {run_b.label[:5] + ' acc':>14}"
        f" | {run_a.label[:3] + ' v':>4} {run_b.label[:3] + ' v':>4}"
    )
    rows.append("=" * len(rows[0]))

    sum_a = sum_b = 0.0
    a_ff_n = a_ff_s = a_dr_n = a_dr_s = 0.0
    b_ff_n = b_ff_s = b_dr_n = b_dr_s = 0.0
    for idx in indices:
        a = a_idx.get(idx)
        b = b_idx.get(idx)
        a_run = a.runtime_s if a else 0.0
        b_run = b.runtime_s if b else 0.0
        delta = b_run - a_run
        sum_a += a_run
        sum_b += b_run

        task = (a or b).task_id[:22]
        a_ff = _fmt_ff(a) if a else "  —      —"
        b_ff = _fmt_ff(b) if b else "  —      —"
        a_dr = _fmt_drain(a) if a else "  —      —"
        b_dr = _fmt_drain(b) if b else "  —      —"
        b_acc = _fmt_accept(b) if b else "—"
        a_v = (a.verification_status or "—")[:4] if a else "—"
        b_v = (b.verification_status or "—")[:4] if b else "—"

        if a:
            a_ff_n += a.fast_forward_skip_count
            a_ff_s += a.fast_forward_saved_s
            a_dr_n += a.total_drain_count
            a_dr_s += a.total_drain_s
        if b:
            b_ff_n += b.fast_forward_skip_count
            b_ff_s += b.fast_forward_saved_s
            b_dr_n += b.total_drain_count
            b_dr_s += b.total_drain_s

        rows.append(
            f"{idx:>3}  {task:22}  "
            f"{_fmt_seconds(a_run)}  {_fmt_seconds(b_run)}  {_fmt_delta_seconds(delta)}"
            f"  | {a_ff:>17}"
            f" | {b_ff:>17}"
            f" | {a_dr:>17}"
            f" | {b_dr:>17}"
            f" | {b_acc:>14}"
            f" | {a_v:>4} {b_v:>4}"
        )

    rows.append("-" * len(rows[0]))
    rows.append(
        f"{'':>3}  {'TOTAL':22}  "
        f"{_fmt_seconds(sum_a)}  {_fmt_seconds(sum_b)}  {_fmt_delta_seconds(sum_b - sum_a)}"
        f"  | {int(a_ff_n):>3} {a_ff_s:>6.1f}s         "
        f" | {int(b_ff_n):>3} {b_ff_s:>6.1f}s         "
        f" | {int(a_dr_n):>3} {a_dr_s:>6.1f}s         "
        f" | {int(b_dr_n):>3} {b_dr_s:>6.1f}s"
    )
    if run_a.run_wall_clock_s is not None and run_b.run_wall_clock_s is not None:
        rows.append(
            f"{'':>3}  {'run wall-clock':22}  "
            f"{_fmt_seconds(run_a.run_wall_clock_s)}  "
            f"{_fmt_seconds(run_b.run_wall_clock_s)}  "
            f"{_fmt_delta_seconds(run_b.run_wall_clock_s - run_a.run_wall_clock_s)}"
        )
    return "\n".join(rows)


def write_paired_csv(path: Path, run_a: RunStats, run_b: RunStats) -> None:
    a_idx = run_a.by_task_index()
    b_idx = run_b.by_task_index()
    indices = sorted(set(a_idx) | set(b_idx))

    fieldnames = ["task_index", "task_id"]
    for label_obj in (run_a, run_b):
        prefix = label_obj.label
        fieldnames.extend(
            [
                f"{prefix}_sandbox_id",
                f"{prefix}_runtime_s",
                f"{prefix}_fast_forward_skip_count",
                f"{prefix}_fast_forward_saved_ms",
                f"{prefix}_pre_fork_drain_count",
                f"{prefix}_pre_fork_drain_ms",
                f"{prefix}_post_match_drain_count",
                f"{prefix}_post_match_drain_ms",
                f"{prefix}_non_spec_drain_count",
                f"{prefix}_non_spec_drain_ms",
                f"{prefix}_spec_total_turns",
                f"{prefix}_spec_accept_count",
                f"{prefix}_verification_status",
            ]
        )
    fieldnames.append("delta_runtime_s")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in indices:
            a = a_idx.get(idx)
            b = b_idx.get(idx)
            row: dict[str, object] = {"task_index": idx}
            row["task_id"] = (a or b).task_id
            for label_obj, t in ((run_a, a), (run_b, b)):
                p = label_obj.label
                if t is None:
                    for k in (
                        "_sandbox_id",
                        "_runtime_s",
                        "_fast_forward_skip_count",
                        "_fast_forward_saved_ms",
                        "_pre_fork_drain_count",
                        "_pre_fork_drain_ms",
                        "_post_match_drain_count",
                        "_post_match_drain_ms",
                        "_non_spec_drain_count",
                        "_non_spec_drain_ms",
                        "_spec_total_turns",
                        "_spec_accept_count",
                        "_verification_status",
                    ):
                        row[p + k] = ""
                    continue
                row[f"{p}_sandbox_id"] = t.sandbox_id_finish or t.sandbox_id_start
                row[f"{p}_runtime_s"] = f"{t.runtime_s:.3f}"
                row[f"{p}_fast_forward_skip_count"] = t.fast_forward_skip_count
                row[f"{p}_fast_forward_saved_ms"] = f"{t.fast_forward_saved_ms:.3f}"
                row[f"{p}_pre_fork_drain_count"] = t.pre_fork_drain_count
                row[f"{p}_pre_fork_drain_ms"] = f"{t.pre_fork_drain_ms:.3f}"
                row[f"{p}_post_match_drain_count"] = t.post_match_drain_count
                row[f"{p}_post_match_drain_ms"] = f"{t.post_match_drain_ms:.3f}"
                row[f"{p}_non_spec_drain_count"] = t.non_spec_drain_count
                row[f"{p}_non_spec_drain_ms"] = f"{t.non_spec_drain_ms:.3f}"
                row[f"{p}_spec_total_turns"] = t.spec_total_turns
                row[f"{p}_spec_accept_count"] = t.spec_accept_count
                row[f"{p}_verification_status"] = t.verification_status
            row["delta_runtime_s"] = (
                f"{(b.runtime_s if b else 0.0) - (a.runtime_s if a else 0.0):.3f}"
            )
            writer.writerow(row)


# ── CLI ─────────────────────────────────────────────────────────────────────


def _default_label(prefix_arg: str) -> str:
    p = _resolve_run(prefix_arg)
    return p.name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two benchmark runs at the per-task level."
    )
    parser.add_argument(
        "run_a",
        help="Run A path: prefix shared by .csv/.log/.telemetry.jsonl/.report",
    )
    parser.add_argument(
        "run_b",
        help="Run B path: same prefix shape as run_a",
    )
    parser.add_argument(
        "--label-a",
        help="Short label for run A in the output (default: filename of run_a)",
    )
    parser.add_argument(
        "--label-b",
        help="Short label for run B in the output (default: filename of run_b)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path to dump structured paired data as CSV.",
    )
    args = parser.parse_args(argv)

    label_a = args.label_a or _default_label(args.run_a)
    label_b = args.label_b or _default_label(args.run_b)

    run_a = _load_run(args.run_a, label_a)
    run_b = _load_run(args.run_b, label_b)

    print(render_table(run_a, run_b))
    if args.csv is not None:
        write_paired_csv(args.csv, run_a, run_b)
        print(f"\nPaired CSV written to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
