from __future__ import annotations

import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .models import CsvSandboxRow, DatasetTaskInfo, MissingTaskRecord


@dataclass(frozen=True)
class CsvReport:
    rows: list[CsvSandboxRow]
    passed_rows: list[CsvSandboxRow]
    failed_rows: list[CsvSandboxRow]
    suspicious_rows: list[CsvSandboxRow]
    missing_tasks: list[MissingTaskRecord]


def _parse_float(raw_value: str | None) -> float | None:
    if raw_value is None:
        return None
    value = raw_value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(raw_value: str | None) -> int | None:
    if raw_value is None:
        return None
    value = raw_value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def classify_csv_row(row: dict[str, str]) -> str:
    verification_status = (row.get("verification_status") or "").strip().lower()
    success_ratio = _parse_float(row.get("success_ratio"))
    task_error = (row.get("task_error") or "").strip()
    verification_exit_code = row.get("verification_exit_code")
    if verification_status and verification_status != "passed":
        return "failed"
    if success_ratio is not None and success_ratio < 1.0:
        return "failed"
    if task_error:
        return "failed"
    if not verification_status and (
        verification_exit_code not in (None, "")
        or row.get("verification_stdout")
        or row.get("verification_stderr")
    ):
        return "suspicious"
    if verification_status == "passed":
        return "passed"
    return "suspicious"


def parse_csv_report(csv_path: Path | None, dataset_tasks: list[DatasetTaskInfo]) -> CsvReport:
    if csv_path is None or not csv_path.exists():
        return CsvReport(
            rows=[],
            passed_rows=[],
            failed_rows=[],
            suspicious_rows=[],
            missing_tasks=_compute_missing_tasks(dataset_tasks, []),
        )
    rows: list[CsvSandboxRow] = []
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, raw in enumerate(reader, start=1):
            row = {str(key): str(value) if value is not None else "" for key, value in raw.items()}
            sandbox_id = (row.get("sandbox_id") or "").strip() or f"row-{row_index}"
            task_id = (row.get("task_id") or "").strip() or sandbox_id
            record = CsvSandboxRow(
                row_index=row_index,
                sandbox_id=sandbox_id,
                task_id=task_id,
                iteration=_parse_int(row.get("iteration")),
                classification=classify_csv_row(row),
                success_ratio=_parse_float(row.get("success_ratio")),
                verification_status=(row.get("verification_status") or "").strip() or None,
                verification_exit_code=_parse_int(row.get("verification_exit_code")),
                task_error=(row.get("task_error") or "").strip() or None,
                event_type=(row.get("event_type") or "").strip() or None,
                raw=row,
            )
            rows.append(record)
    passed_rows = [row for row in rows if row.classification == "passed"]
    failed_rows = [row for row in rows if row.classification == "failed"]
    suspicious_rows = [row for row in rows if row.classification == "suspicious"]
    return CsvReport(
        rows=rows,
        passed_rows=passed_rows,
        failed_rows=failed_rows,
        suspicious_rows=suspicious_rows,
        missing_tasks=_compute_missing_tasks(dataset_tasks, rows),
    )


def _compute_missing_tasks(
    dataset_tasks: list[DatasetTaskInfo],
    csv_rows: list[CsvSandboxRow],
) -> list[MissingTaskRecord]:
    observed_by_task = Counter(row.task_id for row in csv_rows)
    expected_by_task = Counter(task.task_id for task in dataset_tasks)
    missing: list[MissingTaskRecord] = []
    emitted_so_far: Counter[str] = Counter()
    for task in dataset_tasks:
        emitted_so_far[task.task_id] += 1
        observed = observed_by_task.get(task.task_id, 0)
        expected = expected_by_task[task.task_id]
        if observed >= emitted_so_far[task.task_id]:
            continue
        reason = "missing entirely" if observed == 0 else "fewer CSV rows than dataset rows"
        missing.append(
            MissingTaskRecord(
                dataset_index=task.dataset_index,
                task_id=task.task_id,
                reason=reason,
                occurrences_expected=expected,
                occurrences_observed=observed,
                trace_path=task.trace_path,
            )
        )
    return missing
