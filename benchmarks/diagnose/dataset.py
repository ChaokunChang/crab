from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from benchmarks.core import load_task_dataset

from .models import DatasetTaskInfo


@dataclass(frozen=True)
class DatasetIndex:
    tasks: list[DatasetTaskInfo]
    by_task_id: dict[str, list[DatasetTaskInfo]]
    by_trace_path: dict[str, DatasetTaskInfo]


def _prompt_preview(raw_value: object, *, limit: int = 160) -> str:
    prompt = ""
    if isinstance(raw_value, dict):
        raw_prompt = raw_value.get("prompt", "")
        if isinstance(raw_prompt, str):
            prompt = raw_prompt.strip()
    if len(prompt) <= limit:
        return prompt
    return f"{prompt[: max(0, limit - 3)].rstrip()}..."


def load_dataset_index(dataset_path: Path | None) -> DatasetIndex:
    if dataset_path is None:
        return DatasetIndex(tasks=[], by_task_id={}, by_trace_path={})
    records = load_task_dataset(dataset_path)
    raw_rows: list[dict[str, object]] = []
    for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            raw_rows.append(payload)
    tasks: list[DatasetTaskInfo] = []
    by_task_id: dict[str, list[DatasetTaskInfo]] = defaultdict(list)
    by_trace_path: dict[str, DatasetTaskInfo] = {}
    for index, record in enumerate(records):
        raw = raw_rows[index] if index < len(raw_rows) else {}
        trace_path_value = None
        if record.llm_service_config is not None:
            raw_trace_path = record.llm_service_config.get("trace_path")
            if isinstance(raw_trace_path, str) and raw_trace_path:
                trace_path_value = Path(raw_trace_path).expanduser().resolve()
        task = DatasetTaskInfo(
            dataset_index=index,
            task_id=record.task_id or f"dataset-{index}",
            agent_type=record.agent_type,
            llm_service_type=record.llm_service_type,
            trace_path=trace_path_value,
            trace_response_count=record.trace_response_count,
            trace_malformed_line_count=record.trace_malformed_line_count,
            task_root=record.task_root,
            service_name=record.service_name,
            prompt_preview=_prompt_preview(raw.get("task_description")),
            raw=raw,
        )
        tasks.append(task)
        by_task_id[task.task_id].append(task)
        if task.trace_path is not None:
            by_trace_path[str(task.trace_path)] = task
    return DatasetIndex(tasks=tasks, by_task_id=dict(by_task_id), by_trace_path=by_trace_path)
