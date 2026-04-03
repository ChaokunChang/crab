from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from benchmarks.config import BenchmarkConfig, load_config

from .models import DiagnoseRunContext


_RUNC_ROOT_RE = re.compile(r"runc\s+--root\s+([^\s]+?/runtime-state)\b")


@dataclass(frozen=True)
class LoadedArtifacts:
    config: BenchmarkConfig
    context: DiagnoseRunContext


def _resolve_telemetry_path(config: BenchmarkConfig) -> Path | None:
    if config.telemetry_output is not None:
        return config.telemetry_output.resolve()
    if config.output is not None:
        return config.output.with_suffix(".telemetry.jsonl").resolve()
    return config.config_path.with_suffix(".telemetry.jsonl").resolve()


def infer_actual_benchmark_root(log_path: Path | None) -> tuple[Path | None, tuple[str, ...]]:
    if log_path is None or not log_path.exists():
        return (None, ())
    counts: Counter[str] = Counter()
    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _RUNC_ROOT_RE.search(raw_line)
        if not match:
            continue
        runtime_root = Path(match.group(1))
        counts[str(runtime_root.parent)] += 1
    if not counts:
        return (None, ())
    most_common = counts.most_common()
    top_count = most_common[0][1]
    winners = sorted(root for root, count in most_common if count == top_count)
    if len(winners) > 1:
        raise ValueError(
            f"ambiguous benchmark roots inferred from {log_path}: {winners}"
        )
    return (Path(winners[0]), tuple(root for root, _ in most_common))


def load_artifacts(config_path: Path) -> LoadedArtifacts:
    config = load_config(config_path)
    log_path = None if config.log_file is None else config.log_file.resolve()
    csv_path = None if config.output is None else config.output.resolve()
    telemetry_path = _resolve_telemetry_path(config)
    actual_root, inferred_roots = infer_actual_benchmark_root(log_path)
    context = DiagnoseRunContext(
        config_path=config.config_path,
        scenario=config.scenario,
        mode=config.mode,
        provider=config.provider,
        agent=config.agent,
        llm_service=config.llm_service,
        task_dataset_path=None if config.task_dataset is None else config.task_dataset.resolve(),
        log_path=log_path,
        csv_path=csv_path,
        telemetry_path=telemetry_path,
        configured_benchmark_root=None
        if config.benchmark_root is None
        else config.benchmark_root.resolve(),
        actual_benchmark_root=actual_root,
        inferred_run_roots=inferred_roots,
    )
    return LoadedArtifacts(config=config, context=context)
