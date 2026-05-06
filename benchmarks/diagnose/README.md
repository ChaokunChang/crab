# Benchmark Diagnose

`benchmarks.diagnose` is a postmortem-oriented helper package for benchmark runs.

It reads a benchmark YAML config, resolves the corresponding artifact paths, infers the actual benchmark run root from the log, and produces a structured diagnosis report from:

- the benchmark config
- the benchmark CSV
- the benchmark log
- the telemetry JSONL
- the dataset JSONL
- iFlow replay traces
- iFlow session trajectories under `{benchmark_root}/iflow/{sandbox_id}/iflow-state/.iflow/projects/-app`

## CLI

Run the default text report:

```bash
python3 -m benchmarks.diagnose --config benchmarks/examples/iflow/iflow.fault.auto.115tasks.debug.4.yaml
```

Useful options:

- `--sandbox-id fault-54`: focus on one sandbox
- `--task-id compile-compcert`: focus on one task
- `--output-dir out/diagnosis-run`: write `diagnosis.txt/.md/.html/.json` plus one `{sandbox-id}.log` file for each failed sandbox
- `--output-json out/diagnosis.json`: write structured JSON
- `--output-markdown out/diagnosis.md`: write Markdown
- `--output-html out/diagnosis.html`: write an HTML report for human inspection
- `--max-log-lines 40`: increase rendered log excerpts
- `--max-tool-arg-chars 400`: increase stored tool argument previews during parsing
- `--max-visualized-tool-arg-chars 80`: shorten rendered trace/session argument previews
- `--max-tool-comparison-rows 12`: limit the high-signal comparison table
- `--max-visualized-tool-rows 12`: limit the trace/observed detail tables
- `--max-timeline-events 12`: limit rendered timeline rows
- `--include-passed`: include passed sandboxes in the report

## Import API

```python
from pathlib import Path

from benchmarks.diagnose import (
    diagnose_benchmark_config,
    render_run_diagnosis_markdown,
    render_run_diagnosis_text,
)

report = diagnose_benchmark_config(Path("benchmarks/examples/iflow/iflow.fault.auto.115tasks.debug.4.yaml"))
print(render_run_diagnosis_text(report))
```

## What It Extracts

- Run metadata and resolved artifact paths
- Actual run root inferred from `runc --root .../runtime-state`
- Missing dataset tasks by comparing dataset rows against CSV rows
- Failed sandbox summaries from CSV classification
- Sandbox-specific log slices and high-signal excerpts
- Telemetry timelines for checkpoint, restore, recovery, verification, and fault events
- iFlow replay trace summaries
- iFlow session tool-call summaries
- replay-marker presence in the observed session file when fields like `trace_cursor`, `consumed_response_count`, or `action_replay` are present
- trace-vs-session tool-call alignment summaries
- tool duration hints from both replay traces and iFlow session timestamps, plus runtime telemetry
- HTML visualization for timelines and tool-call comparisons
- Per-failed-sandbox extracted log files when using `--output-dir`
- Heuristic findings backed by evidence references

## Notes

- The package is optimized for replay-based iFlow runs first, but it still works for generic benchmark runs with reduced enrichment.
- Missing tasks are reported both in the run summary and as synthetic `missing-task-*` diagnosis entries.
- Tool-call arguments are cropped by default to keep reports readable.
- The heuristics are evidence-backed helpers, not a substitute for reading the raw artifacts when a case is subtle.
