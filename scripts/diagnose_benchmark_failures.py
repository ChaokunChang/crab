#!/usr/bin/env python3
"""Diagnose failed sandboxes in a benchmark run log.

Reads one or more benchmark log files and classifies each failed sandbox
into a diagnostic bucket based on signal lines emitted by the harness.

Buckets it currently recognizes:

  trace_exhausted_with_running_task
      Trace replay finished delivering all responses but the agent's task
      future was still executing a command when the harness's grace period
      expired and `task_run.request_stop()` fired. Symptom in the log:
        WARNING ... Replay is complete but task future is still running ... requesting stop
        ERROR integrations.agents.terminus: Terminus task failed sandbox=...
              error=stop requested while executing command on sandbox ...
      Root cause is generally that command execution on the benchmark host
      is slower than on the host that produced the trace, so the trailing
      tmux quiescence wait runs past the harness grace.

  verification_failed_no_ensurepip
      run-tests.sh exited non-zero AND its stdout contains the canonical
      "ensurepip is not available" message. Fingerprint of the verification
      uv shim's `_create_lightweight_venv` fallback on an image that has
      python3 + python3-pip but no python3.12-venv. The bootstrap shim's
      `python3 -m venv --help` probe returns 0 even when ensurepip is
      missing, so the apt install of python3-venv is skipped.

  verification_failed_other
      run-tests.sh exited non-zero with a different signature.

  verification_uv_bootstrap_failed
      `_ensure_verification_uv` raised after exhausting retries; usually a
      404 fetching an ubuntu archive .deb because the image's apt cache is
      pinned to a version that has been rotated off the mirror.

  task_failed_other
      Some other task-side error surfaced via "Terminus task failed".

  unknown
      Failure was reported in the FAILED summary but no diagnostic signal
      was matched.

Output is tab-separated: SANDBOX TASK BUCKET DETAIL

Examples:
  python3 scripts/diagnose_benchmark_failures.py logs/terminus/*.log
  python3 scripts/diagnose_benchmark_failures.py path/to/run.log --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


_FAILED_RE = re.compile(
    r"FAILED sandbox=(?P<sandbox>\S+) task=(?P<task>\S+) error=(?P<error>.*)$"
)
_INCOMPLETE_RE = re.compile(
    r"INCOMPLETE_REPLAY sandbox=(?P<sandbox>\S+) task=(?P<task>\S+) "
    r"cursor=(?P<cursor>\d+) trace_response_count=(?P<count>\d+) missed=(?P<missed>\d+)"
)
_REPLAY_STOP_RE = re.compile(
    r"Replay is complete but task future is still running; requesting stop "
    r"sandbox=(?P<sandbox>\S+) "
    r"replay_final_trace_cursor=(?P<cursor>\d+) "
    r"trace_response_count=(?P<count>\d+) "
    r"wait_s=(?P<wait>[\d.]+)"
)
_TASK_FAILED_RE = re.compile(
    r"Terminus task failed sandbox=(?P<sandbox>\S+) error=(?P<error>.*)$"
)
_RUN_TESTS_DONE_RE = re.compile(
    r"Completed run-tests\.sh sandbox=(?P<sandbox>\S+) exit_code=(?P<rc>-?\d+) "
)
_BOOTSTRAP_FAIL_RE = re.compile(
    r"Benchmark verification raised an exception sandbox=(?P<sandbox>\S+) error=(?P<error>.*)$"
)
_BOOTSTRAP_RETRY_RE = re.compile(
    r"Retrying transient verification uv bootstrap failure sandbox=(?P<sandbox>\S+) "
    r"attempt=(?P<attempt>\d+) exit_code=(?P<rc>-?\d+) stderr=(?P<stderr>.*)$"
)
_FINALIZED_RE = re.compile(
    r"Replay row finalized sandbox=(?P<sandbox>\S+) task=(?P<task>\S+) "
    r".*? verification_status=(?P<status>\S+)"
)


def _scan_log(path: Path):
    """Return (failed_summary, signals) for a single log file.

    failed_summary: list[tuple[sandbox, task, top_level_error]]
    signals: dict[sandbox] -> dict with diagnostic events (see _classify).
    """
    failed: list[tuple[str, str, str]] = []
    incomplete: list[tuple[str, str, int, int]] = []
    signals: dict[str, dict] = defaultdict(dict)
    finalized_status: dict[str, str] = {}

    # State machine for capturing run-tests stdout/stderr blocks: when we see
    # a stdout/stderr header line, slurp subsequent lines until the next
    # timestamped log line.
    capture: tuple[str, str, str] | None = None  # (sandbox, kind, accumulator)

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            timestamped = bool(re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", line))

            if capture is not None and not timestamped:
                sandbox, kind, acc = capture
                signals[sandbox].setdefault(f"run_tests_{kind}", "")
                signals[sandbox][f"run_tests_{kind}"] += line + "\n"
                continue
            if capture is not None and timestamped:
                capture = None  # fall through to normal handling

            m = _FAILED_RE.search(line)
            if m:
                failed.append((m.group("sandbox"), m.group("task"), m.group("error")))
                continue

            m = _INCOMPLETE_RE.search(line)
            if m:
                incomplete.append(
                    (
                        m.group("sandbox"),
                        m.group("task"),
                        int(m.group("cursor")),
                        int(m.group("count")),
                    )
                )
                continue

            m = _REPLAY_STOP_RE.search(line)
            if m:
                signals[m.group("sandbox")]["replay_stop_requested"] = {
                    "cursor": int(m.group("cursor")),
                    "trace_response_count": int(m.group("count")),
                    "wait_s": float(m.group("wait")),
                }
                continue

            m = _TASK_FAILED_RE.search(line)
            if m:
                signals[m.group("sandbox")]["terminus_task_error"] = m.group("error")
                continue

            m = _RUN_TESTS_DONE_RE.search(line)
            if m:
                signals[m.group("sandbox")]["run_tests_exit_code"] = int(m.group("rc"))
                continue

            m = _BOOTSTRAP_FAIL_RE.search(line)
            if m:
                signals[m.group("sandbox")]["verification_exception"] = m.group("error")
                continue

            m = _BOOTSTRAP_RETRY_RE.search(line)
            if m:
                signals[m.group("sandbox")].setdefault("bootstrap_retries", []).append(
                    {
                        "attempt": int(m.group("attempt")),
                        "rc": int(m.group("rc")),
                        "stderr_head": m.group("stderr")[:200],
                    }
                )
                continue

            m = _FINALIZED_RE.search(line)
            if m:
                finalized_status[m.group("sandbox")] = m.group("status")
                continue

            # run-tests stdout/stderr block headers
            for kind in ("stdout", "stderr"):
                marker = f"run-tests {kind} sandbox="
                idx = line.find(marker)
                if idx >= 0:
                    sandbox = line[idx + len(marker):].split()[0]
                    capture = (sandbox, kind, "")
                    break

    return failed, incomplete, signals, finalized_status


def _classify(
    sandbox: str,
    task: str,
    top_error: str,
    signals: dict,
    incomplete_set: set[str],
    finalized_status: dict[str, str],
) -> tuple[str, str]:
    """Return (bucket, detail_string)."""
    status = finalized_status.get(sandbox)
    stop_event = signals.get("replay_stop_requested")
    task_err = signals.get("terminus_task_error", "")
    rt_rc = signals.get("run_tests_exit_code")
    rt_stdout = signals.get("run_tests_stdout", "")
    rt_stderr = signals.get("run_tests_stderr", "")
    bootstrap_err = signals.get("verification_exception", "")

    if "stop requested while executing command" in top_error or "stop requested while executing command" in task_err:
        cursor = stop_event["cursor"] if stop_event else None
        count = stop_event["trace_response_count"] if stop_event else None
        detail = (
            f"verification_status={status} replay_cursor={cursor}/{count} "
            f"top_error={top_error!r}"
        )
        return "trace_exhausted_with_running_task", detail

    if bootstrap_err:
        return "verification_uv_bootstrap_failed", f"exception={bootstrap_err!r}"

    retries = signals.get("bootstrap_retries") or []
    if retries and rt_rc is None:
        # Bootstrap retried with transient failures and no run-tests.sh
        # ever completed, so verification never ran. Some scenarios swallow
        # the resulting RuntimeError without logging the exception line, so
        # the bootstrap retry trail is the only signal.
        last = retries[-1]
        return (
            "verification_uv_bootstrap_failed",
            f"retries={len(retries)} last_rc={last['rc']} "
            f"last_stderr_head={last['stderr_head']!r}",
        )

    if rt_rc is not None and rt_rc != 0:
        # The python venv error message wraps onto two lines, so match on a
        # collapsed view of stdout to be robust to whitespace.
        rt_stdout_flat = re.sub(r"\s+", " ", rt_stdout)
        if "ensurepip is not available" in rt_stdout_flat:
            ms = re.search(r"apt install (\S+)", rt_stdout_flat)
            need = ms.group(1) if ms else "python3-venv"
            return (
                "verification_failed_no_ensurepip",
                f"run_tests rc={rt_rc} need={need!r} verification_status={status}",
            )
        return "verification_failed_other", f"run_tests rc={rt_rc} verification_status={status}"

    if status == "verification_error":
        return "verification_uv_bootstrap_failed", f"verification_status={status}"

    if task_err and "stop requested" not in task_err:
        return "task_failed_other", f"task_error={task_err!r}"

    if sandbox in incomplete_set:
        return "trace_exhausted_with_running_task", "incomplete replay, no terminus error captured"

    return "unknown", f"top_error={top_error!r} verification_status={status}"


def _emit(rows: list[dict], as_json: bool, fh) -> None:
    if as_json:
        json.dump(rows, fh, indent=2, sort_keys=True)
        fh.write("\n")
        return
    fh.write("LOG\tSANDBOX\tTASK\tBUCKET\tDETAIL\n")
    for r in rows:
        fh.write(
            "\t".join(
                [r["log"], r["sandbox"], r["task"], r["bucket"], r["detail"]]
            )
            + "\n"
        )


def diagnose(paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        failed, incomplete, signals, finalized_status = _scan_log(path)
        incomplete_set = {s for s, _, _, _ in incomplete}
        for sandbox, task, top_error in failed:
            bucket, detail = _classify(
                sandbox,
                task,
                top_error,
                signals.get(sandbox, {}),
                incomplete_set,
                finalized_status,
            )
            rows.append(
                {
                    "log": str(path),
                    "sandbox": sandbox,
                    "task": task,
                    "bucket": bucket,
                    "detail": detail,
                }
            )
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("logs", nargs="+", type=Path, help="benchmark log files")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of TSV")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="also print a per-bucket count summary at the end",
    )
    args = parser.parse_args(argv)

    rows = diagnose(args.logs)
    _emit(rows, args.json, sys.stdout)

    if args.summary and not args.json:
        counts: dict[str, int] = defaultdict(int)
        for r in rows:
            counts[r["bucket"]] += 1
        sys.stdout.write("\n# bucket counts\n")
        for bucket, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            sys.stdout.write(f"{bucket}\t{n}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
