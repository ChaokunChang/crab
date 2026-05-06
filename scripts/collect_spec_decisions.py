"""Collect real draft-model speculative decisions for spec-replay benchmarks.

For each recorded oracle trace we walk every assistant turn, reconstruct the
chat-completions input the original model saw, send it to a configurable draft
model, score the draft response against the oracle, and write a sidecar
``SpeculationSidecar`` JSON. The replay-side spec services consume those
sidecars at run time so benchmarks reflect a real draft model's accept rate
without making any network calls during replay.

Phase A (this script) is intentionally decoupled from replay: collection is
slow and depends on external endpoints; replay must stay deterministic and
hermetic. See ``integrations/llm_services/speculation/`` for the schema.

Usage:
    python3 scripts/collect_spec_decisions.py \\
        --dataset results/datasets/terminus_replay_spec_friendly.jsonl \\
        --draft-tag deepseek-v3.2-fast \\
        --draft-base-url https://api.deepseek.com \\
        --draft-api-key-env DEEPSEEK_API_KEY \\
        --draft-model deepseek-chat \\
        --concurrency 8 \\
        --resume

A single trace can be processed without a dataset via ``--trace path.json
--agent terminus`` (or ``--agent mini_swe``). Multiple ``--trace`` flags are
allowed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from integrations.llm_services.speculation.claude_code_tools import (  # noqa: E402
    CLAUDE_CODE_TOOLS,
)
from integrations.llm_services.speculation.draft_client import (  # noqa: E402
    DraftRequestError,
    OpenAICompatibleDraftClient,
)
from integrations.llm_services.speculation.reconstruct import (  # noqa: E402
    ReconstructedTurn,
    reconstruct_claude_code_turns,
    reconstruct_mini_swe_turns,
    reconstruct_terminus_turns,
)
from integrations.llm_services.speculation.schema import (  # noqa: E402
    AGENT_CLAUDE_CODE,
    AGENT_MINI_SWE,
    AGENT_TERMINUS,
    DEFAULT_REPORT_LEVEL,
    SCORE_LEVELS,
    SpeculationSidecar,
    SpeculationTurn,
    load_sidecar,
    resolve_side_by_side_csv_path,
    resolve_sidecar_path,
    write_side_by_side_csv,
    write_sidecar,
)
from integrations.llm_services.speculation.score import (  # noqa: E402
    extract_first_command,
    score_levels,
)


logger = logging.getLogger("collect_spec_decisions")


_DEFAULT_SIDECAR_ROOT = _REPO_ROOT / "results" / "spec_decisions"
_DEFAULT_CSV_ROOT = _REPO_ROOT / "results" / "spec_decisions_csv"

_AGENT_KIND_FROM_LLM_SERVICE = {
    "terminus_trace_replay": AGENT_TERMINUS,
    "terminus_spec_trace_replay": AGENT_TERMINUS,
    "mini_swe_trace_replay": AGENT_MINI_SWE,
    "mini_swe_spec_trace_replay": AGENT_MINI_SWE,
    "claude_code_trace_replay": AGENT_CLAUDE_CODE,
    "claude_code_spec_trace_replay": AGENT_CLAUDE_CODE,
}


def main() -> int:
    parser = _build_argparser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    targets = list(_resolve_targets(args))
    if not targets:
        parser.error("no traces to process; pass --dataset or one or more --trace")
    logger.info("Resolved %d trace(s) to process", len(targets))

    api_key = _resolve_api_key(args)
    params = _parse_json_dict(args.draft_params, flag="--draft-params")
    client = OpenAICompatibleDraftClient(
        base_url=args.draft_base_url,
        api_key=api_key,
        model=args.draft_model,
        params=params,
        timeout_s=args.draft_timeout,
    )

    sidecar_root = Path(args.sidecar_root).expanduser().resolve()
    sidecar_root.mkdir(parents=True, exist_ok=True)

    csv_root = (
        Path(args.csv_root).expanduser().resolve()
        if args.csv_root and args.csv_root.strip()
        else None
    )
    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if csv_root is not None:
        csv_root.mkdir(parents=True, exist_ok=True)
        logger.info("Side-by-side CSVs will be written under %s/<draft_tag>/%s/", csv_root, run_id)

    overall_summary = _CollectionSummary(report_level=args.report_level)

    # Parallelism strategy:
    #   * Single trace  → parallel turns (use the script's --concurrency for
    #     intra-trace turn parallelism). No prefix-cache benefit to give up,
    #     and we want low latency.
    #   * Multiple traces → parallel *traces*, sequential turns within each
    #     trace. Sequential turns let the LLM engine reuse the previous
    #     turn's prompt prefix as a context cache (DeepSeek and most
    #     OpenAI-compatible servers do this implicitly when the prefix
    #     repeats), which dramatically improves throughput on long traces.
    if len(targets) == 1:
        traj_workers = 1
        intra_concurrency = max(1, args.concurrency)
        if intra_concurrency > 1:
            logger.info("Single-trace mode: %d parallel turn workers", intra_concurrency)
    else:
        traj_workers = max(1, args.concurrency)
        intra_concurrency = 1
        logger.info(
            "Multi-trace mode: %d parallel traces, sequential turns per trace "
            "(prefix-cache friendly)",
            traj_workers,
        )

    process_fn = _make_process_trace_fn(
        client=client,
        sidecar_root=sidecar_root,
        csv_root=csv_root,
        run_id=run_id,
        draft_tag=args.draft_tag,
        draft_model=args.draft_model,
        draft_base_url=args.draft_base_url,
        draft_params=params,
        report_level=args.report_level,
        intra_concurrency=intra_concurrency,
        resume=args.resume,
        limit_turns=args.limit_turns,
        dry_run=args.dry_run,
    )

    if traj_workers == 1:
        for index, target in enumerate(targets, start=1):
            logger.info(
                "[%d/%d] %s (%s)",
                index,
                len(targets),
                target.trace_path,
                target.agent_kind,
            )
            try:
                stats = process_fn(target)
            except Exception:
                logger.exception("trace failed: %s", target.trace_path)
                overall_summary.failed_traces += 1
                continue
            overall_summary.merge(stats)
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=traj_workers) as pool:
            futures = {pool.submit(process_fn, t): t for t in targets}
            for future in as_completed(futures):
                target = futures[future]
                completed += 1
                try:
                    stats = future.result()
                except Exception:
                    logger.exception("trace failed: %s", target.trace_path)
                    overall_summary.failed_traces += 1
                    continue
                logger.info(
                    "[%d/%d] done %s (%s)",
                    completed,
                    len(targets),
                    target.trace_path,
                    target.agent_kind,
                )
                overall_summary.merge(stats)

    overall_summary.log()
    return 0 if overall_summary.failed_traces == 0 else 1


def _make_process_trace_fn(**kwargs):
    """Return a single-arg ``(target) -> stats`` closure over collection config."""

    def _fn(target: "_Target") -> dict[str, object]:
        return _process_trace(target=target, **kwargs)

    return _fn


# ---------------------------------------------------------------------------
# Resolving targets


class _Target:
    __slots__ = ("trace_path", "agent_kind", "task_id")

    def __init__(self, trace_path: Path, agent_kind: str, task_id: str | None) -> None:
        self.trace_path = trace_path
        self.agent_kind = agent_kind
        self.task_id = task_id

    def __repr__(self) -> str:
        return f"_Target({self.trace_path}, {self.agent_kind})"


def _resolve_targets(args: argparse.Namespace) -> Iterable[_Target]:
    seen: set[Path] = set()
    if args.dataset:
        for target in _resolve_dataset_targets(Path(args.dataset).expanduser().resolve()):
            if target.trace_path in seen:
                continue
            seen.add(target.trace_path)
            yield target
    if args.trace:
        if not args.agent and not args.dataset:
            raise SystemExit(
                "--trace requires --agent terminus|mini_swe (no dataset to infer from)"
            )
        for raw in args.trace:
            path = Path(raw).expanduser().resolve()
            if path in seen:
                continue
            seen.add(path)
            agent_kind = args.agent or _infer_agent_from_filename(path)
            yield _Target(path, agent_kind, None)


def _resolve_dataset_targets(dataset_path: Path) -> Iterable[_Target]:
    if not dataset_path.is_file():
        raise SystemExit(f"dataset not found: {dataset_path}")
    dataset_root = dataset_path.parent
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{dataset_path}:{line_number} invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                continue
            llm_service_type = str(payload.get("llm_service_type", ""))
            agent_kind = _AGENT_KIND_FROM_LLM_SERVICE.get(llm_service_type)
            if agent_kind is None:
                # Fall back to top-level agent_type field.
                agent_kind = str(payload.get("agent_type", "")).lower()
            if agent_kind not in {AGENT_TERMINUS, AGENT_MINI_SWE, AGENT_CLAUDE_CODE}:
                logger.warning(
                    "%s:%d skipping row: cannot infer agent kind (llm_service_type=%r, agent_type=%r)",
                    dataset_path,
                    line_number,
                    llm_service_type,
                    payload.get("agent_type"),
                )
                continue
            llm_service_config = payload.get("llm_service_config")
            if not isinstance(llm_service_config, dict):
                continue
            trace_rel = llm_service_config.get("trace_path")
            if not isinstance(trace_rel, str) or not trace_rel.strip():
                continue
            trace_path = (dataset_root / trace_rel).resolve()
            yield _Target(trace_path, agent_kind, payload.get("task_id"))


def _infer_agent_from_filename(path: Path) -> str:
    """Best-effort agent inference for --trace without --dataset.

    Both Terminus and Claude Code traces are named ``trajectory.json`` so
    they cannot be distinguished by filename alone; users must pass
    ``--agent`` explicitly when feeding a single ``trajectory.json``.
    """
    name = path.name.lower()
    if name.endswith(".traj.json"):
        return AGENT_MINI_SWE
    raise SystemExit(
        f"cannot infer agent kind from {path}; pass --agent terminus|mini_swe|claude_code explicitly"
    )


# ---------------------------------------------------------------------------
# Per-trace processing


class _CollectionSummary:
    def __init__(self, *, report_level: str) -> None:
        self.report_level = report_level
        self.traces = 0
        self.failed_traces = 0
        self.turns = 0
        self.scored = 0
        self.errors = 0
        self.accept_counts: dict[str, int] = {lvl: 0 for lvl in SCORE_LEVELS}

    def merge(self, stats: dict[str, object]) -> None:
        self.traces += 1
        self.turns += int(stats.get("turns", 0) or 0)
        self.scored += int(stats.get("scored", 0) or 0)
        self.errors += int(stats.get("errors", 0) or 0)
        per_level = stats.get("accept_counts")
        if isinstance(per_level, dict):
            for lvl, count in per_level.items():
                self.accept_counts[lvl] = self.accept_counts.get(lvl, 0) + int(count)

    def log(self) -> None:
        rates = {
            lvl: (self.accept_counts.get(lvl, 0) / self.scored)
            if self.scored > 0
            else 0.0
            for lvl in SCORE_LEVELS
        }
        rates_str = " ".join(f"{lvl}={rates[lvl]:.4f}" for lvl in SCORE_LEVELS)
        headline = rates.get(self.report_level, 0.0)
        logger.info(
            "Done: traces=%d failed=%d turns=%d scored=%d errors=%d accept_rate[%s]=%.4f (%s)",
            self.traces,
            self.failed_traces,
            self.turns,
            self.scored,
            self.errors,
            self.report_level,
            headline,
            rates_str,
        )


def _process_trace(
    *,
    target: _Target,
    client: OpenAICompatibleDraftClient,
    sidecar_root: Path,
    csv_root: Path | None,
    run_id: str,
    draft_tag: str,
    draft_model: str,
    draft_base_url: str,
    draft_params: dict[str, object],
    report_level: str,
    intra_concurrency: int,
    resume: bool,
    limit_turns: int | None,
    dry_run: bool,
) -> dict[str, object]:
    all_turns = _reconstruct_turns(target)
    # The "final turn" flag must be relative to the *full* trajectory so
    # that --limit-turns runs don't accidentally treat a truncated turn as
    # final.
    final_turn_index = max((t.turn_index for t in all_turns), default=-1)
    turns = all_turns
    if limit_turns is not None and limit_turns >= 0:
        turns = turns[:limit_turns]

    sidecar_path = resolve_sidecar_path(
        sidecar_root=sidecar_root,
        draft_tag=draft_tag,
        trace_path=target.trace_path,
    )

    sidecar = _load_or_init_sidecar(
        sidecar_path=sidecar_path,
        target=target,
        draft_tag=draft_tag,
        draft_model=draft_model,
        draft_base_url=draft_base_url,
        draft_params=draft_params,
        resume=resume,
    )

    covered = sidecar.covered_turn_indices() if resume else set()
    pending = [t for t in turns if t.turn_index not in covered]
    logger.info(
        "Trace turns=%d already=%d pending=%d sidecar=%s",
        len(turns),
        len(covered),
        len(pending),
        sidecar_path,
    )

    if dry_run:
        logger.info("Dry-run: skipping draft requests; not modifying sidecar")
        return {
            "turns": len(turns),
            "scored": 0,
            "errors": 0,
            "accept_counts": {lvl: 0 for lvl in SCORE_LEVELS},
        }

    written_lock = threading.Lock()
    completed_records: list[SpeculationTurn] = []

    tools = CLAUDE_CODE_TOOLS if target.agent_kind == AGENT_CLAUDE_CODE else None

    def _worker(turn: ReconstructedTurn) -> SpeculationTurn:
        oracle_cmd = extract_first_command(
            agent_kind=target.agent_kind, content=turn.oracle_response_content
        )
        try:
            result = client.complete(turn.input_messages, tools=tools)
        except DraftRequestError as exc:
            logger.warning(
                "draft request failed turn=%d trace=%s err=%s",
                turn.turn_index,
                target.trace_path,
                exc,
            )
            return SpeculationTurn(
                turn_index=turn.turn_index,
                oracle_first_command=oracle_cmd,
                draft_response_content="",
                draft_first_command="",
                accepted={lvl: False for lvl in SCORE_LEVELS},
                oracle_latency_ms=turn.oracle_latency_ms,
                error=str(exc),
            )
        draft_cmd = extract_first_command(
            agent_kind=target.agent_kind, content=result.content
        )
        verdicts = score_levels(
            agent_kind=target.agent_kind,
            oracle_content=turn.oracle_response_content,
            draft_content=result.content,
            is_final_turn=(turn.turn_index == final_turn_index),
        )
        return SpeculationTurn(
            turn_index=turn.turn_index,
            oracle_first_command=oracle_cmd,
            draft_response_content=result.content,
            draft_first_command=draft_cmd,
            accepted=verdicts,
            draft_latency_ms=result.latency_ms,
            oracle_latency_ms=turn.oracle_latency_ms,
            draft_prompt_tokens=result.prompt_tokens,
            draft_completion_tokens=result.completion_tokens,
        )

    if pending:
        # Persist incremental progress every N completions so that a crash
        # mid-trace doesn't lose hours of draft inference.
        save_every = max(1, min(50, len(pending) // 4))
        completed_since_save = 0

        def _record_completion(record: SpeculationTurn) -> None:
            nonlocal completed_since_save
            with written_lock:
                sidecar.turns = [
                    t for t in sidecar.turns if t.turn_index != record.turn_index
                ]
                sidecar.turns.append(record)
                completed_records.append(record)
                completed_since_save += 1
                if completed_since_save >= save_every:
                    write_sidecar(sidecar_path, sidecar)
                    completed_since_save = 0

        if intra_concurrency <= 1:
            # Sequential turn execution — preserves prompt-prefix repetition
            # so the server-side context cache can hit on every turn after
            # the first.
            for turn in sorted(pending, key=lambda t: t.turn_index):
                _record_completion(_worker(turn))
        else:
            with ThreadPoolExecutor(max_workers=intra_concurrency) as pool:
                futures = {pool.submit(_worker, t): t for t in pending}
                for future in as_completed(futures):
                    _record_completion(future.result())

    write_sidecar(sidecar_path, sidecar)
    if csv_root is not None:
        csv_path = resolve_side_by_side_csv_path(
            csv_root=csv_root,
            draft_tag=draft_tag,
            run_id=run_id,
            trace_path=target.trace_path,
        )
        write_side_by_side_csv(csv_path, sidecar)
        logger.info("Wrote side-by-side CSV %s", csv_path)
    summary = sidecar.summary()
    rates = summary.get("accept_rates") or {}
    headline = float(rates.get(report_level, 0.0))
    rates_str = " ".join(f"{lvl}={float(rates.get(lvl, 0.0)):.4f}" for lvl in SCORE_LEVELS)
    logger.info(
        "Wrote %s — turns=%s scored=%s errors=%s accept_rate[%s]=%.4f (%s)",
        sidecar_path,
        summary.get("turns"),
        summary.get("scored"),
        summary.get("errors"),
        report_level,
        headline,
        rates_str,
    )
    return {
        "turns": int(summary.get("turns", 0) or 0),
        "scored": int(summary.get("scored", 0) or 0),
        "errors": int(summary.get("errors", 0) or 0),
        "accept_counts": dict(summary.get("accept_counts") or {}),
    }


def _reconstruct_turns(target: _Target) -> list[ReconstructedTurn]:
    if target.agent_kind == AGENT_TERMINUS:
        return reconstruct_terminus_turns(target.trace_path)
    if target.agent_kind == AGENT_MINI_SWE:
        return reconstruct_mini_swe_turns(target.trace_path)
    if target.agent_kind == AGENT_CLAUDE_CODE:
        return reconstruct_claude_code_turns(target.trace_path)
    raise ValueError(f"unsupported agent_kind {target.agent_kind!r}")


def _load_or_init_sidecar(
    *,
    sidecar_path: Path,
    target: _Target,
    draft_tag: str,
    draft_model: str,
    draft_base_url: str,
    draft_params: dict[str, object],
    resume: bool,
) -> SpeculationSidecar:
    draft_descriptor = {
        "tag": draft_tag,
        "name": draft_model,
        "base_url": draft_base_url,
        "params": draft_params,
    }
    if resume and sidecar_path.is_file():
        try:
            existing = load_sidecar(sidecar_path)
        except Exception:
            logger.warning(
                "failed to load existing sidecar %s; rebuilding from scratch",
                sidecar_path,
                exc_info=True,
            )
        else:
            if (
                existing.draft_model.get("name") != draft_model
                or existing.draft_model.get("tag") != draft_tag
            ):
                logger.warning(
                    "existing sidecar has draft_model=%s; current=%s — rebuilding",
                    existing.draft_model,
                    draft_descriptor,
                )
            else:
                existing.draft_model = draft_descriptor
                existing.score_levels = list(SCORE_LEVELS)
                return existing
    return SpeculationSidecar(
        trace_path=str(target.trace_path),
        agent_kind=target.agent_kind,
        draft_model=draft_descriptor,
        score_levels=list(SCORE_LEVELS),
    )


# ---------------------------------------------------------------------------
# CLI plumbing


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", help="task JSONL dataset (e.g. terminus_replay_spec_friendly.jsonl)")
    parser.add_argument("--trace", action="append", help="single trace file (repeatable)")
    parser.add_argument(
        "--agent",
        choices=[AGENT_TERMINUS, AGENT_MINI_SWE, AGENT_CLAUDE_CODE],
        help="agent kind; required when using --trace and not --dataset",
    )
    parser.add_argument(
        "--draft-tag",
        required=True,
        help="short label identifying this draft model (used in sidecar paths and metadata)",
    )
    parser.add_argument(
        "--draft-base-url",
        required=True,
        help="OpenAI-compatible base URL, e.g. https://api.deepseek.com",
    )
    parser.add_argument(
        "--draft-model",
        required=True,
        help="model id passed in the chat-completions request body, e.g. deepseek-chat",
    )
    parser.add_argument(
        "--draft-api-key",
        help="API key (prefer --draft-api-key-env to keep secrets out of argv)",
    )
    parser.add_argument(
        "--draft-api-key-env",
        help="environment variable name to read the API key from",
    )
    parser.add_argument(
        "--draft-params",
        default="{}",
        help='JSON object merged into the request body, e.g. \'{"temperature": 0.0, "max_tokens": 2048}\'',
    )
    parser.add_argument(
        "--draft-timeout",
        type=float,
        default=120.0,
        help="HTTP timeout per draft request in seconds (default: 120)",
    )
    parser.add_argument(
        "--report-level",
        default=DEFAULT_REPORT_LEVEL,
        choices=sorted(SCORE_LEVELS),
        help=(
            "which acceptance level the headline log line uses "
            "(all levels are always computed and stored in the sidecar; "
            f"default: {DEFAULT_REPORT_LEVEL})"
        ),
    )
    parser.add_argument(
        "--sidecar-root",
        default=str(_DEFAULT_SIDECAR_ROOT),
        help="root directory under which JSON sidecars are written",
    )
    parser.add_argument(
        "--csv-root",
        default=str(_DEFAULT_CSV_ROOT),
        help=(
            "root directory under which side-by-side CSVs are written; "
            "pass empty string '' to disable CSV output"
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "run identifier used as the per-invocation sub-directory under "
            "<csv-root>/<draft-tag>/. Default: a timestamp generated at "
            "script start. Pass an explicit value to share a CSV directory "
            "across resumed invocations."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help=(
            "number of parallel workers (default: 4). With a single trace, "
            "workers process turns in parallel. With multiple traces, "
            "workers process traces in parallel and turns within each trace "
            "run sequentially so the LLM server's context cache can hit on "
            "every turn after the first."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse existing sidecar entries; only request turns not yet covered",
    )
    parser.add_argument(
        "--limit-turns",
        type=int,
        default=None,
        help="cap turns per trace (debugging aid)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print plan and reconstruct prompts but make no draft requests",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _parse_json_dict(raw: str, *, flag: str) -> dict[str, object]:
    if raw is None or raw == "":
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{flag} must be a JSON object: {exc}") from exc
    if not isinstance(decoded, dict):
        raise SystemExit(f"{flag} must decode to a JSON object, got {type(decoded).__name__}")
    return decoded


def _resolve_api_key(args: argparse.Namespace) -> str | None:
    if args.draft_api_key:
        return args.draft_api_key
    if args.draft_api_key_env:
        value = os.environ.get(args.draft_api_key_env)
        if not value:
            raise SystemExit(
                f"--draft-api-key-env={args.draft_api_key_env} is unset in the environment"
            )
        return value
    return None


if __name__ == "__main__":
    raise SystemExit(main())
