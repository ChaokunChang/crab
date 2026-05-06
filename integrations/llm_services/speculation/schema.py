from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "2"

AGENT_TERMINUS = "terminus"
AGENT_MINI_SWE = "mini_swe"
AGENT_CLAUDE_CODE = "claude_code"
SUPPORTED_AGENT_KINDS = {AGENT_TERMINUS, AGENT_MINI_SWE, AGENT_CLAUDE_CODE}

# Acceptance levels, ordered strictest -> loosest. Always evaluated together
# so a single collection run produces every level's accept rate.
LEVEL_LITERAL = "literal"
LEVEL_NORMALIZED = "normalized"
LEVEL_SEMANTIC = "semantic"
SCORE_LEVELS: tuple[str, ...] = (LEVEL_LITERAL, LEVEL_NORMALIZED, LEVEL_SEMANTIC)
DEFAULT_REPORT_LEVEL = LEVEL_NORMALIZED


@dataclasses.dataclass
class SpeculationTurn:
    turn_index: int
    oracle_first_command: str
    draft_response_content: str
    draft_first_command: str
    accepted: dict[str, bool] = dataclasses.field(default_factory=dict)
    draft_latency_ms: float | None = None
    oracle_latency_ms: float | None = None
    draft_prompt_tokens: int | None = None
    draft_completion_tokens: int | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "turn_index": self.turn_index,
            "oracle_first_command": self.oracle_first_command,
            "draft_response_content": self.draft_response_content,
            "draft_first_command": self.draft_first_command,
            "accepted": dict(self.accepted),
        }
        if self.draft_latency_ms is not None:
            payload["draft_latency_ms"] = self.draft_latency_ms
        if self.oracle_latency_ms is not None:
            payload["oracle_latency_ms"] = self.oracle_latency_ms
        if self.draft_prompt_tokens is not None:
            payload["draft_prompt_tokens"] = self.draft_prompt_tokens
        if self.draft_completion_tokens is not None:
            payload["draft_completion_tokens"] = self.draft_completion_tokens
        if self.error:
            payload["error"] = self.error
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> SpeculationTurn:
        accepted_raw = payload.get("accepted", {})
        if isinstance(accepted_raw, dict):
            accepted = {str(k): bool(v) for k, v in accepted_raw.items()}
        elif isinstance(accepted_raw, bool):
            # Tolerate v1 sidecars that stored a single boolean.
            accepted = {DEFAULT_REPORT_LEVEL: bool(accepted_raw)}
        else:
            accepted = {}
        return cls(
            turn_index=int(payload["turn_index"]),
            oracle_first_command=str(payload.get("oracle_first_command", "")),
            draft_response_content=str(payload.get("draft_response_content", "")),
            draft_first_command=str(payload.get("draft_first_command", "")),
            accepted=accepted,
            draft_latency_ms=_optional_float(payload.get("draft_latency_ms")),
            oracle_latency_ms=_optional_float(payload.get("oracle_latency_ms")),
            draft_prompt_tokens=_optional_int(payload.get("draft_prompt_tokens")),
            draft_completion_tokens=_optional_int(payload.get("draft_completion_tokens")),
            error=None if payload.get("error") in (None, "") else str(payload["error"]),
        )


@dataclasses.dataclass
class SpeculationSidecar:
    trace_path: str
    agent_kind: str
    draft_model: dict[str, Any]
    score_levels: list[str] = dataclasses.field(
        default_factory=lambda: list(SCORE_LEVELS)
    )
    turns: list[SpeculationTurn] = dataclasses.field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.agent_kind not in SUPPORTED_AGENT_KINDS:
            raise ValueError(
                f"agent_kind must be one of {sorted(SUPPORTED_AGENT_KINDS)}, got {self.agent_kind!r}"
            )
        if not self.score_levels:
            raise ValueError("score_levels must be non-empty")
        unknown = [lvl for lvl in self.score_levels if lvl not in SCORE_LEVELS]
        if unknown:
            raise ValueError(f"unknown score level(s): {unknown}")

    def covered_turn_indices(self) -> set[int]:
        return {t.turn_index for t in self.turns if t.error is None}

    def summary(self) -> dict[str, Any]:
        total = len(self.turns)
        errors = sum(1 for t in self.turns if t.error)
        scored = total - errors
        accept_counts: dict[str, int] = {lvl: 0 for lvl in self.score_levels}
        for turn in self.turns:
            if turn.error:
                continue
            for lvl in self.score_levels:
                if turn.accepted.get(lvl, False):
                    accept_counts[lvl] += 1
        accept_rates = {
            lvl: (accept_counts[lvl] / scored) if scored > 0 else 0.0
            for lvl in self.score_levels
        }
        return {
            "turns": total,
            "scored": scored,
            "errors": errors,
            "accept_counts": accept_counts,
            "accept_rates": accept_rates,
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trace_path": self.trace_path,
            "agent_kind": self.agent_kind,
            "draft_model": self.draft_model,
            "score_levels": list(self.score_levels),
            "summary": self.summary(),
            "turns": [t.to_json() for t in sorted(self.turns, key=lambda t: t.turn_index)],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> SpeculationSidecar:
        version = str(payload.get("schema_version", ""))
        if version not in {SCHEMA_VERSION, "1"}:
            raise ValueError(
                f"unsupported speculation sidecar schema_version {version!r}"
            )
        turns_raw = payload.get("turns") or []
        if not isinstance(turns_raw, list):
            raise ValueError("speculation sidecar 'turns' must be a list")
        turns = [SpeculationTurn.from_json(t) for t in turns_raw if isinstance(t, dict)]
        draft_model = payload.get("draft_model") or {}
        if not isinstance(draft_model, dict):
            raise ValueError("speculation sidecar 'draft_model' must be an object")
        score_levels_raw = payload.get("score_levels")
        if isinstance(score_levels_raw, list) and score_levels_raw:
            score_levels = [str(s) for s in score_levels_raw if isinstance(s, str)]
        else:
            # v1 carried a single match_policy string; fall back to that.
            policy = payload.get("match_policy")
            score_levels = (
                [DEFAULT_REPORT_LEVEL] if not isinstance(policy, str) else [policy]
            )
            score_levels = [
                lvl if lvl in SCORE_LEVELS else DEFAULT_REPORT_LEVEL for lvl in score_levels
            ]
        return cls(
            trace_path=str(payload.get("trace_path", "")),
            agent_kind=str(payload.get("agent_kind", "")),
            draft_model=dict(draft_model),
            score_levels=score_levels,
            turns=turns,
            schema_version=SCHEMA_VERSION,
        )


def resolve_sidecar_path(
    *,
    sidecar_root: Path,
    draft_tag: str,
    trace_path: Path,
) -> Path:
    """Map (root, draft_tag, trace_path) to a stable on-disk sidecar path.

    Layout: ``<sidecar_root>/<draft_tag>/<sha1(abs_trace_path)[:12]>/<trace_basename>.spec.json``
    """
    if not draft_tag.strip():
        raise ValueError("draft_tag must be non-empty")
    abs_trace = str(trace_path.expanduser().resolve())
    digest = hashlib.sha1(abs_trace.encode("utf-8")).hexdigest()[:12]
    base = trace_path.name
    if base.endswith(".json"):
        base = base[: -len(".json")]
    safe_tag = _sanitize_path_segment(draft_tag)
    return (
        sidecar_root.expanduser().resolve()
        / safe_tag
        / digest
        / f"{base}.spec.json"
    )


def load_sidecar(path: Path) -> SpeculationSidecar:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"speculation sidecar {path} did not decode to an object")
    return SpeculationSidecar.from_json(payload)


def write_sidecar(path: Path, sidecar: SpeculationSidecar) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(sidecar.to_json(), ensure_ascii=False, indent=2, sort_keys=False)
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def resolve_side_by_side_csv_path(
    *,
    csv_root: Path,
    draft_tag: str,
    run_id: str,
    trace_path: Path,
) -> Path:
    """Map (csv_root, draft_tag, run_id, trace_path) to a stable CSV path.

    Layout: ``<csv_root>/<draft_tag>/<run_id>/<sha1(abs_trace_path)[:12]>__<basename>.csv``

    The hash prefix prevents collisions when two traces share a basename
    (e.g. several ``trajectory.json`` files from different submissions in the
    same run).
    """
    if not draft_tag.strip():
        raise ValueError("draft_tag must be non-empty")
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    abs_trace = str(trace_path.expanduser().resolve())
    digest = hashlib.sha1(abs_trace.encode("utf-8")).hexdigest()[:12]
    base = trace_path.name
    if base.endswith(".json"):
        base = base[: -len(".json")]
    safe_tag = _sanitize_path_segment(draft_tag)
    safe_run = _sanitize_path_segment(run_id)
    return (
        csv_root.expanduser().resolve()
        / safe_tag
        / safe_run
        / f"{digest}__{base}.csv"
    )


def write_side_by_side_csv(path: Path, sidecar: SpeculationSidecar) -> None:
    """Write a human-readable per-turn side-by-side CSV.

    Columns: ``turn, oracle, draft, <levels...>, oracle_latency_ms,
    draft_latency_ms, speedup, error``.

    * Boolean accept verdicts are emitted as ``true`` / ``false`` strings.
    * Latencies are formatted as floats with one decimal (millisecond
      precision is the unit, fractional ms are kept so per-turn totals stay
      summable). Empty when not recorded.
    * ``speedup`` is ``oracle_latency_ms / draft_latency_ms`` when both are
      known and the draft latency is positive; empty otherwise. >1 means the
      draft was faster than the oracle.
    * ``error`` is empty when the turn was scored successfully.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    levels = list(sidecar.score_levels)
    header = [
        "turn",
        "oracle",
        "draft",
        *levels,
        "oracle_latency_ms",
        "draft_latency_ms",
        "speedup",
        "error",
    ]
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for turn in sorted(sidecar.turns, key=lambda t: t.turn_index):
            row: list[Any] = [
                turn.turn_index,
                turn.oracle_first_command,
                turn.draft_first_command,
            ]
            for lvl in levels:
                row.append("true" if turn.accepted.get(lvl, False) else "false")
            row.append(_format_latency(turn.oracle_latency_ms))
            row.append(_format_latency(turn.draft_latency_ms))
            row.append(_format_speedup(turn.oracle_latency_ms, turn.draft_latency_ms))
            row.append(turn.error or "")
            writer.writerow(row)
    os.replace(tmp_path, path)


def _format_latency(value: float | None) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return ""


def _format_speedup(oracle_ms: float | None, draft_ms: float | None) -> str:
    if oracle_ms is None or draft_ms is None:
        return ""
    try:
        d = float(draft_ms)
    except (TypeError, ValueError):
        return ""
    if d <= 0:
        return ""
    try:
        return f"{float(oracle_ms) / d:.4f}"
    except (TypeError, ValueError):
        return ""


def _sanitize_path_segment(value: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-._" else "_" for c in value.strip())
    return cleaned or "draft"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
