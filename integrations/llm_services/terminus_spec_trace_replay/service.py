from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from integrations.llm_services.terminus_trace_replay.service import (
    _completion_response,
    _config_with_legacy_alias,
    parse_replay_trace,
)
from integrations.llm_services.speculation.schema import (
    DEFAULT_REPORT_LEVEL,
    SCORE_LEVELS,
    SpeculationTurn,
    load_sidecar,
)

_logger = logging.getLogger(__name__)

_DEFAULT_ACCEPTANCE_RATE = 0.5
_DEFAULT_DRAFT_DELAY_SCALING_FACTOR = 1.0
_DEFAULT_RESPONSE_DELAY_MS = 0
_DEFAULT_MINIMAL_DELAY_MS = 0.0
_DEFAULT_MAXIMAL_DELAY_MS = 1_000_000_000.0
_RESPONSE_DELAY_POLICIES = {"fixed", "trace_replay"}
_SUPPORTED_MISMATCH_POLICIES = {"preserve_command_class"}
_SPEC_PAIR_HEADER = "X-AgentCR-Spec-Pair-Id"
_SPEC_ROLE_HEADER = "X-AgentCR-Spec-Role"
_SPEC_ROLE_ORACLE = "oracle"
_SPEC_ROLE_DRAFT = "draft"


def _decode_response(content: str) -> dict[str, Any] | None:
    """Best-effort load of a recorded Terminus assistant response."""
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_first_command(content: str) -> str:
    payload = _decode_response(content)
    if not payload:
        return ""
    commands = payload.get("commands")
    if not isinstance(commands, list) or not commands:
        return ""
    first = commands[0]
    if not isinstance(first, dict):
        return ""
    keystrokes = first.get("keystrokes")
    return str(keystrokes) if isinstance(keystrokes, str) else ""


def _is_task_complete(content: str) -> bool:
    payload = _decode_response(content)
    if not payload:
        return False
    raw = payload.get("task_complete", False)
    if isinstance(raw, str):
        return raw.strip().lower() in {"true", "1", "yes"}
    return bool(raw)


def _normalize_command(command: str) -> str:
    return " ".join(command.split())


def _command_bucket(command: str) -> str:
    """Coarse-grained classification used to pick a same-class mismatch command."""
    normalized = _normalize_command(command)
    lowered = normalized.lower()
    if lowered.startswith("apt-get") or lowered.startswith("apt ") or lowered.startswith("dpkg "):
        return "package"
    if lowered.startswith("git ") or " git " in lowered:
        return "git"
    if (
        lowered.startswith("pytest")
        or lowered.startswith("rscript ")
        or " rscript " in lowered
        or lowered.startswith("python ")
        or lowered.startswith("python3 ")
        or lowered.startswith("npm test")
    ):
        return "test_or_run"
    if lowered.startswith(("cat <<", "cat <", "cat >", "cat >>", "cat - ", "tee ", "tee -")):
        return "heredoc_write"
    read_only_prefixes = (
        "pwd",
        "ls",
        "cat ",
        "head ",
        "tail ",
        "sed -n",
        "grep ",
        "rg ",
        "find ",
        "which ",
        "git status",
        "git diff",
        "git show",
        "wc ",
        "stat ",
        "file ",
    )
    if lowered.startswith(read_only_prefixes):
        return "read_only_shell"
    return "stateful_shell"


def _fallback_command_for_bucket(bucket: str) -> str:
    return {
        "package": "apt-get --version\n",
        "git": "git status --short\n",
        "test_or_run": "echo 'draft test placeholder'\n",
        "heredoc_write": "echo 'draft heredoc placeholder' > /tmp/.draft\n",
        "read_only_shell": "pwd\n",
        "stateful_shell": "touch /tmp/.terminus-spec-draft\n",
    }.get(bucket, "pwd\n")


def _header_value(headers: dict[str, str], name: str, default: str = "") -> str:
    direct = headers.get(name)
    if isinstance(direct, str):
        return direct
    normalized_name = name.lower()
    for key, value in headers.items():
        if key.lower() == normalized_name and isinstance(value, str):
            return value
    return default


def _replace_first_command(content: str, replacement_keystrokes: str) -> str:
    """Replace the first command's keystrokes in a Terminus JSON response."""
    payload = _decode_response(content)
    if not payload:
        return content
    commands = payload.get("commands")
    if not isinstance(commands, list) or not commands:
        return content
    first = commands[0]
    if not isinstance(first, dict):
        return content
    first["keystrokes"] = replacement_keystrokes
    return json.dumps(payload, ensure_ascii=False)


class TraceReplayLLMState:
    def __init__(self, *, llm_service_config: dict[str, object] | None = None) -> None:
        config = dict(llm_service_config or {})
        raw_trace_path = config.get("trace_path")
        if not isinstance(raw_trace_path, str) or not raw_trace_path.strip():
            raise ValueError("terminus_spec_trace_replay requires llm_service_config.trace_path")
        self._parsed = parse_replay_trace(Path(raw_trace_path).expanduser().resolve())

        # Phase B: optional sidecar for real draft-model decisions.
        # Must be explicitly enabled via use_spec_sidecar=true; defaults to simulated decisions.
        self._use_spec_sidecar = bool(config.get("use_spec_sidecar", False))
        self._sidecar_by_turn: dict[int, SpeculationTurn] | None = None
        raw_sidecar_path = config.get("sidecar_path")
        if self._use_spec_sidecar and isinstance(raw_sidecar_path, str) and raw_sidecar_path.strip():
            sidecar_path = Path(raw_sidecar_path).expanduser().resolve()
            try:
                sidecar = load_sidecar(sidecar_path)
                self._sidecar_by_turn = {t.turn_index: t for t in sidecar.turns if t.error is None}
                _logger.info(
                    "Loaded speculation sidecar %s (%d turns)", sidecar_path, len(self._sidecar_by_turn)
                )
            except Exception as exc:
                _logger.warning("Failed to load sidecar %s: %s — falling back to simulated decisions", sidecar_path, exc)

        raw_score_level = str(config.get("score_level", DEFAULT_REPORT_LEVEL)).strip().lower()
        if raw_score_level not in SCORE_LEVELS:
            raise ValueError(
                f"score_level must be one of {list(SCORE_LEVELS)}, got {raw_score_level!r}"
            )
        self._score_level = raw_score_level

        raw_policy = str(config.get("response_delay_policy", "fixed")).strip().lower()
        if raw_policy not in _RESPONSE_DELAY_POLICIES:
            raise ValueError(
                f"response_delay_policy must be one of {sorted(_RESPONSE_DELAY_POLICIES)}, got {raw_policy!r}"
            )
        raw_mismatch_policy = str(
            config.get("mismatch_policy", "preserve_command_class")
        ).strip().lower()
        if raw_mismatch_policy not in _SUPPORTED_MISMATCH_POLICIES:
            raise ValueError(
                f"mismatch_policy must be one of {sorted(_SUPPORTED_MISMATCH_POLICIES)}, got {raw_mismatch_policy!r}"
            )
        self._response_delay_policy = raw_policy
        self._mismatch_policy = raw_mismatch_policy
        try:
            self._response_delay_ms = max(
                0, int(config.get("response_delay_ms", _DEFAULT_RESPONSE_DELAY_MS))
            )
        except (TypeError, ValueError):
            self._response_delay_ms = _DEFAULT_RESPONSE_DELAY_MS
        try:
            self._response_delay_scaling_factor = max(
                0.0, float(config.get("response_delay_scaling_factor", 1.0))
            )
        except (TypeError, ValueError):
            self._response_delay_scaling_factor = 1.0
        try:
            self._draft_response_delay_scaling_factor = max(
                0.0,
                float(
                    _config_with_legacy_alias(
                        config,
                        "draft_response_delay_scaling_factor",
                        "speculative_delay_scaling_factor",
                        _DEFAULT_DRAFT_DELAY_SCALING_FACTOR,
                    )
                ),
            )
        except (TypeError, ValueError):
            self._draft_response_delay_scaling_factor = _DEFAULT_DRAFT_DELAY_SCALING_FACTOR
        try:
            self._acceptance_rate = max(
                0.0,
                min(
                    1.0,
                    float(
                        _config_with_legacy_alias(
                            config,
                            "acceptance_rate",
                            "accept_rate",
                            _DEFAULT_ACCEPTANCE_RATE,
                        )
                    ),
                ),
            )
        except (TypeError, ValueError):
            self._acceptance_rate = _DEFAULT_ACCEPTANCE_RATE
        try:
            self._minimal_delay_ms = max(
                0.0,
                float(
                    _config_with_legacy_alias(
                        config, "minimal_delay", "minimal-delay", _DEFAULT_MINIMAL_DELAY_MS
                    )
                ),
            )
        except (TypeError, ValueError):
            self._minimal_delay_ms = _DEFAULT_MINIMAL_DELAY_MS
        try:
            self._maximal_delay_ms = max(
                0.0,
                float(
                    _config_with_legacy_alias(
                        config, "maximal_delay", "maximal-delay", _DEFAULT_MAXIMAL_DELAY_MS
                    )
                ),
            )
        except (TypeError, ValueError):
            self._maximal_delay_ms = _DEFAULT_MAXIMAL_DELAY_MS
        if self._maximal_delay_ms < self._minimal_delay_ms:
            raise ValueError(
                "maximal_delay must be greater than or equal to minimal_delay, "
                f"got {self._maximal_delay_ms!r} < {self._minimal_delay_ms!r}"
            )
        # Pre-bucket the trace's commands so we can fabricate same-class mismatches.
        self._bucket_to_indices: dict[str, list[int]] = {}
        for index, message in enumerate(self._parsed.assistant_messages):
            command = _extract_first_command(str(message.get("content", "")))
            self._bucket_to_indices.setdefault(_command_bucket(command), []).append(index)
        self._lock = threading.Lock()
        self._oracle_trace_cursor = 0
        self._draft_trace_cursor = 0
        self._accept_count = 0
        self._reject_count = 0
        self._pair_cache: dict[tuple[str, str], tuple[dict[str, Any], float]] = {}

    @property
    def initial_user_prompt(self) -> str:
        return self._parsed.initial_user_prompt

    def reset(self) -> None:
        with self._lock:
            self._oracle_trace_cursor = 0
            self._draft_trace_cursor = 0
            self._accept_count = 0
            self._reject_count = 0
            self._pair_cache = {}

    def restore(self, *, consumed_response_count: int) -> None:
        # Like the non-spec terminus replay, the agent process and chat history
        # survive restore, so we don't rewind the cursor here.
        _ = consumed_response_count
        return

    def export_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "oracle_trace_cursor": self._oracle_trace_cursor,
                "draft_trace_cursor": self._draft_trace_cursor,
                "accept_count": self._accept_count,
                "reject_count": self._reject_count,
            }

    def import_state(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._oracle_trace_cursor = max(0, int(state.get("oracle_trace_cursor", 0)))
            self._draft_trace_cursor = max(
                0, int(state.get("draft_trace_cursor", self._oracle_trace_cursor))
            )
            self._accept_count = max(0, int(state.get("accept_count", 0)))
            self._reject_count = max(0, int(state.get("reject_count", 0)))
            self._pair_cache = {}

    def snapshot(self, *, include_events: bool = True) -> dict[str, Any]:
        _ = include_events
        with self._lock:
            oracle_trace_cursor = self._oracle_trace_cursor
            draft_trace_cursor = self._draft_trace_cursor
            accept_count = self._accept_count
            reject_count = self._reject_count
        total_pairs = accept_count + reject_count
        return {
            "trace_path": str(self._parsed.trace_path),
            "model_name": self._parsed.model_name,
            "response_delay_policy": self._response_delay_policy,
            "response_delay_ms": self._response_delay_ms,
            "response_delay_scaling_factor": self._response_delay_scaling_factor,
            "draft_response_delay_scaling_factor": self._draft_response_delay_scaling_factor,
            "minimal_delay": self._minimal_delay_ms,
            "maximal_delay": self._maximal_delay_ms,
            "acceptance_rate": self._acceptance_rate,
            "mismatch_policy": self._mismatch_policy,
            "use_spec_sidecar": self._use_spec_sidecar,
            "score_level": self._score_level,
            "sidecar_turns": len(self._sidecar_by_turn) if self._sidecar_by_turn is not None else None,
            "trace_cursor": oracle_trace_cursor,
            "oracle_trace_cursor": oracle_trace_cursor,
            "draft_trace_cursor": draft_trace_cursor,
            "total_responses": len(self._parsed.assistant_messages),
            "is_complete": oracle_trace_cursor >= len(self._parsed.assistant_messages),
            "accept_count": accept_count,
            "reject_count": reject_count,
            "accept_rate": 0.0 if total_pairs <= 0 else accept_count / total_pairs,
            "initial_user_prompt": self._parsed.initial_user_prompt,
        }

    def handle_request(
        self, *, path: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> dict[str, Any]:
        _ = (path, payload)
        role_value = _header_value(headers, _SPEC_ROLE_HEADER, _SPEC_ROLE_ORACLE)
        role = role_value.strip().lower() if isinstance(role_value, str) else _SPEC_ROLE_ORACLE
        if role not in {_SPEC_ROLE_ORACLE, _SPEC_ROLE_DRAFT}:
            raise ValueError(f"unsupported speculative role {role!r}")
        pair_id = str(_header_value(headers, _SPEC_PAIR_HEADER, "")).strip()
        cache_key = None if not pair_id else (role, pair_id)
        with self._lock:
            cached = None if cache_key is None else self._pair_cache.get(cache_key)
            if cached is not None:
                response, effective_delay_ms = cached
            else:
                response, effective_delay_ms = self._build_response_locked(
                    role=role, pair_id=pair_id
                )
                if cache_key is not None:
                    self._pair_cache[cache_key] = (copy.deepcopy(response), effective_delay_ms)
        if effective_delay_ms > 0:
            time.sleep(effective_delay_ms / 1000.0)
        return copy.deepcopy(response)

    def _build_response_locked(self, *, role: str, pair_id: str) -> tuple[dict[str, Any], float]:
        if role == _SPEC_ROLE_ORACLE:
            if self._oracle_trace_cursor >= len(self._parsed.assistant_messages):
                raise RuntimeError(
                    "terminus_spec_trace_replay exhausted all oracle assistant responses"
                )
            trace_index = self._oracle_trace_cursor
            message = copy.deepcopy(self._parsed.assistant_messages[trace_index])
            trace_delay_ms = self._parsed.assistant_trace_delays_ms[trace_index]
            self._oracle_trace_cursor += 1
            effective_delay_ms = self._effective_delay_ms(trace_delay_ms, role=role)
            return (
                _completion_response(
                    message, trace_index=trace_index, model_name=self._parsed.model_name
                ),
                effective_delay_ms,
            )
        if self._draft_trace_cursor >= len(self._parsed.assistant_messages):
            raise RuntimeError(
                "terminus_spec_trace_replay exhausted all draft assistant responses"
            )
        trace_index = self._draft_trace_cursor
        original_message = copy.deepcopy(self._parsed.assistant_messages[trace_index])
        message = copy.deepcopy(original_message)
        trace_delay_ms = self._parsed.assistant_trace_delays_ms[trace_index]
        self._draft_trace_cursor += 1
        original_content = str(original_message.get("content", ""))
        accepted = True
        if not _is_task_complete(original_content):
            accepted = self._draft_decision(trace_index=trace_index, pair_id=pair_id)
        if accepted:
            self._accept_count += 1
        else:
            self._reject_count += 1
            replacement = self._replacement_keystrokes(
                original_command=_extract_first_command(original_content),
                trace_index=trace_index,
            )
            message["content"] = _replace_first_command(original_content, replacement)
        sidecar_turn = self._sidecar_by_turn.get(trace_index) if self._sidecar_by_turn is not None else None
        sidecar_draft_delay_ms = sidecar_turn.draft_latency_ms if sidecar_turn is not None else None
        effective_delay_ms = self._effective_delay_ms(
            trace_delay_ms, role=role, sidecar_draft_delay_ms=sidecar_draft_delay_ms
        )
        return (
            _completion_response(
                message, trace_index=trace_index, model_name=self._parsed.model_name
            ),
            effective_delay_ms,
        )

    def _draft_decision(self, *, trace_index: int, pair_id: str) -> bool:
        _ = pair_id
        # Sidecar-based real decision (Phase B).
        if self._sidecar_by_turn is not None:
            turn = self._sidecar_by_turn.get(trace_index)
            if turn is not None:
                return turn.accepted.get(self._score_level, False)
        # Fallback: simulated SHA256 decision.
        key = f"turn-{trace_index}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        value = int(digest[:8], 16) / float(0xFFFFFFFF)
        return value < self._acceptance_rate

    def _replacement_keystrokes(self, *, original_command: str, trace_index: int) -> str:
        bucket = _command_bucket(original_command)
        normalized_original = _normalize_command(original_command)
        for candidate_index in self._bucket_to_indices.get(bucket, []):
            if candidate_index == trace_index:
                continue
            candidate_command = _extract_first_command(
                str(self._parsed.assistant_messages[candidate_index].get("content", ""))
            )
            if candidate_command and _normalize_command(candidate_command) != normalized_original:
                return candidate_command
        fallback = _fallback_command_for_bucket(bucket)
        if _normalize_command(fallback) != normalized_original:
            return fallback
        return "pwd\n"

    def _effective_delay_ms(
        self,
        trace_delay_ms: float | None,
        *,
        role: str,
        sidecar_draft_delay_ms: float | None = None,
    ) -> float:
        if role == _SPEC_ROLE_DRAFT and sidecar_draft_delay_ms is not None:
            # Use real draft-model latency from the sidecar, scaled by
            # draft_response_delay_scaling_factor (oracle response_delay_scaling_factor
            # is not applied — it's oracle-specific).
            delay_ms = sidecar_draft_delay_ms * self._draft_response_delay_scaling_factor
        else:
            if self._response_delay_policy == "trace_replay":
                if trace_delay_ms is not None:
                    delay_ms = trace_delay_ms * self._response_delay_scaling_factor
                else:
                    delay_ms = float(self._response_delay_ms)
            else:
                delay_ms = float(self._response_delay_ms)
            if role == _SPEC_ROLE_DRAFT:
                delay_ms *= self._draft_response_delay_scaling_factor
        return min(self._maximal_delay_ms, max(self._minimal_delay_ms, delay_ms))
