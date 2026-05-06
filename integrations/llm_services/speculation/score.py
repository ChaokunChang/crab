"""Score draft model responses against the recorded oracle response.

Multiple acceptance *levels* are computed per turn so a single collection run
yields a spectrum of accept-rate numbers from strict to lenient. This lets the
benchmark distinguish between rejections that are real model disagreements vs.
ones that are just whitespace or syntactic-equivalence noise.

Levels (each strictly looser than the previous):

* ``literal`` — the extracted first command must match byte-for-byte.
* ``normalized`` — first commands match after whitespace collapse
  (``" ".join(s.split())``).
* ``semantic`` — ``normalized`` plus heredoc-operator normalization (``<<
  'EOF'`` ≡ ``<<'EOF'``) and a task-complete short-circuit (when *both*
  responses signal task completion, count as accept regardless of keystrokes).
"""

from __future__ import annotations

import json
import re

from integrations.llm_services.speculation.claude_code_tools import primary_arg
from integrations.llm_services.speculation.schema import (
    AGENT_CLAUDE_CODE,
    AGENT_MINI_SWE,
    AGENT_TERMINUS,
    LEVEL_LITERAL,
    LEVEL_NORMALIZED,
    LEVEL_SEMANTIC,
    SCORE_LEVELS,
)

_MSWEA_COMMAND_PATTERN = re.compile(
    r"<mswea_bash_command>(.*?)</mswea_bash_command>", re.DOTALL
)
_HEREDOC_OP_RE = re.compile(r"(<<-?)\s+")
_INLINE_WS_RE = re.compile(r"[ \t]+")
_MSWEA_SUBMIT_TOKEN = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def extract_first_command(*, agent_kind: str, content: str) -> str:
    """Pull the first executable command out of an assistant response."""
    if not isinstance(content, str) or not content.strip():
        return ""
    if agent_kind == AGENT_TERMINUS:
        return _extract_terminus_command(content)
    if agent_kind == AGENT_MINI_SWE:
        return _extract_mini_swe_command(content)
    if agent_kind == AGENT_CLAUDE_CODE:
        return _extract_claude_code_command(content)
    raise ValueError(f"unsupported agent_kind {agent_kind!r}")


def extract_response_text(*, agent_kind: str, content: str) -> str:
    """Extract the free-form text portion of a response.

    Used to compare two final-response turns when neither emits a tool
    call. For each agent the text is the prose-y part of the response:
    terminus's analysis+plan, mini_swe's THOUGHT, claude_code's thinking.
    """
    if not isinstance(content, str) or not content.strip():
        return ""
    if agent_kind == AGENT_TERMINUS:
        parsed = _safe_json_loads(content)
        if not isinstance(parsed, dict):
            recovered = _recover_json_block(content)
            if not isinstance(recovered, dict):
                return content
            parsed = recovered
        analysis = parsed.get("analysis")
        plan = parsed.get("plan")
        parts = [
            p.strip()
            for p in (analysis if isinstance(analysis, str) else "", plan if isinstance(plan, str) else "")
            if isinstance(p, str) and p.strip()
        ]
        return "\n".join(parts)
    if agent_kind == AGENT_MINI_SWE:
        # The THOUGHT lives outside the <mswea_bash_command>...</mswea_bash_command>
        # block. Strip the command block and return what's left.
        without_cmd = _MSWEA_COMMAND_PATTERN.sub("", content)
        return without_cmd.strip()
    if agent_kind == AGENT_CLAUDE_CODE:
        parsed = _safe_json_loads(content)
        if not isinstance(parsed, dict):
            return content
        thinking = parsed.get("thinking")
        return thinking.strip() if isinstance(thinking, str) else ""
    raise ValueError(f"unsupported agent_kind {agent_kind!r}")


def is_task_complete(*, agent_kind: str, content: str) -> bool:
    """Return True iff the response signals task completion in the agent's protocol."""
    if not isinstance(content, str) or not content.strip():
        return False
    if agent_kind == AGENT_TERMINUS:
        parsed = _safe_json_loads(content)
        if not isinstance(parsed, dict):
            parsed = _recover_json_block(content)
        if not isinstance(parsed, dict):
            return False
        raw = parsed.get("task_complete", False)
        if isinstance(raw, str):
            return raw.strip().lower() in {"true", "1", "yes"}
        return bool(raw)
    if agent_kind == AGENT_MINI_SWE:
        return _MSWEA_SUBMIT_TOKEN in content
    if agent_kind == AGENT_CLAUDE_CODE:
        return _is_claude_code_final(content)
    return False


def score_levels(
    *,
    agent_kind: str,
    oracle_content: str,
    draft_content: str,
    is_final_turn: bool = False,
) -> dict[str, bool]:
    """Compute the full per-level acceptance verdict for one turn.

    Returns a dict keyed by every name in ``SCORE_LEVELS``. Inclusion is
    monotonic by construction: ``literal ⇒ normalized ⇒ semantic``.

    Three cases:

    1. **Final-response turn** (``is_final_turn=True``) where the oracle
       emitted no tool call: compare the prose-y *response text* directly.
       Two different free-form summaries from two different models almost
       never match, so this path usually rejects at all three levels —
       which is the correct behaviour, since rejecting a final turn carries
       essentially no spec-replay penalty (the task is over).

    2. **Mid-trajectory wait turn** (not final) where *both* responses have
       no tool call / no command: this typically signals "I'm waiting for
       something to finish, no action this turn". When both sides agree
       on this, we count it as accept at all three levels.

    3. **Standard command turn**: extract first commands and compare them
       under the literal / normalized / semantic transforms.
    """
    oracle_cmd = extract_first_command(agent_kind=agent_kind, content=oracle_content)
    draft_cmd = extract_first_command(agent_kind=agent_kind, content=draft_content)

    if is_final_turn and not oracle_cmd:
        oracle_text = extract_response_text(
            agent_kind=agent_kind, content=oracle_content
        )
        draft_text = extract_response_text(
            agent_kind=agent_kind, content=draft_content
        )
        has_text = bool(oracle_text or draft_text)
        literal = has_text and oracle_text == draft_text
        normalized = has_text and _normalize_text(oracle_text) == _normalize_text(
            draft_text
        )
        # No additional loosening for free-form text yet; semantic mirrors
        # normalized so the level columns stay informative even though they
        # agree here.
        semantic = normalized
    elif (not is_final_turn) and not oracle_cmd and not draft_cmd:
        # Mid-trajectory wait turn: both sides predict "no action".
        literal = True
        normalized = True
        semantic = True
    else:
        has_command = bool(oracle_cmd or draft_cmd)
        literal = has_command and oracle_cmd == draft_cmd
        normalized = has_command and _normalize_command(oracle_cmd) == _normalize_command(
            draft_cmd
        )
        semantic = (
            has_command
            and _semantic_normalize(oracle_cmd) == _semantic_normalize(draft_cmd)
        )

    levels = {
        LEVEL_LITERAL: literal,
        LEVEL_NORMALIZED: normalized,
        LEVEL_SEMANTIC: semantic,
    }
    # Monotonicity guard.
    if levels[LEVEL_LITERAL]:
        levels[LEVEL_NORMALIZED] = True
    if levels[LEVEL_NORMALIZED]:
        levels[LEVEL_SEMANTIC] = True
    return {name: levels[name] for name in SCORE_LEVELS}


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _extract_terminus_command(content: str) -> str:
    parsed = _safe_json_loads(content)
    if not isinstance(parsed, dict):
        recovered = _recover_json_block(content)
        if recovered is None:
            return ""
        parsed = recovered
    commands = parsed.get("commands")
    if not isinstance(commands, list) or not commands:
        return ""
    first = commands[0]
    if not isinstance(first, dict):
        return ""
    keystrokes = first.get("keystrokes")
    return keystrokes if isinstance(keystrokes, str) else ""


def _extract_mini_swe_command(content: str) -> str:
    matches = _MSWEA_COMMAND_PATTERN.findall(content)
    if len(matches) != 1:
        return ""
    return matches[0].strip()


def _extract_claude_code_command(content: str) -> str:
    """Render a claude_code response as a comparable ``"Tool: primary_arg"`` string.

    Both oracle and draft responses are stored as JSON envelopes:
    ``{"thinking": ..., "tool_calls": [{"name": ..., "arguments": {...}}]}``.
    Returns ``""`` for an empty (final-text) response so the semantic-level
    short-circuit can match two final responses without false-positives at
    literal/normalized.
    """
    parsed = _safe_json_loads(content)
    if not isinstance(parsed, dict):
        return ""
    tool_calls = parsed.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return ""
    first = tool_calls[0]
    if not isinstance(first, dict):
        return ""
    name = first.get("name")
    if not isinstance(name, str) or not name:
        return ""
    arguments = first.get("arguments") if isinstance(first.get("arguments"), dict) else {}
    arg_value = primary_arg(name, arguments)
    if not arg_value:
        return name
    return f"{name}: {arg_value}"


def _is_claude_code_final(content: str) -> bool:
    parsed = _safe_json_loads(content)
    if not isinstance(parsed, dict):
        return False
    tool_calls = parsed.get("tool_calls")
    return isinstance(tool_calls, list) and len(tool_calls) == 0


def _normalize_command(command: str) -> str:
    return " ".join(command.split())


def _semantic_normalize(command: str) -> str:
    """Looser-than-normalized comparison key.

    Currently:
    * Removes whitespace immediately after ``<<`` / ``<<-`` so heredoc operator
      spacing doesn't differentiate otherwise-identical commands.
    * Collapses runs of horizontal whitespace to a single space.
    * Strips leading/trailing whitespace (including newlines).

    Newlines inside the command are preserved so heredoc bodies stay
    distinguishable.
    """
    out = _HEREDOC_OP_RE.sub(r"\1", command)
    out = _INLINE_WS_RE.sub(" ", out)
    return out.strip()


def _safe_json_loads(text: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _recover_json_block(text: str) -> dict | None:
    """Best-effort: find the first balanced ``{...}`` substring and decode it.

    Real models occasionally wrap JSON in markdown fences or trailing prose.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                parsed = _safe_json_loads(candidate)
                return parsed if isinstance(parsed, dict) else None
    return None
