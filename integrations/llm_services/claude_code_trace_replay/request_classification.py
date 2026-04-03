from __future__ import annotations

import re

DEFAULT_REPLAY_MODEL_NAME = "claude-opus-4-6"

REQUEST_KIND_MAIN_LOOP = "main_loop"
REQUEST_KIND_HELPER = "helper"
REQUEST_KIND_COUNT_TOKENS = "count_tokens"
REQUEST_KIND_OTHER = "other"


def normalize_model_family(model_name: str | None) -> str | None:
    if not isinstance(model_name, str):
        return None
    stripped = model_name.strip()
    if not stripped:
        return None
    if re.search(r"-\d{8}$", stripped):
        return stripped.rsplit("-", 1)[0]
    return stripped


def is_helper_model_request(requested_model: str | None, *, replay_model: str) -> bool:
    requested_family = normalize_model_family(requested_model)
    replay_family = normalize_model_family(replay_model)
    return requested_family is not None and replay_family is not None and requested_family != replay_family


def classify_replay_request(*, path: str, requested_model: str | None, replay_model: str) -> str:
    """Classify Claude Code replay requests for scheduler-safe progress tracking."""
    if path == "/v1/messages/count_tokens":
        return REQUEST_KIND_COUNT_TOKENS
    if path != "/v1/messages":
        return REQUEST_KIND_OTHER
    if is_helper_model_request(requested_model, replay_model=replay_model):
        return REQUEST_KIND_HELPER
    return REQUEST_KIND_MAIN_LOOP


def infer_live_request_kind(
    *,
    path: str,
    requested_model: str | None,
    default_main_model: str = DEFAULT_REPLAY_MODEL_NAME,
) -> str:
    """Infer Claude request kind when only the live payload is available.

    For the replayable Claude Code benchmark corpus, auxiliary `/v1/messages`
    traffic is emitted through Haiku-family helper models while user-visible
    replay turns use the main Opus model family. We mirror that distinction here
    so helper/count-token requests do not participate in response gating.
    """
    return classify_replay_request(
        path=path,
        requested_model=requested_model,
        replay_model=default_main_model,
    )


def should_gate_live_request(
    *,
    path: str,
    requested_model: str | None,
    default_main_model: str = DEFAULT_REPLAY_MODEL_NAME,
) -> bool:
    return (
        infer_live_request_kind(
            path=path,
            requested_model=requested_model,
            default_main_model=default_main_model,
        )
        == REQUEST_KIND_MAIN_LOOP
    )
