from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

try:
    import orjson as _orjson
except ImportError:  # pragma: no cover - optional dependency
    _orjson = None


_SUPPORTED_JSON_SERIALIZERS = {"auto", "stdlib", "orjson"}


class JsonCodec:
    def __init__(self, serializer: str = "auto") -> None:
        normalized = str(serializer).strip().lower()
        if normalized not in _SUPPORTED_JSON_SERIALIZERS:
            raise ValueError(
                f"serializer must be one of {sorted(_SUPPORTED_JSON_SERIALIZERS)}, got {serializer!r}"
            )
        if normalized == "orjson" and _orjson is None:
            raise ValueError("serializer='orjson' requires the optional orjson dependency")
        self._serializer = normalized

    @property
    def serializer(self) -> str:
        if self._serializer == "auto":
            return "orjson" if _orjson is not None else "stdlib"
        return self._serializer

    def dumps(self, payload: object, *, sort_keys: bool = False) -> str:
        return self.dumps_bytes(payload, sort_keys=sort_keys).decode("utf-8")

    def dumps_bytes(self, payload: object, *, sort_keys: bool = False) -> bytes:
        if self.serializer == "orjson":
            option = _orjson.OPT_SORT_KEYS if sort_keys else 0
            return _orjson.dumps(payload, option=option)
        return json.dumps(payload, sort_keys=sort_keys, separators=(",", ":")).encode("utf-8")

    def loads(self, payload: str | bytes | bytearray | memoryview) -> Any:
        raw: str | bytes
        if isinstance(payload, memoryview):
            raw = payload.tobytes()
        else:
            raw = payload
        if self.serializer == "orjson":
            return _orjson.loads(raw)
        return json.loads(raw)


@lru_cache(maxsize=4)
def get_json_codec(serializer: str = "auto") -> JsonCodec:
    return JsonCodec(serializer=serializer)
