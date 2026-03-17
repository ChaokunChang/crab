from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    stateless: bool
    changes_process_state: bool
    changes_filesystem: bool
    idempotent: bool
    uses_network: bool

    def openai_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def anthropic_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def effect_metadata(self) -> dict[str, Any]:
        return {
            "stateless": self.stateless,
            "changes_process_state": self.changes_process_state,
            "changes_filesystem": self.changes_filesystem,
            "idempotent": self.idempotent,
            "uses_network": self.uses_network,
        }


TOOL_DEFINITIONS: list[ToolDefinition] = [
    ToolDefinition(
        name="read_workdir",
        description="Read the current workdir listing without mutating state.",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        stateless=True,
        changes_process_state=False,
        changes_filesystem=False,
        idempotent=True,
        uses_network=False,
    ),
    ToolDefinition(
        name="show_pwd",
        description="Return the current working directory.",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        stateless=True,
        changes_process_state=False,
        changes_filesystem=False,
        idempotent=True,
        uses_network=False,
    ),
    ToolDefinition(
        name="remember_note",
        description="Record a note in the agent's in-memory state.",
        input_schema={
            "type": "object",
            "properties": {
                "note": {"type": "string"},
            },
            "required": ["note"],
            "additionalProperties": False,
        },
        stateless=False,
        changes_process_state=False,
        changes_filesystem=False,
        idempotent=False,
        uses_network=False,
    ),
    ToolDefinition(
        name="overwrite_artifact",
        description="Overwrite a deterministic artifact in the work directory.",
        input_schema={
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["filename", "content"],
            "additionalProperties": False,
        },
        stateless=False,
        changes_process_state=False,
        changes_filesystem=True,
        idempotent=True,
        uses_network=False,
    ),
    ToolDefinition(
        name="append_journal",
        description="Append a line to a workdir journal file.",
        input_schema={
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "line": {"type": "string"},
            },
            "required": ["filename", "line"],
            "additionalProperties": False,
        },
        stateless=False,
        changes_process_state=False,
        changes_filesystem=True,
        idempotent=False,
        uses_network=False,
    ),
    ToolDefinition(
        name="mkdir_cache",
        description="Ensure a cache directory exists.",
        input_schema={
            "type": "object",
            "properties": {
                "dirname": {"type": "string"},
            },
            "required": ["dirname"],
            "additionalProperties": False,
        },
        stateless=False,
        changes_process_state=False,
        changes_filesystem=True,
        idempotent=True,
        uses_network=False,
    ),
    ToolDefinition(
        name="spawn_probe",
        description="Spawn a short-lived process probe.",
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
            "additionalProperties": False,
        },
        stateless=False,
        changes_process_state=True,
        changes_filesystem=False,
        idempotent=False,
        uses_network=False,
    ),
    ToolDefinition(
        name="fetch_proxy_health",
        description="Fetch the interceptor health endpoint.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        stateless=True,
        changes_process_state=False,
        changes_filesystem=False,
        idempotent=True,
        uses_network=True,
    ),
]

_TOOL_MAP = {tool.name: tool for tool in TOOL_DEFINITIONS}


def provider_tools(provider: str) -> list[dict[str, Any]]:
    if provider == "openai":
        return [tool.openai_spec() for tool in TOOL_DEFINITIONS]
    if provider == "anthropic":
        return [tool.anthropic_spec() for tool in TOOL_DEFINITIONS]
    raise ValueError(f"unsupported provider: {provider}")


def get_tool(name: str) -> ToolDefinition:
    return _TOOL_MAP[name]


def allowed_tool_names(provider: str, request_payload: dict[str, Any]) -> list[str]:
    raw_tools = request_payload.get("tools", [])
    names: list[str] = []
    for item in raw_tools:
        if provider == "openai":
            function = dict(item.get("function", {}))
            name = function.get("name")
        else:
            name = item.get("name")
        if isinstance(name, str) and name in _TOOL_MAP:
            names.append(name)
    return names


def default_input_for_tool(name: str, turn_index: int) -> dict[str, Any]:
    if name == "remember_note":
        return {"note": f"remembered note {turn_index}"}
    if name == "overwrite_artifact":
        return {"filename": "tool_artifact.txt", "content": f"artifact updated {turn_index}"}
    if name == "append_journal":
        return {"filename": "journal.log", "line": f"journal line {turn_index}"}
    if name == "mkdir_cache":
        return {"dirname": "cache/runtime"}
    if name == "spawn_probe":
        return {"message": f"probe {turn_index}"}
    if name == "fetch_proxy_health":
        return {"path": "/healthz"}
    return {}
