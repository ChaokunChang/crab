"""OpenAI-compatible tool schemas for the Claude Code agent.

These are *minimal, comparison-oriented* schemas — they exist so the draft
model (DeepSeek-Chat / any OpenAI-compatible chat-completions endpoint) can
emit a structured tool call that we can extract and compare against the
oracle's recorded tool call. They are not faithful 1:1 replicas of the real
Claude Code tool surface; we only describe the fields we care about for
acceptance scoring.

Tool inventory was derived empirically from the
``tbench-claude-code-claude-opus4.6-trajectories`` traces (Bash, TodoWrite,
Read, Write, Edit, Task, Grep, Glob in descending frequency).
"""

from __future__ import annotations

CLAUDE_CODE_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a shell command in the persistent terminal session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to run.",
                    },
                    "description": {
                        "type": "string",
                        "description": "A short human-readable description of what the command does.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in milliseconds (optional).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read the contents of a file from disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to read.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start from (1-indexed, optional).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of lines to read (optional).",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file, overwriting any existing content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Edit",
            "description": "Replace exactly one occurrence of old_string with new_string in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Grep",
            "description": "Search file contents using a regex pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string"},
                    "output_mode": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Glob",
            "description": "Find files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "TodoWrite",
            "description": "Replace the current todo list. Use to plan or track multi-step work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {"type": "string"},
                                "activeForm": {"type": "string"},
                            },
                        },
                    },
                },
                "required": ["todos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Task",
            "description": "Delegate a sub-task to a specialized sub-agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "prompt": {"type": "string"},
                    "subagent_type": {"type": "string"},
                },
                "required": ["description", "prompt"],
            },
        },
    },
]


CLAUDE_CODE_SYSTEM_PROMPT = (
    "You are an autonomous coding agent solving a task in a Linux sandbox. "
    "Read files, run shell commands, and edit files using the available tools. "
    "Each turn you may emit at most one tool call (or, when the task is fully "
    "complete, a final text response with no tool call). Match the working "
    "style of the prior assistant turns in this conversation."
)


_PRIMARY_ARG_BY_TOOL: dict[str, tuple[str, ...]] = {
    "Bash": ("command",),
    "Read": ("file_path",),
    "Write": ("file_path",),
    "Edit": ("file_path",),
    "Grep": ("pattern",),
    "Glob": ("pattern",),
    "Task": ("description",),
    # TodoWrite intentionally has no primary arg — todo lists are too noisy
    # to compare verbatim, so we let function-name match alone determine
    # acceptance for it.
    "TodoWrite": (),
}


def primary_arg(tool_name: str, arguments: dict) -> str:
    """Return a canonical "primary argument" string for a tool call.

    For Bash this is the ``command``; for Read/Write/Edit it's ``file_path``;
    etc. Used for human-readable side-by-side display and for the per-level
    acceptance comparison. ``""`` means *no primary arg* (e.g. TodoWrite),
    so two tool calls with the same function name but different bodies will
    still match — that's intentional for housekeeping tools.
    """
    keys = _PRIMARY_ARG_BY_TOOL.get(tool_name)
    if not keys:
        # Unknown tool: fall back to the first present argument value.
        if not isinstance(arguments, dict):
            return ""
        for key in sorted(arguments.keys()):
            value = arguments[key]
            if isinstance(value, (str, int, float, bool)):
                return str(value)
        return ""
    if not isinstance(arguments, dict):
        return ""
    for key in keys:
        value = arguments.get(key)
        if isinstance(value, str):
            return value
        if value is not None:
            return str(value)
    return ""
