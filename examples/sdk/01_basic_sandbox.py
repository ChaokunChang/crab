"""Minimal Agent-CR SDK demo — bare sandbox, no agent.

Runs against the in-memory runtime by default so it executes anywhere without
runc / ZFS / CRIU installed. Real sandbox isolation requires
EngineConfig(runtime="runc") and operator-provided host setup.

    python3 examples/sdk/01_basic_sandbox.py
"""
from __future__ import annotations

from agent_cr import Engine, EngineConfig, Sandbox


def main() -> None:
    with Engine.start(EngineConfig(runtime="docker")) as engine:
        sbx = Sandbox(image="ubuntu:22.04", engine=engine)
        try:
            print("sandbox launched:", sbx.sandbox_id)
            print("interceptor base URL:", engine.interceptor_base_url)
            # With the in-memory runtime, commands.run will raise
            # NotImplementedError. Swap to runtime="runc" for real exec.
        finally:
            sbx.kill()


if __name__ == "__main__":
    main()
