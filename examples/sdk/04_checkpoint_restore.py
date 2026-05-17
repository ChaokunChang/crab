"""Checkpoint and restore — the Agent-CR value-prop.

Shows the C/R API on a bare sandbox. Real C/R requires runc + CRIU + ZFS;
this example uses the in-memory runtime so it runs anywhere, demonstrating
the shape and the no-op happy path.

    python3 examples/sdk/04_checkpoint_restore.py
"""
from __future__ import annotations

from agent_cr import Engine, EngineConfig, Sandbox


def main() -> None:
    with Engine.start(EngineConfig(runtime="docker")) as engine:
        sbx = Sandbox(image="ubuntu:22.04", engine=engine)
        try:
            ckpt_id = sbx.checkpoint(label="initial-state")
            print("checkpoint:", ckpt_id)
            print("known checkpoints:", sbx.checkpoints.list())
            sbx.restore(ckpt_id)
            print("restored to:", ckpt_id)
        finally:
            sbx.kill()


if __name__ == "__main__":
    main()
