"""Minimal Crab SDK demo: launch a sandbox, run a command, clean up.

Before running, start the daemon in another terminal:

    sudo crab daemon start --foreground --config /etc/crab/config.yaml

Then run this script:

    sudo --preserve-env=PYTHONPATH PYTHONPATH=. \
      python3 examples/sdk/01_basic_sandbox.py
"""
from __future__ import annotations

from crab import Engine, Sandbox


def main() -> None:
    with Engine.connect() as engine:
        sbx = Sandbox(image="ubuntu:22.04", engine=engine)
        try:
            print("sandbox:", sbx.sandbox_id)
            result = sbx.commands.run("uname -a && whoami")
            print(result.stdout.rstrip())
        finally:
            sbx.kill()


if __name__ == "__main__":
    main()
