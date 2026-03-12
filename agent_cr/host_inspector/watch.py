from __future__ import annotations

import argparse
import time

from ..ids import SandboxId
from ..remote_inspector import HostInspectorServiceClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch registered sandbox status via host inspector")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=0, help="0 means forever")
    parser.add_argument("sandbox_ids", nargs="+")
    args = parser.parse_args(argv)

    client = HostInspectorServiceClient(args.base_url)
    iteration = 0
    while args.iterations == 0 or iteration < args.iterations:
        for sandbox_id_raw in args.sandbox_ids:
            payload = client.get_proc_and_fs_status(SandboxId(sandbox_id_raw))
            status = payload["status"]
            print(
                "sandbox={sandbox} running={running} process_changed={process} filesystem_changed={filesystem}".format(
                    sandbox=sandbox_id_raw,
                    running=status["is_running"],
                    process=status["process_changed"],
                    filesystem=status["filesystem_changed"],
                ),
                flush=True,
            )
        iteration += 1
        if args.iterations == 0 or iteration < args.iterations:
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
