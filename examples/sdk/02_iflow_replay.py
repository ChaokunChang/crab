"""Run iFlow against a recorded trace without a model API key.

The example accepts Terminal-Bench task assets and a proxy trajectory as
explicit inputs; it does not rely on a developer-specific dataset path.

Start the Crab daemon and replay router first, then run:

    python3 examples/sdk/02_iflow_replay.py \
      --task-root /path/to/original-tasks/crack-7z-hash \
      --trace /path/to/crack-7z-hash/agent-logs/proxy_server_trajectory.log
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from crab import Engine, Sandbox
from crab.agents_builtin.iflow import IFlowAgent
from crab.templates import DockerComposeTemplate
from integrations.llm_services.router import BenchmarkLLMRouterClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-root",
        type=Path,
        required=True,
        help="Terminal-Bench task directory containing task.yaml and docker-compose.yaml.",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="Recorded proxy_server_trajectory.log.",
    )
    parser.add_argument("--replay-url", default="http://127.0.0.1:18080")
    parser.add_argument("--name", default=None, help="Sandbox id (derived from task id by default).")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Do not run the Terminal-Bench run-tests.sh after the agent exits.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the sandbox and checkpoints for manual inspection.",
    )
    return parser.parse_args()


def _load_task(task_root: Path) -> tuple[str, str, Path, str]:
    task_file = task_root / "task.yaml"
    compose_file = task_root / "docker-compose.yaml"
    for required in (task_file, compose_file, task_root / "run-tests.sh"):
        if not required.is_file():
            raise FileNotFoundError(f"missing Terminal-Bench task file: {required}")

    payload = yaml.safe_load(task_file.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"task YAML must contain an object: {task_file}")
    prompt = payload.get("instruction")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"task instruction is missing: {task_file}")

    compose = yaml.safe_load(compose_file.read_text(encoding="utf-8")) or {}
    services = compose.get("services") if isinstance(compose, dict) else None
    if not isinstance(services, dict) or not services:
        raise ValueError(f"compose file has no services: {compose_file}")
    if "client" in services:
        service_name = "client"
    elif len(services) == 1:
        service_name = str(next(iter(services)))
    else:
        raise ValueError(f"compose file has multiple services and no 'client': {compose_file}")

    return task_root.name, prompt, compose_file, service_name


def main() -> None:
    args = _parse_args()
    task_root = args.task_root.expanduser().resolve()
    trace_path = args.trace.expanduser().resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(f"trace not found: {trace_path}")

    task_id, prompt, compose_file, service_name = _load_task(task_root)
    name = args.name or f"sdk-iflow-replay-{task_id}"

    replay = BenchmarkLLMRouterClient(args.replay_url, timeout_seconds=30.0)
    replay.register_sandbox(
        sandbox_id=name,
        llm_service_type="iflow_trace_replay",
        llm_service_config={
            "trace_path": str(trace_path),
            "response_delay_policy": "trace_replay",
            "response_delay_scaling_factor": 1.0,
        },
    )

    try:
        with Engine.connect() as engine:
            sandbox = Sandbox(
                template=DockerComposeTemplate(
                    compose_file=compose_file,
                    service_name=service_name,
                    task_root=task_root,
                ),
                engine=engine,
                name=name,
            )
            try:
                agent = IFlowAgent(timeout=args.timeout).bind(sandbox, llm_url=args.replay_url)
                result = agent.run(prompt)
                print("agent exit:", result.exit_code)
                print(result.output.rstrip())
                if result.exit_code != 0:
                    raise RuntimeError(
                        f"iFlow exited with {result.exit_code}: {result.extra.get('stderr', '')}"
                    )
                checkpoints = sandbox.checkpoints.list()
                print("checkpoints:")
                print(json.dumps(checkpoints, indent=2))
                if not checkpoints:
                    raise RuntimeError("trace completed without recording any checkpoints")
                if not any(item.get("has_filesystem") for item in checkpoints):
                    raise RuntimeError("trace checkpoints do not contain filesystem state")
                if not args.skip_verification:
                    verification = sandbox.commands.run(
                        "bash /tests/run-tests.sh",
                        cwd=sandbox.process_cwd,
                        timeout=600.0,
                        check=False,
                    )
                    print(verification.stdout.rstrip())
                    if verification.returncode != 0:
                        raise RuntimeError(
                            f"task verification failed (rc={verification.returncode}): "
                            f"{verification.stderr.rstrip()}"
                        )
            finally:
                if args.keep:
                    print(f"kept sandbox: {sandbox.sandbox_id}")
                else:
                    sandbox.kill()
    finally:
        replay.unregister_sandbox(name)
        replay.close()


if __name__ == "__main__":
    main()
