from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from agent_cr import AgentCRRequestInterceptorServer, InMemoryRequestStateStore

from .harness import prepare_iflow_runtime, prepare_iflow_state
from .image import build_image, export_image_rootfs
from .manual import (
    checkpoint_manual_iflow,
    launch_manual_iflow,
    list_manual_checkpoints,
    load_session,
    manual_shell,
    restore_manual_iflow,
    session_summary,
    stop_manual_iflow,
)
from .service import default_script_steps, serve, serve_manual


def _post_json(base_url: str, path: str, payload: dict[str, object]) -> dict[str, object]:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(base_url: str, path: str) -> dict[str, object]:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}{path}", timeout=10.0) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve_parser = subparsers.add_parser("serve-scripted-llm")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, required=True)
    serve_parser.add_argument("--idle-delay-ms", type=int, default=2000)

    manual_serve_parser = subparsers.add_parser("serve-manual-llm")
    manual_serve_parser.add_argument("--host", default="127.0.0.1")
    manual_serve_parser.add_argument("--port", type=int, required=True)

    manual_interceptor_parser = subparsers.add_parser("serve-manual-interceptor")
    manual_interceptor_parser.add_argument("--host", default="0.0.0.0")
    manual_interceptor_parser.add_argument("--port", type=int, required=True)
    manual_interceptor_parser.add_argument("--upstream-url", required=True)
    manual_interceptor_parser.add_argument("--sandbox-id", required=True)
    manual_interceptor_parser.add_argument("--sandbox-ip", required=True)

    build_parser = subparsers.add_parser("build-rootfs")
    build_parser.add_argument("--tag", required=True)
    build_parser.add_argument("--output-dir", required=True)

    runtime_parser = subparsers.add_parser("prepare-runtime")
    runtime_parser.add_argument("--work-root", required=True)
    runtime_parser.add_argument("--base-url", required=True)
    runtime_parser.add_argument("--model-name", default=os.environ.get("AGENT_CR_IFLOW_MODEL_NAME", "agent-cr-iflow-scripted"))
    runtime_parser.add_argument("--alternate-node-runtime-dir", default=os.environ.get("AGENT_CR_IFLOW_NODE_RUNTIME_DIR"))

    launch_parser = subparsers.add_parser("launch-manual-sandbox")
    launch_parser.add_argument("--work-root", required=True)
    launch_parser.add_argument("--llm-base-url", required=True)
    launch_parser.add_argument("--sandbox-id", default="sbx-iflow-manual")
    launch_parser.add_argument(
        "--task",
        default="Wait for the next tool instruction, execute it exactly once, then ask for the next instruction.",
    )
    launch_parser.add_argument("--host-inspector-url", default=None)
    launch_parser.add_argument("--image-tag", default=None)
    launch_parser.add_argument("--model-name", default=os.environ.get("AGENT_CR_IFLOW_MODEL_NAME", "agent-cr-iflow-manual"))
    launch_parser.add_argument("--sandbox-ip", default=os.environ.get("AGENT_CR_IFLOW_SANDBOX_IP", "172.17.0.240"))
    launch_parser.add_argument("--alternate-node-runtime-dir", default=os.environ.get("AGENT_CR_IFLOW_NODE_RUNTIME_DIR"))

    stop_parser = subparsers.add_parser("stop-manual-sandbox")
    stop_parser.add_argument("--work-root", required=True)
    stop_parser.add_argument("--remove-image", action="store_true")

    checkpoint_parser = subparsers.add_parser("checkpoint-manual-sandbox")
    checkpoint_parser.add_argument("--work-root", required=True)

    restore_parser = subparsers.add_parser("restore-manual-sandbox")
    restore_parser.add_argument("--work-root", required=True)
    restore_parser.add_argument("--checkpoint-id", default=None)

    checkpoints_parser = subparsers.add_parser("list-manual-checkpoints")
    checkpoints_parser.add_argument("--work-root", required=True)

    shell_parser = subparsers.add_parser("manual-shell")
    shell_parser.add_argument("--work-root", required=True)
    shell_parser.add_argument("exec_argv", nargs=argparse.REMAINDER)

    session_parser = subparsers.add_parser("show-manual-session")
    session_parser.add_argument("--work-root", required=True)

    enqueue_parser = subparsers.add_parser("enqueue-run-shell-command")
    enqueue_parser.add_argument("--base-url", required=True)
    enqueue_parser.add_argument("--command", dest="shell_command", required=True)
    enqueue_parser.add_argument("--sandbox-id", required=True)
    enqueue_parser.add_argument("--content", default="Run the requested shell command and report the result.")
    enqueue_parser.add_argument("--response-delay-ms", type=int, default=0)

    final_parser = subparsers.add_parser("enqueue-final-response")
    final_parser.add_argument("--base-url", required=True)
    final_parser.add_argument("--content", required=True)
    final_parser.add_argument("--sandbox-id", required=True)
    final_parser.add_argument("--response-delay-ms", type=int, default=0)

    state_parser = subparsers.add_parser("manual-llm-state")
    state_parser.add_argument("--base-url", required=True)

    args = parser.parse_args()
    if args.command == "serve-scripted-llm":
        server = serve(host=args.host, port=args.port, steps=default_script_steps(idle_delay_ms=args.idle_delay_ms))
        server.serve_forever()
        return
    if args.command == "serve-manual-llm":
        server = serve_manual(host=args.host, port=args.port)
        server.serve_forever()
        return
    if args.command == "serve-manual-interceptor":
        interceptor = AgentCRRequestInterceptorServer(
            upstream_url=args.upstream_url,
            request_state_store=InMemoryRequestStateStore(),
            sandbox_id_resolver=lambda client_host, headers, body: args.sandbox_id if client_host == args.sandbox_ip else None,
            host=args.host,
            port=args.port,
        )
        interceptor.start()
        try:
            while True:
                time.sleep(3600.0)
        except KeyboardInterrupt:
            return
        finally:
            interceptor.stop()
    if args.command == "prepare-runtime":
        work_root = Path(args.work_root)
        runtime = prepare_iflow_runtime(
            work_root=work_root,
            alternate_node_runtime_dir=None if args.alternate_node_runtime_dir is None else Path(args.alternate_node_runtime_dir),
        )
        state = prepare_iflow_state(work_root=work_root, base_url=args.base_url, model_name=args.model_name)
        print(
            json.dumps(
                {
                    "runtime_root": str(runtime.root),
                    "runtime_strategy": runtime.runtime_strategy,
                    "state_root": str(state.root),
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "launch-manual-sandbox":
        session = launch_manual_iflow(
            work_root=Path(args.work_root),
            llm_base_url=args.llm_base_url,
            sandbox_id=args.sandbox_id,
            task_description=args.task,
            host_inspector_url=args.host_inspector_url,
            image_tag=args.image_tag,
            model_name=args.model_name,
            sandbox_ip=args.sandbox_ip,
            alternate_node_runtime_dir=None
            if args.alternate_node_runtime_dir is None
            else Path(args.alternate_node_runtime_dir),
        )
        print(json.dumps(session_summary(session), sort_keys=True, indent=2))
        return
    if args.command == "stop-manual-sandbox":
        session = stop_manual_iflow(work_root=Path(args.work_root), remove_image=args.remove_image)
        print(json.dumps(session_summary(session), sort_keys=True, indent=2))
        return
    if args.command == "checkpoint-manual-sandbox":
        print(json.dumps(checkpoint_manual_iflow(work_root=Path(args.work_root)), sort_keys=True, indent=2))
        return
    if args.command == "restore-manual-sandbox":
        print(
            json.dumps(
                restore_manual_iflow(work_root=Path(args.work_root), checkpoint_id=args.checkpoint_id),
                sort_keys=True,
                indent=2,
            )
        )
        return
    if args.command == "list-manual-checkpoints":
        print(json.dumps({"checkpoints": list_manual_checkpoints(work_root=Path(args.work_root))}, sort_keys=True, indent=2))
        return
    if args.command == "manual-shell":
        shell_command = [item for item in args.exec_argv if item != "--"]
        raise SystemExit(manual_shell(work_root=Path(args.work_root), command=shell_command or None))
    if args.command == "show-manual-session":
        print(json.dumps(session_summary(load_session(Path(args.work_root))), sort_keys=True, indent=2))
        return
    if args.command == "enqueue-run-shell-command":
        print(
            json.dumps(
                _post_json(
                    args.base_url,
                        "/control/run_shell_command",
                        {
                        "command": args.shell_command,
                        "sandbox_id": args.sandbox_id,
                        "content": args.content,
                        "response_delay_ms": args.response_delay_ms,
                    },
                ),
                sort_keys=True,
                indent=2,
            )
        )
        return
    if args.command == "enqueue-final-response":
        print(
            json.dumps(
                _post_json(
                    args.base_url,
                    "/control/final_response",
                    {
                        "content": args.content,
                        "sandbox_id": args.sandbox_id,
                        "response_delay_ms": args.response_delay_ms,
                    },
                ),
                sort_keys=True,
                indent=2,
            )
        )
        return
    if args.command == "manual-llm-state":
        print(json.dumps(_get_json(args.base_url, "/control/state"), sort_keys=True, indent=2))
        return

    build_image(tag=args.tag)
    rootfs = export_image_rootfs(tag=args.tag, output_dir=Path(args.output_dir))
    print(rootfs)


if __name__ == "__main__":
    main()
