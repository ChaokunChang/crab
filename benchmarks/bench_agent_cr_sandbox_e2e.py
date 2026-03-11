#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_cr import (
    AgentCRRequestInterceptorServer,
    AgentCRSystem,
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    CRExecutor,
    CRScheduler,
    CompositeRequestInterceptorHook,
    DefaultCWorker,
    DefaultRWorker,
    EBPFEvent,
    EBPFEventKind,
    EBPFSandboxInspector,
    ExecutorConfig,
    InMemoryEBPFEventCollector,
    InMemoryRequestStateStore,
    InMemorySchedulerStateStore,
    InMemoryTelemetrySink,
    LocalCheckpointManager,
    RequestAwareSandboxInspector,
    RuncRuntimeAdapter,
    RuncRuntimePaths,
    RuncSandboxManager,
    RuncSandboxManagerPaths,
    SandboxId,
    SandboxSnapshot,
    SchedulerConfig,
    StorageConfig,
    TelemetryRequestInterceptorHook,
)
from agent_cr.ids import JobId
from agent_cr.models import RestoreJob, utc_now
from simulated_agent.image import build_image, export_image_rootfs
from simulated_agent.service import serve


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent-CR sandboxed simulated-agent benchmark")
    parser.add_argument("--sandboxes", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
    )
    return parser.parse_args()


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_http_json(url: str, *, timeout_s: float = 30.0) -> dict[str, object]:
    deadline = time.time() + timeout_s
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_exc = exc
            time.sleep(0.2)
    raise RuntimeError(f"timed out waiting for {url}: {last_exc}")


def wait_for(
    predicate,
    *,
    timeout_s: float = 30.0,
    interval_s: float = 0.2,
    raise_on_timeout: bool = True,
) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    if not raise_on_timeout:
        return False
    raise RuntimeError("timed out waiting for predicate")


def enough_progress(payload: dict[str, object]) -> bool:
    return (
        int(payload["total_actions"]) >= 6
        and int(payload["filesystem_actions"]) >= 1
        and int(payload["process_actions"]) >= 1
        and int(payload["network_actions"]) >= 1
        and int(payload["stateful_actions"]) >= 1
    )


def write_bundle_config(
    *,
    bundle_dir: Path,
    interceptor_port: int,
    provider: str,
    sandbox_name: str,
    status_port: int,
    cgroup_path: str,
) -> None:
    config_path = bundle_dir / "config.json"
    cfg = json.loads(config_path.read_text())
    linux_cfg = cfg.get("linux", {})
    linux_cfg["namespaces"] = [
        ns
        for ns in linux_cfg.get("namespaces", [])
        if ns.get("type") not in {"network", "cgroup"}
    ]
    linux_cfg["cgroupsPath"] = cgroup_path
    linux_cfg.pop("seccomp", None)
    cfg["linux"] = linux_cfg
    cfg["process"]["terminal"] = False
    cfg["process"]["cwd"] = "/work"
    cfg["process"]["args"] = [
        "/bin/sh",
        "-lc",
        f"exec /usr/local/bin/agent-cli run --provider {provider} >/dev/null 2>/dev/null",
    ]
    cfg["process"]["env"] = [
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "PYTHONUNBUFFERED=1",
        f"INTERCEPTOR_URL=http://127.0.0.1:{interceptor_port}",
        f"STATUS_PORT={status_port}",
        "POLL_INTERVAL_S=0.2",
        "AGENT_WORK_DIR=/work",
        f"AGENT_SANDBOX_ID={sandbox_name}",
        f"AGENT_PROVIDER={provider}",
    ]
    cfg["root"]["path"] = "rootfs"
    cfg["root"]["readonly"] = False
    config_path.write_text(json.dumps(cfg, indent=2))


def record_activity_events(
    *,
    collector: InMemoryEBPFEventCollector,
    sandbox_id: SandboxId,
    previous: dict[str, object],
    current: dict[str, object],
) -> None:
    mappings = [
        ("filesystem_actions", EBPFEventKind.FILE_WRITE, {"path": "/work/tool_artifact.txt"}),
        ("process_actions", EBPFEventKind.PROCESS_EXEC, {"command": "/bin/sh"}),
        ("network_actions", EBPFEventKind.NETWORK_EGRESS, {"address": "127.0.0.1"}),
    ]
    for field, kind, metadata in mappings:
        if int(current.get(field, 0)) > int(previous.get(field, 0)):
            collector.record(
                EBPFEvent(
                    sandbox_id=sandbox_id,
                    kind=kind,
                    observed_at=utc_now(),
                    metadata=metadata,
                )
            )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    required = ["docker", "runc", "criu", "zfs"]
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        raise SystemExit(f"missing required binaries: {', '.join(missing)}")

    with tempfile.TemporaryDirectory(prefix="agent_cr_sandbox_bench_") as tmp:
        root = Path(tmp)
        pool_name = f"agentcrbench{int(time.time())}"
        image_tag = f"agent-cr-simulated-agent-bench:{int(time.time())}"
        llm_server = serve(host="127.0.0.1", port=0, response_delay_ms=250)
        llm_thread = threading.Thread(target=llm_server.serve_forever, daemon=True)
        llm_thread.start()
        rows: list[dict[str, object]] = []
        sandboxes: list[SandboxId] = []
        status_ports: dict[SandboxId, int] = {}
        last_status: dict[SandboxId, dict[str, object]] = {}
        runtime_state_root = root / "runtime-state"
        executor: CRExecutor | None = None
        interceptor: AgentCRRequestInterceptorServer | None = None
        try:
            build_image(tag=image_tag)
            exported_rootfs = export_image_rootfs(tag=image_tag, output_dir=root / "image")
            subprocess.run(["truncate", "-s", "10G", str(root / "zpool.img")], check=True)
            subprocess.run(["zpool", "create", "-f", pool_name, str(root / "zpool.img")], check=True)
            subprocess.run(["zfs", "create", f"{pool_name}/agent-cr"], check=True)

            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            request_state_store = InMemoryRequestStateStore()
            base_inspector = EBPFSandboxInspector(collector)
            inspector = RequestAwareSandboxInspector(base_inspector, request_state_store)
            runtime = RuncRuntimeAdapter(
                paths=RuncRuntimePaths(
                    state_root=runtime_state_root,
                    bundle_root=root / "bundles",
                    checkpoint_root=root / "checkpoints",
                    zfs_dataset_prefix=f"{pool_name}/agent-cr",
                )
            )
            storage = LocalCheckpointManager(StorageConfig(root_dir=root / "storage"))
            executor = CRExecutor(
                ExecutorConfig(max_workers=max(1, args.sandboxes)),
                DefaultCWorker(
                    AdapterProcessCWorker(runtime),
                    AdapterFileSystemCWorker(runtime),
                    storage,
                    runtime,
                ),
                DefaultRWorker(
                    AdapterProcessRWorker(runtime),
                    AdapterFileSystemRWorker(runtime),
                    storage,
                ),
                telemetry,
            )
            sandbox_manager = RuncSandboxManager(
                paths=RuncSandboxManagerPaths(
                    state_root=runtime_state_root,
                    bundle_root=root / "bundles",
                    metadata_root=root / "sandbox-meta",
                    zfs_dataset_prefix=f"{pool_name}/agent-cr",
                )
            )
            scheduler = CRScheduler(
                SchedulerConfig(
                    min_checkpoint_interval_seconds=0.0,
                    force_checkpoint_after_seconds=0.0,
                    require_change_signal=True,
                    prefer_checkpoint_during_llm_request=True,
                    require_llm_request_for_checkpoint=False,
                ),
                inspector,
                sandbox_manager,
                InMemorySchedulerStateStore(),
                telemetry,
            )
            system = AgentCRSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                sandbox_manager=sandbox_manager,
                telemetry=telemetry,
                request_state_store=request_state_store,
            )
            interceptor = AgentCRRequestInterceptorServer(
                upstream_url=f"http://127.0.0.1:{llm_server.server_address[1]}",
                request_state_store=request_state_store,
                hook=CompositeRequestInterceptorHook([TelemetryRequestInterceptorHook(telemetry)]),
                on_state_change=system.notify_interceptor_state_change,
                host="127.0.0.1",
                port=0,
            )
            interceptor.start()
            wait_for_http_json(f"http://127.0.0.1:{interceptor.port}/healthz")

            for index in range(args.sandboxes):
                sandbox_name = f"sandbox-{index}"
                sandbox_id = SandboxId(sandbox_name)
                status_port = find_free_port()
                sandboxes.append(sandbox_id)
                status_ports[sandbox_id] = status_port
                bundle_dir = root / "bundles" / sandbox_name
                bundle_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(["runc", "spec"], cwd=bundle_dir, check=True)
                write_bundle_config(
                    bundle_dir=bundle_dir,
                    interceptor_port=interceptor.port,
                    provider=args.provider,
                    sandbox_name=sandbox_name,
                    status_port=status_port,
                    cgroup_path=f"agent-cr-bench/{pool_name}/{sandbox_name}",
                )
                inspector.upsert_snapshot(
                    SandboxSnapshot(
                        sandbox_id=sandbox_id,
                        runtime_name="runc",
                        is_running=True,
                        process_changed=False,
                        filesystem_changed=False,
                        observed_at=utc_now(),
                    )
                )
                system.sandbox_manager.launch(
                    "runc",
                    {
                        "sandbox_id": sandbox_name,
                        "bundle_path": str(bundle_dir),
                        "rootfs_init_dirs": [
                            "work",
                            "tmp",
                            "proc",
                            "dev",
                            "dev/pts",
                            "dev/shm",
                            "dev/mqueue",
                            "sys",
                            "run",
                            "var",
                        ],
                        "rootfs_copy_paths": [{"source": str(exported_rootfs), "destination": "/"}],
                    },
                )
                payload = wait_for_http_json(f"http://127.0.0.1:{status_port}/status")
                last_status[sandbox_id] = payload

            for iteration in range(args.iters):
                checkpoint_results = []
                for sandbox_id in sandboxes:
                    payload = wait_for_http_json(f"http://127.0.0.1:{status_ports[sandbox_id]}/status")
                    wait_for(
                        lambda sid=sandbox_id: enough_progress(
                            wait_for_http_json(f"http://127.0.0.1:{status_ports[sid]}/status", timeout_s=2.0)
                        ),
                        timeout_s=45.0,
                    )
                    if not wait_for(
                        lambda sid=sandbox_id: request_state_store.get(sid).llm_request_in_flight,
                        timeout_s=20.0,
                        raise_on_timeout=False,
                    ):
                        logger.warning(
                            "sandbox %s did not enter an in-flight LLM request window before checkpoint; continuing",
                            sandbox_id,
                        )
                    current = wait_for_http_json(f"http://127.0.0.1:{status_ports[sandbox_id]}/status")
                    record_activity_events(
                        collector=collector,
                        sandbox_id=sandbox_id,
                        previous=last_status[sandbox_id],
                        current=current,
                    )
                    last_status[sandbox_id] = current

                t0 = time.perf_counter()
                for sandbox_id in sandboxes:
                    result = system.checkpoint_if_due(sandbox_id)
                    if result is not None:
                        checkpoint_results.append(result)
                t1 = time.perf_counter()
                logger.debug("checkpoint results: %s", checkpoint_results)

                restore_results = [
                    system.restore_once(result.sandbox_id, result.checkpoint_id)
                    for result in checkpoint_results
                ]
                t2 = time.perf_counter()
                logger.debug("restore results: %s", restore_results)

                for sandbox_id in sandboxes:
                    inspector.upsert_snapshot(
                        SandboxSnapshot(
                            sandbox_id=sandbox_id,
                            runtime_name="runc",
                            is_running=True,
                            process_changed=False,
                            filesystem_changed=False,
                            observed_at=utc_now(),
                            last_checkpoint_at=utc_now(),
                        )
                    )

                rows.append(
                    {
                        "iter": iteration,
                        "provider": args.provider,
                        "sandboxes": args.sandboxes,
                        "tool_actions": sum(int(last_status[sid]["total_actions"]) for sid in sandboxes),
                        "fs_actions": sum(int(last_status[sid]["filesystem_actions"]) for sid in sandboxes),
                        "process_actions": sum(int(last_status[sid]["process_actions"]) for sid in sandboxes),
                        "network_actions": sum(int(last_status[sid]["network_actions"]) for sid in sandboxes),
                        "checkpoints": len(checkpoint_results),
                        "restores": len(restore_results),
                        "checkpoint_batch_ms": (t1 - t0) * 1000.0,
                        "restore_batch_ms": (t2 - t1) * 1000.0,
                        "success_ratio": (
                            sum(1 for item in restore_results if item.status.value == "succeeded")
                            / max(1, len(restore_results))
                        ),
                    }
                )

            if args.out:
                with open(args.out, "w", newline="") as fh:
                    writer = csv.DictWriter(
                        fh,
                        fieldnames=[
                            "iter",
                            "provider",
                            "sandboxes",
                            "tool_actions",
                            "fs_actions",
                            "process_actions",
                            "network_actions",
                            "checkpoints",
                            "restores",
                            "checkpoint_batch_ms",
                            "restore_batch_ms",
                            "success_ratio",
                        ],
                    )
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)

            avg_ckpt = sum(float(row["checkpoint_batch_ms"]) for row in rows) / len(rows)
            avg_restore = sum(float(row["restore_batch_ms"]) for row in rows) / len(rows)
            avg_success = sum(float(row["success_ratio"]) for row in rows) / len(rows)
            print(f"checkpoint_batch_ms_avg: {avg_ckpt:.3f}")
            print(f"restore_batch_ms_avg:    {avg_restore:.3f}")
            print(f"restore_success_ratio_avg: {avg_success:.3f}")
        finally:
            if interceptor is not None:
                interceptor.stop()
            if executor is not None:
                executor.shutdown()
            for sandbox_id in sandboxes:
                subprocess.run(
                    ["runc", "--root", str(runtime_state_root), "delete", "-f", str(sandbox_id)],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            subprocess.run(["docker", "rmi", "-f", image_tag], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["zfs", "destroy", "-r", f"{pool_name}/agent-cr"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["zpool", "destroy", "-f", pool_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            llm_server.shutdown()
            llm_server.server_close()
            llm_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
