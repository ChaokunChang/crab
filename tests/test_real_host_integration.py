from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

from agent_cr import (
    AgentCRSystem,
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    CRExecutor,
    CRScheduler,
    DefaultCWorker,
    DefaultHeuristicPolicy,
    DefaultRWorker,
    EBPFSandboxInspector,
    EBPFEvent,
    EBPFEventKind,
    ExecutorConfig,
    InMemoryEBPFEventCollector,
    InMemorySchedulerStateStore,
    InMemoryTelemetrySink,
    LocalCheckpointManager,
    PolicyConfig,
    RuncRuntimeAdapter,
    RuncRuntimePaths,
    RuncSandboxManager,
    RuncSandboxManagerPaths,
    SandboxId,
    SandboxSnapshot,
    SchedulerConfig,
    StorageConfig,
)
from agent_cr.models import JobStatus, utc_now


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "agent_sandbox"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http_json(url: str, *, timeout_s: float = 20.0) -> dict[str, object]:
    deadline = time.time() + timeout_s
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - exercised in real integration only
            last_exc = exc
            time.sleep(0.2)
    raise RuntimeError(f"timed out waiting for {url}: {last_exc}")


def _wait_for(predicate, *, timeout_s: float = 20.0, interval_s: float = 0.2) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(interval_s)
    raise RuntimeError("timed out waiting for predicate")


class RealHostIntegrationTests(unittest.TestCase):
    def test_real_runc_criu_zfs_checkpoint_restore(self) -> None:
        if shutil.which("runc") is None or shutil.which("criu") is None or shutil.which("zfs") is None:
            self.skipTest("runc/criu/zfs not installed")

        tmpdir = tempfile.TemporaryDirectory(prefix="agent_cr_real_it_")
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name)
        sandbox_id = SandboxId("sbx-real")
        pool_name = f"agentcrit{int(time.time())}"
        try:
            llm_port = _find_free_port()
            proxy_port = _find_free_port()
            status_port = _find_free_port()
            pool_file = root / "zpool.img"
            bundle_dir = root / "bundles" / "sbx-real"
            runtime_state_root = root / "runtime-state"
            checkpoint_root = root / "checkpoints"
            storage_root = root / "storage"
            sandbox_metadata_root = root / "sandbox-meta"

            llm_proc = subprocess.Popen(
                ["python3", str(FIXTURE_DIR / "simulated_llm_service.py"), "--port", str(llm_port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            proxy_proc = subprocess.Popen(
                [
                    "python3",
                    str(FIXTURE_DIR / "proxy_server.py"),
                    "--port",
                    str(proxy_port),
                    "--llm-url",
                    f"http://127.0.0.1:{llm_port}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self.addCleanup(lambda: llm_proc.poll() is None and llm_proc.terminate())
            self.addCleanup(lambda: proxy_proc.poll() is None and proxy_proc.terminate())
            _wait_for_http_json(f"http://127.0.0.1:{llm_port}/healthz")
            _wait_for_http_json(f"http://127.0.0.1:{proxy_port}/healthz")

            subprocess.run(["truncate", "-s", "512M", str(pool_file)], check=True)
            subprocess.run(["zpool", "create", "-f", pool_name, str(pool_file)], check=True)
            self.addCleanup(
                lambda: subprocess.run(
                    ["zpool", "destroy", "-f", pool_name],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
            subprocess.run(["zfs", "create", f"{pool_name}/agent-cr"], check=True)

            bundle_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["runc", "spec"], cwd=bundle_dir, check=True)
            self._write_bundle_config(
                bundle_dir=bundle_dir,
                proxy_port=proxy_port,
                status_port=status_port,
            )

            telemetry = InMemoryTelemetrySink()
            collector = InMemoryEBPFEventCollector()
            inspector = EBPFSandboxInspector(collector)
            runtime = RuncRuntimeAdapter(
                paths=RuncRuntimePaths(
                    state_root=runtime_state_root,
                    bundle_root=root / "bundles",
                    checkpoint_root=checkpoint_root,
                    zfs_dataset_prefix=f"{pool_name}/agent-cr",
                )
            )
            storage = LocalCheckpointManager(StorageConfig(root_dir=storage_root))
            executor = CRExecutor(
                ExecutorConfig(max_workers=1),
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
            scheduler = CRScheduler(
                SchedulerConfig(),
                DefaultHeuristicPolicy(
                    PolicyConfig(
                        min_checkpoint_interval_seconds=0.0,
                        force_checkpoint_after_seconds=0.0,
                        require_change_signal=True,
                    )
                ),
                inspector,
                InMemorySchedulerStateStore(),
                telemetry,
            )
            sandbox_manager = RuncSandboxManager(
                paths=RuncSandboxManagerPaths(
                    state_root=runtime_state_root,
                    bundle_root=root / "bundles",
                    metadata_root=sandbox_metadata_root,
                    zfs_dataset_prefix=f"{pool_name}/agent-cr",
                )
            )
            system = AgentCRSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                sandbox_manager=sandbox_manager,
                telemetry=telemetry,
            )

            rootfs_copy_paths = self._build_rootfs_copy_paths()
            sandbox_id = system.sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-real",
                    "bundle_path": str(bundle_dir),
                    "rootfs_init_dirs": [
                        "usr",
                        "bin",
                        "lib",
                        "lib64",
                        "etc",
                        "app",
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
                    "rootfs_copy_paths": rootfs_copy_paths,
                },
            )
            self.addCleanup(
                lambda: subprocess.run(
                    ["runc", "--root", str(runtime_state_root), "delete", "-f", str(sandbox_id)],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
            self.assertEqual(sandbox_id, SandboxId("sbx-real"))

            status_before = self._wait_for_sandbox_status(
                status_port=status_port,
                runtime_state_root=runtime_state_root,
                sandbox_id=sandbox_id,
                work_dir=bundle_dir / "rootfs" / "work",
            )
            self.assertGreaterEqual(int(status_before["total_actions"]), 0)

            _wait_for(
                lambda: self._enough_agent_progress(status_port),
                timeout_s=30.0,
            )
            status_before = _wait_for_http_json(f"http://127.0.0.1:{status_port}/status")
            total_before = int(status_before["total_actions"])
            runtime_id_before = str(status_before["runtime_id"])

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
            collector.record(
                EBPFEvent(
                    sandbox_id=sandbox_id,
                    kind=EBPFEventKind.FILE_WRITE,
                    observed_at=utc_now(),
                    metadata={"path": "/work/agent_state.json"},
                )
            )

            checkpoint_result = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(checkpoint_result)
            assert checkpoint_result is not None
            self.assertEqual(checkpoint_result.status, JobStatus.SUCCEEDED)

            work_dir = bundle_dir / "rootfs" / "work"
            tamper_path = work_dir / "host_tamper.txt"
            tamper_path.write_text("tampered\n", encoding="utf-8")
            self.assertTrue(tamper_path.exists())

            restore_result = system.restore_once(sandbox_id, checkpoint_result.checkpoint_id)
            self.assertEqual(restore_result.status, JobStatus.SUCCEEDED)
            self.assertFalse(tamper_path.exists(), "zfs rollback should remove host-side tamper")

            status_after = _wait_for_http_json(f"http://127.0.0.1:{status_port}/status")
            self.assertEqual(str(status_after["runtime_id"]), runtime_id_before)
            self.assertGreater(int(status_after["total_actions"]), total_before)
            self.assertGreaterEqual(int(status_after["stateful_actions"]), 1)
            self.assertGreaterEqual(int(status_after["side_effectful_actions"]), 1)
            self.assertTrue(bool(status_after["state_file_exists"]))
            self.assertTrue(bool(status_after["side_effect_log_exists"]))

            state_path = work_dir / "agent_state.json"
            side_effect_log = work_dir / "side_effects.log"
            self.assertTrue(state_path.exists())
            self.assertTrue(side_effect_log.exists())

            system.sandbox_manager.stop(sandbox_id)
            self.assertEqual(system.sandbox_manager.describe(sandbox_id).status, "stopped")
            system.sandbox_manager.delete(sandbox_id)
            executor.shutdown()
        finally:
            subprocess.run(
                ["runc", "--root", str(root / "runtime-state"), "delete", "-f", str(sandbox_id)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["zfs", "destroy", "-r", f"{pool_name}/agent-cr/sbx-real"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["zfs", "destroy", "-r", f"{pool_name}/agent-cr"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["zpool", "destroy", "-f", pool_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def _write_bundle_config(self, *, bundle_dir: Path, proxy_port: int, status_port: int) -> None:
        config_path = bundle_dir / "config.json"
        cfg = json.loads(config_path.read_text())
        python_bin = str(Path(sys.executable).resolve())
        linux_cfg = cfg.get("linux", {})
        linux_cfg["namespaces"] = [ns for ns in linux_cfg.get("namespaces", []) if ns.get("type") != "network"]
        linux_cfg.pop("seccomp", None)
        cfg["linux"] = linux_cfg
        cfg["process"]["terminal"] = False
        cfg["process"]["cwd"] = "/work"
        cfg["process"]["args"] = [python_bin, "/app/sandbox_agent.py"]
        cfg["process"]["env"] = [
            "PATH=/usr/bin:/bin",
            "PYTHONUNBUFFERED=1",
            f"PROXY_URL=http://127.0.0.1:{proxy_port}",
            f"STATUS_PORT={status_port}",
            "POLL_INTERVAL_S=0.2",
            "AGENT_WORK_DIR=/work",
        ]
        cfg["root"]["path"] = "rootfs"
        cfg["root"]["readonly"] = False
        config_path.write_text(json.dumps(cfg, indent=2))

    def _enough_agent_progress(self, status_port: int) -> bool:
        try:
            payload = _wait_for_http_json(f"http://127.0.0.1:{status_port}/status", timeout_s=1.0)
        except Exception:
            return False
        return (
            int(payload["total_actions"]) >= 4
            and int(payload["stateful_actions"]) >= 1
            and int(payload["side_effectful_actions"]) >= 1
        )

    def _build_rootfs_copy_paths(self) -> list[dict[str, str]]:
        major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
        python_bin = Path(sys.executable).resolve()
        candidates = [
            (python_bin, str(python_bin)),
            (Path(f"/usr/lib/python{major_minor}"), f"/usr/lib/python{major_minor}"),
            (Path(f"/usr/lib/python{major_minor}.zip"), f"/usr/lib/python{major_minor}.zip"),
            (Path("/usr/lib/python3/dist-packages"), "/usr/lib/python3/dist-packages"),
            (FIXTURE_DIR, "/app"),
        ]
        copy_paths: list[dict[str, str]] = []
        for source, destination in candidates:
            if source.exists():
                copy_paths.append({"source": str(source), "destination": destination})

        ldd = subprocess.run(
            ["ldd", str(python_bin)],
            check=True,
            capture_output=True,
            text=True,
        )
        seen: set[str] = set()
        for line in ldd.stdout.splitlines():
            line = line.strip()
            if "=>" in line:
                candidate = line.split("=>", 1)[1].split("(", 1)[0].strip()
            else:
                candidate = line.split("(", 1)[0].strip()
            if not candidate.startswith("/"):
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            copy_paths.append({"source": candidate, "destination": candidate})
        return copy_paths

    def _wait_for_sandbox_status(
        self,
        *,
        status_port: int,
        runtime_state_root: Path,
        sandbox_id: SandboxId,
        work_dir: Path,
    ) -> dict[str, object]:
        try:
            return _wait_for_http_json(f"http://127.0.0.1:{status_port}/status")
        except Exception as exc:
            state = subprocess.run(
                ["runc", "--root", str(runtime_state_root), "state", str(sandbox_id)],
                check=False,
                capture_output=True,
                text=True,
            )
            work_listing = sorted(x.name for x in work_dir.glob("*")) if work_dir.exists() else []
            raise RuntimeError(
                "sandbox status endpoint did not come up\n"
                f"runc_state_rc={state.returncode}\n"
                f"runc_state_stdout={state.stdout.strip()}\n"
                f"runc_state_stderr={state.stderr.strip()}\n"
                f"work_dir={work_dir}\n"
                f"work_listing={work_listing}\n"
                f"original_error={exc}"
            ) from exc


if __name__ == "__main__":
    unittest.main()
