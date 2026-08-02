from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from crab import RuncRestoreOptions, RuncRuntime, RuncRuntimePaths, SandboxId
from crab.ids import CheckpointId
from crab.runtime.base import CommandResult, CommandRunner


class _CapturingRunner(CommandRunner):
    """Records every command and returns a configurable result. Anything
    not seeded returns success; the test asserts on `commands`.
    """

    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        _ = (cwd, timeout_seconds)
        key = tuple(command)
        self.commands.append(key)
        return CommandResult(command=key, returncode=0, stdout="", stderr="")


class RuncCheckpointCommandTests(unittest.TestCase):
    def _make_runtime(self, runner: CommandRunner, root: Path) -> RuncRuntime:
        runtime = RuncRuntime(
            command_runner=runner,
            paths=RuncRuntimePaths(
                state_root=root / "state",
                bundle_root=root / "bundles",
                checkpoint_root=root / "checkpoints",
                metadata_root=root / "metadata",
                zfs_dataset_prefix="pool/crab",
            ),
        )
        runtime.launch(
            "runc",
            {"sandbox_id": "sbx-incr", "bundle_path": str(root / "bundles" / "sbx-incr")},
        )
        return runtime

    def _last_checkpoint_command(self, runner: _CapturingRunner) -> tuple[str, ...]:
        for cmd in reversed(runner.commands):
            if "checkpoint" in cmd:
                return cmd
        raise AssertionError("no checkpoint command captured")

    def test_full_checkpoint_omits_pre_dump_and_parent_path(self) -> None:
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            runtime = self._make_runtime(runner, Path(raw))
            sbx = SandboxId("sbx-incr")
            runtime.checkpoint_process(sbx, CheckpointId("ck-1"), leave_running=True)
        cmd = self._last_checkpoint_command(runner)
        self.assertIn("--leave-running=true", cmd)
        self.assertNotIn("--pre-dump", cmd)
        self.assertNotIn("--parent-path", cmd)

    def test_pre_dump_chain_root_omits_parent_path(self) -> None:
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            runtime = self._make_runtime(runner, Path(raw))
            sbx = SandboxId("sbx-incr")
            runtime.pre_dump_process(sbx, CheckpointId("ck-1"))
        cmd = self._last_checkpoint_command(runner)
        self.assertIn("--pre-dump", cmd)
        # Pre-dump implies process keeps running; we must not also pass
        # --leave-running, which runc rejects in combination.
        self.assertFalse(any(x.startswith("--leave-running") for x in cmd))
        self.assertNotIn("--parent-path", cmd)

    def test_pre_dump_with_parent_resolves_relative_path(self) -> None:
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = self._make_runtime(runner, root)
            sbx = SandboxId("sbx-incr")
            # Materialize the parent's pre_dump dir so the runtime doesn't
            # raise FileNotFoundError before constructing the command.
            parent_pd = Path(runtime.pre_dump_location(sbx, CheckpointId("ck-1")))
            parent_pd.mkdir(parents=True, exist_ok=True)
            runtime.pre_dump_process(
                sbx,
                CheckpointId("ck-2"),
                parent_checkpoint_id=CheckpointId("ck-1"),
            )
        cmd = self._last_checkpoint_command(runner)
        self.assertIn("--pre-dump", cmd)
        idx = cmd.index("--parent-path")
        # Two levels up from <ckpt-2>/pre_dump to reach <ckpt-1>/pre_dump.
        self.assertEqual(cmd[idx + 1], "../../ck-1/pre_dump")

    def test_incremental_dump_parent_points_at_sibling_pre_dump(self) -> None:
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = self._make_runtime(runner, root)
            sbx = SandboxId("sbx-incr")
            # Final dump's parent is THIS checkpoint's own pre_dump sibling.
            sibling_pd = Path(runtime.pre_dump_location(sbx, CheckpointId("ck-2")))
            sibling_pd.mkdir(parents=True, exist_ok=True)
            runtime.checkpoint_process(
                sbx,
                CheckpointId("ck-2"),
                leave_running=False,
                parent_checkpoint_id=CheckpointId("ck-2"),
            )
        cmd = self._last_checkpoint_command(runner)
        idx = cmd.index("--parent-path")
        self.assertEqual(cmd[idx + 1], "../pre_dump")
        self.assertIn("--leave-running=false", cmd)
        self.assertNotIn("--pre-dump", cmd)

    def test_missing_parent_pre_dump_dir_raises(self) -> None:
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            runtime = self._make_runtime(runner, Path(raw))
            sbx = SandboxId("sbx-incr")
            with self.assertRaises(FileNotFoundError):
                runtime.pre_dump_process(
                    sbx,
                    CheckpointId("ck-2"),
                    parent_checkpoint_id=CheckpointId("ck-missing"),
                )

    def test_runtime_capability_advertises_incremental_process(self) -> None:
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            runtime = self._make_runtime(runner, Path(raw))
            self.assertTrue(runtime.capabilities().supports_incremental_process)

    def test_link_ancestor_pre_dump_creates_relative_symlink(self) -> None:
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = self._make_runtime(runner, root)
            source_id = SandboxId("sbx-incr")
            target_id = SandboxId("sbx-fork")
            ckpt = CheckpointId("ck-anc")
            # Populate the source's runtime checkpoint dir as if a pre-dump
            # had landed there. link_ancestor_pre_dump only mints the
            # symlink — actual CRIU output is the runtime's responsibility.
            source_dir = root / "checkpoints" / str(source_id) / str(ckpt)
            (source_dir / "pre_dump").mkdir(parents=True)
            (source_dir / "pre_dump" / "pages-1.img").write_bytes(b"PAGES")

            self.assertTrue(runtime.link_ancestor_pre_dump(source_id, target_id, ckpt))
            target_dir = root / "checkpoints" / str(target_id) / str(ckpt)
            self.assertTrue(target_dir.is_symlink())
            # Reading through the link sees the source's bytes, validating
            # that CRIU's `--parent-path` walk would resolve correctly.
            self.assertEqual(
                (target_dir / "pre_dump" / "pages-1.img").read_bytes(), b"PAGES"
            )

    def test_link_ancestor_pre_dump_replaces_prior_symlink(self) -> None:
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = self._make_runtime(runner, root)
            source_id = SandboxId("sbx-incr")
            target_id = SandboxId("sbx-fork")
            ckpt = CheckpointId("ck-anc")
            source_dir = root / "checkpoints" / str(source_id) / str(ckpt)
            source_dir.mkdir(parents=True)
            runtime.link_ancestor_pre_dump(source_id, target_id, ckpt)
            # Re-linking must not raise even though the symlink already exists.
            self.assertTrue(runtime.link_ancestor_pre_dump(source_id, target_id, ckpt))

    def test_link_ancestor_pre_dump_raises_when_source_missing(self) -> None:
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = self._make_runtime(runner, root)
            with self.assertRaises(FileNotFoundError):
                runtime.link_ancestor_pre_dump(
                    SandboxId("nope"), SandboxId("fork"), CheckpointId("ck")
                )

    def test_lazy_pages_flag_threads_through_to_runc_restore(self) -> None:
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = RuncRuntime(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    checkpoint_root=root / "checkpoints",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/crab",
                ),
                restore_options=RuncRestoreOptions(lazy_pages=True),
            )
            runtime.launch(
                "runc",
                {"sandbox_id": "sbx-lazy", "bundle_path": str(root / "bundles" / "sbx-lazy")},
            )
            sbx = SandboxId("sbx-lazy")
            status = runtime.restore_process(sbx, CheckpointId("ck-1"))
        # Find the restore command (clear that pre-launch commands are
        # captured first); the last 'restore' is what runc would run.
        restore_cmd = next(
            cmd for cmd in reversed(runner.commands) if "restore" in cmd
        )
        self.assertIn("--lazy-pages", restore_cmd)
        # Operation status records the lazy-pages flag for downstream
        # telemetry / observability.
        self.assertEqual(status.metadata.get("lazy_pages"), True)

    def test_lazy_pages_default_off_omits_flag(self) -> None:
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = self._make_runtime(runner, Path(raw))
            sbx = SandboxId("sbx-incr")
            status = runtime.restore_process(sbx, CheckpointId("ck-1"))
        restore_cmd = next(
            cmd for cmd in reversed(runner.commands) if "restore" in cmd
        )
        self.assertNotIn("--lazy-pages", restore_cmd)
        self.assertEqual(status.metadata.get("lazy_pages"), False)

    def test_lazy_restore_capability_advertised_by_runc_runtime(self) -> None:
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            runtime = self._make_runtime(runner, Path(raw))
            self.assertTrue(runtime.capabilities().supports_lazy_restore)

    def test_runtime_image_path_in_use_default_returns_false(self) -> None:
        # No daemons registered → predicate is uniformly False.
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            runtime = self._make_runtime(runner, Path(raw))
            self.assertFalse(
                runtime.runtime_image_path_in_use(Path(raw) / "checkpoints" / "anywhere")
            )

    def test_runtime_image_path_in_use_after_register_and_reap(self) -> None:
        # Spawn a benign sleep PID into the daemon registry by hand,
        # exercise both directions of path containment, then reap and
        # confirm the registry empties out.
        from crab.runtime.runc import _LazyPagesDaemonHandle

        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = self._make_runtime(runner, root)
            sandbox = SandboxId("sbx-fork")
            ckpt = CheckpointId("ck-leaf")
            image_path = root / "checkpoints" / str(sandbox) / str(ckpt) / "process"
            image_path.mkdir(parents=True)

            # Use a real subprocess so reap_lazy_pages_daemon actually has
            # something to signal; bash sleep is reliable enough.
            import subprocess
            proc = subprocess.Popen(["sleep", "30"])
            try:
                runtime._register_lazy_pages_daemon(
                    pid=proc.pid,
                    sandbox_id=sandbox,
                    checkpoint_id=ckpt,
                    image_path=image_path,
                    work_path=image_path.parent / "work",
                )
                # Daemon image_path is the leaf process dir. Pruning the
                # checkpoint dir (its parent) must report in-use because
                # ``rmtree`` on the parent would break the daemon's reads.
                checkpoint_dir = image_path.parent
                self.assertTrue(runtime.runtime_image_path_in_use(checkpoint_dir))
                # The image dir itself is also in use.
                self.assertTrue(runtime.runtime_image_path_in_use(image_path))
                # An unrelated sibling is not.
                sibling = root / "checkpoints" / "other-sbx" / "ck"
                sibling.mkdir(parents=True)
                self.assertFalse(runtime.runtime_image_path_in_use(sibling))
            finally:
                runtime.reap_lazy_pages_daemon(proc.pid)
                proc.wait(timeout=5)
            # Reaper unregisters; further checks are False.
            self.assertFalse(runtime.runtime_image_path_in_use(image_path.parent))

    def test_runtime_image_path_in_use_drops_dead_daemons(self) -> None:
        # If a daemon exits on its own (CRIU does this once all faults
        # are served) the registry can still hold its handle until the
        # next reap call. ``runtime_image_path_in_use`` snapshots the
        # registry and prunes dead PIDs lazily so retention isn't
        # blocked forever waiting for an explicit reap.
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = self._make_runtime(runner, root)
            image_path = root / "checkpoints" / "sbx" / "ck" / "process"
            image_path.mkdir(parents=True)
            import subprocess
            proc = subprocess.Popen(["true"])
            proc.wait()
            # Register the now-dead pid; the predicate must observe its
            # death and prune itself instead of falsely guarding.
            runtime._register_lazy_pages_daemon(
                pid=proc.pid,
                sandbox_id=SandboxId("sbx"),
                checkpoint_id=CheckpointId("ck"),
                image_path=image_path,
                work_path=image_path.parent / "work",
            )
            self.assertFalse(runtime.runtime_image_path_in_use(image_path))
            # The follow-up live check has already pruned the entry.
            self.assertEqual(runtime._lazy_pages_daemons, {})

    def test_materialize_linked_pre_dumps_replaces_symlinks_with_copies(self) -> None:
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = self._make_runtime(runner, root)
            source_id = SandboxId("sbx-incr")
            target_id = SandboxId("sbx-fork")
            ckpt = CheckpointId("ck-anc")
            source_dir = root / "checkpoints" / str(source_id) / str(ckpt)
            (source_dir / "pre_dump").mkdir(parents=True)
            (source_dir / "pre_dump" / "pages-1.img").write_bytes(b"BYTES")
            runtime.link_ancestor_pre_dump(source_id, target_id, ckpt)

            count = runtime.materialize_linked_pre_dumps(target_id)
            self.assertEqual(count, 1)
            target_dir = root / "checkpoints" / str(target_id) / str(ckpt)
            self.assertFalse(target_dir.is_symlink())
            # After materialization the fork owns the bytes; destroying the
            # source's tree must not affect the fork.
            import shutil

            shutil.rmtree(source_dir.parent)
            self.assertEqual(
                (target_dir / "pre_dump" / "pages-1.img").read_bytes(), b"BYTES"
            )


if __name__ == "__main__":
    unittest.main()
