# Contributing to Crab

Thank you for helping make recoverable agent sandboxes easier to use.

Crab v0 touches privileged Linux subsystems, so changes should be small,
reviewable, and explicit about what was tested on a real host.

## Before you start

- Open an issue for large behavioral changes or new checkpoint substrates.
- Keep refactors separate from behavior changes. Repository-wide renames
  should be their own commit so history remains reviewable.
- Do not commit API keys, private traces, benchmark datasets, generated
  rootfs trees, ZFS pool images, logs, or machine-specific absolute paths.
- Preserve the distinction between the control layer and checkpoint
  substrates. A new runtime or storage backend should not silently weaken
  checkpoint guarantees.

## Development setup

For dependency-light SDK and unit work:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

For real runc/CRIU/ZFS work on a disposable Ubuntu x86-64 host:

```bash
sudo ./scripts/install-ubuntu.sh
sudo ./scripts/smoke-rollback.sh
```

The installer creates privileged host state under `/opt/crab`,
`/var/lib/crab`, `/etc/crab`, and a dedicated ZFS pool. Read
[`docs/installation.md`](docs/installation.md) before running it on a machine
with existing ZFS workloads.

## Tests

Run the dependency-light suite documented for contributors:

```bash
.venv/bin/python -m unittest -v \
  tests.test_remote_engine_checkpoint \
  tests.test_image_runtime \
  tests.test_sdk_sandbox \
  tests.test_iflow_trace_replay
```

Then run focused tests for every module you changed. Real-host tests require
the installer dependencies and root privileges. Some historical benchmark
tests require optional SWE-bench packages and external datasets; state those
requirements in your pull request instead of adding datasets to the repo.

For changes to checkpoint/restore semantics, include at least one real
rollback test that mutates state after checkpoint and proves the mutation is
undone. A successful checkpoint command alone is not sufficient evidence.

## Documentation

- User-facing behavior belongs under `docs/` and must match a command or API
  that exists in the current tree.
- Experiment reports, PR notes, and run-specific tuning belong under
  `legacy/docs/` and must be marked as archived.
- Examples must not contain developer home directories, private hostnames, or
  paths to external datasets.
- Call out checkpoint boundaries: host bind mounts and external side effects
  are not rolled back.
- Keep English and Chinese README links working when changing entry points.

## Pull requests

Describe:

1. the user-visible problem and chosen behavior;
2. the checkpoint boundary and failure behavior;
3. unit tests run;
4. real-host tests run, including OS, CRIU, runc, and ZFS when relevant;
5. any optional dependency or external trace needed to reproduce the result.

Before requesting review:

```bash
git diff --check
```

Also inspect `git status` for generated artifacts and verify Markdown links.
Do not mix generated benchmark outputs or unrelated cleanup into a feature
change.

## License

By contributing, you agree that your contributions will be licensed under the
repository's [MIT License](LICENSE).
