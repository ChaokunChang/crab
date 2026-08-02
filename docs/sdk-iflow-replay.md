# Replay an iFlow trace

This example runs a recorded iFlow tool trajectory without a model API key.
The trace and Terminal-Bench task assets are inputs; datasets and local machine
paths are not embedded in the repository.

The `crack-7z-hash` trace is a useful multi-turn example because it includes
shell and file-write tool calls.

## Inputs

You need:

- a Terminal-Bench task directory containing `task.yaml`,
  `docker-compose.yaml`, and `run-tests.sh`;
- the matching `agent-logs/proxy_server_trajectory.log`.

Set paths appropriate to your checkout or dataset location:

```bash
TASK_ROOT=/path/to/terminal-bench/original-tasks/crack-7z-hash
TRACE=/path/to/crack-7z-hash/agent-logs/proxy_server_trajectory.log
```

Other parquet datasets are not required for this example.

## Run

Terminal 1 — start the replay router from the Crab repository root:

```bash
PYTHONPATH=. python3 -m integrations.llm_services.router \
  --host 127.0.0.1 \
  --port 18080 \
  --telemetry-jsonl /tmp/crab-iflow-router.telemetry.jsonl
```

Terminal 2 — start the Crab daemon:

```bash
sudo crab daemon start --foreground \
  --config examples/sdk/configs/iflow_replay_engine.runc.yaml
```

Terminal 3 — run the task and trace:

```bash
sudo --preserve-env=PYTHONPATH \
  PYTHONPATH=. python3 examples/sdk/02_iflow_replay.py \
  --task-root "$TASK_ROOT" \
  --trace "$TRACE"
```

The script registers the sandbox with the replay router, translates the task's
Compose service into a runc sandbox, runs all recorded tool calls, prints the
checkpoints created at LLM request boundaries, runs the task's
`/tests/run-tests.sh`, and unregisters the trace on exit. The tracked iFlow
runtime archives are packaged with Crab, so the replay does not install a live
iFlow release or require an iFlow account.

Keep the sandbox and its checkpoints for CLI inspection:

```bash
sudo --preserve-env=PYTHONPATH \
  PYTHONPATH=. python3 examples/sdk/02_iflow_replay.py \
  --task-root "$TASK_ROOT" \
  --trace "$TRACE" \
  --keep

sudo crab checkpoint ls sdk-iflow-replay-crack-7z-hash
```

You can then prove that the checkpoint following the final `write_file` tool
call contains the solution: overwrite `/app/solution.txt`, restore the latest
filesystem checkpoint shown by `checkpoint ls`, and read the file again.

```bash
sudo crab sandbox exec sdk-iflow-replay-crack-7z-hash -- \
  sh -lc 'echo corrupted > /app/solution.txt'
sudo crab restore sdk-iflow-replay-crack-7z-hash <checkpoint-id>
sudo crab sandbox exec sdk-iflow-replay-crack-7z-hash -- \
  cat /app/solution.txt
```

For `crack-7z-hash`, the restored value must be `honeybear`. Remove the kept
sandbox with `sudo crab sandbox rm sdk-iflow-replay-crack-7z-hash`.

Runtime state and logs use `/var/lib/crab/examples/iflow-replay`; the ZFS
datasets use the explicitly configured `crab/iflow-replay` prefix.

Stop the daemon after the example:

```bash
sudo crab daemon stop
```
