# SDK iFlow Replay Example

This shows the SDK path using the existing benchmark LLM router as an
external replay service. The replay server is not part of the sandbox
example; it is a host-side service, like a local OpenAI-compatible endpoint.

## 1. Start The Replay Router

In terminal 1, start the existing benchmark LLM router from the repo root:

```bash
cd /root/workspace/acr-deploy/agent-cr

PYTHONPATH=. python3 -m integrations.llm_services.router \
  --host 127.0.0.1 \
  --port 18080 \
  --telemetry-jsonl /tmp/agentcr-iflow-router.telemetry.jsonl
```

Leave this process running. In another shell, check that it is ready:

```bash
curl -s http://127.0.0.1:18080/healthz
```

This is the same host-side router that the benchmark harness starts in
`benchmarks/real_host_scenario_base.py`; the SDK example just connects to it
as an external OpenAI-compatible service.

The router exposes:

- `POST /v1/chat/completions` for iFlow's OpenAI-compatible calls.
- `POST /control/register` to bind a sandbox id to an `iflow_trace_replay`
  trace.
- `GET /control/state?sandbox_id=...` to inspect replay progress.
- `POST /control/unregister` to remove a sandbox registration.

The SDK example registers its two sandbox ids with `/control/register`
before launching them, then unregisters them on exit.

If you prefer a background process in your own shell, redirect the logs
explicitly:

```bash
cd /root/workspace/acr-deploy/agent-cr

PYTHONPATH=. python3 -m integrations.llm_services.router \
  --host 127.0.0.1 \
  --port 18080 \
  --telemetry-jsonl /tmp/agentcr-iflow-router.telemetry.jsonl \
  > /tmp/agentcr-iflow-router.log 2>&1 &

echo $! > /tmp/agentcr-iflow-router.pid
```

## 2. Run The SDK Example

In terminal 2:

```bash
cd /root/workspace/acr-deploy/agent-cr

AGENT_CR_REPLAY_BASE_URL=http://127.0.0.1:18080 \
  PYTHONPATH=. python3 examples/sdk/06_iflow_replay_dataset_runc.py
```

The example starts the in-process runc engine from:

```text
examples/sdk/configs/iflow_replay_engine.runc.yaml
```

Override that with:

```bash
AGENT_CR_ENGINE_CONFIG=/path/to/engine.yaml
```

The example reads the first two rows from:

```text
/root/workspace/agent-cr/results/datasets/termnius_iflow_replay_128tasks_light.jsonl
```

Override that with:

```bash
AGENT_CR_REPLAY_DATASET=/path/to/dataset.jsonl
```

Expected success signal:

- `sdk-iflow-replay-0-analyze-access-logs`: agent exit `0`, verifier exit `0`,
  replay progress `7/7`.
- `sdk-iflow-replay-1-crack-7z-hash`: agent exit `0`, verifier exit `0`,
  replay progress `23/23`.

## 3. Stop The Replay Router

If the router is running in the foreground, stop it with `Ctrl-C`.

If you used the background command:

```bash
kill "$(cat /tmp/agentcr-iflow-router.pid)"
rm -f /tmp/agentcr-iflow-router.pid
```

## Telemetry And Logs

For this external router process:

- Router stdout/stderr stay in terminal 1 when run in the foreground.
- With the background command above, router stdout/stderr are redirected to
  `/tmp/agentcr-iflow-router.log`.
- Router telemetry JSONL is written to
  `/tmp/agentcr-iflow-router.telemetry.jsonl` because the start command passes
  `--telemetry-jsonl`.

For the SDK engine inside the example:

- This example uses
  `examples/sdk/configs/iflow_replay_engine.runc.yaml`, so engine telemetry is
  written to
  `/root/workspace/agent-cr/logs/sdk/iflow-replay-runc/engine.telemetry.jsonl`.
- Engine telemetry is stamped with `run_id: sdk-iflow-replay-runc`, which lets
  the existing telemetry report CLI analyze it.
- Engine logs are written to
  `/root/workspace/agent-cr/logs/sdk/iflow-replay-runc/engine.log`.
- Runtime/checkpoint/agent state is kept under
  `/root/workspace/agent-cr/data/agent_cr/sdk/iflow-replay-runc/`.
- The runc `--root` path is
  `/root/workspace/agent-cr/data/agent_cr/sdk/iflow-replay-runc/runtime/runtime-state`.
- Sandbox ZFS datasets use the prefix
  `agentcr-300/agent-cr-sdk-iflow-replay`.
- The config's `host_inspector.launch_mode: in_process` documents the current
  SDK behavior: the in-process engine uses `EBPFSandboxInspector` directly.
  The benchmark host-inspector process launcher is still benchmark-only.

Generate the same style of telemetry report bundle manually with:

```bash
cd /root/workspace/acr-deploy/agent-cr

PYTHONPATH=. python3 -m benchmarks.telemetry_analysis.report \
  --input /root/workspace/agent-cr/logs/sdk/iflow-replay-runc/engine.telemetry.jsonl \
  --output-dir /root/workspace/agent-cr/logs/sdk/iflow-replay-runc/report \
  --top-k 50 \
  --figure-window-seconds 10
```
- With plain `EngineConfig(runtime="runc")` and no YAML file, telemetry falls
  back to an in-memory sink and runtime state defaults under a temporary
  `/tmp/agentcr-engine-*/` directory.
- To persist SDK engine telemetry, pass a `TelemetryConfig`:

```python
from pathlib import Path
from agent_cr import EngineConfig, TelemetryConfig

config = EngineConfig(
    runtime="runc",
    telemetry_config=TelemetryConfig(
        jsonl_path=Path("/tmp/agentcr-sdk.telemetry.jsonl"),
        keep_in_memory_copy=True,
    ),
)
```

Benchmark runs are different: `benchmarks/run.py` wires log files, CSV output,
router logs, and telemetry paths from YAML, for example
`telemetry.output`, `log_file`, and `output`.
