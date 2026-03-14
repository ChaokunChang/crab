#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ "${EUID}" -ne 0 ]]; then
  echo "start-manual-demo.sh requires root" >&2
  exit 1
fi

WORK_ROOT="${WORK_ROOT:-/tmp/iflow-manual}"
SANDBOX_ID="${SANDBOX_ID:-sbx-iflow-manual}"
HOST_INSPECTOR_PORT="${HOST_INSPECTOR_PORT:-9782}"
MANUAL_LLM_PORT="${MANUAL_LLM_PORT:-8091}"
INTERCEPTOR_PORT="${INTERCEPTOR_PORT:-8092}"
SANDBOX_IP="${SANDBOX_IP:-172.17.0.240}"
HOST_INSPECTOR_URL="${HOST_INSPECTOR_URL:-http://127.0.0.1:${HOST_INSPECTOR_PORT}}"
MANUAL_LLM_URL="${MANUAL_LLM_URL:-http://127.0.0.1:${MANUAL_LLM_PORT}}"
INTERCEPTOR_URL="${INTERCEPTOR_URL:-http://127.0.0.1:${INTERCEPTOR_PORT}}"
SANDBOX_LLM_BASE_URL="${SANDBOX_LLM_BASE_URL:-http://172.17.0.1:${INTERCEPTOR_PORT}/v1}"
TASK_DESCRIPTION="${TASK_DESCRIPTION:-Wait for the next tool instruction, execute it exactly once, then ask for the next instruction.}"

LOG_DIR="$WORK_ROOT/logs"
HOST_INSPECTOR_LOG="$LOG_DIR/host-inspector.log"
MANUAL_LLM_LOG="$LOG_DIR/manual-llm.log"
INTERCEPTOR_LOG="$LOG_DIR/interceptor.log"
SESSION_SUMMARY="$WORK_ROOT/session-summary.json"
KEEP_WORK_ROOT="${KEEP_WORK_ROOT:-0}"

HOST_INSPECTOR_PID=""
MANUAL_LLM_PID=""
INTERCEPTOR_PID=""
STARTED_DEMO="0"

mkdir -p "$LOG_DIR"

wait_for_http_json() {
  local url="$1"
  local label="$2"
  python3 - "$url" "$label" <<'PY'
import json
import sys
import time
import urllib.request

url = sys.argv[1]
label = sys.argv[2]
deadline = time.time() + 30.0
last_error = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            json.loads(response.read().decode("utf-8"))
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        last_error = exc
        time.sleep(0.2)
raise SystemExit(f"timed out waiting for {label} at {url}: {last_error}")
PY
}

cleanup() {
  local exit_code="$?"
  set +e
  if [[ -f "$WORK_ROOT/manual_session.json" ]]; then
    python3 -m agents.iflow_integration stop-manual-sandbox \
      --work-root "$WORK_ROOT" >/dev/null 2>&1 || true
  fi
  if [[ -n "$MANUAL_LLM_PID" ]]; then
    kill "$MANUAL_LLM_PID" >/dev/null 2>&1 || true
    wait "$MANUAL_LLM_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$INTERCEPTOR_PID" ]]; then
    kill "$INTERCEPTOR_PID" >/dev/null 2>&1 || true
    wait "$INTERCEPTOR_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$HOST_INSPECTOR_PID" ]]; then
    kill "$HOST_INSPECTOR_PID" >/dev/null 2>&1 || true
    wait "$HOST_INSPECTOR_PID" >/dev/null 2>&1 || true
  fi
  if [[ "$STARTED_DEMO" == "1" && "$KEEP_WORK_ROOT" != "1" ]]; then
    rm -rf "$WORK_ROOT"
  fi
  exit "$exit_code"
}

trap cleanup EXIT INT TERM

if [[ -f "$WORK_ROOT/manual_session.json" ]]; then
  echo "existing manual session found at $WORK_ROOT/manual_session.json" >&2
  echo "stop it first with:" >&2
  echo "  python3 -m agents.iflow_integration stop-manual-sandbox --work-root \"$WORK_ROOT\"" >&2
  exit 1
fi

echo "Building host inspector helper"
make -C "$REPO_ROOT/agent_cr/host_inspector/bpf" >/dev/null

echo "Starting host inspector on $HOST_INSPECTOR_URL"
python3 -m agent_cr.host_inspector \
  --port "$HOST_INSPECTOR_PORT" \
  --runc-state-root "$WORK_ROOT/runtime-state" \
  >"$HOST_INSPECTOR_LOG" 2>&1 &
HOST_INSPECTOR_PID="$!"
wait_for_http_json "$HOST_INSPECTOR_URL/healthz" "host inspector"

echo "Starting manual LLM server on $MANUAL_LLM_URL"
python3 -m agents.iflow_integration serve-manual-llm \
  --host 0.0.0.0 \
  --port "$MANUAL_LLM_PORT" \
  >"$MANUAL_LLM_LOG" 2>&1 &
MANUAL_LLM_PID="$!"
wait_for_http_json "$MANUAL_LLM_URL/healthz" "manual llm server"

echo "Starting interceptor on $INTERCEPTOR_URL"
python3 -m agents.iflow_integration serve-manual-interceptor \
  --host 0.0.0.0 \
  --port "$INTERCEPTOR_PORT" \
  --upstream-url "$MANUAL_LLM_URL" \
  --sandbox-id "$SANDBOX_ID" \
  --sandbox-ip "$SANDBOX_IP" \
  >"$INTERCEPTOR_LOG" 2>&1 &
INTERCEPTOR_PID="$!"
wait_for_http_json "$INTERCEPTOR_URL/healthz" "interceptor"

echo "Launching manual iflow sandbox"
python3 -m agents.iflow_integration launch-manual-sandbox \
  --work-root "$WORK_ROOT" \
  --sandbox-id "$SANDBOX_ID" \
  --host-inspector-url "$HOST_INSPECTOR_URL" \
  --llm-base-url "$SANDBOX_LLM_BASE_URL" \
  --sandbox-ip "$SANDBOX_IP" \
  --task "$TASK_DESCRIPTION" \
  | tee "$SESSION_SUMMARY"
STARTED_DEMO="1"

cat <<EOF

Manual iflow demo is running.

Terminal 2:

export WORK_ROOT="$WORK_ROOT"
export HOST_INSPECTOR_URL="$HOST_INSPECTOR_URL"
export MANUAL_LLM_URL="$MANUAL_LLM_URL"
export INTERCEPTOR_URL="$INTERCEPTOR_URL"
export SANDBOX_ID="$SANDBOX_ID"

# Reset baseline before each tool call
curl -s -X POST "\$HOST_INSPECTOR_URL/reset" \\
  -H 'Content-Type: application/json' \\
  -d '{"sandbox_id":"$SANDBOX_ID"}'

# Watch inspector status
python3 -m agent_cr.host_inspector.watch \\
  --base-url "\$HOST_INSPECTOR_URL" \\
  --interval 0.5 \\
  "\$SANDBOX_ID"

# Send a transient tool call
python3 -m agents.iflow_integration enqueue-run-shell-command \\
  --base-url "\$MANUAL_LLM_URL" \\
  --sandbox-id "\$SANDBOX_ID" \\
  --command 'sh -lc "printf noop >/dev/null"'

# Send a filesystem-writing tool call
python3 -m agents.iflow_integration enqueue-run-shell-command \\
  --base-url "\$MANUAL_LLM_URL" \\
  --sandbox-id "\$SANDBOX_ID" \\
  --command 'sh -lc "mkdir -p /work/iflow-probe && printf iflow-artifact >/work/iflow-probe/artifact.txt"'

# Send a detached-daemon tool call
python3 -m agents.iflow_integration enqueue-run-shell-command \\
  --base-url "\$MANUAL_LLM_URL" \\
  --sandbox-id "\$SANDBOX_ID" \\
  --command 'sh -lc "mkdir -p /work/iflow-probe && python3 -m http.server 8123 >/work/iflow-probe/http.log 2>&1 & echo \$! >/work/iflow-probe/http.pid"'

# Send a long-running counter workload for manual checkpoint/restore observation
python3 -m agents.iflow_integration enqueue-run-shell-command \\
  --base-url "\$MANUAL_LLM_URL" \\
  --sandbox-id "\$SANDBOX_ID" \\
  --command 'bash -lc '"'"'mkdir -p /work/iflow-observe; count=0; while [ "\$count" -lt 200 ]; do count=\$((count + 1)); printf "%s\n" "\$count" >/work/iflow-observe/counter.txt; sleep 1; done'"'"''

# Checkpoint the running sandbox
python3 -m agents.iflow_integration checkpoint-manual-sandbox \\
  --work-root "\$WORK_ROOT"

# List available checkpoints
python3 -m agents.iflow_integration list-manual-checkpoints \\
  --work-root "\$WORK_ROOT"

# Restore the latest checkpoint
python3 -m agents.iflow_integration restore-manual-sandbox \\
  --work-root "\$WORK_ROOT"

# Enter the sandbox and watch the counter continue after restore
python3 -m agents.iflow_integration manual-shell \\
  --work-root "\$WORK_ROOT"

# Stop the agent
python3 -m agents.iflow_integration enqueue-final-response \\
  --base-url "\$MANUAL_LLM_URL" \\
  --sandbox-id "\$SANDBOX_ID" \\
  --content 'The session is complete. Summarize briefly and stop.'

Logs:
  host inspector: $HOST_INSPECTOR_LOG
  interceptor:    $INTERCEPTOR_LOG
  manual llm:     $MANUAL_LLM_LOG
  session summary:$SESSION_SUMMARY

Press Ctrl-C in this terminal to stop the sandbox and all background services.
Set KEEP_WORK_ROOT=1 before starting if you want to preserve $WORK_ROOT after exit.
EOF

while true; do
  if ! kill -0 "$HOST_INSPECTOR_PID" >/dev/null 2>&1; then
    echo "host inspector exited unexpectedly; see $HOST_INSPECTOR_LOG" >&2
    exit 1
  fi
  if ! kill -0 "$MANUAL_LLM_PID" >/dev/null 2>&1; then
    echo "manual llm server exited unexpectedly; see $MANUAL_LLM_LOG" >&2
    exit 1
  fi
  if ! kill -0 "$INTERCEPTOR_PID" >/dev/null 2>&1; then
    echo "interceptor exited unexpectedly; see $INTERCEPTOR_LOG" >&2
    exit 1
  fi
  sleep 2
done
