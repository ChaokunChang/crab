# Host Inspector

This package contains the host-side inspector server for sandbox process/filesystem change detection.

## Prerequisites

- Linux host
- root
- Docker
- build deps for `agent_cr/host_inspector/bpf`
- test image `agent-sandbox-bench:latest`

Build the image if needed:

```bash
docker build -t agent-sandbox-bench:latest -f simulated_agent/Dockerfile simulated_agent
```

Build the eBPF helper:

```bash
make -C agent_cr/host_inspector/bpf
```

## Start The Server

```bash
python3 -m agent_cr.host_inspector --port 9782 --process-poll-interval 0.1 --log-level INFO
```

`--process-poll-interval` is kept for CLI compatibility, but `process_changed` is evaluated on demand when `/get_proc_and_fs_status` is queried. It no longer drives a sticky polling result.

## Launch A Test Container

```bash
docker run -d --name hi-demo agent-sandbox-bench:latest sleep 10000000
```

Use the same value for `sandbox_id` and container name unless you want a different external identifier:

```bash
export HOST_INSPECTOR_URL=http://127.0.0.1:9782
export SANDBOX_ID=hi-demo
export CONTAINER_ID=hi-demo
```

## Register / Reset / Status / Unregister

Register:

```bash
curl -s -X POST "$HOST_INSPECTOR_URL/register" \
  -H 'Content-Type: application/json' \
  -d "{\"sandbox_id\":\"$SANDBOX_ID\",\"runtime\":\"docker\",\"object_id\":\"$CONTAINER_ID\"}"
```

Reset:

```bash
curl -s -X POST "$HOST_INSPECTOR_URL/reset" \
  -H 'Content-Type: application/json' \
  -d "{\"sandbox_id\":\"$SANDBOX_ID\"}"
```

Get status:

```bash
curl -s -X POST "$HOST_INSPECTOR_URL/get_proc_and_fs_status" \
  -H 'Content-Type: application/json' \
  -d "{\"sandbox_id\":\"$SANDBOX_ID\"}"
```

Unregister:

```bash
curl -s -X POST "$HOST_INSPECTOR_URL/unregister" \
  -H 'Content-Type: application/json' \
  -d "{\"sandbox_id\":\"$SANDBOX_ID\"}"
```

Watch:

```bash
python3 -m agent_cr.host_inspector.watch --base-url "$HOST_INSPECTOR_URL" "$SANDBOX_ID"
```

## Manual Command Matrix

Call `/reset` before each command below. Expected result is shown as `process_changed/filesystem_changed`.

- stdout only: `docker exec "$CONTAINER_ID" sh -lc 'echo 123; sleep 1'` -> `True/False`
- stderr only: `docker exec "$CONTAINER_ID" sh -lc 'echo err 1>&2; sleep 1'` -> `True/False`
- read only: `docker exec "$CONTAINER_ID" sh -lc 'cat /etc/hostname; sleep 1'` -> `True/False`
- memory only: `docker exec "$CONTAINER_ID" python3 -B -c "import time; buf=bytearray(8*1024*1024); buf[4096]=1; time.sleep(2)"` -> `True/False`
- mkdir in `/tmp`: `docker exec "$CONTAINER_ID" sh -lc 'mkdir -p /tmp/hi-dir; sleep 1'` -> `True/True`
- file write in `/tmp`: `docker exec "$CONTAINER_ID" sh -lc 'echo 123 >/tmp/hi-out.txt; sleep 1'` -> `True/True`
- file write in `/root`: `docker exec "$CONTAINER_ID" sh -lc 'echo 123 >/root/hi-root.txt; sleep 1'` -> `True/True`
- file write in `/workspace`: `docker exec "$CONTAINER_ID" sh -lc 'mkdir -p /workspace && echo 123 >/workspace/hi-workspace.txt; sleep 1'` -> `True/True`
- rename: `docker exec "$CONTAINER_ID" sh -lc 'echo abc >/tmp/hi-a.txt && mv /tmp/hi-a.txt /tmp/hi-b.txt; sleep 1'` -> `True/True`
- hard link: `docker exec "$CONTAINER_ID" sh -lc 'echo abc >/tmp/hi-link-src.txt && ln /tmp/hi-link-src.txt /tmp/hi-link-hard.txt; sleep 1'` -> `True/True`
- soft link: `docker exec "$CONTAINER_ID" sh -lc 'echo abc >/tmp/hi-symlink-src.txt && ln -s /tmp/hi-symlink-src.txt /tmp/hi-link-soft.txt; sleep 1'` -> `True/True`
- rm in `/tmp`: `docker exec "$CONTAINER_ID" sh -lc 'echo abc >/tmp/hi-rm.txt && rm /tmp/hi-rm.txt; sleep 1'` -> `True/True`
- rm in `/workspace`: `docker exec "$CONTAINER_ID" sh -lc 'mkdir -p /workspace && echo abc >/workspace/hi-rm.txt && rm /workspace/hi-rm.txt; sleep 1'` -> `True/True`
- rmdir in `/tmp`: `docker exec "$CONTAINER_ID" sh -lc 'mkdir -p /tmp/hi-rmdir && rmdir /tmp/hi-rmdir; sleep 1'` -> `True/True`
- rmdir in `/root`: `docker exec "$CONTAINER_ID" sh -lc 'mkdir -p /root/hi-rmdir && rmdir /root/hi-rmdir; sleep 1'` -> `True/True`

After each command, query `/get_proc_and_fs_status` or use `watch`.

## Test Commands

Real correctness matrix:

```bash
python3 -m unittest \
  tests.test_host_inspector_real_integration.HostInspectorRealIntegrationTests.test_real_docker_command_matrix_for_process_and_filesystem_changes
```

Perf test:

```bash
AGENT_CR_RUN_PERF=1 python3 -m unittest \
  tests.test_host_inspector_real_integration.HostInspectorRealIntegrationTests.test_real_docker_remote_inspect_latency
```

## Cleanup

Remove the container:

```bash
docker rm -f "$CONTAINER_ID"
```

Remove common test artifacts if needed:

```bash
docker exec "$CONTAINER_ID" sh -lc 'rm -rf /tmp/hi-* /root/hi-* /workspace/hi-*'
```
