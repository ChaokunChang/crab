# Getting started

This walkthrough uses no model API key. It exercises the same real runc,
CRIU, and ZFS path used by agent sandboxes.

Install Crab first:

```bash
sudo ./scripts/install-ubuntu.sh
```

## Start the daemon

```bash
sudo crab daemon start --config /etc/crab/config.yaml
sudo crab info
```

The daemon is root-owned in v0, so the CLI and SDK must use the same user and
socket.

## Launch a sandbox

```bash
SBX=$(sudo crab sandbox run --detach ubuntu:22.04)
sudo crab sandbox ls
```

CLI options currently go before the image name. For example, use
`crab sandbox run --detach ubuntu:22.04`, not
`crab sandbox run ubuntu:22.04 --detach`.

Create a file and a long-running process:

```bash
sudo crab sandbox exec "$SBX" -- sh -lc \
  'echo before > /root/state.txt; nohup sleep 100000 >/root/worker.log 2>&1 & echo $! >/root/worker.pid'
```

## Create and inspect a checkpoint

```bash
CKPT=$(sudo crab checkpoint create "$SBX")
sudo crab checkpoint ls "$SBX"
```

The list shows whether each manifest contains process and filesystem state.

## Mutate and restore

```bash
sudo crab sandbox exec "$SBX" -- sh -lc \
  'echo after > /root/state.txt; kill "$(cat /root/worker.pid)"'

sudo crab restore "$SBX" "$CKPT"
sudo crab sandbox exec "$SBX" -- cat /root/state.txt
sudo crab sandbox exec "$SBX" -- sh -lc 'kill -0 "$(cat /root/worker.pid)"'
```

The file prints `before`, and `kill -0` succeeds because the saved process was
restored.

## Clean up

```bash
sudo crab sandbox rm "$SBX"
sudo crab daemon stop
```

## Files that are not rolled back

Do not use `--work-dir` for this recovery test. That option bind-mounts a host
directory at `/work`, outside the sandbox's ZFS root filesystem. Host bind
mounts and external services are not part of a Crab checkpoint.

To protect a project in v0, clone or copy it into the sandbox root filesystem
and let the agent work on that internal copy.
