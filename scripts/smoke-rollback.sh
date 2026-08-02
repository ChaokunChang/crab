#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG_PATH=/etc/crab/config.yaml
IMAGE=ubuntu:22.04
KEEP=0

usage() {
  echo "Usage: sudo $0 [--config PATH] [--image IMAGE] [--keep]"
}

die() {
  echo "smoke-rollback.sh: $*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --config)
      (($# >= 2)) || die "--config requires a value"
      CONFIG_PATH=$2
      shift 2
      ;;
    --image)
      (($# >= 2)) || die "--image requires a value"
      IMAGE=$2
      shift 2
      ;;
    --keep)
      KEEP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ ${EUID} -eq 0 ]] || die "run this smoke test with sudo"
[[ -f ${CONFIG_PATH} ]] || die "config not found: ${CONFIG_PATH}"
command -v crab >/dev/null 2>&1 || die "crab is not on PATH"

started_daemon=0
sandbox_id=

cleanup() {
  rc=$?
  if [[ -n ${sandbox_id} ]] && ((KEEP == 0)); then
    crab sandbox rm "${sandbox_id}" >/dev/null 2>&1 || true
  fi
  if ((started_daemon == 1)); then
    crab daemon stop >/dev/null 2>&1 || true
  fi
  exit "${rc}"
}
trap cleanup EXIT INT TERM

if ! crab daemon status >/dev/null 2>&1; then
  echo "==> Starting Crab daemon"
  crab daemon start --config "${CONFIG_PATH}"
  started_daemon=1
fi

echo "==> Launching sandbox from ${IMAGE}"
sandbox_id=$(crab sandbox run --detach --name "crab-smoke-$$" "${IMAGE}")

echo "==> Creating filesystem and process state"
crab sandbox exec "${sandbox_id}" -- sh -lc \
  'echo before > /root/crab-state.txt; nohup sh -c "while :; do sleep 1; done" >/root/crab-worker.log 2>&1 & echo $! >/root/crab-worker.pid'
before_pid=$(crab sandbox exec "${sandbox_id}" -- cat /root/crab-worker.pid)

echo "==> Taking full checkpoint"
checkpoint_id=$(crab checkpoint create "${sandbox_id}")
[[ -n ${checkpoint_id} ]] || die "checkpoint command returned an empty id"

echo "==> Mutating state after ${checkpoint_id}"
crab sandbox exec "${sandbox_id}" -- sh -lc \
  'echo after > /root/crab-state.txt; kill "$(cat /root/crab-worker.pid)"; echo replaced >/root/post-checkpoint.txt'

echo "==> Restoring ${checkpoint_id}"
crab restore "${sandbox_id}" "${checkpoint_id}" >/dev/null

restored_value=$(crab sandbox exec "${sandbox_id}" -- cat /root/crab-state.txt)
restored_pid=$(crab sandbox exec "${sandbox_id}" -- cat /root/crab-worker.pid)
crab sandbox exec "${sandbox_id}" -- kill -0 "${restored_pid}"

[[ ${restored_value} = before ]] || die "expected file value 'before', got ${restored_value@Q}"
[[ ${restored_pid} = "${before_pid}" ]] || die "expected restored pid ${before_pid}, got ${restored_pid}"
if crab sandbox exec "${sandbox_id}" -- test -e /root/post-checkpoint.txt; then
  die "post-checkpoint file still exists after restore"
fi

echo
echo "PASS: filesystem content and the background process were restored."
echo "  sandbox:   ${sandbox_id}"
echo "  checkpoint: ${checkpoint_id}"
if ((KEEP == 1)); then
  echo "  kept sandbox; remove it with: crab sandbox rm ${sandbox_id}"
fi
