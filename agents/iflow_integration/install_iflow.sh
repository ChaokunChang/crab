#!/bin/sh
set -eu

CACHE_DIR="${AGENT_CR_IFLOW_CACHE_DIR:-/opt/cache}"
NODE_PKG="${CACHE_DIR}/node-v22.18.0-linux-x64.tar.xz"
IFLOW_PKG="${CACHE_DIR}/iflow-ai-iflow-cli-for-roll-0-4-4-v4.tgz"
NODE_ROOT="/opt/nodejs"
SETTINGS_DIR="${HOME:-/root}/.iflow"
SETTINGS_PATH="${SETTINGS_DIR}/settings.json"

if [ ! -f "${NODE_PKG}" ]; then
  echo "missing node package: ${NODE_PKG}" >&2
  exit 1
fi

if [ ! -f "${IFLOW_PKG}" ]; then
  echo "missing iflow package: ${IFLOW_PKG}" >&2
  exit 1
fi

rm -rf "${NODE_ROOT}"
mkdir -p /opt "${SETTINGS_DIR}"
tar --no-same-owner --no-same-permissions -xf "${NODE_PKG}" -C /opt/
mv /opt/node-v22.18.0-linux-* "${NODE_ROOT}"
ln -sf "${NODE_ROOT}/bin/node" /usr/local/bin/node
ln -sf "${NODE_ROOT}/bin/npm" /usr/local/bin/npm
ln -sf "${NODE_ROOT}/bin/npx" /usr/local/bin/npx
ln -sf "${NODE_ROOT}/bin/corepack" /usr/local/bin/corepack

npm config set fund false
npm config set audit false
npm install -g "${IFLOW_PKG}" --verbose

NPM_PREFIX="$(npm config get prefix)"
IFLOW_BIN="${NPM_PREFIX}/bin/iflow"
if [ ! -x "${IFLOW_BIN}" ] && [ -x "${NODE_ROOT}/bin/iflow" ]; then
  IFLOW_BIN="${NODE_ROOT}/bin/iflow"
fi
if [ ! -x "${IFLOW_BIN}" ]; then
  echo "installed iflow binary not found under ${NPM_PREFIX}/bin or ${NODE_ROOT}/bin" >&2
  exit 1
fi
ln -sf "${IFLOW_BIN}" /usr/local/bin/iflow

cat > "${SETTINGS_PATH}" <<EOF
{
  "selectedAuthType": "openai-compatible",
  "apiKey": "${AGENT_CR_IFLOW_API_KEY:-sk-agent-cr-iflow}",
  "baseUrl": "${AGENT_CR_IFLOW_BASE_URL:-http://172.17.0.1:8081/v1}",
  "modelName": "${AGENT_CR_IFLOW_MODEL_NAME:-agent-cr-iflow-scripted}",
  "bootAnimationShown": true,
  "disableAutoUpdate": true,
  "maxSessionTurns": ${AGENT_CR_IFLOW_MAX_SESSION_TURNS:-32},
  "approvalMode": "yolo",
  "mcpServers": {}
}
EOF
