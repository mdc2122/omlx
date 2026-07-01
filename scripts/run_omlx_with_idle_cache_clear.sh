#!/bin/bash
# Launch oMLX server (port 8000) and clear model cache after a period of inactivity.
#
# oMLX has no built-in idle-restart. This wrapper starts the server and runs a
# background watchdog that polls /v1/models/status. When a loaded model has been
# idle (no new request) for IDLE_SECONDS, it POSTs /v1/models/{id}/unload, which
# tears down the engine + scheduler and frees Metal memory -> clears the model's
# in-memory KV / prefix cache. The model auto-reloads on the next request.
#
# This keeps GLM's KV/prefix cache from growing unbounded across idle gaps while
# preserving warm prefix-cache hits during active agent sessions.

set -u

OMLX_DIR="/Users/studio2/omlx"
OMLX_BIN="${OMLX_DIR}/.venv/bin/omlx"
MODEL_DIR="/Users/studio2/.mlx-models"
CACHE_DIR="${HOME}/.omlx/cache"
HOST="127.0.0.1"
PORT="8000"
HOT_CACHE_MAX_SIZE="${OMLX_HOT_CACHE_MAX_SIZE:-140GB}"
BASE_URL="http://${HOST}:${PORT}"

# Set OMLX_PURGE_CACHE=1 before restart to move aside stale SSD prefix-cache blocks
# (e.g. after switching model quant or mlx-lm/oMLX patches).
OMLX_PURGE_CACHE="${OMLX_PURGE_CACHE:-0}"

# Idle threshold before unloading the model (seconds). Default 86400 = 24 h.
# NOTE: unload frees the WEIGHTS (306 GB) too — next request pays a multi-minute
# reload. With the hot cache bounded (140 GB) there is no cache-growth reason to
# unload aggressively; this is only a safety valve for multi-day idle stretches.
IDLE_SECONDS="${OMLX_IDLE_SECONDS:-86400}"
# How often the watchdog checks (seconds).
POLL_SECONDS="${OMLX_POLL_SECONDS:-60}"

LOG="/tmp/omlx-glm52-server.log"

purge_cache_if_requested() {
    if [ "${OMLX_PURGE_CACHE}" != "1" ]; then
        return 0
    fi
    local stamp backup
    stamp=$(date '+%Y%m%d-%H%M%S')
    backup="${HOME}/.omlx/cache.purged-${stamp}"
    if [ -d "${CACHE_DIR}" ]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [wrapper] OMLX_PURGE_CACHE=1 -> moving ${CACHE_DIR} to ${backup}" >> "${LOG}"
        mv "${CACHE_DIR}" "${backup}"
        mkdir -p "${CACHE_DIR}"
    fi
}

check_cache_health_from_log() {
    # Only inspect log lines from this wrapper invocation (ignore historical stale-cache events).
    local recent
    recent=$(awk '/\[wrapper\] starting oMLX serve/{buf=$0; next} {buf=buf ORS $0} END{print buf}' "${LOG}" 2>/dev/null)
    if echo "${recent}" | grep -qE 'skipped_incompatible=[1-9][0-9]* blocks|Failed to load block|Partial cache reconstruction'; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [wrapper] WARNING: prefix cache unhealthy (incompatible/stale blocks in log). Purge with: OMLX_PURGE_CACHE=1 launchctl kickstart -k gui/\$(id -u)/com.studio2.omlx-server" >> "${LOG}"
    fi
}

cd "${OMLX_DIR}" || exit 1

purge_cache_if_requested

echo "$(date '+%Y-%m-%d %H:%M:%S') [wrapper] starting oMLX serve on ${BASE_URL} (idle-clear=${IDLE_SECONDS}s hot-cache=${HOT_CACHE_MAX_SIZE})" >> "${LOG}"

# Start the oMLX server.
"${OMLX_BIN}" serve \
    --model-dir "${MODEL_DIR}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --hot-cache-max-size "${HOT_CACHE_MAX_SIZE}" \
    >> "${LOG}" 2>&1 &
SERVER_PID=$!

# Make sure the server is killed if this wrapper exits (launchd KeepAlive restarts it).
cleanup() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [wrapper] shutting down (server pid ${SERVER_PID})" >> "${LOG}"
    kill "${SERVER_PID}" 2>/dev/null
    wait "${SERVER_PID}" 2>/dev/null
}
trap cleanup EXIT INT TERM

# Wait for the server to come up.
for _ in $(seq 1 60); do
    if curl -sf -m 5 "${BASE_URL}/v1/models" >/dev/null 2>&1; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [wrapper] server is up" >> "${LOG}"
        check_cache_health_from_log
        break
    fi
    # If the server died during startup, exit so launchd can restart us.
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [wrapper] server exited during startup" >> "${LOG}"
        exit 1
    fi
    sleep 2
done

# Idle watchdog loop.
while kill -0 "${SERVER_PID}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"

    NOW=$(date +%s)
    STATUS_JSON=$(curl -sf -m 10 "${BASE_URL}/v1/models/status" 2>/dev/null)
    [ -z "${STATUS_JSON}" ] && continue

    # For each loaded model idle longer than IDLE_SECONDS, request an unload.
    echo "${STATUS_JSON}" | IDLE_SECONDS="${IDLE_SECONDS}" NOW="${NOW}" python3 -c '
import json, os, sys
data = json.load(sys.stdin)
idle = float(os.environ["IDLE_SECONDS"])
now = float(os.environ["NOW"])
for m in data.get("models", []):
    if not m.get("loaded"):
        continue
    la = m.get("last_access")
    if not la:
        continue
    if now - float(la) >= idle:
        print(m["id"])
' | while read -r MODEL_ID; do
        [ -z "${MODEL_ID}" ] && continue
        echo "$(date '+%Y-%m-%d %H:%M:%S') [wrapper] model '${MODEL_ID}' idle >= ${IDLE_SECONDS}s -> unloading to clear cache" >> "${LOG}"
        curl -sf -m 60 -X POST "${BASE_URL}/v1/models/$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=''))" "${MODEL_ID}")/unload" >> "${LOG}" 2>&1
        echo "" >> "${LOG}"
    done
done

echo "$(date '+%Y-%m-%d %H:%M:%S') [wrapper] server process ended" >> "${LOG}"
