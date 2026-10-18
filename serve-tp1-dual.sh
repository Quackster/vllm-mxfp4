#!/bin/bash
# serve-tp1-dual.sh -- BOTH cards, one TP=1 container each: gpu0 on :8080, gpu1 on :8081.
#
# This is the two-card counterpart of serve-tp1.sh. Each card runs the same measured TP=1
# single-GPU profile (fp16 ssm cache, MAXSEQS 3, MAXLEN 220000, CHUNK 2560, libr4d rx9, the
# fp8 residual stream), in its own container, with its own name and its own cache root:
#
#   vllm-mxfp4-gpu0   :8080   HIP 0   ~/.radiance-cache-gpu0-tp1s
#   vllm-mxfp4-gpu1   :8081   HIP 1   ~/.radiance-cache-gpu1-tp1s
#
# STAGGERED ON PURPOSE. The host has 60 GiB of RAM and each server reads a 19.4 GiB checkpoint
# plus the 2 GiB drafter through its own container; both loads at once OOM the host (the same
# note is in ~/models/README.md for the previous g0a/g1a pair). So gpu0 starts, is polled to
# /health, and only then gpu1 starts. A launch that is already healthy is not restarted.
#
#   ./serve-tp1-dual.sh                 # start both (first start compiles, allow ~15 min)
#   ./serve-tp1-dual.sh stop            # stop and remove both containers
#   HEALTH_TIMEOUT=3600 ./serve-tp1-dual.sh
#   PORT0=9300 PORT1=9301 ./serve-tp1-dual.sh    # override the ports (must differ)
#
# Both endpoints serve the same model names, so clients pick a card by port:
#   curl http://localhost:8080/v1/models
#   curl http://localhost:8081/v1/models
# Logs:  docker logs -f vllm-mxfp4-gpu0   /   docker logs -f vllm-mxfp4-gpu1

set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
RUNTIME=${RUNTIME:-docker}
PORT0=${PORT0:-8080}
PORT1=${PORT1:-8081}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-1800}
NAME0=vllm-mxfp4-gpu0
NAME1=vllm-mxfp4-gpu1

if [ "${1:-}" = stop ]; then
  for n in "$NAME0" "$NAME1"; do
    "$RUNTIME" stop "$n" >/dev/null 2>&1 || true
    "$RUNTIME" rm -f "$n" >/dev/null 2>&1 || true
    echo "[dual] stopped $n"
  done
  exit 0
fi

healthy() { curl -fsS -o /dev/null --max-time 2 "http://127.0.0.1:$1/health"; }

wait_healthy() {
  local port=$1 name=$2 waited=0
  printf '[dual] waiting for %s on :%s ' "$name" "$port"
  while ! healthy "$port"; do
    if [ "$waited" -ge "$HEALTH_TIMEOUT" ]; then
      echo
      echo "[dual] ERROR: $name not healthy after ${HEALTH_TIMEOUT}s; last log lines:" >&2
      "$RUNTIME" logs --tail 40 "$name" >&2 || true
      exit 1
    fi
    sleep 5
    waited=$((waited + 5))
    [ $((waited % 30)) -eq 0 ] && printf '.'
  done
  echo " ready (${waited}s)"
}

bring_up() {
  local script=$1 port=$2 name=$3
  if healthy "$port"; then
    echo "[dual] $name already healthy on :$port -- leaving it alone"
    return 0
  fi
  echo "[dual] starting $name on :$port"
  DETACH=1 PORT="$port" "$here/$script"
  wait_healthy "$port" "$name"
}

bring_up serve-tp1-gpu0.sh "$PORT0" "$NAME0"
bring_up serve-tp1-gpu1.sh "$PORT1" "$NAME1"

echo
echo "[dual] both cards are serving:"
echo "  gpu0  http://localhost:$PORT0/v1   ($NAME0)"
echo "  gpu1  http://localhost:$PORT1/v1   ($NAME1)"
echo "  stop  ./serve-tp1-dual.sh stop"
