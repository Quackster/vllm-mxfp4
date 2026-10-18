#!/bin/bash
# serve-tp1-gpu1.sh -- ONE card, TP=1, pinned to HIP index 1, as its own container.
#
# The second half of the dual-card setup; see serve-tp1-gpu0.sh for the full rationale. The
# two differ ONLY in the HIP index, the container name, the port and the cache root -- which
# is exactly the set that must differ, and nothing else, so both serves are the same measured
# TP=1 single-GPU profile.
#
#   HIP index      GPUS=1 -> ROCR_VISIBLE_DEVICES=1, HIP_VISIBLE_DEVICES=0 (indexes INTO the
#                  filtered list; the launcher derives it, so "1,1" cannot happen)
#   port           8081 -- 8080 is gpu0's; the containers share the host network namespace
#   cache root     ~/.radiance-cache-gpu1, effective ~/.radiance-cache-gpu1-tp1s (-tp1s is
#                  appended by the launcher at TP=1)
#
# HOST MEMORY: this box has 60 GiB of RAM. Loading a 19.4 GiB checkpoint plus the drafter in
# each container at the same time OOMs the host, so bring this one up only after gpu0 is
# healthy -- serve-tp1-dual.sh does that for you. Once both are resident they coexist.
#
#   ./serve-tp1-gpu1.sh                    # foreground, http://localhost:8081/v1
#   DETACH=1 ./serve-tp1-gpu1.sh           # background (logs: docker logs -f vllm-mxfp4-gpu1)
#   ./serve-tp1-gpu1.sh --enforce-eager    # extra args go to `vllm serve`
#   DRY_RUN=1 ./serve-tp1-gpu1.sh          # print the container command, do not run it

exec env \
  TP=1 \
  GPUS=1 \
  RUNTIME="${RUNTIME:-docker}" \
  NAME="${NAME:-vllm-mxfp4-gpu1}" \
  PORT="${PORT:-8081}" \
  MODELS="${MODELS:-$HOME/models}" \
  SNAP="${SNAP:-$HOME/models/Qwen3.8-27B-MXFP4-mtpfp8}" \
  CACHE="${CACHE:-$HOME/.radiance-cache-gpu1}" \
  "$(cd "$(dirname "$0")" && pwd)/serve-mxfp4.sh" "$@"
