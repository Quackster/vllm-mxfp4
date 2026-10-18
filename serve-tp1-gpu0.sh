#!/bin/bash
# serve-tp1-gpu0.sh -- ONE card, TP=1, pinned to HIP index 0, as its own container.
#
# The dual-card companion of serve-tp1.sh: that one serves on whichever single card
# gpu-detect.sh picks first, this one names the card so a second container can take the
# other. serve-tp1-gpu1.sh is the same thing on index 1; serve-tp1-dual.sh brings up both.
#
# What this pins, and why each one has to be pinned:
#   TP=1           one card per container; the single-GPU profile comes on automatically
#                  (fp16 ssm cache, MAXSEQS 3, MAXLEN 220000, CHUNK 2560, libr4d rx9)
#   GPUS=0         ROCR_VISIBLE_DEVICES=0, HIP_VISIBLE_DEVICES=0 (index into the filtered list)
#   NAME=...       a distinct container name -- the launcher's default name would make each
#                  start replace the OTHER container (podman --replace / docker rm -f)
#   PORT=8080      --network=host, so the two containers must listen on different ports
#   CACHE=~/.radiance-cache-gpu0
#                  a distinct cache root. The launcher appends -tp1s at TP=1 (the fp8-stream
#                  epilogue without an all-reduce keys the traced graph), so the directory
#                  that actually gets used is ~/.radiance-cache-gpu0-tp1s. Cache dirs are
#                  keyed on the exact configuration and MUST NOT be shared between serves.
#   RUNTIME=docker
#                  the pulled image lives in docker's store, and the existing cache roots are
#                  root-owned (rootful docker). Override RUNTIME=podman if you pulled there.
#
# The checkpoint is the AMD Quark AWQ MXFP4 release as rebuilt by ./setup-mxfp4.sh
# (README: "Qwen3.8-27B-MXFP4-mtpfp8: AMD's Qwen3.8-27B-Quark-AWQ-MXFP4 body with the MTP
# head requantized to fp8"). AMD's raw release does not load as-is; the rebuilt config
# declares mtp.* in layer_quant_config and loads. The DFlash2 drafter sits next to it.
#
#   ./serve-tp1-gpu0.sh                    # foreground, http://localhost:8080/v1
#   DETACH=1 ./serve-tp1-gpu0.sh           # background (logs: docker logs -f vllm-gpu0)
#   ./serve-tp1-gpu0.sh --enforce-eager    # extra args go to `vllm serve`
#   DRY_RUN=1 ./serve-tp1-gpu0.sh          # print the container command, do not run it
#
# Every other serve-mxfp4.sh knob applies (SPEC, MAXSEQS, MAXLEN, CHUNK, GPU_UTIL, KV_MEM, ...).

exec env \
  TP=1 \
  GPUS=0 \
  RUNTIME="${RUNTIME:-docker}" \
  NAME="${NAME:-vllm-gpu0}" \
  PORT="${PORT:-8080}" \
  MODELS="${MODELS:-$HOME/models}" \
  SNAP="${SNAP:-$HOME/models/Qwen3.8-27B-MXFP4-mtpfp8}" \
  CACHE="${CACHE:-$HOME/.radiance-cache-gpu0}" \
  "$(cd "$(dirname "$0")" && pwd)/serve-mxfp4.sh" "$@"
