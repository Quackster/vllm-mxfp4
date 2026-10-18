# CREATE_TWO_TP1.md — two TP=1 containers, one per R9700

How to bring up **two independent vLLM servers, each on one card**: `TP=1` on HIP index 0 and
`TP=1` on HIP index 1, in separate containers, with separate cache roots, on separate ports.
Written for (and verified on) the two-card box in this repo: 2 x Radeon AI PRO R9700 (gfx1201,
32 GiB each), 60 GiB host RAM.

Everything below is `VAR=${VAR:-default}` in the scripts, so knobs can be overridden from the
environment without editing anything.

```
container           GPU (HIP)   endpoint                    cache root
vllm-gpu0     0           http://localhost:8080/v1    ~/.radiance-cache-gpu0-tp1s
vllm-gpu1     1           http://localhost:8081/v1    ~/.radiance-cache-gpu1-tp1s
```

Both serve the same model names (`Qwen3.8`, `Qwen3.6`, `Qwen3.8-MXFP4`) — clients pick a card by
port. The two serves differ **only** in the four values above; everything else (the measured TP=1
profile) is identical by construction.

---

## 1. Prerequisites

* amdgpu driver loaded: `/dev/kfd` and `/dev/dri` exist, both cards on `gfx1201`.
* `docker` (or `podman`; see the runtime note in §4). The wrappers default to **docker**, because
  that is where the image is pulled and where the existing cache roots live.
* This repo checked out on the host (the container bind-mounts it at `/patches`).
* ~45 GiB free disk: ~13.7 GiB image, ~19.4 GiB served checkpoint, ~2 GiB drafter.
* Host RAM: **60 GiB is enough for both servers resident, but not for both loads at once** —
  see §6, the launcher staggers them.

---

## 2. Models — download into `~/models/` (skip if already there)

Two checkpoints are needed. Both are public, no token required.

| what | source | size |
|---|---|---|
| AMD Quark AWQ MXFP4 release (the body) | <https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4> | ~19 GiB |
| DFlash2 FP8 speculative drafter | <https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8> | ~2 GiB |

**AMD's raw release does not load as-is.** Its `quantization_config.exclude` names the bf16
`mtp.*` layers as *tensor* names (`mtp.fc.weight`) among *module* names; quark matches modules,
so the exclusion never fires, vLLM applies the mxfp4 scheme to a bf16 MTP head and asserts on a
half-width parameter. The supported form is the rebuilt checkpoint the launcher defaults to:

```
~/models/Qwen3.8-27B-MXFP4-mtpfp8/        <- SNAP      (AMD's body, MTP head requantized to fp8)
~/models/Qwen3.8-27B-DFlash2-FP8/         <- DRAFTER   (drafter)
```

Check first, download only what is missing:

```bash
test -f ~/models/Qwen3.8-27B-MXFP4-mtpfp8/config.json \
  && echo "target checkpoint OK" \
  || echo "target checkpoint MISSING -> see below"
test -f ~/models/Qwen3.8-27B-DFlash2-FP8/config.json \
  && echo "drafter OK" \
  || echo "drafter MISSING -> see below"
```

### Recommended: let setup-mxfp4.sh do it

```bash
./setup-mxfp4.sh --yes
```

It pulls the image, downloads AMD's release into the HF cache, builds
`~/models/Qwen3.8-27B-MXFP4-mtpfp8` from it with `fp8_mtp.py` (~15 min, one file rewritten),
downloads the drafter into `~/models/`, and builds the pinned libr4d. It skips every step whose
output already exists, so it is safe to re-run.

### Manual download (if you already have the source, or prefer the `hf` CLI)

Downloads land in `~/models/` and the checkpoints are used from there; nothing is read from
anywhere else at serve time (`MODELS` is bind-mounted at `/models`).

```bash
mkdir -p ~/models

# drafter — used directly from this directory
test -f ~/models/Qwen3.8-27B-DFlash2-FP8/config.json \
  || hf download tcclaviger/Qwen3.8-27B-DFlash2-FP8 \
       --local-dir ~/models/Qwen3.8-27B-DFlash2-FP8

# body — this is the SOURCE, it still needs the MTP rewrite before it will load
test -f ~/models/Qwen3.8-27B-Quark-AWQ-MXFP4/config.json \
  || hf download amd/Qwen3.8-27B-Quark-AWQ-MXFP4 \
       --local-dir ~/models/Qwen3.8-27B-Quark-AWQ-MXFP4
```

Then build the loadable checkpoint (needs torch; easiest inside the image, which is what
`setup-mxfp4.sh` does, or call it directly with host torch installed):

```bash
python3 ./fp8_mtp.py \
  ~/models/Qwen3.8-27B-Quark-AWQ-MXFP4 \
  ~/models/Qwen3.8-27B-MXFP4-mtpfp8
```

Without host torch, use the container path from `setup-mxfp4.sh` step 4/6, or just re-run
`./setup-mxfp4.sh` (it will see the already-downloaded source in the HF cache only if it fetched
it; a `--local-dir` download is fine as the input to `fp8_mtp.py` above).

A checkpoint that already declares `mtp.*` in `layer_quant_config` needs no rewrite — point
`SNAP` at it and serve:

```bash
SNAP=~/models/<that-checkpoint> ./serve-tp1-gpu0.sh
```

`~/models/Qwen3.8-27B-Quark-AWQ-MXFP4/` containing only a 41 MB `model.safetensors` and no
`config.json` is a broken partial download — delete it and re-download.

---

## 3. The image

One image serves both containers:

```bash
docker pull stilldeadcode/vllm-radiance:0.9.3      # ~13.7 GB
```

The launcher builds the pinned libr4d (`b9e42ab-rx9`, the narrow-state GDN kernels the TP=1
profile needs) into `~/.cache/radiance-libr4d/b9e42ab-rx9` on first use — a few minutes, once,
then reused. Do not use the image's stock libr4d: it predates the GDN overflow fix and NaNs this
model's output.

---

## 4. The launchers

| file | what it does |
|---|---|
| `serve-tp1-gpu0.sh` | `TP=1 GPUS=0`, name `vllm-gpu0`, port `8080`, cache `~/.radiance-cache-gpu0` |
| `serve-tp1-gpu1.sh` | `TP=1 GPUS=1`, name `vllm-gpu1`, port `8081`, cache `~/.radiance-cache-gpu1` |
| `serve-tp1-dual.sh` | starts gpu0, waits for `/health`, starts gpu1; `stop` subcommand stops both |

Each wrapper is a thin `exec env ... ./serve-mxfp4.sh`, which pins only what **must** differ:

* **GPU** — `GPUS=0` / `GPUS=1`. The launcher sets `ROCR_VISIBLE_DEVICES=<abs id>` and
  `HIP_VISIBLE_DEVICES=0` (indices *into* the filtered list; passing `1,1` would leave the engine
  with no device).
* **Container name** — the launcher's default name would make each start replace the *other*
  container (`podman --replace` / `docker rm -f`).
* **Port** — `--network=host`, so the two listeners must not collide.
* **Cache root** — cache dirs are keyed on the exact traced graph and **must not be shared**
  between serves. The launcher appends `-tp1s` at TP=1 (the fp8-residual-stream arm without an
  all-reduce), so the directories actually used are `~/.radiance-cache-gpu0-tp1s` and
  `~/.radiance-cache-gpu1-tp1s`.

Runtime: the wrappers default to `RUNTIME=docker` (override with `RUNTIME=podman`). Pull the
image in whichever store you serve from — the two stores are separate.

`~/models`, the chat template and the repo itself are bind-mounted read paths; only the cache
root and the HF cache are writable.

---

## 5. What each serve actually runs

`TP=1` turns on the single-GPU profile automatically (`SINGLE_GPU_PROFILE=auto`). Resolved and
verified against this box:

```
checkpoint        /models/Qwen3.8-27B-MXFP4-mtpfp8      (AMD Quark body, fp8 MTP head)
drafter           /models/Qwen3.8-27B-DFlash2-FP8       dflash, spec=7, TRITON_ATTN
tensor parallel   1                                      one HIP card per container
attention         R4D                                    libr4d b9e42ab-rx9
kv cache          fp8, pinned --kv-cache-memory 9126805504
                  (row 1x7551-32624 / seqs 3 / chunk 2560 / maxlen 220000 in kv-profiles.tsv;
                   ~220k KV tokens. Reported, not measured here — re-calibrate with
                   ./calibrate-kv.sh if you change the batch shape)
max model len     220000
max num seqs      3
batched tokens    2560
mamba cache       conv bfloat16, ssm float16   (halves the attention block -> more sequences)
cudagraphs        [1,2,4,8,16,24]
gpu util          0.98
norm/quant        fuse_norm_quant + fuse_act_quant, GDN in_proj merge, fp8 stream (TP1 arm)
hf/inductor/etc   bind-mounted cache root at /cache (vllm, inductor, triton, aiter)
```

First start compiles Triton/inductor kernels and autotunes, so it takes several extra minutes
per container. Later starts reuse the cache.

---

## 6. Start both

```bash
./serve-tp1-dual.sh
```

The script starts **gpu0 first, polls `/health`, and only then starts gpu1**. This is deliberate:
60 GiB of host RAM does not survive two 19.4 GiB checkpoint loads at the same time. A server
that is already healthy is left alone rather than restarted.

Expected output when both are up:

```
[dual] starting vllm-gpu0 on :8080
[dual] waiting for vllm-gpu0 on :8080 ...... ready (300s)
[dual] starting vllm-gpu1 on :8081
[dual] waiting for vllm-gpu1 on :8081 ...... ready (310s)

[dual] both cards are serving:
  gpu0  http://localhost:8080/v1   (vllm-gpu0)
  gpu1  http://localhost:8081/v1   (vllm-gpu1)
```

## 7. Verify

```bash
curl -fsS http://localhost:8080/health && echo " gpu0 up"
curl -fsS http://localhost:8081/health && echo " gpu1 up"

curl -s http://localhost:8080/v1/models
curl -s http://localhost:8081/v1/models

curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8","messages":[{"role":"user","content":"Hello!"}]}'
```

Worth grepping for in the logs (`docker logs vllm-gpu0 | less`):

* `Using RadianceMxfp4W4A8LinearKernel for MXFP4 GEMM` — the fork's kernel won selection.
* `[radiance] native MXFP4 enabled on gfx12x` — the aiter fp4 gate was relaxed.
* `[run] single-GPU profile ON (TP=1)` — the fp16 ssm profile is active.
* the stock `current platform does not support native MXFP4/MXFP6` notice still prints and is a
  false alarm (it comes from a separate `supports_mx()` call).
* R4D selections table — `RADIANCE_R4D_REPORT=1` is set by the launcher.

## 8. Logs, stop, restart

```bash
docker logs -f vllm-gpu0
docker logs -f vllm-gpu1

./serve-tp1-dual.sh stop                 # stop + remove both
docker stop vllm-gpu0              # or one at a time

./serve-tp1-gpu0.sh                      # foreground, one card only
DETACH=1 ./serve-tp1-gpu0.sh             # background, returns once the container is RUNNING
                                         # (that is not the same as /health being ready)
```

Extra arguments go to `vllm serve` (`./serve-tp1-gpu0.sh --enforce-eager`), and any
`serve-mxfp4.sh` env var applies (`SPEC=5`, `MAXSEQS=8 MAXLEN=65536 CHUNK=4096` for the
throughput shape instead of the long-context one, `PORT0`/`PORT1` for the dual script, ...).

`DRY_RUN=1 ./serve-tp1-gpu0.sh` prints the exact `docker run` command without running it —
useful when auditing a configuration change.

## 9. Gotchas

* **Do not run the two loads concurrently.** The dual script stagger exists because of this; if
  you start the wrappers by hand, wait for gpu0's `/health` first.
* **Cache dirs are configuration-keyed.** If you change `TP`, `RADIANCE_FP8_STREAM`,
  `RADIANCE_GDN_MERGE_INPROJ`, `RADIANCE_NORMQUANT_FUSION`, `SINGLE_GPU_PROFILE` or the capture
  set, a new suffix directory is built — expected. The dangerous direction is a serve writing a
  *stock* graph into an `-fp8s`/`-tp1s` dir (documented at `RADIANCE_FP8_STREAM` in
  `serve-mxfp4.sh`); if you suspect it, delete that cache dir and let it rebuild.
* **The pre-existing `~/.radiance-cache-gpu0` / `-gpu1` directories (root-owned, no `-tp1s`
  suffix) are not used by this setup.** They were built by an older bake with a different traced
  graph. Leave them alone, or `sudo rm -rf` them to reclaim the space.
* **Port in use.** The launcher names the container holding the port and the runtime command to
  stop it. A container started outside systemd is stopped with the runtime, not a unit.
* **Wrong runtime.** Image present under podman but not docker (or vice versa) ⇒ run
  `RUNTIME=podman ...` or pull into the right store. A rootless podman cannot write a root-owned
  cache root; use a new `CACHE=...` in that case.
* **`~/models/Qwen3.8-27B-Quark-AWQ-MXFP4` is the source, not the served checkpoint.** The
  served form is `~/models/Qwen3.8-27B-MXFP4-mtpfp8`. Once the latter exists the source download
  can be deleted (~19 GiB).
* **Optional baked images.** `Dockerfile.int5-tp1` + `.gpu0` / `.gpu1` bake the same TP=1 serve
  args into per-GPU images (`ROCR_VISIBLE_DEVICES=0/1`, ports 8080/8081) for a launcher-free
  `docker run`. Not needed here — the wrappers reproduce that arg list exactly — and building
  them takes a full image build.
