# Qwen 3.8 Flash Next
Purpose of this repo is to run a quantized version of Qwen3.8-Flash-Next using `llama.cpp` in docker.

Serves an OpenAI-compatible API on **port 8000** as a drop-in replacement for the vLLM service.

Sibling repo `../deepseek-v4-flash/` does the same job for DeepSeek V4 Flash and shares this layout.
Machine-level lessons carry over between them; **per-model numbers do not.**

## Machine
- headless machine running Ubuntu Server
- 96gb system memory - not all of it is available (~91gb total, ~82gb available)
- RTX 5090 with 32gb VRAM - **shared, not exclusive**: `vllm_gemma4-31b-nvfp4-mtp` normally owns
  all of it (`--gpu-memory-utilization 0.982`, `restart: unless-stopped`). One GPU LLM at a time.
  `start.sh` stops vLLM, `stop.sh` restores it.
- docker and nvidia docker tools already installed
- Ryzen 9 9900X (12C/24T), driver 610.43.02 / CUDA 13.3, GPU is Blackwell **sm_120**
- Crucial T705 Gen5 NVMe, ~369gb free (the DeepSeek repo's 104gb quant is also on this disk)

## Architecture
```
qwen-3-8-flash-next/
├── llama.cpp/          upstream submodule, pinned to a release tag (b11719). Never edited.
├── build.sh            builds upstream's .devops/cuda.Dockerfile (target: server) via build args
├── build-async.sh      start|status [--json]|wait|log|stop — build takes ~10-20min
├── update.sh           move submodule to a newer tag (refuses dirty submodule) + rebuild
├── download-model.sh   throw-away container pulls the GGUF (+ MTP draft) into ~/models/gguf
├── verify-model.sh     shard count + GGUF magic check, incl. optional draft/mmproj sidecars
├── start.sh            free GPU → run container → wait healthy → warm up → re-point agents
├── stop.sh             stop container → restart vLLM  (`--no-restore` to skip)
├── warmup.sh           repeat a short request until decode settles (mmap pages in lazily)
├── bench.sh            single-request prefill/decode + PCIe counters against :8000
├── bench-spec.sh       echo-a-file decode bench + MTP acceptance rate
├── tune.sh             sweep server flags, restarting the container per variant → tune.results
└── build.number/history
```

No custom Dockerfile: upstream's `.devops/cuda.Dockerfile` already does what we need, driven by
`--build-arg CUDA_VERSION` / `CUDA_DOCKER_ARCH`. If a patch ever becomes necessary, add
`<file>.patched.<ext>` here plus a thin Dockerfile that `COPY`s it over the submodule — never edit
`llama.cpp/` in place.

Image tags per build: `qwen-3-8-flash-next:b<N>`, `:<upstream-describe>`, `:latest`.
`build.number` bumps only on success.

`llama.cpp/` ships its own `CLAUDE.md`/`AGENTS.md`. Those are upstream's contributor guide for
hacking on llama.cpp itself and **do not govern this repo** — we never edit the submodule.

**Mainline llama.cpp is enough — do not switch to the `unslothai/llama.cpp` fork.** Unsloth's model
card tells you to, because at the time of writing mainline lacked MTP. Since b10752 mainline has both
`src/models/qwen4exp.cpp` and `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` (`--spec-type draft-mtp`), so the
fork buys nothing and would cost us the "pinned upstream tag" property.

## Model
`unsloth/Qwen3.8-Flash-Next-GGUF`, quant **UD-IQ4_XS** (93.7gb, 3 shards). Released 2026-08-26 as a
preview of the Qwen4 architecture (`model_type: qwen4_exp`).

Unsloth re-uploaded all shards 2026-10-05 "due to upstream changes" (llama.cpp's qwen4exp rework,
#29751/#29761). Current download is from then, served on b11719. Old b10752 + old shards are gone.
If Unsloth re-uploads again, re-download **and** `./update.sh` together — the two move in lockstep.

From the upstream `config.json` — these are read off the config, not measured:
- **48 layers, hybrid attention.** `layer_types` cycles `[linear, linear, linear, full]`, so only
  **12 layers are full attention**; the other 36 are linear (Gated DeltaNet-style) with fixed-size
  recurrent state. QSA sparse attention, `full_attention_interval: 4`.
- **512 experts, 10 active per token**, `moe_intermediate_size` 640 (+ one shared expert, 640).
- hidden 2560, 24 query heads / 2 KV heads, head_dim 256, `partial_rotary_factor` 0.25.
- vocab 248320, native ctx **262144** (extensible to 1M).
- **~51B n-gram embedding parameters** on top of the 125B main model — `ngram_size` 3,
  `ngram_vocab_size_base` 20M, `split_ngram_parts` 128. See below; these dominate the memory plan.
- Multimodal: 27-layer vision tower (`mmproj-F16.gguf`, 0.9gb). We serve text-only — see Vision.
- 1 MTP layer, shipped separately as a draft GGUF. See MTP below.

Expert weights are 3 x 2560 x 640 = 4.92M params each; x512 x48 layers = **~121B, i.e. ~97% of the
125B main model**. Same shape as DeepSeek V4: dense part is tiny, experts are everything.

### Quant ladder
`UD-IQ4_XS` is the default because it leaves real headroom. Change `QUANT` + `MODEL_FILE` in
`download-model.sh` and `start.sh` together.

| quant | size | shards | note |
|---|---|---|---|
| UD-IQ1_S / UD-IQ1_M | 72.5 / 74.5gb | 3 | lots of room, quality cost |
| UD-Q2_K_XL | 78.9gb | 3 | |
| UD-IQ3_XXS | 82.0gb | 3 | matches the DeepSeek repo's tier |
| UD-Q3_K_XL | 90.0gb | 3 | |
| **UD-IQ4_XS (current)** | **93.7gb** | **3** | ~20gb headroom |
| UD-Q4_K_XL | 111.3gb | 4 | Unsloth's pick, but ~114gb usable total — **no headroom, expect paging** |
| UD-Q5_K_XL and up | 158gb+ | 6+ | does not fit |

### Memory budget
Usable = ~82gb RAM + 32gb VRAM = ~114gb > 93.7gb.

At IQ4_XS the 93.7gb spreads over 176B params (125B + 51B n-gram) ≈ 4.26 bits/param. Applying that
rate uniformly — Unsloth Dynamic quants are *not* uniform, so treat these as ±20% estimates:
- experts ~121B → **~64gb**, i.e. ~1.34gb per layer
- n-gram table ~51B → **~27gb** (RAM-pinned, see below)
- everything else (dense, attention, embeddings) → ~2gb

GPU: ~2gb dense + ~1.6gb KV @64k + ~2gb compute buffers + ~0.6gb overhead → ~26gb left for experts
→ ~19 expert layers fit → `--n-cpu-moe 29-30`. `start.sh` ships **30** as the conservative start.
RAM then holds 27gb n-gram + 30 x 1.34 = ~40gb experts = **~67gb vs ~82gb available**, ~15gb headroom.

**Tuned 2026-10-18 on b11719 (`tune.results`), ctx 65536:**

| variant | VRAM | decode 512→64k | prefill 4.6k / 18k / 64k |
|---|---|---|---|
| ncmoe30 ub2048 | 30.5gb | 46 / 45 / 47 / 46 | 1654 / 1819 / 1669 |
| ncmoe26 ub2048 | OOM (KV alloc) | | |
| ncmoe34 ub2048 | 25.5gb | 43 / 43 / 44 / 42 | 1469 / 1602 / 1472 |
| ncmoe30 ub512 | 29.8gb | 47 / 46 / 46 / 46 | 819 / 765 / 676 |
| **ncmoe30 ub4096 (shipped)** | **31.9gb** | **47 / 46 / 47 / 46** | **2081 / 2479 / 2394** |
| MTP n2, ncmoe34 ub2048 | 30.2gb | 51 / 48 / 50 / 44 | 1367 / 1529 / 1410 |

`-ncmoe 30` is the floor (26 OOMs; 28 untried). ub4096 is pure win on prefill but leaves only ~0.7gb
VRAM — anything that grows GPU use (bigger ctx, mmproj, MTP) needs `-ncmoe` raised to compensate.

RAM behaves as designed: ~5gb actually used, ~86gb of mmap'd weights sitting in reclaimable page
cache, ~85gb still reported available. That is the OOM safety net working — see `--load-mode` below.

### The ~51B n-gram table lives on CPU
This is the big structural difference from DeepSeek V4. The n-gram embeddings are **not experts**,
so `--n-cpu-moe` does not move them. They are sparse row lookups, so serving them from RAM is cheap
in a way that streaming expert matrices is not.

**Measured: llama.cpp already keeps them on CPU.** `tune.sh`'s `ngram-on-gpu` variant (no `-ot`)
matched `ncmoe34` exactly — same 25512MiB VRAM, same speed. `per_layer_token_embd` is classed
`LLM_TENSOR_LAYER_INPUT` (a GET_ROWS input, like `token_embd`), which stays on CPU by default. The
original "it would eat ~27gb of VRAM" claim was wrong. `start.sh` keeps
`--override-tensor per_layer_token_embd=CPU` as an explicit guard only.

Verified against the pinned tag: `src/models/qwen4exp.cpp` creates a single flat
`per_layer_token_embd.weight` of `[ple_head_dim, ple_rows]`, so the `-ot` regex matches exactly one
tensor. The `%s.ple.*` GGUF keys (`ngram_size`, `heads_per_ngram`, `head_vocab_sizes`) line up with
the config's n-gram fields — PLE *is* the n-gram table.

### KV cache is cheap here — spend context freely
Only 12 of 48 layers keep a KV cache: 12 x 2 (K+V) x 2 KV heads x 256 head_dim x 2 bytes (f16)
= **24kb/token**. So 64k ctx ≈ 1.6gb and even the native 262k ≈ 6.4gb.

Consequences:
- `CONTEXT_SIZE=65536` costs ~1.6gb of VRAM. The DeepSeek repo runs 32768 because its KV is far
  more expensive; do not copy that number over as if it were a constraint here.
- **No `--cache-type-k/-v`.** f16, like the sibling repo, but for a different reason: there the
  concern was garbage output from quantized KV, here it is simply that q8_0 would save under a
  gigabyte and is not worth any quality risk. Other users do run q8_0 on this model successfully.

### Decode is ~2x DeepSeek V4 — the experts are tiny
Decode reads active expert weights from RAM: 10 of 512 experts x 4.92M params x ~0.53 bytes
= ~26mb per CPU layer, x30 CPU layers = **~0.79gb/token**. DeepSeek V4 reads ~1.9gb/token and lands
at 23-24 tok/s, so the RAM-streaming floor here is ~2.4x looser.

**Measured, `-ncmoe 30 -ub 2048`, IQ4_XS** (warmup.sh short prompt; bench.sh rows, thinking off):

| build | warm decode | bench.sh n=512 / 4.6k / 18.5k / 64.3k tokens | VRAM |
|---|---|---|---|
| b10752 (2026-09-01) | 47.3 tok/s | not run | 31.6gb |
| b11719 (2026-10-18) | 49.1 tok/s | decode 42 / 40 / 42 / 45, prefill 588 / 1444 / 1748 / 1654 tok/s | 30.5gb |

**No decode decay with depth on b11719** — 64k decodes as fast as 512. bench.sh's prompt ("Count
slowly.") decodes a bit slower than warmup.sh's; compare rows within one script only.
The ~1gb VRAM drop is plausibly #29825 ("halve the indexer score memory") — not isolated.

The dual-channel ceiling is unchanged (~96 GB/s theoretical, ~60-70 practical; RAM already at its
rated 6000 MT/s, so no BIOS win). What changed is bytes-per-token, which is the only lever that ever
mattered.

### MTP speculative decoding is worth measuring here (unlike DeepSeek)
The model's own MTP head ships as standalone draft GGUFs under `MTP/`; `-shared-` variants borrow
embeddings from the main model (see below — mainline can't). Unsloth reports 64-66% acceptance and 1.34-1.67x **at greedy sampling**,
falling off as temperature rises, and a *loss* (0.81-0.87x) at concurrency 8 — we run `--parallel 1`,
so the concurrency caveat does not apply.

The DeepSeek repo proves speculation cannot help *there*, and the batching half of that argument
holds here too: with 10-of-512 routing, verifying K drafts touches `512*(1-(1-10/512)^K)` experts per
layer — 19.8 for K=2 vs 20 sequential, a ~1% saving. So MTP wins **only** to the extent that decode
is not purely expert-streaming-bound. Because bytes/token here are ~2.4x lower, the non-streaming
part of the budget (48 layers of compute, 36 of them linear-attention, plus n-gram lookups) is a much
bigger share than DeepSeek's ~0%, and there is real room for a draft to amortise it.

**Measured 2026-10-18, b11719 — shipped OFF (`SPEC=0`):**
- **Mainline cannot use the `-shared-` heads** (`tensor 'token_embd.weight' not found`): it has no
  cross-model tensor borrowing. Unsloth's MTP/README still says mainline lacks MTP at all — stale.
  We use the self-contained `MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (4.1gb).
- It only fits at `-ncmoe 34 -ub 2048`; ncmoe32 and any ub4096 combo OOM (draft compute buf 1-2gb).
- Result vs shipped config: decode +4-11% to 18k, -4% at 64k; bench-spec echo 55 vs 45 tok/s, 100%
  acceptance (greedy best case); prefill ~-40%. For long-context agent work at temp 1.0, prefill wins.
- Revisit if upstream adds tensor borrowing (saves 1.3gb, may let it fit at ub4096).

### `--load-mode none` / `--no-mmap`: do not enable
Not yet tested on this model, and the sibling repo's incident is reason enough to leave it alone.
There, pinning ~78gb of host memory dropped MemAvailable to ~10gb and the OOM killer took the
`claude` agent process (`oom_score_adj:200` makes it the preferred victim) four minutes after load.
Here the CPU-resident set is ~67gb rather than ~78gb, so it is *less* suicidal but still leaves only
~15gb of slack against a kernel that can no longer reclaim any of it.

mmap stays enabled (the default) so RAM overflow degrades to NVMe paging instead of OOM. Do not add
`--mlock` or `--no-mmap`. A 128gb RAM upgrade is what would make this safe, and it remains the single
highest-value hardware change for this box.

### Models live in `~/models/gguf`, NOT the HF cache
vLLM bind-mounts `~/.cache/huggingface` and pulls by repo id, but `~/scripts/hf-cache-cleanup.sh`
builds its KEEP set by grepping `MODEL_ID=` out of `~/scripts/vllm.sh` and `rm -rf`s everything else
under `hub/`. A GGUF there would be treated as an orphan and deleted. Keeping it at `~/models/gguf`
(the path `~/scripts/llama.sh` already declares) avoids that, and llama.cpp wants a plain file path
anyway rather than the hub's blobs/snapshots symlink layout.

`~/models/gguf` is safe from the HF cleanup but **not** from `~/scripts/disk-cleanup.sh` (an interactive
Claude session): it deleted the whole dir 2026-09-27 (logged in `~/scripts/DISK-CLEANUP.md`).
Re-downloaded 2026-10-18. If `start.sh` says the model is missing, that is the likely cause.

## Flag choices worth remembering
- `--override-tensor per_layer_token_embd=CPU`: the n-gram table. Redundant on b11719 (already
  CPU by default), kept as a guard. See above.
- **f16 KV cache** (no `--cache-type-k/-v`): KV is only 24kb/token, so quantizing saves ~nothing.
- `--jinja`: required for the chat template / tool calling.
- **Thinking mode is ON by default.** Sampling defaults follow the model card's thinking preset:
  `--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0`. Instruct preset would be 0.7 / 0.80 / 20.
  Per-request override: `chat_template_kwargs: {"enable_thinking": false}` — both bench scripts do
  this so their token counts stay comparable.
- `--parallel 1`: also a correctness guard, see Known upstream issues.
- `--restart no`: vLLM's container is `unless-stopped` and also binds 8000 — two auto-restarting
  containers would race for the port after a reboot.
- `--threads 12`: physical cores only; SMT siblings thrash cache.
- Health start period is generous, but mmap means the server reports healthy in ~10s — long before
  the weights are resident. A cold first request measured 18 tok/s against 47 warm, which is why
  `start.sh` runs `warmup.sh` before re-pointing the agents.
- `-b/-ub 4096`: prefill streams CPU experts over PCIe once per ubatch, so bigger amortises it.
  Measured: 512 → 2048 → 4096 gives ~700 → ~1700 → ~2400 tok/s prefill at 64k; decode unaffected.
- `start.sh` always recreates the container: the flags are baked in at create time, so reusing a
  stale one would silently ignore edits to the constants.

## Known upstream issues (llama.cpp, statuses rechecked 2026-10-18)
These are why some flags look the way they do.
- **#27911** — CUDA `invalid configuration argument` in `ggml_cuda_op_rms_norm_fused` under
  concurrent batching (`np=10`). Open. We run `--parallel 1`, which avoids it.
- **#27871** — abort at the native 262144 depth on sm_121. **Closed 2026-10-16.** We still cap at
  65536; raising it is untested (KV would be ~6.4gb at 262k, see KV section).
- **#27886** — vision issues. **Closed 2026-10-12.** Vision stays off to match the vLLM service.
- **#27964** — `--split-mode tensor` aborts on multi-GPU CUDA. Open. N/A, single GPU.
- **#28158 / #28206** — MTP speculative emitting an out-of-bounds token id on Vulkan; fix merged.
  CUDA-only here, but worth knowing if MTP output ever looks corrupt.
- Users reported **decode decay with depth** on llama.cpp for this model. Not reproduced on b11719
  (see decode table).

## Vision (not enabled)
The model is multimodal but we serve text-only, matching the vLLM service this replaces and dodging
issue #27886. To enable: set `WITH_MMPROJ=1` in `download-model.sh`, then add
`--mmproj /models/mmproj-F16.gguf` to `start.sh`'s docker run. Costs ~1gb VRAM.

## Agent config sync
`start.sh` calls `~/scripts/update-agent-models.sh` (same as vllm.sh / colibri) to point pi + OpenClaw
at whatever :8000 serves. Two llama.cpp-specific details:
- `--alias` is what makes `/v1/models` report `Qwen3.8-Flash-Next`; without it the id is the raw
  container path of the GGUF, which is what would land in the agent configs.
- That script reads the context window from vLLM's `max_model_len`, **which llama.cpp does not report**.
  It would silently fall back to its own 32768 default, so `start.sh` passes `CONTEXT_WINDOW` explicitly.
  Change `CONTEXT_SIZE` in `start.sh` and both agents follow.

## Ports
8000 = this service / vLLM / the DeepSeek repo (all mutually exclusive). 8080 = open-webui, do not
use. owui already has `http://host.docker.internal:8000/v1` registered, so it picks this up with no
config change.

## Usage
```
./build-async.sh start && ./build-async.sh wait
./download-model.sh
./start.sh
./stop.sh
```

## CLAUDE.md Guidance
- broad architecture of this repo must be recorded in `CLAUDE.md`
- file must be kept in sync with the repo, apply edits to `CLAUDE.md` alongside changes to the repo
- use terse language
- **mark estimates as estimates.** Much of this file is arithmetic from `config.json`, not
  measurement. Replace it with real numbers as they come in, and do not let a number from
  `../deepseek-v4-flash/CLAUDE.md` get restated here as if it were measured on this model.

## Work Guidance
- scripts are argument-free and configured by the `# --- configuration ---` constants at the top
  - do NOT use env variables to tune our scripts — edit the constant, so there is nothing to memorize
- skip code comments if the code is self-explanatory
- code comments must be extremely terse
- less code is better code
- create and persist small scripts instead of running medium to long bash commands ad-hoc
  - e.g. create `start.sh`, `stop.sh`, `rebuild.sh`, ...
- do not install anything globally
  - example: if a 1-time dependency is required to download a model from hugging-face, just write a script to create and start a throw-away container to get it done
- prefer additive patches over editing
  - example: instead of cloning a repo and applying changes in place, prefer cloning a repo in a subfolder, and "override" files from the subfolder via Dockerfile instructions at build time
