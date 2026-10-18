# Qwen 3.8 Flash Next

Run a quantized [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
with `llama.cpp` in docker, serving an OpenAI-compatible API on **port 8000**.

Built for a single-box hybrid setup: a 125B MoE (+51B n-gram embeddings) split between
32gb of VRAM and system RAM, as a drop-in replacement for a vLLM service on the same port.

## Layout

| | |
|---|---|
| `llama.cpp/` | upstream submodule, pinned to a release tag. Never edited. |
| `build.sh` | builds upstream's `.devops/cuda.Dockerfile` (target: `server`) via build args |
| `build-async.sh` | `start` \| `status [--json]` \| `wait` \| `log` \| `stop` — the build takes ~10-20min |
| `update.sh` | move the submodule to a newer tag (ff-only) + rebuild |
| `download-model.sh` | throw-away container pulls the GGUF (+ MTP draft) into `~/models/gguf` |
| `verify-model.sh` | shard count + GGUF magic check, incl. optional draft/mmproj sidecars |
| `start.sh` | free the GPU → run container → wait healthy → re-point agents |
| `stop.sh` | stop container → restart vLLM (`--no-restore` to skip) |
| `bench.sh` | single-request prefill/decode + PCIe counters against `:8000` |
| `bench-spec.sh` | echo-a-file decode bench + MTP acceptance rate |
| `tune.sh` | sweep server flags, restarting the container per variant |

There is no custom Dockerfile — upstream's CUDA image is driven entirely by
`--build-arg CUDA_VERSION` / `CUDA_DOCKER_ARCH`.

## Usage

```sh
git clone --recurse-submodules https://github.com/j-tesse/qwen-3-8-flash-next
cd qwen-3-8-flash-next
./build-async.sh start && ./build-async.sh wait
./download-model.sh
./start.sh     # serves http://localhost:8000/v1
./stop.sh
```

Scripts take no arguments; everything is a constant in the `# --- configuration ---` block
at the top of each one.

## Notes for other machines

This is tuned for one specific box (Ryzen 9 9900X, 96gb RAM, RTX 5090 / sm_120, Ubuntu Server)
and hardcodes some personal paths (`/home/joran/...`, an agent-config sync hook in `start.sh`).
Expect to edit at least `MODELS_DIR`, `N_CPU_MOE` and `THREADS`.

Choices worth carrying over regardless of hardware:

- **Big ubatch (`-b/-ub 4096`)** — prefill streams the RAM-resident experts over PCIe once per
  ubatch. On a 5090 this gave ~40% more prefill than 2048 at no decode cost, if VRAM allows.
- **MTP needs the self-contained draft on mainline llama.cpp** — the `-shared-` heads fail to load
  (`token_embd.weight not found`); use `MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`.
- **Context is cheap on this model.** Only 12 of 48 layers are full-attention, so the KV cache is
  ~24kb/token — 64k context costs ~1.6gb, and quantizing the KV cache saves next to nothing.

[`CLAUDE.md`](CLAUDE.md) carries the full reasoning: memory budget, quant ladder, MTP speculative
decoding, and the relevant open upstream issues. Numbers in it that are arithmetic rather than
measurement are marked as such.

## License

MIT — see [LICENSE](LICENSE). The `llama.cpp` submodule is MIT-licensed upstream; model weights
carry their own license.
