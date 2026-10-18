#!/usr/bin/env bash
# Start Qwen3.8-Flash-Next on port 8000 as a drop-in replacement for vLLM.
# The GPU holds one LLM at a time, so this stops any running vLLM first.
set -euo pipefail
cd "$(dirname "$0")"

# --- configuration ---------------------------------------------------------
IMAGE="qwen-3-8-flash-next:latest"
CONTAINER="llama_qwen38_flash_next"
MODELS_DIR="/home/joran/models/gguf"
QUANT="UD-IQ4_XS"
MODEL_FILE="Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
ALIAS="Qwen3.8-Flash-Next"
PORT=8000
CONTEXT_SIZE=65536  # KV is only ~24kb/token (12 of 48 layers are full-attention), so
                    # 64k costs ~1.6gb VRAM. Cheap here, unlike a dense-attention model.
THREADS=12          # physical cores only; SMT siblings thrash cache
N_CPU_MOE=30        # tuned 2026-10-18: 26 OOMs, 34 decodes ~7% slower. See CLAUDE.md.
UBATCH_SIZE=4096    # tuned: +40% prefill vs 2048, decode unchanged. VRAM 31.9 of 32.6gb.
BATCH_SIZE=4096
LOAD_MODE=mmap      # DO NOT set to `none`/--no-mmap on this box. It pins the CPU-resident
                    # weights unreclaimably; on the DeepSeek repo that left ~10gb and the
                    # OOM killer took out the agent 4min in. See CLAUDE.md.

# ~51B n-gram table. b11719 already keeps it on CPU (measured: same VRAM without this);
# kept as an explicit guard in case upstream placement changes.
NGRAM_TENSORS="per_layer_token_embd=CPU"

# MTP head. Measured OFF: needs N_CPU_MOE=34 + UBATCH/BATCH 2048 to fit, which costs ~40%
# prefill for +4-11% decode (and -4% at 64k). Set those three together to try it.
SPEC=0
SPEC_DRAFT="MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
SPEC_N_MAX=2
# ---------------------------------------------------------------------------

C='\033[96m'; G='\033[92m'; Y='\033[93m'; R='\033[91m'; RS='\033[0m'
say() { echo -e "${C}==>${RS} $*"; }

[ -f "$MODELS_DIR/$QUANT/$MODEL_FILE" ] || {
  echo -e "${R}model missing:${RS} $MODELS_DIR/$QUANT/$MODEL_FILE"
  echo "run: ./download-model.sh"
  exit 1
} >&2

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo -e "${R}image missing:${RS} $IMAGE — run ./build-async.sh start && ./build-async.sh wait" >&2
  exit 1
}

spec_args=()
if [ "$SPEC" = 1 ]; then
  [ -f "$MODELS_DIR/$SPEC_DRAFT" ] || {
    echo -e "${R}draft missing:${RS} $MODELS_DIR/$SPEC_DRAFT — set WITH_DRAFT=1 in download-model.sh" >&2
    exit 1
  }
  spec_args=( --model-draft "/models/${SPEC_DRAFT}" --spec-type draft-mtp
              --spec-draft-n-max "$SPEC_N_MAX" --n-gpu-layers-draft 999 )
fi

# 1. Free the GPU. Our own container first — the flags below are baked in at create
#    time, so we always recreate and must release its VRAM before the drain check.
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

# `docker stop` (not kill) so `unless-stopped` won't resurrect it.
vllm_running=$(docker ps -q --filter "name=vllm_")
if [ -n "$vllm_running" ]; then
  say "stopping vLLM to free the GPU"
  echo "$vllm_running" | xargs -r docker stop
fi

say "waiting for VRAM to drain"
for _ in $(seq 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 1000 ] && break
  sleep 2
done
[ "$used" -lt 1000 ] || { echo -e "${R}VRAM still ${used} MiB in use${RS} — check nvidia-smi" >&2; exit 1; }
say "VRAM free (${used} MiB used)"

# 2. Create.
say "creating container (model load takes several minutes)"
# --restart no: vLLM's container is `unless-stopped` and also binds 8000;
# two auto-restarting containers would race for the port after a reboot.
# No --cache-type-k/-v: f16 KV. The hybrid attention makes KV tiny (see CONTEXT_SIZE),
# so quantizing it buys nothing worth the quality risk.
# --parallel 1: upstream #27911 reports a CUDA rms_norm_fused crash on this
# architecture under concurrent batching.
docker run -d --name "$CONTAINER" \
  --runtime nvidia --gpus all \
  --restart no \
  -v "${MODELS_DIR}:/models:ro" \
  -p "${PORT}:${PORT}" \
  --ipc=host \
  --health-cmd "curl -f http://localhost:${PORT}/health || exit 1" \
  --health-interval 30s \
  --health-start-interval 5s \
  --health-timeout 5s \
  --health-retries 60 \
  --health-start-period 30s \
  "$IMAGE" \
  --model "/models/${QUANT}/${MODEL_FILE}" \
  --alias "$ALIAS" \
  --n-gpu-layers 999 \
  --n-cpu-moe "$N_CPU_MOE" \
  --override-tensor "$NGRAM_TENSORS" \
  --threads "$THREADS" \
  --ctx-size "$CONTEXT_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --ubatch-size "$UBATCH_SIZE" \
  --load-mode "$LOAD_MODE" \
  --cont-batching \
  --parallel 1 \
  --jinja \
  --flash-attn on \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  "${spec_args[@]}" \
  --host 0.0.0.0 --port "$PORT" >/dev/null

# 3. Wait for healthy, logging startup duration like vllm.sh/llama.sh do.
START_TIME=$(date +%s)
say "waiting for health"
while true; do
  status=$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo gone)
  elapsed=$(( $(date +%s) - START_TIME ))
  case "$status" in
    healthy) echo -e "\r${G}healthy${RS} after ${elapsed}s        "; break ;;
    gone|unhealthy)
      echo -e "\r${R}container ${status}${RS} after ${elapsed}s"
      docker logs --tail 40 "$CONTAINER" 2>&1 || true
      exit 1 ;;
  esac
  if ! docker ps -q -f name="^${CONTAINER}$" | grep -q .; then
    echo -e "\r${R}container exited${RS} after ${elapsed}s"
    docker logs --tail 40 "$CONTAINER" 2>&1 || true
    exit 1
  fi
  printf "\r  %ss elapsed (%s)" "$elapsed" "$status"
  sleep 2
done

CSV=/home/joran/scripts/llama-startups.csv
[ -f "$CSV" ] || echo "timestamp,container,duration_seconds" > "$CSV"
echo "$(date -Is),${CONTAINER},$(( $(date +%s) - START_TIME ))" >> "$CSV"

# 4. Page the weights in — mmap is lazy, so without this the first agent request
#    pays for faulting ~94gb off NVMe.
say "warming up"
./warmup.sh || echo -e "${Y}warmup failed (non-fatal)${RS}"

# 5. Re-point pi + OpenClaw at whatever :8000 now serves.
if [ -x /home/joran/scripts/update-agent-models.sh ]; then
  say "re-pointing agents at :${PORT}"
  # That script reads the context from vLLM's `max_model_len`, which llama.cpp
  # does not report; without this it silently falls back to its 32768 default.
  CONTEXT_WINDOW="$CONTEXT_SIZE" /home/joran/scripts/update-agent-models.sh \
    || echo -e "${Y}agent sync failed (non-fatal)${RS}"
fi

say "serving on http://localhost:${PORT}/v1  — ./stop.sh returns the GPU to vLLM"
