#!/usr/bin/env bash
# Sweep llama-server flags and benchmark each. Restarts the container per variant,
# so do not run this while agents depend on :8000. Fold the winner into start.sh.
set -euo pipefail
cd "$(dirname "$0")"

# --- configuration ---------------------------------------------------------
IMAGE="qwen-3-8-flash-next:latest"
CONTAINER="llama_tune"
MODELS_DIR="/home/joran/models/gguf"
QUANT="UD-IQ4_XS"
MODEL_FILE="Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf"
DRAFT="MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
ALIAS="Qwen3.8-Flash-Next"
PORT=8000
CONTEXT_SIZE=65536
THREADS=12
RESULTS="tune.results"

# name|extra llama-server args
# Nothing here is measured yet — this is the first sweep to run on this box.
# Order matters only for readability; each variant is a fresh container.
VARIANTS=(
  "ncmoe30-ub2048  |-ncmoe 30 -b 2048 -ub 2048 -ot per_layer_token_embd=CPU"
  "ncmoe26-ub2048  |-ncmoe 26 -b 2048 -ub 2048 -ot per_layer_token_embd=CPU"
  "ncmoe34-ub2048  |-ncmoe 34 -b 2048 -ub 2048 -ot per_layer_token_embd=CPU"
  "ncmoe30-ub512   |-ncmoe 30 -b 2048 -ub 512  -ot per_layer_token_embd=CPU"
  "ncmoe30-ub4096  |-ncmoe 30 -b 4096 -ub 4096 -ot per_layer_token_embd=CPU"
  # Is pinning the n-gram table to CPU actually right, or does the VRAM pay off?
  "ngram-on-gpu    |-ncmoe 34 -b 2048 -ub 2048"
  # MTP speculative decoding. Expect a decode win only if decode is not purely
  # expert-streaming-bound — see CLAUDE.md.
  # The 4.1gb self-contained draft does not fit next to -ncmoe 30; give back 2-4 expert layers.
  "mtp-n2-ncmoe32  |-ncmoe 32 -b 2048 -ub 2048 -ot per_layer_token_embd=CPU -md /models/${DRAFT} --spec-type draft-mtp --spec-draft-n-max 2 -ngld 999"
  "mtp-n2-ncmoe34  |-ncmoe 34 -b 2048 -ub 2048 -ot per_layer_token_embd=CPU -md /models/${DRAFT} --spec-type draft-mtp --spec-draft-n-max 2 -ngld 999"
  "mtp-n2-ub4096   |-ncmoe 34 -b 4096 -ub 4096 -ot per_layer_token_embd=CPU -md /models/${DRAFT} --spec-type draft-mtp --spec-draft-n-max 2 -ngld 999"
  "mtp-n3-ub4096   |-ncmoe 34 -b 4096 -ub 4096 -ot per_layer_token_embd=CPU -md /models/${DRAFT} --spec-type draft-mtp --spec-draft-n-max 3 -ngld 999"
)
# ---------------------------------------------------------------------------

C='\033[96m'; G='\033[92m'; R='\033[91m'; RS='\033[0m'
say() { echo -e "${C}==>${RS} $*"; }

cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker ps -q --filter "name=vllm_" | xargs -r docker stop >/dev/null
docker ps -q --filter "name=llama_qwen38_flash_next" | xargs -r docker stop >/dev/null

: >"$RESULTS"
for v in "${VARIANTS[@]}"; do
  name="${v%%|*}"; args="${v#*|}"; name="${name// /}"
  say "variant ${name}: ${args}"
  cleanup
  sleep 3

  # shellcheck disable=SC2086
  docker run -d --name "$CONTAINER" \
    --runtime nvidia --gpus all --restart no \
    -v "${MODELS_DIR}:/models:ro" -p "${PORT}:${PORT}" --ipc=host \
    --health-cmd "curl -f http://localhost:${PORT}/health || exit 1" \
    --health-interval 30s --health-start-interval 5s --health-timeout 5s \
    --health-retries 120 --health-start-period 30s \
    "$IMAGE" \
    --model "/models/${QUANT}/${MODEL_FILE}" --alias "$ALIAS" \
    --n-gpu-layers 999 --threads "$THREADS" --ctx-size "$CONTEXT_SIZE" \
    --parallel 1 --jinja --flash-attn on \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
    --host 0.0.0.0 --port "$PORT" $args >/dev/null

  ok=0
  for _ in $(seq 180); do
    s=$(docker inspect --format='{{.State.Health.Status}}' "$CONTAINER" 2>/dev/null || echo gone)
    [ "$s" = healthy ] && { ok=1; break; }
    docker ps -q -f name="^${CONTAINER}$" | grep -q . || break
    sleep 2
  done
  if [ "$ok" != 1 ]; then
    echo -e "${R}${name}: failed to start${RS}"
    { echo "### $name ($args)"; echo "FAILED TO START"; docker logs --tail 15 "$CONTAINER" 2>&1; } >>"$RESULTS"
    continue
  fi

  vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  ram=$(free -g | awk '/^Mem:/{print $7}')
  { echo "### $name ($args)  vram=${vram}MiB  ram_avail=${ram}G"
    ./bench.sh 2>&1 | grep -E "n=|avg sm"
    echo -n "  echo-test: "; ./bench-spec.sh 2>&1 | head -1; } >>"$RESULTS"
  tail -n 10 "$RESULTS"
done

say "results in ${RESULTS}"
cat "$RESULTS"
