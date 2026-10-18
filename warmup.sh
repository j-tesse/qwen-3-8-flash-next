#!/usr/bin/env bash
# Page the model in after a cold start and report when decode settles.
# mmap is lazy: the first requests fault ~94gb off NVMe and read far below
# steady state, so a cold first number is meaningless. start.sh calls this so
# an agent request is never the one paying for it.
set -euo pipefail
cd "$(dirname "$0")"

# --- configuration ---------------------------------------------------------
ENDPOINT="http://localhost:8000"
MODEL="Qwen3.8-Flash-Next"
MAX_RUNS=6          # give up after this many even if still climbing
DECODE_TOKENS=200   # long enough for a stable tok/s reading
SETTLE_PCT=8        # warm once decode moves less than this % between runs
# ---------------------------------------------------------------------------

C='\033[96m'; G='\033[92m'; Y='\033[93m'; R='\033[91m'; RS='\033[0m'
say() { echo -e "${C}==>${RS} $*"; }

req() {
  curl -s --max-time 600 "$ENDPOINT/v1/chat/completions" \
    -H 'Content-Type: application/json' -d '{
      "model":"'"$MODEL"'",
      "messages":[{"role":"user","content":"Count from 1 to 80, separated by spaces."}],
      "max_tokens":'"$DECODE_TOKENS"',"temperature":0,
      "chat_template_kwargs":{"enable_thinking":false}}' \
  | jq -r '.timings | "\(.prompt_per_second // 0) \(.predicted_per_second // 0)"'
}

say "warming $MODEL at $ENDPOINT"
prev=0
for i in $(seq "$MAX_RUNS"); do
  read -r pf dc < <(req) || true
  [ -n "${dc:-}" ] && [ "$dc" != "0" ] || {
    echo -e "${R}no timings from $ENDPOINT${RS} — is the server up and is MODEL right?" >&2
    exit 1
  }
  printf "  run %d: prefill=%.0f tok/s  decode=%.1f tok/s\n" "$i" "$pf" "$dc"
  # settled once the run-to-run change drops below SETTLE_PCT
  if awk -v a="$prev" -v b="$dc" -v p="$SETTLE_PCT" \
       'BEGIN{exit !(a>0 && b>0 && (b>a?b-a:a-b)*100/b < p)}'; then
    say "${G}warm${RS} — decode settled at $(printf '%.1f' "$dc") tok/s"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/  vram: /'
    free -g | awk '/^Mem:/{print "  ram:  " $7 "G available, " $6 "G page cache"}'
    exit 0
  fi
  prev=$dc
done
echo -e "${Y}still climbing after ${MAX_RUNS} runs${RS} — NVMe may still be paging; re-run to confirm"
