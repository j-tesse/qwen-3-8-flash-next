#!/usr/bin/env bash
# Download the Qwen3.8-Flash-Next GGUF into ~/models/gguf using a throw-away
# container (nothing is installed on the host). Resumable — safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

# --- configuration ---------------------------------------------------------
REPOSITORY="unsloth/Qwen3.8-Flash-Next-GGUF"
QUANT="UD-IQ4_XS"                        # 93.7 GB, 3 shards. See the quant ladder in CLAUDE.md.
DRAFT="MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"   # 2.8 GB MTP head, for --spec-type draft-mtp
WITH_DRAFT=1                             # 0 skips the MTP draft entirely
WITH_MMPROJ=0                            # 1 also pulls mmproj-F16.gguf (vision) — see CLAUDE.md
DESTINATION="/home/joran/models/gguf"    # NOT the HF cache — see CLAUDE.md
REQUIRED_FREE_GB=110
MAX_WORKERS=8
# ---------------------------------------------------------------------------

mkdir -p "$DESTINATION"

avail_gb=$(df -BG --output=avail "$DESTINATION" | tail -1 | tr -dc '0-9')
if [ "$avail_gb" -lt "$REQUIRED_FREE_GB" ]; then
  echo "need ${REQUIRED_FREE_GB}G free at $DESTINATION, have ${avail_gb}G" >&2
  exit 1
fi

# Single-quoted for the *container's* shell so the globs reach hf, not sh.
INCLUDES="--include '${QUANT}/*'"
[ "$WITH_DRAFT" = 1 ]  && INCLUDES="$INCLUDES --include '$DRAFT'"
[ "$WITH_MMPROJ" = 1 ] && INCLUDES="$INCLUDES --include 'mmproj-F16.gguf'"

echo "downloading $REPOSITORY [$QUANT] -> $DESTINATION (${avail_gb}G free)"

# --user keeps the downloaded files owned by us, not root.
# HF_HOME is parked on the same big filesystem so the xet chunk cache can't fill /.
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HOME=/out/.hfcache \
  -v "$DESTINATION:/out" \
  python:3.12-slim \
  sh -c "pip install --quiet --user huggingface_hub && \
         /tmp/.local/bin/hf download '$REPOSITORY' $INCLUDES \
           --local-dir /out \
           --max-workers $MAX_WORKERS"

rm -rf "${DESTINATION:?}/.hfcache"
exec ./verify-model.sh
