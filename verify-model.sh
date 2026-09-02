#!/usr/bin/env bash
# Check the GGUF shards are all present and self-consistent.
set -euo pipefail
cd "$(dirname "$0")"

QUANT="${QUANT:-UD-IQ4_XS}"
DESTINATION="${DESTINATION:-/home/joran/models/gguf}"
DRAFT="${DRAFT:-MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}"
DIR="$DESTINATION/$QUANT"

magic_ok() { head -c4 "$1" | grep -qa GGUF; }

[ -d "$DIR" ] || { echo "missing $DIR — run ./download-model.sh" >&2; exit 1; }

shopt -s nullglob
shards=("$DIR"/*.gguf)
(( ${#shards[@]} )) || { echo "no .gguf in $DIR — run ./download-model.sh" >&2; exit 1; }

# Shard names end in -0000N-of-0000M.gguf; M must match what we actually have.
expected=$(basename "${shards[0]}" | sed -nE 's/.*-of-0*([0-9]+)\.gguf/\1/p')
if [ -n "$expected" ] && [ "$expected" != "${#shards[@]}" ]; then
  echo "incomplete: found ${#shards[@]} shard(s), manifest says $expected — re-run ./download-model.sh" >&2
  exit 1
fi

# A truncated download leaves a short final shard; llama.cpp only fails minutes in.
for f in "${shards[@]}"; do
  [ -s "$f" ] || { echo "empty shard: $f" >&2; exit 1; }
  magic_ok "$f" || { echo "bad magic (not a GGUF): $f" >&2; exit 1; }
done

echo "OK  ${#shards[@]} shard(s), $(du -shc "${shards[@]}" | tail -1 | cut -f1) total"
echo "$DIR"

# Optional sidecars — absent is fine, corrupt is not.
for extra in "$DRAFT" "mmproj-F16.gguf"; do
  p="$DESTINATION/$extra"
  [ -f "$p" ] || continue
  magic_ok "$p" || { echo "bad magic (not a GGUF): $p" >&2; exit 1; }
  echo "OK  $extra ($(du -sh "$p" | cut -f1))"
done
