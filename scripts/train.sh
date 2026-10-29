#!/usr/bin/env bash
# Train one seed for the full schedule.  Usage: scripts/train.sh SEED   (RESUME=1 to continue a crashed run)
set -euo pipefail
source "$(dirname "$0")/_env.sh"
seed=${1:?usage: train.sh SEED}
out=results/per_seed/seed_${seed}
mkdir -p "$out"
extra=()
if [[ "${RESUME:-0}" == 1 ]]; then
  extra+=(--resume)
else
  rm -f "$out/eval.json" "$out/meta.json"   # never leave stale results next to a fresh run
fi
"$PY" -m repro.train --config "$CONFIG" --seed "$seed" --out "$out" "${extra[@]}" 2>&1 | tee "$out/train.stdout.log"
