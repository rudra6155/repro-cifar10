#!/usr/bin/env bash
# Full reproduction: train + eval every seed, then aggregate, plot, and regenerate FINDINGS.md.
#   SEEDS="0 1 2" (default)   RESUME=1 keeps finished seeds and resumes an interrupted one.
set -euo pipefail
source "$(dirname "$0")/_env.sh"
SEEDS=${SEEDS:-"0 1 2"}
t0=$(date +%s)
for seed in $SEEDS; do
  out=results/per_seed/seed_${seed}
  if [[ "${RESUME:-0}" == 1 && -f "$out/eval.json" && -f "$out/model_final.pt" && ! -f "$out/ckpt_last.pt" ]]; then
    echo "== seed $seed already complete, skipping (RESUME=1)"
    continue
  fi
  echo "== seed $seed: train"
  bash scripts/train.sh "$seed"
  echo "== seed $seed: eval (from saved checkpoint)"
  bash scripts/eval.sh "$seed"
done
echo "== aggregate + plots + FINDINGS table"
"$PY" -m repro.report --results results --findings FINDINGS.md --readme README.md
echo "== repro finished in $(( ($(date +%s) - t0) / 60 )) min"
