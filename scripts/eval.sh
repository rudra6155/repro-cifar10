#!/usr/bin/env bash
# Re-evaluate a seed's final checkpoint from disk.  Usage: scripts/eval.sh SEED
set -euo pipefail
source "$(dirname "$0")/_env.sh"
seed=${1:?usage: eval.sh SEED}
out=results/per_seed/seed_${seed}
"$PY" -m repro.eval --ckpt "$out/model_final.pt" --out "$out/eval.json"
