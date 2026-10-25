#!/usr/bin/env bash
# FROZEN test evaluation on the leave-one-document-out engineering-manual benchmark. Run ONCE.
# Usage:  nohup bash run_test.sh > logs/run_emanual_test.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
G1=${G1:-1}; G2=${G2:-2}
DATA=data/emanual_loo; WD=runs/emanual; CFG=configs/emanual.yaml
OUT=$WD/eval/emanual_loo_test
mkdir -p logs
if [ -d "$OUT" ] && [ -z "${FORCE_RERUN:-}" ]; then
  echo "ERROR: $OUT exists. The test split must be touched once. Set FORCE_RERUN=1 only for a documented rerun."; exit 1
fi
J="--set eval.use_llm_judge=true"      # LLM judge is logged as a SECONDARY metric
SYSTEMS="B0 B1 B2 B3 B4 B6 B7 B8 B9 UACT A1 A2 A3 A4 A5 A6 A7"
r() { local g=$1; shift; CUDA_VISIBLE_DEVICES=$g "$@"; }
echo "START $(date)"

( echo "[GPU$G1] main"; r $G1 python scripts/run_eval.py --config $CFG --data $DATA --split test --systems $SYSTEMS $J > logs/emloo_test_main.log 2>&1
  echo "[GPU$G1] forced"; r $G1 python scripts/run_eval.py --config $CFG --data $DATA --split test --systems B6 B8 B9 UACT --forced $J > logs/emloo_test_forced.log 2>&1 ) & p1=$!

( echo "[GPU$G2] latency"; r $G2 python scripts/run_eval.py --config $CFG --data $DATA --split test --systems B2 B8 UACT --batch_size 1 --max_q 300 --tag _latency > logs/emloo_test_latency.log 2>&1
  for B in 1 2 3 4; do
    echo "[GPU$G2] budget $B"; r $G2 python scripts/run_eval.py --config $CFG --data $DATA --split test --systems B7 UACT --set control.max_rounds=$B --tag _B$B > logs/emloo_test_budget$B.log 2>&1
  done ) & p2=$!

wait $p1; wait $p2
python scripts/analyze.py --eval_dir $OUT > logs/emloo_analyze.log 2>&1 || true
grep -E "^(B[0-9]|UACT|A[0-9])" logs/emloo_test_main.log || true
echo "TEST ALL DONE $(date)"
