#!/usr/bin/env bash
# Re-run stages 3-4 after the correctness rule changed. Estimators do not depend on correctness, so they are kept.
set -euo pipefail
cd "$(dirname "$0")"
G1=${G1:-1}; G2=${G2:-2}
DATA=data/techqa; WD=runs/main
mkdir -p logs
r() { local g=$1; shift; CUDA_VISIBLE_DEVICES=$g "$@"; }

rollout() {  # variant gpu
  local V=$1 g=$2
  for S in train val; do
    r $g python scripts/build_controller_data.py --data $DATA --split $S --estimator $WD/$V/estimator.pt --out $WD/$V/rollout_$S.pkl > logs/roll_${V}_$S.log 2>&1
  done
  python scripts/train_controller.py --train $WD/$V/rollout_train.pkl --val $WD/$V/rollout_val.pkl --outdir $WD/$V > logs/ctl_$V.log 2>&1
}

echo "[1/2] rollouts, controllers, baselines: $(date)"
( rollout full $G1; rollout A7 $G1 ) & p1=$!
( rollout A1 $G2; r $G2 python scripts/fit_baselines.py --data $DATA --estimator $WD/full/estimator.pt > logs/baselines.log 2>&1 ) & p2=$!
wait $p1; wait $p2

echo "[2/2] smoke evaluation on VALIDATION (20 questions): $(date)"
r $G1 python scripts/run_eval.py --data $DATA --split val --systems B0 B2 B8 UACT A7 --max_q 20 --tag _smoke2 > logs/smoke_val2.log 2>&1
grep -E "^(B[0-9]|UACT|A[0-9])" logs/smoke_val2.log || true
tail -n 2 logs/ctl_full.log
echo "ALL DONE: $(date)"
