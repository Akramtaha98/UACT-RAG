#!/usr/bin/env bash
# Stages 1-4 on two GPUs (default: GPUs 1 and 2). Never touches the TEST split.
set -euo pipefail
cd "$(dirname "$0")"
G1=${G1:-1}; G2=${G2:-2}
DATA=data/techqa; WD=runs/main
STATES="data/states data/techqa/states"
mkdir -p $WD/full $WD/A1 $WD/A7 logs
r() { local g=$1; shift; CUDA_VISIBLE_DEVICES=$g "$@"; }

echo "[1/4] full estimator + feature cache: $(date)"
r $G1 python scripts/train_estimator.py --states_dir $STATES --kind five --features all --out $WD/full/estimator.pt > logs/est_full.log 2>&1
cp $WD/full/feat_*.npy $WD/A1/
cp $WD/full/feat_*.npy $WD/A7/

echo "[2/4] ablation estimators A1, A7: $(date)"
r $G1 python scripts/train_estimator.py --states_dir $STATES --kind five --features baseline --out $WD/A1/estimator.pt > logs/est_A1.log 2>&1 & p1=$!
r $G2 python scripts/train_estimator.py --states_dir $STATES --kind three --features all --out $WD/A7/estimator.pt > logs/est_A7.log 2>&1 & p2=$!
wait $p1; wait $p2

rollout() {  # variant gpu
  local V=$1 g=$2
  for S in train val; do
    r $g python scripts/build_controller_data.py --data $DATA --split $S --estimator $WD/$V/estimator.pt --out $WD/$V/rollout_$S.pkl > logs/roll_${V}_$S.log 2>&1
  done
  python scripts/train_controller.py --train $WD/$V/rollout_train.pkl --val $WD/$V/rollout_val.pkl --outdir $WD/$V > logs/ctl_$V.log 2>&1
}

echo "[3/4] rollouts, controllers, baselines: $(date)"
( rollout full $G1; rollout A7 $G1 ) & p1=$!
( rollout A1 $G2; r $G2 python scripts/fit_baselines.py --data $DATA --estimator $WD/full/estimator.pt > logs/baselines.log 2>&1 ) & p2=$!
wait $p1; wait $p2

echo "[4/4] smoke evaluation on VALIDATION (20 questions): $(date)"
r $G1 python scripts/run_eval.py --data $DATA --split val --systems B0 B2 B8 UACT A7 --max_q 20 --tag _smoke > logs/smoke_val.log 2>&1
tail -n 25 logs/smoke_val.log
echo "ALL DONE: $(date)"
