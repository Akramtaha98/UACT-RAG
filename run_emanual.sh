#!/usr/bin/env bash
# Leave-one-document-out engineering-manual benchmark: prepare, index, rollouts, controllers, baselines (NO test data used).
# Usage:  nohup bash run_emanual.sh > logs/run_emanual.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
G1=${G1:-1}; G2=${G2:-2}
DATA=data/emanual_loo; WD=runs/emanual; CFG=configs/emanual.yaml
mkdir -p logs $WD/full $WD/A1 $WD/A7
sed "s#work_dir: runs/main#work_dir: $WD#" configs/default.yaml > $CFG
grep -n "work_dir" $CFG
for V in full A1 A7; do cp runs/main/$V/estimator.pt $WD/$V/estimator.pt; done   # estimators are domain-trained already; reused as-is
r() { local g=$1; shift; CUDA_VISIBLE_DEVICES=$g "$@"; }

echo "[1/4] prepare: $(date)"
python scripts/make_emanual_loo.py --data data --out $DATA | tee logs/emloo_prepare.log

echo "[2/4] index: $(date)"
r $G1 python scripts/build_index.py --config $CFG --data $DATA > logs/emloo_index.log 2>&1

rollout() {  # variant gpu
  local V=$1 g=$2
  for S in train val; do
    r $g python scripts/build_controller_data.py --config $CFG --data $DATA --split $S --estimator $WD/$V/estimator.pt --out $WD/$V/rollout_$S.pkl > logs/emloo_roll_${V}_$S.log 2>&1
  done
  python scripts/train_controller.py --config $CFG --train $WD/$V/rollout_train.pkl --val $WD/$V/rollout_val.pkl --outdir $WD/$V --risk_target 0.40 --min_coverage 0.20 > logs/emloo_ctl_$V.log 2>&1
}
echo "[3/4] rollouts, controllers, baselines: $(date)"
( rollout full $G1; rollout A7 $G1 ) & p1=$!
( rollout A1 $G2; r $G2 python scripts/fit_baselines.py --config $CFG --data $DATA --estimator $WD/full/estimator.pt --risk_target 0.40 --min_coverage 0.20 > logs/emloo_baselines.log 2>&1 ) & p2=$!
wait $p1; wait $p2

echo "[4/4] selected controllers (validation only): $(date)"
for V in full A1 A7; do echo $V; tail -n 1 logs/emloo_ctl_$V.log; done
echo "EMLOO PREP DONE: $(date)"
