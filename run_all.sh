#!/usr/bin/env bash
# End-to-end pipeline on one H200. Usage:  bash run_all.sh data/techqa
# Prerequisites: data prepared (see README), indexes built, state data built.
set -euo pipefail
DATA=${1:-data/techqa}
CFG=${CFG:-configs/default.yaml}
WD=$(python - <<EOF
import yaml;print(yaml.safe_load(open("$CFG"))["paths"]["work_dir"])
EOF
)
STATES=${STATES:-"data/states $DATA/states"}
mkdir -p "$WD/full" "$WD/A1" "$WD/A7"

echo "== 1. estimators (full five-state, A1 relevance-only features, A7 three-state) =="
python scripts/train_estimator.py --config $CFG --states_dir $STATES --kind five  --features all      --out $WD/full/estimator.pt
python scripts/train_estimator.py --config $CFG --states_dir $STATES --kind five  --features baseline --out $WD/A1/estimator.pt
python scripts/train_estimator.py --config $CFG --states_dir $STATES --kind three --features all      --out $WD/A7/estimator.pt

echo "== 2. rollouts of all feasible actions (train + val) and controller training, per estimator variant =="
for V in full A1 A7; do
  python scripts/build_controller_data.py --config $CFG --data $DATA --split train --estimator $WD/$V/estimator.pt --out $WD/$V/rollout_train.pkl
  python scripts/build_controller_data.py --config $CFG --data $DATA --split val   --estimator $WD/$V/estimator.pt --out $WD/$V/rollout_val.pkl
  python scripts/train_controller.py --config $CFG --train $WD/$V/rollout_train.pkl --val $WD/$V/rollout_val.pkl --outdir $WD/$V
done

echo "== 3. baseline confidence model and thresholds (validation only) =="
python scripts/fit_baselines.py --config $CFG --data $DATA --estimator $WD/full/estimator.pt

echo "== 4. FREEZE: everything above is tuned. Final test evaluation (touch the test split once) =="
SYSTEMS="B0 B1 B2 B3 B4 B6 B7 B8 B9 UACT A1 A2 A3 A4 A5 A6 A7"
python scripts/run_eval.py --config $CFG --data $DATA --split test --systems $SYSTEMS
python scripts/run_eval.py --config $CFG --data $DATA --split test --systems B6 B8 B9 UACT --forced      # for risk-coverage / AURC
python scripts/run_eval.py --config $CFG --data $DATA --split test --systems B2 B8 UACT --batch_size 1 --tag _latency --max_q 300
for B in 1 2 3 4; do   # budget sweep (RQ3). NOTE: controllers were trained for the default max_rounds; retrain for strict comparisons
  python scripts/run_eval.py --config $CFG --data $DATA --split test --systems B7 UACT --set control.max_rounds=$B --tag _B$B
done

echo "== 5. tables, curves, paired statistics =="
python scripts/analyze.py --eval_dir $WD/eval/$(basename $DATA)_test
