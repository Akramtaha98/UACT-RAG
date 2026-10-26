#!/usr/bin/env bash
# Post-hoc: re-select the controller with the verifier rejection/regeneration step INSIDE the validation/training rollouts,
# then evaluate it once on the test split as system UACTG (documented second test run; report it as post hoc).
#   nohup bash run_gated.sh > logs/run_gated.out 2>&1 &
# Step 1-3 need GPU (several hours); STEP=1 rollouts, STEP=2 train controller, STEP=3 test evaluation.
set -euo pipefail
cd "$(dirname "$0")"
DATA=data/techqa; WD=runs/main; G=${G:-1}
mkdir -p logs $WD/full_gated
cp $WD/full/estimator.pt $WD/full_gated/estimator.pt
r() { CUDA_VISIBLE_DEVICES=$G "$@"; }
STEP=${STEP:-all}
if [ "$STEP" = all ] || [ "$STEP" = 1 ]; then
  for S in train val; do
    r python scripts/build_controller_data.py --gate --data $DATA --split $S --estimator $WD/full/estimator.pt --out $WD/full_gated/rollout_$S.pkl > logs/gated_roll_$S.log 2>&1
  done
fi
if [ "$STEP" = all ] || [ "$STEP" = 2 ]; then
  python scripts/train_controller.py --train $WD/full_gated/rollout_train.pkl --val $WD/full_gated/rollout_val.pkl --outdir $WD/full_gated --risk_target 0.40 --min_coverage 0.20 > logs/gated_ctl.log 2>&1
  tail -n 3 logs/gated_ctl.log
fi
if [ "$STEP" = all ] || [ "$STEP" = 3 ]; then
  r python scripts/run_eval.py --data $DATA --split test --systems UACTG --tag _gated --set eval.use_llm_judge=true > logs/gated_test.log 2>&1
  grep -E "^UACTG" logs/gated_test.log || true
fi
echo "GATED DONE $(date)"
