#!/usr/bin/env bash
# Two robustness experiments for the revision. Both are CPU-only, use cached features and rollouts, and do not touch
# the test split or modify anything under runs/main.
#   bash run_robustness.sh              # both experiments
#   STEP=1 bash run_robustness.sh       # only the document-grouped, multi-seed estimator evaluation
#   STEP=2 bash run_robustness.sh       # only the cross-validated controller / state-representation comparison
#   STEP=3 bash run_robustness.sh       # only the training-composition experiment (synthetic-only / real-only / mixed)
# Needs: runs/main/full/feat_{train,val,cal,test}.npy and rollout_{train,val}.pkl (made by run_all.sh), data/states, data/techqa.
set -euo pipefail
cd "$(dirname "$0")"
WD=runs/main; DATA=data/techqa; OUT=$WD/revision
mkdir -p "$OUT" logs
STEP=${STEP:-all}

if [ "$STEP" = all ] || [ "$STEP" = 1 ]; then
  echo "== 1. estimator: document-grouped 5-fold CV, 5 seeds (trains 25 small MLPs; CPU is fine) =="
  python scripts/docsplit_seeds.py --states_dir data/states $DATA/states --techqa_dir $DATA --variant $WD/full \
         --seeds ${SEEDS:-5} --folds 5 --out $OUT 2>&1 | tee logs/rev_docsplit.log
fi
if [ "$STEP" = all ] || [ "$STEP" = 2 ]; then
  echo "== 2. controller: cross-validated lambda/mu/delta and five/three/binary/none state comparison =="
  python scripts/cv_controller.py --variant $WD/full --folds 5 --repeats ${REPEATS:-3} --out $OUT \
         --save_controllers runs/cv_select 2>&1 | tee logs/rev_cv_controller.log
fi
if [ "$STEP" = 3 ]; then
  echo "== 3. transfer isolation: estimator trained on synthetic-only / real-only / mixed, tested on held-out real TechQA states =="
  python scripts/train_composition.py --states_dir data/states $DATA/states --techqa_dir $DATA --variant $WD/full \
         --seeds ${SEEDS:-5} --folds 5 --max_df 5 --out $OUT 2>&1 | tee logs/rev_composition.log
fi
echo ROBUSTNESS DONE
