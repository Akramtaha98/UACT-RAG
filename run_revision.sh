#!/usr/bin/env bash
# Revision analyses. Part A is read-only (no GPU, safe). Part B refits the baseline thresholds under the controller's rule
# and re-runs B6/B8/B9 on the test split; it is a documented, post-hoc rerun and must be reported as such.
#   bash run_revision.sh            # part A
#   REFIT=1 bash run_revision.sh    # part A + part B
set -euo pipefail
cd "$(dirname "$0")"
WD=runs/main; DATA=data/techqa; EV=$WD/eval/techqa_test; OUT=$WD/revision
mkdir -p "$OUT" logs

echo "== A1. ECE under both definitions, per slice =="
python scripts/plot_reliability.py --states_dir data/states $DATA/states --variant $WD/full --out figures | tee logs/rev_ece.log
echo "== A2. passage / technote overlap between splits =="
python scripts/revision_analyses.py overlap --states_dir data/states $DATA/states --techqa_dir $DATA --out $OUT
echo "== A2b. question-text collisions and overlap-free AUROC =="
python scripts/revision_analyses.py collisions --states_dir data/states $DATA/states --out $OUT
python scripts/revision_analyses.py overlap_free_auroc --states_dir data/states $DATA/states --variant $WD/full --out $OUT
echo "== A3. AUROC with question-level (cluster) bootstrap =="
python scripts/revision_analyses.py cluster_auroc --states_dir data/states $DATA/states --variant $WD/full --out $OUT
echo "== A4. risk at matched coverage (forced runs) =="
python scripts/revision_analyses.py matched_coverage --eval_dir $EV --systems UACT A7 B6 B8 B9 --out $OUT
echo "== A5. actions and retrieval outcomes =="
python scripts/revision_analyses.py actions --eval_dir $EV --systems UACT A6 A7 B8 --out $OUT
echo "== A6. paired tests (McNemar + paired bootstrap, Holm-adjusted) =="
python scripts/analyze.py --eval_dir $EV --ref B2 B6 B8 B9 --main UACT | tee logs/rev_paired.log

if [ -n "${REFIT:-}" ]; then
  echo "== B. refit baseline thresholds under the controller rule (risk<=0.40, coverage>=0.20) in a COPY =="
  RW=runs/revision_refit
  [ -e "$RW" ] && { echo "$RW exists; move it first"; exit 1; }
  cp -r $WD $RW
  sed "s#work_dir: runs/main#work_dir: $RW#" configs/default.yaml > configs/revision.yaml
  python scripts/fit_baselines.py --config configs/revision.yaml --data $DATA --estimator $RW/full/estimator.pt --risk_target 0.40 --min_coverage 0.20
  G=${G1:-1}
  CUDA_VISIBLE_DEVICES=$G python scripts/run_eval.py --config configs/revision.yaml --data $DATA --split test --systems B6 B8 B9 --set eval.use_llm_judge=true > logs/rev_refit_main.log 2>&1
  CUDA_VISIBLE_DEVICES=$G python scripts/run_eval.py --config configs/revision.yaml --data $DATA --split test --systems B6 B8 B9 --forced > logs/rev_refit_forced.log 2>&1
  echo "Refit results: $RW/eval/techqa_test  (compare with $EV; report as a post-hoc rerun)"
fi
echo REVISION DONE
