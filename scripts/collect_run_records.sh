#!/usr/bin/env bash
# Collect the records of the reported runs into reproducibility/ so that code, settings and numbers can be checked together.
# Run from the repository root with the venv active:  bash scripts/collect_run_records.sh
set -u
D=reproducibility/run_records; mkdir -p $D
git rev-parse HEAD > $D/git_commit.txt; git status --short > $D/git_status.txt
cp configs/*.yaml $D/ 2>/dev/null
for V in full A1 A7; do [ -f runs/main/$V/controller_grid.json ] && cp runs/main/$V/controller_grid.json $D/controller_grid_$V.json; done
cp runs/main/revision/*.json $D/ 2>/dev/null
python scripts/reconcile_selection.py --variant full > $D/reconcile_full.txt 2>&1
python scripts/env_report.py > $D/environment.txt 2>&1
for f in runs/main/eval/techqa_test/*.json; do cp "$f" $D/ 2>/dev/null; done
python - <<'PY'
import glob, json, os
out = {}
for f in sorted(glob.glob("runs/main/eval/techqa_test/*.jsonl")):
    n = a = 0
    for l in open(f):
        r = json.loads(l); n += 1; a += bool(r.get("answered"))
    out[os.path.basename(f)] = {"records": n, "answered": a}
json.dump(out, open("reproducibility/run_records/eval_counts.json", "w"), indent=1)
PY
sha256sum runs/main/eval/techqa_test/*.jsonl > $D/eval_sha256.txt 2>/dev/null
ls $D
