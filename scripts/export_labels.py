"""Write a de-identified table of the claim-support audit labels (no passages, no claim text) for release.

  python scripts/export_labels.py --sheetA data/review/faith1_A.csv --sheetC data/review/faith1_C.csv \
      --sheetD data/review/faith1_annotator_D_labeled.csv --key data/review/faith1_key.json \
      --nli_trunc data/review/faith1_nli_trunc.json --out release/claim_support_labels.csv
Columns: claim id, system, TechQA question id, annotator 1, annotator 2, annotator 3 (1 = fully supported), majority label of the three,
NLI verdict (full cited passages), NLI verdict (text shown to the annotators, if --nli_trunc is given).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from claim_three_way_io import read_any  # noqa: E402

ap = argparse.ArgumentParser()
for k in ("sheetA", "sheetC", "sheetD", "key", "out"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--nli_trunc", default=None)
a = ap.parse_args()
A, C, D = read_any(a.sheetA), read_any(a.sheetC), read_any(a.sheetD)
key = {int(k): v for k, v in json.load(open(a.key)).items()}
T = {int(k): v["nli_trunc_supported"] for k, v in json.load(open(a.nli_trunc)).items()} if a.nli_trunc else {}
ids = sorted(set(key) & set(A) & set(C) & set(D))
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
import csv  # noqa: E402
with open(a.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["claim_id", "system", "techqa_question_id", "annotator1", "annotator2", "annotator3", "majority", "nli_full_evidence", "nli_matched_evidence"])
    for i in ids:
        m = int((int(A[i]) + int(C[i]) + int(D[i])) >= 2)
        w.writerow([i, key[i]["system"], key[i]["qid"], int(A[i]), int(C[i]), int(D[i]), m, int(key[i]["nli_supported"]), (int(T[i]) if i in T else "")])
print("wrote", a.out, "with", len(ids), "claims")
