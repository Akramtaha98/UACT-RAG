"""Write a de-identified table of the claim-support audit labels (no passages, no claim text) for release.

  python scripts/export_labels.py --sheetA data/review/faith1_A.csv --sheetC data/review/faith1_C.csv \
      --final data/review/faith1_final.csv --key data/review/faith1_key.json --out release/claim_support_labels.csv
Columns: claim id, system, TechQA question id, annotator 1 label, annotator 2 label, settled label, NLI verdict (original evidence),
NLI verdict (matched evidence, if --nli_trunc is given), whether the claim was in the 60-claim double-labelled subset of the first round.
Only publish it if you and your co-author agree; the paper currently says the labels are not released.
"""
import argparse
import csv
import json
import os

ap = argparse.ArgumentParser()
for k in ("sheetA", "sheetC", "final", "key", "out"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--sheetB", default="data/review/faith1_B.csv")
ap.add_argument("--nli_trunc", default=None)
a = ap.parse_args()


def lab(path):
    d = {}
    for r in csv.DictReader(open(path, newline="", encoding="utf-8-sig")):
        d[int(r["id"])] = (r.get("decision (y/n)") or "").strip().lower()
    return d


A, C, F = lab(a.sheetA), lab(a.sheetC), lab(a.final)
B = set(lab(a.sheetB)) if os.path.exists(a.sheetB) else set()
key = {int(k): v for k, v in json.load(open(a.key)).items()}
T = {int(k): v["nli_trunc_supported"] for k, v in json.load(open(a.nli_trunc)).items()} if a.nli_trunc else {}
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["claim_id", "system", "techqa_question_id", "annotator1", "annotator2", "settled", "nli_original_evidence", "nli_matched_evidence", "in_first_round_double_labelled_subset"])
    for i in sorted(key):
        w.writerow([i, key[i]["system"], key[i]["qid"], A.get(i), C.get(i), F.get(i), int(key[i]["nli_supported"]), (int(T[i]) if i in T else ""), int(i in B)])
print("wrote", a.out)
