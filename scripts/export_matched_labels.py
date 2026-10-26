"""De-identified release of the matched-coverage audit labels (no question, reference or answer text).
  python scripts/export_matched_labels.py --key data/review/matched1_key.json --sheets data/review/matched1_A_done.csv data/review/matched1_B_done.csv data/review/matched1_annotator_C.csv --out release/matched_audit_labels.csv
Columns: item id, system, TechQA question id, rank by confidence, label of annotator 1, 2, 3 (y/n), majority label.
Annotator identities are not recorded."""
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from review_sheet import read_sheet  # noqa: E402
from matched_audit import read_sheet as read_any  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--key", required=True); ap.add_argument("--sheets", nargs=3, required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
K = json.load(open(a.key))["items"]; S = [read_any(p) for p in a.sheets]
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["item", "system", "techqa_qid", "rank_by_confidence", "annotator1", "annotator2", "annotator3", "majority"])
    for i, v in sorted(K.items(), key=lambda t: int(t[0])):
        ys = [s.get(int(i)) for s in S]
        w.writerow([i, v["system"], v["qid"], v["rank"]] + [{True: "y", False: "n", None: ""}[y] for y in ys] + [{True: "y", False: "n"}[sum(bool(y) for y in ys) >= 2] if None not in ys else ""])
print("wrote", a.out, len(K), "items")
