"""Offline human review of contradiction candidates via spreadsheets, with inter-reviewer agreement.

  # 1. export: sheet for reviewer A (all items) and reviewer B (a random sample for the agreement check)
  python scripts/review_sheet.py export --sample 60
  #    -> data/review/review_A.csv , data/review/review_B.csv   (open in Excel / Google Sheets)

  # 2. each reviewer fills the 'decision' column with y or n (blank = skip), saves as CSV (UTF-8)

  # 3. import: merges decisions back into data/contradiction_candidates.jsonl and reports Cohen's kappa
  python scripts/review_sheet.py import --a data/review/review_A.csv [--b data/review/review_B.csv]

Rule when both reviewers labelled an item and disagree: it is set back to pending and written to
data/review/disagreements.csv for discussion (adjudicate by editing the 'decision' in review_A.csv and re-importing).
"""
import argparse
import csv
import json
import os
import random
import shutil

CAND = "data/contradiction_candidates.jsonl"
OUT = "data/review"
YES, NO = {"y", "yes", "1", "true", "t"}, {"n", "no", "0", "false", "f"}


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def save(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_sheet(path, rows, idxs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig so Excel reads accents correctly
        w = csv.writer(f)
        w.writerow(["id", "original", "rewritten_to_contradict", "decision (y/n)", "notes"])
        for i in idxs:
            r = rows[i]
            cur = {True: "y", False: "n", None: ""}[r["reviewed"]]
            w.writerow([i, r["original"], r["contradicting"], cur, ""])


def read_sheet(path):
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            d = (row.get("decision (y/n)") or row.get("decision") or "").strip().lower()
            if d in YES:
                out[int(row["id"])] = True
            elif d in NO:
                out[int(row["id"])] = False
    return out


def kappa(a, b):
    ids = sorted(set(a) & set(b))
    n = len(ids)
    if n == 0:
        return None, 0
    po = sum(a[i] == b[i] for i in ids) / n
    pa, pb = sum(a[i] for i in ids) / n, sum(b[i] for i in ids) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (1.0 if pe == 1 else (po - pe) / (1 - pe)), n


def export(a):
    rows = load(a.cand)
    idxs = [i for i, r in enumerate(rows) if r["reviewed"] is None] if a.pending else list(range(len(rows)))
    write_sheet(os.path.join(OUT, "review_A.csv"), rows, idxs)
    sample = sorted(random.Random(0).sample(idxs, min(a.sample, len(idxs))))
    write_sheet(os.path.join(OUT, "review_B.csv"), rows, sample)
    print(f"wrote {OUT}/review_A.csv ({len(idxs)} items) and review_B.csv ({len(sample)} items)")


def do_import(a):
    rows = load(a.cand)
    shutil.copy(a.cand, a.cand + ".bak")
    A = read_sheet(a.a)
    B = read_sheet(a.b) if a.b else {}
    disagree = []
    for i, r in enumerate(rows):
        if i in A and i in B and A[i] != B[i]:
            r["reviewed"] = None
            disagree.append(i)
        elif i in A:
            r["reviewed"] = A[i]
        elif i in B:
            r["reviewed"] = B[i]
    save(a.cand, rows)
    if disagree:
        write_sheet(os.path.join(OUT, "disagreements.csv"), rows, disagree)
    k, n = kappa(A, B)
    print(f"accepted {sum(r['reviewed'] is True for r in rows)}, rejected {sum(r['reviewed'] is False for r in rows)}, "
          f"pending {sum(r['reviewed'] is None for r in rows)}")
    if k is not None:
        print(f"double-labelled: {n}, disagreements: {len(disagree)}, Cohen's kappa = {k:.3f}")
    else:
        print("no overlap between sheets A and B, so no agreement statistic")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("export"); e.add_argument("--sample", type=int, default=60); e.add_argument("--cand", default=CAND); e.add_argument("--pending", action="store_true")
    i = sp.add_parser("import"); i.add_argument("--a", required=True); i.add_argument("--b"); i.add_argument("--cand", default=CAND)
    a = ap.parse_args()
    export(a) if a.cmd == "export" else do_import(a)
