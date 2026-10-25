"""Matched-coverage human audit: compare systems at the SAME number of answered questions, with answers judged by two
independent annotators. Fixes the 'only 20 answered' problem (risk at 20 answers has a +-0.2 interval).

  python scripts/matched_audit.py export --eval_dir runs/main/eval/techqa_test --data data/techqa --split test \
         --systems UACT B9 B6 B8 B2 --coverage 0.2 --name matched1
     -> data/review/matched1_A.csv (annotator 1), matched1_B.csv (annotator 2, SAME items, shuffled order is shared), matched1_key.json (hide)
  python scripts/matched_audit.py import --sheet_a data/review/matched1_A_done.csv --sheet_b data/review/matched1_B_done.csv \
         --key data/review/matched1_key.json
Each system contributes its top-k most confident answers (k = coverage * N, by the 'conf' field of <system>_forced.jsonl).
Answers to unanswerable questions are errors by construction and are NOT shown; humans judge only answerable ones.
"""
import argparse, csv, json, math, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from review_sheet import kappa, read_sheet  # noqa: E402

OUT = "data/review"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def export(a):
    qs = {q["qid"]: q for q in map(json.loads, open(os.path.join(a.data, f"{a.split}.jsonl")))}
    k = round(a.coverage * len(qs))
    rng = random.Random(a.seed)
    items, key, tops = [], {}, {}
    for s in a.systems:
        rows = [json.loads(l) for l in open(os.path.join(a.eval_dir, f"{s}_forced.jsonl")) if l.strip()]
        rows = [r for r in rows if r.get("answer") and r["qid"] in qs]
        rows.sort(key=lambda r: -(r.get("conf") if r.get("conf") is not None else -1))
        top = rows[:k]
        tops[s] = {"k": len(top), "unanswerable_auto_error": sum(not qs[r["qid"]]["answerable"] for r in top)}
        items += [(s, rank, r) for rank, r in enumerate(top) if qs[r["qid"]]["answerable"]]
    rng.shuffle(items)
    os.makedirs(OUT, exist_ok=True)
    hdr = ["id", "question", "reference_answer", "candidate_answer", "decision (y/n)", "notes"]
    rows_out = []
    for i, (s, rank, r) in enumerate(items):
        q = qs[r["qid"]]
        rows_out.append([i, " ".join(q["question"].split()), " ".join(q["reference_answer"].split()), " ".join(r["answer"].split()), "", ""])
        key[i] = {"system": s, "qid": r["qid"], "rank": rank}
    for suf in ("A", "B"):
        with open(os.path.join(OUT, f"{a.name}_{suf}.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(hdr); w.writerows(rows_out)
    json.dump({"k": k, "n_questions": len(qs), "tops": tops, "items": key}, open(os.path.join(OUT, f"{a.name}_key.json"), "w"), indent=1)
    print(f"k={k} per system; {len(rows_out)} items to judge; per system:", json.dumps(tops))
    print(f"identical sheets {a.name}_A.csv / {a.name}_B.csv written; give each annotator ONE file; keep {a.name}_key.json hidden")


def risk(sys_, K, H):
    """errors = unanswerable in top-k (auto) + answerable judged wrong; H: id -> bool (None = unjudged)"""
    ids = [i for i, v in K["items"].items() if v["system"] == sys_ and int(i) in H]
    wrong = sum(not H[int(i)] for i in ids) + K["tops"][sys_]["unanswerable_auto_error"]
    return wrong, K["tops"][sys_]["k"]


def do_import(a):
    from scipy.stats import fisher_exact
    K = json.load(open(a.key))
    A, B = read_sheet(a.sheet_a), read_sheet(a.sheet_b)
    kap, n = kappa(A, B)
    print(f"annotators judged {len(A)} / {len(B)} items; overlap {n}; Cohen kappa BEFORE adjudication = {kap:.2f}" if kap is not None else "no overlap")
    both = {i: (A[i] and B[i]) for i in A if i in B}      # conservative: correct only if both say yes
    anyy = {i: (A[i] or B[i]) for i in A if i in B}       # lenient
    systems = sorted({v["system"] for v in K["items"].values()} | set(K["tops"]))
    ref = a.ref
    print(f"\nSelective risk at matched coverage k={K['k']} (wrong / answered). Unanswerable answers count as wrong.")
    print(f"{'system':8s} {'ann1':>16s} {'ann2':>16s} {'both-correct':>16s} {'either-correct':>16s}")
    res = {}
    for s in systems:
        cells = []
        for H in (A, B, both, anyy):
            w, k = risk(s, K, H); lo, hi = wilson(w, k); cells.append((w, k)); 
            res.setdefault(s, []).append((w, k))
        print(f"{s:8s} " + " ".join(f"{w:3d}/{k:3d}={w / k:.2f}[{wilson(w, k)[0]:.2f},{wilson(w, k)[1]:.2f}]".rjust(16) for w, k in cells))
    if ref in res:
        print(f"\nFisher exact test vs {ref} (conservative both-correct rule, two-sided):")
        w0, k0 = res[ref][2]
        for s in systems:
            if s != ref:
                w, k = res[s][2]
                print(f"  {ref} vs {s}: risk {w0 / k0:.2f} vs {w / k:.2f}, p={fisher_exact([[w0, k0 - w0], [w, k - w]])[1]:.3f}")
    print("\nReport kappa BEFORE adjudication; adjudicate disagreements only after printing this table, and say so in the paper.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("export"); e.add_argument("--eval_dir", default="runs/main/eval/techqa_test"); e.add_argument("--data", default="data/techqa")
    e.add_argument("--split", default="test"); e.add_argument("--systems", nargs="+", default=["UACT", "B9", "B6", "B8", "B2"])
    e.add_argument("--coverage", type=float, default=0.2); e.add_argument("--name", default="matched1"); e.add_argument("--seed", type=int, default=0)
    i = sp.add_parser("import"); i.add_argument("--sheet_a", required=True); i.add_argument("--sheet_b", required=True)
    i.add_argument("--key", required=True); i.add_argument("--ref", default="UACT")
    a = ap.parse_args(); {"export": export, "import": do_import}[a.cmd](a)
