"""Claim-support audit with a third annotator: pairwise kappas, majority-vote labels, and the verifier compared with the MAJORITY.
  python scripts/claim_three_way.py --sheetA data/review/faith1_A.csv --sheetC data/review/faith1_C.csv --sheetD data/review/faith1_annotator_D.xlsx \
      --key data/review/faith1_key.json --nli_trunc data/review/faith1_nli_trunc.json
sheetA = annotator 1 (final own labels), sheetC = annotator 2's independent relabelling of all 200 claims, sheetD = annotator 3. Reports kappas BEFORE any discussion."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from faithfulness_audit import kappa, kappa_vec, read as read_csv, wilson  # noqa: E402


def read_any(p):
    if not p.lower().endswith(".xlsx"):
        return read_csv(p)
    import openpyxl
    ws = openpyxl.load_workbook(p, data_only=True)["Items"]; rows = list(ws.iter_rows(values_only=True))
    h = [str(x or "").strip().lower() for x in rows[0]]; di = next(k for k, x in enumerate(h) if x.startswith("decision"))
    out = {}
    for r in rows[1:]:
        d = str(r[di] or "").strip().lower()
        if d in ("y", "yes", "1", "true"): out[int(r[h.index("id")])] = True
        elif d in ("n", "no", "0", "false"): out[int(r[h.index("id")])] = False
    return out


ap = argparse.ArgumentParser()
for k in ("sheetA", "sheetC", "sheetD", "key"): ap.add_argument("--" + k, required=True)
ap.add_argument("--nli_trunc", default=None); ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
A, C, D = read_any(a.sheetA), read_any(a.sheetC), read_any(a.sheetD)
key = {int(k): v for k, v in json.load(open(a.key)).items()}
T = {int(i): v["nli_trunc_supported"] for i, v in json.load(open(a.nli_trunc)).items()} if a.nli_trunc else None
ids = sorted(set(A) & set(C) & set(D) & set(key))
print(f"{len(ids)} claims labelled by all three annotators")
for nm, X, Y in (("1-2", A, C), ("1-3", A, D), ("2-3", C, D)):
    k, n = kappa(X, Y); print(f"kappa {nm} = {k:.2f} (n={n}); supported rate {np.mean([X[i] for i in ids]):.2f} vs {np.mean([Y[i] for i in ids]):.2f}")
M = {i: (A[i] + C[i] + D[i]) >= 2 for i in ids}
print("\nPer system: share of claims judged supported by each annotator, majority vote with answer-clustered bootstrap 95% interval,")
print("and how many claims the verifier passed (full evidence / matched evidence) and how many of those the majority judged supported.")
rng = np.random.default_rng(0)
cl = {i: f"{key[i]['system']}|{key[i]['qid']}" for i in ids}
for s in sorted({key[i]["system"] for i in ids}) + ["ALL"]:
    v = [i for i in ids if s == "ALL" or key[i]["system"] == s]
    g = {}
    for i in v: g.setdefault(cl[i], []).append(i)
    ks = list(g); bs = []; kn = []; kt = []
    for _ in range(a.boot):
        ii = [i for j in rng.integers(0, len(ks), len(ks)) for i in g[ks[j]]]; bs.append(np.mean([M[i] for i in ii]))
        if s == "ALL":
            kn.append(kappa_vec([M[i] for i in ii], [key[i]["nli_supported"] for i in ii]))
            if T and all(i in T for i in ii): kt.append(kappa_vec([M[i] for i in ii], [T[i] for i in ii]))
    nli = [i for i in v if key[i]["nli_supported"]]
    tm = [i for i in v if T and T.get(i)]
    print(f"{s:6s} n={len(v):3d} | ann1 {np.mean([A[i] for i in v]):.2f} ann2 {np.mean([C[i] for i in v]):.2f} ann3 {np.mean([D[i] for i in v]):.2f} | majority {np.mean([M[i] for i in v]):.3f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}] | verifier full {len(nli)} (majority-supported {sum(M[i] for i in nli)}) | verifier matched {len(tm)} (majority-supported {sum(M[i] for i in tm)})")
    if s == "ALL":
        k0 = kappa_vec([M[i] for i in ids], [key[i]["nli_supported"] for i in ids])
        print(f"\nVerifier vs MAJORITY label, full evidence: kappa = {k0:.2f} [{np.nanpercentile(kn, 2.5):.2f}, {np.nanpercentile(kn, 97.5):.2f}]")
        if T and all(i in T for i in ids):
            k1 = kappa_vec([M[i] for i in ids], [T[i] for i in ids])
            print(f"Verifier vs MAJORITY label, matched evidence: kappa = {k1:.2f} [{np.nanpercentile(kt, 2.5):.2f}, {np.nanpercentile(kt, 97.5):.2f}]")
