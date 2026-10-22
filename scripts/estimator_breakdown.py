"""Per-source breakdown of the estimator on the state-test split (read-only; no refitting, no tuning).
  python scripts/estimator_breakdown.py --states_dir data/states data/techqa/states --variant runs/main/full
Splits test states into synthetic RAGBench states vs real TechQA retrieval states; bootstrap 95% CIs."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from uact.estimator import Estimator, bad_indices, map_label, state_names
from uact.features import FEATURE_NAMES
from uact.utils import read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--states_dir", nargs="+", required=True)
ap.add_argument("--variant", default="runs/main/full")
a = ap.parse_args()
rows = []
for d in a.states_dir:
    for suf in ("", "_retrieved"):
        p = os.path.join(d, f"test{suf}.jsonl")
        if os.path.exists(p):
            rows += read_jsonl(p)
X = np.load(os.path.join(a.variant, "feat_test.npy"))
assert len(rows) == len(X), (len(rows), len(X))
est = Estimator.load(os.path.join(a.variant, "estimator.pt"))
kind = est.kind
names = state_names(kind)
y = np.array([names.index(map_label(r["state"], kind)) for r in rows])
P = est.predict_proba(X)
bad = np.isin(y, bad_indices(kind)).astype(int)
pbad = P[:, bad_indices(kind)].sum(1)
fi = {n: i for i, n in enumerate(FEATURE_NAMES)}
base = {"estimator": pbad, "top1_sim": 1 - X[:, fi["rerank_top1"]], "llm_autorater": 1 - X[:, fi["llm_suff"]]}
src = np.array([r.get("source", "") for r in rows])
rng = np.random.default_rng(0)

def ci(fn, idx):
    v = [fn(rng.choice(idx, len(idx))) for _ in range(500)]
    return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]

out = {}
for gname, mask in (("all", np.ones(len(rows), bool)), ("techqa_retrieved", src == "techqa_retrieved"), ("synthetic_ragbench", src != "techqa_retrieved")):
    idx = np.where(mask)[0]
    g = {"n": int(len(idx)), "insufficient_prevalence": round(float(bad[idx].mean()), 3),
         "accuracy": round(float((P[idx].argmax(1) == y[idx]).mean()), 3),
         "macro_f1": round(float(f1_score(y[idx], P[idx].argmax(1), average="macro")), 3)}
    if 0 < bad[idx].sum() < len(idx):
        for k, s in base.items():
            g[f"auroc_{k}"] = [round(float(roc_auc_score(bad[idx], s[idx])), 3), *ci(lambda i: roc_auc_score(bad[i], s[i]) if 0 < bad[i].sum() < len(i) else np.nan, idx)]
            g[f"auprc_{k}"] = round(float(average_precision_score(bad[idx], s[idx])), 3)
    out[gname] = g
ci_ = names.index("contradictory") if "contradictory" in names else None
if ci_ is not None:
    m = y == ci_
    out["contradictory"] = {"n": int(m.sum()), "recall": round(float((P[m].argmax(1) == ci_).mean()), 3),
                            "precision": round(float((y[P.argmax(1) == ci_] == ci_).mean()), 3)}
print(json.dumps(out, indent=2))
json.dump(out, open(os.path.join(a.variant, "estimator_breakdown.json"), "w"), indent=2)
