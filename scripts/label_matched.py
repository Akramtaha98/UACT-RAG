"""Label-matched comparison of constructed and real evidence states.

Review point: constructed and real states are labelled by different rules, so the AUROC gap between them might reflect the
definitions and not a failure to transfer. This script compares the two slices on contrasts whose label rule is as similar as the
data allow, using the same document-grouped K-fold protocol and seeds as docsplit_seeds.py (every state is scored by a model that
never saw its group; mixed training, and real-only training for the real contrasts).

Contrasts (positive class first):
  constructed  irrelevant (swapped context)      vs sufficient   -> "the right passage is absent"
  real         irrelevant (gold technote missed) vs sufficient   -> same idea, answerable questions only
  constructed  missing (relevant text removed)   vs sufficient
  real         missing (unanswerable question)   vs sufficient   -> NOT the same thing: answerability, not retrieval
  constructed  contradictory                     vs sufficient   (no real counterpart)
Each contrast gives AUROC of the estimator, top-1 similarity and the LLM autorater with question-level bootstrap intervals,
plus estimator-minus-autorater. If the real "irrelevant vs sufficient" AUROC is close to the constructed one, the gap is mostly
due to the rule for missing/unanswerable; if it is still much lower, the gap is not just a label-definition effect.

  python scripts/label_matched.py --states_dir data/states data/techqa/states --techqa_dir data/techqa \
         --variant runs/main/full --seeds 5 --folds 5 --max_df 5 --out runs/main/revision

CPU only. Needs feat_{train,val,cal,test}.npy. Writes <out>/label_matched.json.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docsplit_seeds import assign_folds, auroc, build_groups, load_pooled, norm_q, split_rest  # noqa: E402
from uact.utils import load_config  # noqa: E402

MIXES = ("mixed", "real_only")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--states_dir", nargs="+", required=True)
    ap.add_argument("--techqa_dir", default=None)
    ap.add_argument("--variant", default="runs/main/full")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--max_df", type=int, default=5)
    ap.add_argument("--max_group_share", type=float, default=0.35)
    ap.add_argument("--out", default="runs/main/revision")
    a = ap.parse_args()
    cfg = load_config(a.config)
    from uact.estimator import Estimator, bad_indices, map_label, p_bad, state_names
    from uact.features import FEATURE_NAMES

    rows, X, _ = load_pooled(a.states_dir, a.variant)
    N = len(rows)
    names = state_names("five")
    y = np.array([names.index(map_label(r["state"], "five")) for r in rows])
    lab = np.array([names[i] for i in y])
    qid = np.array([norm_q(r["question"]) for r in rows])
    real = np.array([r.get("source", "") == "techqa_retrieved" for r in rows])
    gid = build_groups(rows, a.techqa_dir, a.max_df)
    share = float(np.bincount(gid).max() / N)
    if share > a.max_group_share:
        sys.exit("largest group too large; lower --max_df")
    fi = {n: i for i, n in enumerate(FEATURE_NAMES)}
    base = {"top1_similarity": 1 - X[:, fi["rerank_top1"]], "llm_autorater": 1 - X[:, fi["llm_suff"]]}

    oof = {m: np.full((a.seeds, N), np.nan) for m in MIXES}
    for seed in range(a.seeds):
        rng = np.random.default_rng(1000 + seed)
        fold = assign_folds(gid, a.folds, rng)
        for f in range(a.folds):
            te = fold == f
            tr, va, ca = split_rest(gid, ~te, rng)
            for mix in MIXES:
                keep = np.ones(N, bool) if mix == "mixed" else real
                t_, v_, c_ = tr & keep, va & keep, ca & keep
                est = Estimator("five", cfg["estimator"]["hidden"], None)
                est.fit(X[t_], y[t_], X[v_], y[v_], X[c_], y[c_], epochs=cfg["estimator"]["epochs"],
                        lr=cfg["estimator"]["lr"], wd=cfg["estimator"]["weight_decay"], seed=seed * 100 + f)
                oof[mix][seed, te] = p_bad(est.predict_proba(X[te]), "five")
        print("seed", seed, "done", flush=True)

    contrasts = [
        ("constructed: irrelevant vs sufficient", ~real, "irrelevant"),
        ("real: irrelevant vs sufficient (answerable questions)", real, "irrelevant"),
        ("constructed: missing vs sufficient", ~real, "missing"),
        ("real: missing (unanswerable) vs sufficient", real, "missing"),
        ("constructed: contradictory vs sufficient", ~real, "contradictory"),
    ]
    rng = np.random.default_rng(0)
    rep = {"n_states": int(N), "seeds": a.seeds, "folds": a.folds, "contrasts": {}}
    for nm, sl, pos in contrasts:
        m = sl & np.isin(lab, [pos, "sufficient"])
        idx = np.where(m)[0]
        if len(idx) == 0 or (lab[idx] == pos).sum() == 0:
            rep["contrasts"][nm] = "no states"
            continue
        yy = (lab == pos).astype(int)
        qs = np.unique(qid[idx])
        by_q = {q: idx[qid[idx] == q] for q in qs}

        def pt(ii):
            d = {}
            for mx in MIXES:
                d["estimator_" + mx] = float(np.mean([auroc(yy[ii], oof[mx][k, ii]) for k in range(a.seeds)]))
            for b, sc in base.items():
                d[b] = auroc(yy[ii], sc[ii])
            return d

        point = pt(idx)
        B = []
        for _ in range(a.boot):
            ii = np.concatenate([by_q[q] for q in rng.choice(qs, len(qs))])
            if 0 < yy[ii].sum() < len(ii):
                B.append(pt(ii))

        def ci(fn):
            v = [fn(b) for b in B]
            return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]

        ent = {k: round(v, 3) for k, v in point.items()}
        rep["contrasts"][nm] = {
            "positives": int((lab[idx] == pos).sum()), "negatives": int((lab[idx] == "sufficient").sum()),
            "questions": int(len(qs)), "auroc": ent,
            "ci95": {k: ci(lambda b, k=k: b[k]) for k in point},
            "mixed_minus_autorater": ci(lambda b: b["estimator_mixed"] - b["llm_autorater"]),
            "mixed_minus_similarity": ci(lambda b: b["estimator_mixed"] - b["top1_similarity"])}
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out, "label_matched.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
