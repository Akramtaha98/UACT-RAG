"""Isolating 'transfer': does the training mix (synthetic only, real only, mixed) change how well the estimator detects
insufficient evidence on REAL TechQA retrieval states?  (Review item: the transfer claim is not isolated.)

Same document-grouped K-fold protocol as docsplit_seeds.py (no document or question is shared between training and the held-out
fold). For every fold and seed the estimator is trained three times, always on the same training groups but with different
composition of the training rows:
    synthetic_only : only constructed RAGBench states
    real_only      : only real TechQA retrieval states
    mixed          : both (what the main experiments used)
and each model scores the held-out fold. We report AUROC for insufficient evidence on the held-out REAL states (and on the
held-out synthetic states for reference), next to the top-1 similarity and LLM-autorater baselines on the same states, with
question-level bootstrap intervals for each training mix and for the paired differences.

  python scripts/train_composition.py --states_dir data/states data/techqa/states --techqa_dir data/techqa \
         --variant runs/main/full --seeds 5 --folds 5 --max_df 5 --out runs/main/revision

CPU is fine (15 small MLPs per seed). Needs feat_{train,val,cal,test}.npy from train_estimator.py. Writes <out>/composition.json.
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

MIXES = ("synthetic_only", "real_only", "mixed")


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
    bad = np.isin(y, bad_indices("five")).astype(int)
    qid = np.array([norm_q(r["question"]) for r in rows])
    real = np.array([r.get("source", "") == "techqa_retrieved" for r in rows])
    gid = build_groups(rows, a.techqa_dir, a.max_df)
    share = float(np.bincount(gid).max() / N)
    print(f"{N} states ({int(real.sum())} real, {int((~real).sum())} synthetic), {len(np.unique(gid))} groups, largest {share:.1%}")
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
                keep = {"synthetic_only": ~real, "real_only": real, "mixed": np.ones(N, bool)}[mix]
                t_, v_, c_ = tr & keep, va & keep, ca & keep
                est = Estimator("five", cfg["estimator"]["hidden"], None)
                est.fit(X[t_], y[t_], X[v_], y[v_], X[c_], y[c_], epochs=cfg["estimator"]["epochs"],
                        lr=cfg["estimator"]["lr"], wd=cfg["estimator"]["weight_decay"], seed=seed * 100 + f)
                oof[mix][seed, te] = p_bad(est.predict_proba(X[te]), "five")
        print("seed", seed, {m: round(auroc(bad[real], oof[m][seed, real]), 3) for m in MIXES}, flush=True)

    rng = np.random.default_rng(0)
    rep = {"n_states": int(N), "n_real": int(real.sum()), "seeds": a.seeds, "folds": a.folds, "max_df": a.max_df,
           "slices": {}}
    for sname, m in (("real_techqa", real), ("synthetic", ~real)):
        idx = np.where(m)[0]
        qs = np.unique(qid[idx])
        by_q = {q: idx[qid[idx] == q] for q in qs}

        def pt(ii):
            d = {mx: float(np.mean([auroc(bad[ii], oof[mx][k, ii]) for k in range(a.seeds)])) for mx in MIXES}
            for nm, sc in base.items():
                d[nm] = auroc(bad[ii], sc[ii])
            return d

        point = pt(idx)
        B = []
        for _ in range(a.boot):
            ii = np.concatenate([by_q[q] for q in rng.choice(qs, len(qs))])
            if 0 < bad[ii].sum() < len(ii):
                B.append(pt(ii))

        def ci(fn):
            v = [fn(b) for b in B]
            return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]

        rep["slices"][sname] = {
            "states": int(len(idx)), "questions": int(len(qs)),
            "auroc": {k: round(v, 3) for k, v in point.items()},
            "ci95": {k: ci(lambda b, k=k: b[k]) for k in point},
            "mixed_minus_real_only": ci(lambda b: b["mixed"] - b["real_only"]),
            "mixed_minus_synthetic_only": ci(lambda b: b["mixed"] - b["synthetic_only"]),
            "real_only_minus_synthetic_only": ci(lambda b: b["real_only"] - b["synthetic_only"]),
            "mixed_minus_autorater": ci(lambda b: b["mixed"] - b["llm_autorater"]),
            "real_only_minus_autorater": ci(lambda b: b["real_only"] - b["llm_autorater"]),
            "synthetic_only_minus_autorater": ci(lambda b: b["synthetic_only"] - b["llm_autorater"])}
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out, "composition.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
