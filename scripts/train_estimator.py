"""Train + calibrate the evidence-state estimator and evaluate it (RQ2).

  python scripts/train_estimator.py --states_dir data/states [--extra data/techqa/states] --out runs/main/estimator.pt
Variants:  --kind three   (ablation A7)    --features baseline   (ablation A1: relevance-only features)
Splits used: train -> fit; val -> early stopping; cal -> temperature scaling; test -> final report (touch once).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uact.estimator import Estimator, bad_indices, map_label, state_names  # noqa: E402
from uact.features import BASELINE_FEATURES, FEATURE_NAMES, feature_mask  # noqa: E402
from uact.metrics import estimator_report  # noqa: E402
from uact.utils import load_config, read_jsonl, set_seed  # noqa: E402


def load_split(dirs, split):
    rows = []
    for d in dirs:
        for suffix in ("", "_retrieved"):
            p = os.path.join(d, f"{split}{suffix}.jsonl")
            if os.path.exists(p):
                rows += read_jsonl(p)
    return rows


def featurize_cached(fx, rows, cache):
    if cache and os.path.exists(cache):
        return np.load(cache)
    X = np.concatenate([fx.featurize([{"question": r["question"], "equipment_model": r.get("equipment_model"),
                                       "passages": r["passages"]} for r in rows[i : i + 64]]) for i in range(0, len(rows), 64)])
    if cache:
        np.save(cache, X)
    return X


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--states_dir", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--kind", default=None, choices=["five", "three"])
    ap.add_argument("--features", default="all", choices=["all", "baseline"])
    ap.add_argument("--allow_unreviewed", action="store_true", help="development only")
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    kind = a.kind or cfg["estimator"]["states"]
    out = a.out or os.path.join(cfg["paths"]["work_dir"], f"estimator_{kind}_{a.features}.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    from uact.loading import load_ctx

    ctx = load_ctx(cfg, with_llm=cfg["estimator"]["use_llm_sufficiency_feature"], with_verifier=False)
    names = state_names(kind)
    data = {}
    for sp in ("train", "val", "cal", "test"):
        rows = load_split(a.states_dir, sp)
        if not a.allow_unreviewed:
            rows = [r for r in rows if r.get("reviewed", True)]
        cache = os.path.join(os.path.dirname(out), f"feat_{sp}.npy")
        X = featurize_cached(ctx.fx, rows, cache)
        y = np.array([names.index(map_label(r["state"], kind)) for r in rows])
        data[sp] = (X, y, rows)
        print(sp, X.shape, np.bincount(y, minlength=len(names)).tolist())

    mask = feature_mask(BASELINE_FEATURES) if a.features == "baseline" else None
    est = Estimator(kind, cfg["estimator"]["hidden"], mask)
    est.fit(data["train"][0], data["train"][1], data["val"][0], data["val"][1], data["cal"][0], data["cal"][1],
            epochs=cfg["estimator"]["epochs"], lr=cfg["estimator"]["lr"], wd=cfg["estimator"]["weight_decay"], seed=cfg["seed"])
    est.save(out)

    Xt, yt, _ = data["test"]
    bad = bad_indices(kind)
    report = {"kind": kind, "features": a.features, "temperature": est.T,
              "uact_calibrated": estimator_report(est.predict_proba(Xt), yt, bad, cfg["eval"]["ece_bins"]),
              "uact_uncalibrated": estimator_report(est.predict_proba(Xt, calibrated=False), yt, bad, cfg["eval"]["ece_bins"])}

    # Baseline estimators for insufficient-evidence detection (binary AUROC/AUPRC/Brier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    ybad_t = np.isin(yt, bad).astype(int)
    ybad_tr = np.isin(data["train"][1], bad).astype(int)
    fi = {n: i for i, n in enumerate(FEATURE_NAMES)}
    base = {"top1_similarity_threshold": 1 - Xt[:, fi["rerank_top1"]], "llm_sufficiency_autorater": 1 - Xt[:, fi["llm_suff"]]}
    lr = LogisticRegression(max_iter=2000).fit(data["train"][0][:, [fi[f] for f in BASELINE_FEATURES]], ybad_tr)
    base["reranker_features_logreg"] = lr.predict_proba(Xt[:, [fi[f] for f in BASELINE_FEATURES]])[:, 1]
    report["baselines_insufficient_detection"] = {
        k: {"auroc": float(roc_auc_score(ybad_t, s)), "auprc": float(average_precision_score(ybad_t, s)),
            "brier": float(np.mean((np.clip(s, 0, 1) - ybad_t) ** 2))} for k, s in base.items()}
    pred = est.predict_proba(Xt).argmax(1)
    cm = np.zeros((len(names), len(names)), int)
    for t, p in zip(yt, pred):
        cm[t, p] += 1
    report["confusion_matrix_rows_true"] = {"labels": names, "matrix": cm.tolist()}
    json.dump(report, open(out.replace(".pt", "_report.json"), "w"), indent=2)
    print(json.dumps(report["uact_calibrated"], indent=2))
