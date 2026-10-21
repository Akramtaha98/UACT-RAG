"""Fit the confidence model used by B6/B8 and tune baseline thresholds on VALIDATION data.

  python scripts/fit_baselines.py --data data/techqa --estimator runs/main/estimator_five_all.pt
Writes conf_model.pkl and baseline_thresholds.json in the work dir.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uact.loading import load_ctx  # noqa: E402
from uact.report import summarize  # noqa: E402
from uact.systems import CalibConfController, ConfModel, HybridRAG, SufficiencyAbstain, score_records  # noqa: E402
from uact.utils import load_config, read_jsonl, set_seed  # noqa: E402


def pick(rows, risk_target, min_cov):
    ok = [r for r in rows if r["coverage"] >= min_cov and r["selective_risk"] == r["selective_risk"] and r["selective_risk"] <= risk_target]
    if ok:
        return min(ok, key=lambda r: r["cost_mean"])
    return min(rows, key=lambda r: (np.nan_to_num(r["selective_risk"], nan=9), r["cost_mean"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--estimator", default=None)
    ap.add_argument("--max_q", type=int, default=1500)
    ap.add_argument("--risk_target", type=float, default=0.10)
    ap.add_argument("--min_coverage", type=float, default=0.50)
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    wd = cfg["paths"]["work_dir"]
    os.makedirs(wd, exist_ok=True)
    ctx = load_ctx(cfg, os.path.join(a.data, "index"), os.path.join(a.data, "corpus.jsonl"), estimator=a.estimator)
    train = read_jsonl(os.path.join(a.data, "train.jsonl"))[: a.max_q]
    val = read_jsonl(os.path.join(a.data, "val.jsonl"))

    def run_scored(qs):
        recs = HybridRAG(ctx).run(qs)
        return score_records(ctx, qs, recs)

    rt, rv = run_scored(train), run_scored(val)
    F = lambda rs: np.array([r["feat"] for r in rs])
    Y = lambda rs: np.array([int(r["correct"]) for r in rs])
    cm = ConfModel().fit(F(rt), Y(rt), F(rv), Y(rv))
    cm.save(os.path.join(wd, "conf_model.pkl"))
    ctx.conf_model = cm
    cost = ctx.cost
    out = {}

    # B6: threshold on calibrated P(correct) of the hybrid-RAG answer
    conf = cm.predict(F(rv))
    rows = []
    for tau in np.unique(np.quantile(conf, np.linspace(0, 0.9, 19))):
        keep = conf >= tau
        recs = [dict(r, answered=bool(k)) for r, k in zip(rv, keep)]
        s = summarize(recs, cost); s["tau"] = float(tau); rows.append(s)
    out["B6_tau"] = pick(rows, a.risk_target, a.min_coverage)["tau"]

    # B9: threshold on the LLM sufficiency probability
    sv = SufficiencyAbstain(ctx, tau=-1.0).run(val)
    score_records(ctx, val, sv)
    ps = np.array([r["conf"] for r in sv])
    rows = []
    for tau in np.unique(np.quantile(ps, np.linspace(0, 0.9, 19))):
        recs = [dict(r, answered=bool(p >= tau)) for r, p in zip(sv, ps)]
        s = summarize(recs, cost); s["tau"] = float(tau); rows.append(s)
    out["B9_tau"] = pick(rows, a.risk_target, a.min_coverage)["tau"]

    # B8: calibrated-confidence controller grid (full re-runs)
    rows = []
    for hi, lo in itertools.product([0.5, 0.65, 0.8], [0.1, 0.25]):
        recs = CalibConfController(ctx, hi, lo).run(val)
        score_records(ctx, val, recs)
        s = summarize(recs, cost); s.update({"tau_hi": hi, "tau_lo": lo}); rows.append(s)
    b = pick(rows, a.risk_target, a.min_coverage)
    out["B8_tau_hi"], out["B8_tau_lo"] = b["tau_hi"], b["tau_lo"]
    json.dump(out, open(os.path.join(wd, "baseline_thresholds.json"), "w"), indent=2)
    print(out)
