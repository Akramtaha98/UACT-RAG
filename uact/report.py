"""Aggregate per-question records into the metrics reported in the paper."""
import numpy as np

from .metrics import aurc, e_aurc, percentile, selective_stats


def summarize(recs, cost_fn=None):
    n = len(recs)
    answered = np.array([r["answered"] for r in recs])
    error = np.array([r.get("error", 0) for r in recs], float)
    correct = np.array([bool(r.get("correct", False)) for r in recs])
    out = {"n": n, "accuracy_all": float(correct.mean())}  # correct answers over ALL questions
    out.update(selective_stats(answered, error))
    out["precision_answered"] = float(correct[answered].mean()) if answered.any() else float("nan")
    out["f1_mean_answered"] = float(np.mean([r.get("f1", 0.0) for r in recs if r["answered"]])) if answered.any() else float("nan")
    sup = [r["support_rate"] for r in recs if r["answered"] and "support_rate" in r]
    out["faithfulness"] = float(np.mean(sup)) if sup else float("nan")
    ncl = sum(r.get("n_claims", 0) for r in recs if r["answered"])
    out["unsupported_claim_rate"] = float(sum(r.get("n_unsupported_claims", 0) for r in recs if r["answered"]) / ncl) if ncl else float("nan")
    out["retrieval_calls_mean"] = float(np.mean([r["retrieval_calls"] for r in recs]))
    out["prompt_tokens_mean"] = float(np.mean([r["prompt_tokens"] for r in recs]))
    out["gen_tokens_mean"] = float(np.mean([r["gen_tokens"] for r in recs]))
    if cost_fn:
        out["cost_mean"] = float(np.mean([cost_fn(r["prompt_tokens"], r["gen_tokens"], r["retrieval_calls"]) for r in recs]))
    lat = [r["latency"] for r in recs if "latency" in r]
    if lat:
        out.update({"latency_mean": float(np.mean(lat)), "latency_p50": percentile(lat, 50), "latency_p95": percentile(lat, 95)})
    gr = [r["gold_recall"] for r in recs if "gold_recall" in r]
    if gr:
        out["gold_recall_mean"] = float(np.mean(gr))
    conf = np.array([r["conf"] for r in recs], float)
    # AURC/E-AURC are only defined here for FORCED-ANSWER runs (run_eval.py --forced), where every question
    # received an answer and the system's confidence score ranks them; the curve is then swept over thresholds.
    if answered.all() and n > 1 and len(np.unique(conf)) > 1:
        out["aurc"] = aurc(conf, error)
        out["e_aurc"] = e_aurc(conf, error)
    return out
