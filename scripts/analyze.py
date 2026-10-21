"""Aggregate evaluation outputs into paper tables (Markdown + CSV), risk-coverage plots and paired statistics.

  python scripts/analyze.py --eval_dir runs/main/eval/techqa_test --ref B2 --main UACT
"""
import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uact.metrics import holm, mcnemar_exact, paired_bootstrap_diff, reliability_bins, risk_coverage  # noqa: E402
from uact.utils import read_jsonl  # noqa: E402

COLS = ["system", "accuracy_all", "coverage", "selective_risk", "wrong_answer_rate", "faithfulness", "unsupported_claim_rate",
        "retrieval_calls_mean", "gen_tokens_mean", "cost_mean", "latency_p50", "latency_p95", "aurc", "e_aurc"]


def table(eval_dir):
    rows = []
    for p in sorted(glob.glob(os.path.join(eval_dir, "summary_*.json"))):
        s = json.load(open(p))
        s["system"] = s["system"] + ("_forced" if s.get("forced") else "")
        rows.append(s)
    df = pd.DataFrame(rows)
    return df[[c for c in COLS if c in df.columns]].sort_values("system")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--ref", nargs="+", default=["B2", "B6", "B8"], help="baselines to compare UACT against")
    ap.add_argument("--main", default="UACT")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    df = table(a.eval_dir)
    df.to_csv(os.path.join(a.eval_dir, "results_table.csv"), index=False)
    with open(os.path.join(a.eval_dir, "results_table.md"), "w") as f:
        f.write(df.round(4).to_markdown(index=False))
    print(df.round(4).to_string(index=False))

    # risk-coverage curves from FORCED runs
    plt.figure(figsize=(5.2, 4))
    for p in sorted(glob.glob(os.path.join(a.eval_dir, "*_forced.jsonl"))):
        recs = read_jsonl(p)
        cov, risk = risk_coverage([r["conf"] for r in recs], [r.get("error", 1) for r in recs])
        plt.plot(cov, risk, label=os.path.basename(p).replace("_forced.jsonl", ""), lw=1.5)
    plt.xlabel("Coverage"); plt.ylabel("Selective risk"); plt.legend(fontsize=7); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(a.eval_dir, "risk_coverage.png"), dpi=200)

    # paired comparisons (same questions): correctness, error, support
    main_p = os.path.join(a.eval_dir, f"{a.main}.jsonl")
    if os.path.exists(main_p):
        m = {r["qid"]: r for r in read_jsonl(main_p)}
        rows, pv = [], []
        for ref in a.ref:
            rp = os.path.join(a.eval_dir, f"{ref}.jsonl")
            if not os.path.exists(rp):
                continue
            b = {r["qid"]: r for r in read_jsonl(rp)}
            ids = sorted(set(m) & set(b))
            for metric, fn in (("correct_over_all", lambda r: float(r["correct"])),
                               ("wrong_answer_rate", lambda r: float(r["answered"] and not r["correct"])),
                               ("retrieval_calls", lambda r: float(r["retrieval_calls"]))):
                d, lo, hi = paired_bootstrap_diff([fn(m[i]) for i in ids], [fn(b[i]) for i in ids], a.boot)
                row = {"vs": ref, "metric": metric, "mean_diff": d, "ci_lo": lo, "ci_hi": hi}
                if metric == "correct_over_all":
                    t = mcnemar_exact([m[i]["correct"] for i in ids], [b[i]["correct"] for i in ids])
                    row.update(t); pv.append(t["p_value"])
                rows.append(row)
        if pv:
            adj = iter(holm(pv))
            for r in rows:
                if "p_value" in r:
                    r["p_holm"] = next(adj)
        pd.DataFrame(rows).to_csv(os.path.join(a.eval_dir, "paired_stats.csv"), index=False)
        print(pd.DataFrame(rows).round(4).to_string(index=False))
