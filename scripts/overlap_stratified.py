"""Frozen end-to-end results restricted to test questions WITHOUT document overlap with training.

Review point: some TechQA test questions share their gold technote with a training question (38 of 160 answerable dev questions),
so part of the controller's performance could come from shared source material, and the full pipeline was not re-evaluated under a
grouped split. This script does not retrain anything. It splits the frozen test results by overlap and recomputes coverage and
proxy selective risk (and, if present, human audit labels are NOT used) for every system on:
    all 310 test questions | questions WITHOUT gold-technote overlap | questions WITH overlap
Unanswerable questions have no gold technote and count as non-overlapping. Intervals are question-level bootstrap.

  python scripts/overlap_stratified.py --eval_dir runs/main/eval/techqa_test --techqa_dir data/techqa \
         --systems B2 B6 B8 B9 UACT A3 --out runs/main/revision

Reads <eval_dir>/<system>.jsonl (fields qid, answered, error) and <techqa_dir>/{train,dev}.jsonl. Writes overlap_stratified.json.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uact.utils import read_jsonl  # noqa: E402


def load_q(d, names):
    for nm in names:
        p = os.path.join(d, nm)
        if os.path.exists(p):
            return read_jsonl(p)
    sys.exit(f"none of {names} found under {d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", default="runs/main/eval/techqa_test")
    ap.add_argument("--techqa_dir", default="data/techqa")
    ap.add_argument("--systems", nargs="+", default=["B2", "B6", "B8", "B9", "UACT", "A3"])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="runs/main/revision")
    a = ap.parse_args()
    train = load_q(a.techqa_dir, ["train.jsonl", "questions_train.jsonl"])
    dev = load_q(a.techqa_dir, ["dev.jsonl", "test.jsonl", "questions_dev.jsonl"])
    gtr = {g for r in train for g in (r.get("gold_evidence_ids") or [])}
    overlap = {r["qid"]: bool(r.get("answerable", True)) and any(g in gtr for g in (r.get("gold_evidence_ids") or [])) for r in dev}
    print(f"{len(dev)} dev questions, {sum(overlap.values())} share a gold technote with training")
    rng = np.random.default_rng(0)
    rep = {"n_questions": len(dev), "n_overlap": int(sum(overlap.values())), "systems": {}}
    for s in a.systems:
        p = os.path.join(a.eval_dir, f"{s}.jsonl")
        if not os.path.exists(p):
            print("missing", p, file=sys.stderr)
            continue
        recs = {r["qid"]: r for r in read_jsonl(p)}
        qids = [q for q in overlap if q in recs]
        ans = np.array([bool(recs[q].get("answered")) for q in qids])
        err = np.array([recs[q].get("error", 1) for q in qids], float)
        ov = np.array([overlap[q] for q in qids])

        def stat(mask, ii=None):
            idx = np.where(mask)[0] if ii is None else ii[mask[ii]]
            if len(idx) == 0:
                return (np.nan, np.nan, 0)
            a_ = ans[idx]
            cov = float(a_.mean())
            risk = float(err[idx][a_].mean()) if a_.any() else np.nan
            return (cov, risk, int(a_.sum()))

        out = {}
        for nm, m in (("all", np.ones(len(qids), bool)), ("no_overlap", ~ov), ("overlap", ov)):
            cov, risk, k = stat(m)
            bs = []
            for _ in range(a.boot):
                ii = rng.integers(0, len(qids), len(qids))
                bs.append(stat(m, ii)[:2])
            bs = np.array(bs, float)

            def ci(c):
                v = bs[:, c]
                return None if np.isnan(v).all() else [round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3)]
            out[nm] = {"questions": int(m.sum()), "answered": k, "coverage": round(cov, 3), "coverage_ci95": ci(0),
                       "selective_risk_proxy": None if np.isnan(risk) else round(risk, 3), "risk_ci95": ci(1)}
        rep["systems"][s] = out
        print(s, {k: (v["answered"], v["coverage"], v["selective_risk_proxy"]) for k, v in out.items()})
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out, "overlap_stratified.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
