"""Roll out ALL feasible actions on TRAIN (and VAL) questions and store per-state outcomes.

For each question the evidence sequence D_0 .. D_B is produced by always retrieving (same RETRIEVE procedure as at
test time). At every t we also generate and score the ANSWER outcome, so utilities for ANSWER, ABSTAIN and
RETRIEVE (via a reference continuation) can be computed offline for any (lambda, mu) without re-running the LLM.

  python scripts/build_controller_data.py --data data/techqa --split train --estimator runs/main/estimator_five_all.pt
Use --split val for the validation rollouts that select lambda, mu and the abstention margin. NEVER use test.
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uact.loading import load_ctx  # noqa: E402
from uact.systems import (finalize_answer, gen_answers, initial_retrieval, new_rec, retrieve_more,  # noqa: E402
                          score_records)
from uact.utils import load_config, read_jsonl, set_seed  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--estimator", required=True)
    ap.add_argument("--max_q", type=int, default=3000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    ctx = load_ctx(cfg, os.path.join(a.data, "index"), os.path.join(a.data, "corpus.jsonl"), estimator=a.estimator)
    qs = read_jsonl(os.path.join(a.data, f"{a.split}.jsonl"))[: a.max_q]
    B = ctx.B
    N = len(qs)
    recs = [new_rec(q, "rollout") for q in qs]
    evs = initial_retrieval(ctx, qs, recs)
    states = []  # states[t] = dict of arrays over questions
    for t in range(B + 1):
        snap_tokens = [(r["prompt_tokens"], r["gen_tokens"], r["retrieval_calls"]) for r in recs]
        items = [{"question": q["question"], "equipment_model": q.get("equipment_model"), "revision": q.get("revision"),
                  "passages": evs[i]} for i, q in enumerate(qs)]
        X = ctx.fx.featurize(items)
        probs = ctx.estimator.predict_proba(X)
        # ANSWER outcome at this t (separate record objects so token counts of answering are isolated)
        arecs = [new_rec(q, "answer_probe") for q in qs]
        outs = gen_answers(ctx, qs, evs, arecs, range(N))
        for i, o in outs.items():
            finalize_answer(arecs[i], o, evs[i], ctx)
        score_records(ctx, qs, arecs)
        # answering cost counts only the answer call; retrieval cost = tokens of the rewrite call + 1 search
        states.append({
            "X": X, "probs": probs,
            "ans_answered": np.array([r["answered"] for r in arecs]),
            "ans_correct": np.array([r["correct"] for r in arecs]),
            "ans_support": np.array([r.get("support_rate", 1.0) for r in arecs]),
            "ans_tokens": np.array([r["prompt_tokens"] + r["gen_tokens"] for r in arecs]),
            "answerable": np.array([q["answerable"] for q in qs]),
        })
        if t < B:
            before = [(r["prompt_tokens"] + r["gen_tokens"]) for r in recs]
            retrieve_more(ctx, qs, evs, recs, list(range(N)))
            states[t]["ret_tokens"] = np.array([(r["prompt_tokens"] + r["gen_tokens"]) - b for r, b in zip(recs, before)])
    out = a.out or os.path.join(cfg["paths"]["work_dir"], f"rollout_{a.split}.pkl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pickle.dump({"states": states, "B": B, "kind": ctx.estimator.kind, "mask": ctx.estimator.mask,
                 "qids": [q["qid"] for q in qs]}, open(out, "wb"))
    print("saved", out)
