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
from uact.systems import (_ctx_passages, finalize_answer, gen_answers, initial_retrieval, new_rec, retrieve_more,  # noqa: E402
                          score_records)
from uact.utils import load_config, read_jsonl, set_seed  # noqa: E402

def gate_answers(ctx, qs, evs, arecs, outs):
    """Apply the SAME verifier gate as UACT.run: verify the draft; if any claim is unsupported regenerate once with the
    unsupported claims flagged, verify again, and abstain if it still fails (so the rollout matches the final system)."""
    n_ctx = ctx.cfg["retrieval"]["n_ctx"]
    redo = []
    for i, o in outs.items():
        if o["abstained"]:
            finalize_answer(arecs[i], o, evs[i], ctx)
            continue
        v = ctx.verifier.verify(o["text"], _ctx_passages(evs[i], n_ctx))
        if v["all_supported"]:
            finalize_answer(arecs[i], o, evs[i], ctx)
        else:
            redo.append((i, o, v))
    if redo:
        extra = {i: [{"role": "assistant", "content": o["text"]},
                     {"role": "user", "content": "These statements are NOT supported by the passages:\n- " + "\n- ".join(v["unsupported"]) +
                      "\nRewrite the answer using only supported statements, or reply INSUFFICIENT_EVIDENCE."}] for i, o, v in redo}
        outs2 = gen_answers(ctx, qs, evs, arecs, [i for i, _, _ in redo], extra=extra)
        for i, o, v in redo:
            o2 = outs2[i]
            v2 = ctx.verifier.verify(o2["text"], _ctx_passages(evs[i], n_ctx)) if not o2["abstained"] else None
            if v2 is not None and v2["all_supported"]:
                finalize_answer(arecs[i], o2, evs[i], ctx)
            else:
                arecs[i]["answer"], arecs[i]["answered"] = "", False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true", help="apply the verifier rejection/regeneration step to the ANSWER probes (matches the final system)")
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
        if a.gate:
            gate_answers(ctx, qs, evs, arecs, outs)
        else:
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
