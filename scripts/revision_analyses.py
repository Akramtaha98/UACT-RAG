"""Analyses added for the revision. All are read-only: no model is refitted and nothing is re-run on the test split
except what the subcommands read from existing eval outputs.

  python scripts/revision_analyses.py overlap --states_dir data/states data/techqa/states --techqa_dir data/techqa
  python scripts/revision_analyses.py cluster_auroc --states_dir data/states data/techqa/states --variant runs/main/full
  python scripts/revision_analyses.py matched_coverage --eval_dir runs/main/eval/techqa_test --systems UACT A7 B6 B8 B9
  python scripts/revision_analyses.py actions --eval_dir runs/main/eval/techqa_test

Each subcommand prints a JSON report and writes it to <out>/revision_<name>.json (default --out runs/main/revision).
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uact.utils import read_jsonl  # noqa: E402

SPLITS = ("train", "val", "calib", "test")


def save(out, name, rep):
    os.makedirs(out, exist_ok=True)
    json.dump(rep, open(os.path.join(out, f"revision_{name}.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))


def load_states(dirs, split):
    rows = []
    for d in dirs:
        for suf in ("", "_retrieved"):
            for nm in ([split, "validation"] if split == "val" else [split, "calibration"] if split == "calib" else [split]):
                p = os.path.join(d, f"{nm}{suf}.jsonl")
                if os.path.exists(p):
                    rows += read_jsonl(p)
    return rows


def passages_of(r):
    ps = r.get("passages") or r.get("contexts") or []
    out = []
    for p in ps:
        out.append(p if isinstance(p, str) else p.get("text", p.get("content", "")))
    return out


def shingles(text, k=8):
    w = text.lower().split()
    if len(w) < k:
        return {hashlib.md5(" ".join(w).encode()).hexdigest()[:12]} if w else set()
    return {hashlib.md5(" ".join(w[i:i + k]).encode()).hexdigest()[:12] for i in range(len(w) - k + 1)}


# ------------------------------------------------------------------ overlap
def cmd_overlap(a):
    """How much of each evaluation split's passage text also occurs in the training states, and how many TechQA dev
    questions share a gold technote with a training question. Questions are separated across splits; documents may not be."""
    rep = {}
    train = load_states(a.states_dir, "train")
    train_sh, train_hash = set(), set()
    for r in train:
        for p in passages_of(r):
            train_hash.add(hashlib.md5(p.strip().lower().encode()).hexdigest())
            train_sh |= shingles(p)
    for split in ("val", "calib", "test"):
        rows = load_states(a.states_dir, split)
        n_p = exact = hi = 0
        q_hi = 0
        for r in rows:
            flag = False
            for p in passages_of(r):
                n_p += 1
                if hashlib.md5(p.strip().lower().encode()).hexdigest() in train_hash:
                    exact += 1
                sh = shingles(p)
                frac = len(sh & train_sh) / len(sh) if sh else 0.0
                if frac >= 0.5:
                    hi += 1
                    flag = True
            q_hi += flag
        rep[split] = {"states": len(rows), "passages": n_p,
                      "passages_exact_in_train": exact, "passages_exact_share": round(exact / max(n_p, 1), 4),
                      "passages_ge50pct_shingles_in_train": hi, "passages_ge50pct_share": round(hi / max(n_p, 1), 4),
                      "states_with_any_such_passage": q_hi, "states_share": round(q_hi / max(len(rows), 1), 4)}
    if a.techqa_dir:
        def gold(split):
            for nm in (f"{split}.jsonl", f"questions_{split}.jsonl", os.path.join(split, "questions.jsonl")):
                p = os.path.join(a.techqa_dir, nm)
                if os.path.exists(p):
                    return read_jsonl(p)
            return None
        tr, dv = gold("train"), gold("dev") or gold("test")
        if tr and dv:
            gtr = {g for r in tr for g in (r.get("gold_evidence_ids") or [])}
            ans = [r for r in dv if r.get("answerable", True) and r.get("gold_evidence_ids")]
            sh = [r for r in ans if any(g in gtr for g in r["gold_evidence_ids"])]
            rep["techqa_gold_technote_overlap"] = {"dev_answerable": len(ans), "dev_sharing_gold_with_train": len(sh),
                                                   "share": round(len(sh) / max(len(ans), 1), 4)}
        else:
            rep["techqa_gold_technote_overlap"] = "question files not found under --techqa_dir; pass the folder that holds train/dev jsonl"
    save(a.out, "overlap", rep)


# ------------------------------------------------------------------ cluster bootstrap
def cmd_cluster_auroc(a):
    """AUROC for insufficient-evidence detection with a bootstrap that resamples QUESTIONS (all states of a question move
    together), for the estimator and the two baselines, pooled and by source."""
    from sklearn.metrics import roc_auc_score
    from uact.estimator import Estimator, bad_indices, map_label, state_names
    from uact.features import FEATURE_NAMES

    rows = load_states(a.states_dir, "test")
    X = np.load(os.path.join(a.variant, "feat_test.npy"))
    assert len(rows) == len(X), (len(rows), len(X))
    est = Estimator.load(os.path.join(a.variant, "estimator.pt"))
    names = state_names(est.kind)
    y = np.array([names.index(map_label(r["state"], est.kind)) for r in rows])
    bad = np.isin(y, bad_indices(est.kind)).astype(int)
    P = est.predict_proba(X)
    fi = {n: i for i, n in enumerate(FEATURE_NAMES)}
    scores = {"estimator": P[:, bad_indices(est.kind)].sum(1), "top1_sim": 1 - X[:, fi["rerank_top1"]],
              "llm_autorater": 1 - X[:, fi["llm_suff"]]}
    src = np.array([r.get("source", "") for r in rows])
    qid = np.array([r["question"].strip().lower() for r in rows])
    rng = np.random.default_rng(0)
    rep = {}
    for g, mask in (("all", np.ones(len(rows), bool)), ("synthetic", src != "techqa_retrieved"), ("techqa", src == "techqa_retrieved")):
        idx = np.where(mask)[0]
        qs = np.unique(qid[idx])
        by_q = {q: idx[qid[idx] == q] for q in qs}
        rep[g] = {"states": int(len(idx)), "questions": int(len(qs)), "states_per_question": round(len(idx) / len(qs), 2)}
        for nm, s in scores.items():
            pt = float(roc_auc_score(bad[idx], s[idx]))
            vals, diffs = [], []
            for _ in range(a.boot):
                pick = rng.choice(qs, len(qs))
                ii = np.concatenate([by_q[q] for q in pick])
                if 0 < bad[ii].sum() < len(ii):
                    vals.append(roc_auc_score(bad[ii], s[ii]))
                    if nm != "estimator":
                        diffs.append(roc_auc_score(bad[ii], scores["estimator"][ii]) - vals[-1])
            rep[g][nm] = {"auroc": round(pt, 3), "ci95": [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]}
            if diffs:
                rep[g][nm]["estimator_minus_this_ci95"] = [round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)]
    save(a.out, "cluster_auroc", rep)


# ------------------------------------------------------------------ matched coverage
def cmd_matched_coverage(a):
    """Selective risk of each system at the SAME coverage, from the forced-answer runs (answers ranked by each system's own
    confidence), with a question-level bootstrap. Risk uses the 'error' field of the eval records (grounded-success proxy).
    Also gives the paired difference to the main system at each coverage."""
    recs = {}
    for s in a.systems:
        p = os.path.join(a.eval_dir, f"{s}_forced.jsonl")
        if os.path.exists(p):
            recs[s] = {r["qid"]: r for r in read_jsonl(p)}
        else:
            print("missing", p, file=sys.stderr)
    qids = sorted(set.intersection(*[set(v) for v in recs.values()]))
    N = len(qids)
    conf = {s: np.array([recs[s][q].get("conf", 0.0) if recs[s][q].get("answered", True) else -np.inf for q in qids]) for s in recs}
    err = {s: np.array([recs[s][q].get("error", 1) for q in qids], float) for s in recs}

    def risk_at(s, c, idx):
        k = int(round(c * len(idx)))
        c_, e_ = conf[s][idx], err[s][idx]
        order = np.argsort(-c_, kind="stable")[:k]
        if k == 0 or not np.isfinite(c_[order]).all():
            return np.nan
        return float(e_[order].mean())

    rng = np.random.default_rng(0)
    rep = {"n_questions": N, "note": "NaN means the forced run answered fewer questions than the target coverage"}
    base = np.arange(N)
    for c in a.coverages:
        row = {}
        boots = {s: [] for s in recs}
        for _ in range(a.boot):
            idx = rng.integers(0, N, N)
            for s in recs:
                boots[s].append(risk_at(s, c, idx))
        for s in recs:
            b = np.array(boots[s], float)
            row[s] = {"risk": None if np.isnan(risk_at(s, c, base)) else round(risk_at(s, c, base), 3),
                      "ci95": None if np.isnan(b).all() else [round(float(np.nanpercentile(b, 2.5)), 3), round(float(np.nanpercentile(b, 97.5)), 3)]}
            if s != a.main and a.main in recs:
                d = np.array(boots[s], float) - np.array(boots[a.main], float)
                if not np.isnan(d).all():
                    row[s]["minus_main_ci95"] = [round(float(np.nanpercentile(d, 2.5)), 3), round(float(np.nanpercentile(d, 97.5)), 3)]
        rep[f"coverage_{c}"] = row
    save(a.out, "matched_coverage", rep)


# ------------------------------------------------------------------ actions
def cmd_actions(a):
    """Action frequencies, retrieval outcomes and cost for UACT-RAG (and other controllers) from the eval logs."""
    rep = {}
    for s in a.systems:
        p = os.path.join(a.eval_dir, f"{s}.jsonl")
        if not os.path.exists(p):
            continue
        recs = read_jsonl(p)
        n = len(recs)
        acts = {}
        first = {}
        helped = harmed = neutral = 0
        for r in recs:
            tr = r.get("trace") or []
            for i, step in enumerate(tr):
                acts[step["action"]] = acts.get(step["action"], 0) + 1
                if i == 0:
                    first[step["action"]] = first.get(step["action"], 0) + 1
            if any(st["action"] == "RETRIEVE" for st in tr) and len(tr) >= 2:
                # change in P(sufficient) between the first state and the last state seen
                p0, p1 = tr[0]["probs"], tr[-1]["probs"]
                d = p1[0] - p0[0]  # index 0 assumed to be 'sufficient'; check uact.estimator.state_names
                if d > 0.05:
                    helped += 1
                elif d < -0.05:
                    harmed += 1
                else:
                    neutral += 1
        rep[s] = {"n": n, "step_actions": acts, "first_action": first,
                  "questions_with_retrieve": helped + harmed + neutral,
                  "retrieve_then_P_sufficient_up": helped, "retrieve_then_P_sufficient_down": harmed, "retrieve_unchanged": neutral,
                  "answered": int(sum(bool(r.get("answered")) for r in recs)),
                  "answered_correct_proxy": int(sum(bool(r.get("answered")) and not r.get("error", 1) for r in recs)),
                  "mean_retrieval_calls": round(float(np.mean([r.get("retrieval_calls", 0) for r in recs])), 3),
                  "mean_prompt_tokens": round(float(np.mean([r.get("prompt_tokens", 0) for r in recs])), 1),
                  "mean_gen_tokens": round(float(np.mean([r.get("gen_tokens", 0) for r in recs])), 1)}
        # outcome by whether retrieval was used
        w = [r for r in recs if (r.get("retrieval_calls", 0) or 0) > 1]
        wo = [r for r in recs if (r.get("retrieval_calls", 0) or 0) <= 1]
        for nm, g in (("with_extra_retrieval", w), ("without", wo)):
            ans = [r for r in g if r.get("answered")]
            rep[s][nm] = {"n": len(g), "answered": len(ans),
                          "selective_risk_proxy": None if not ans else round(float(np.mean([r.get("error", 1) for r in ans])), 3)}
    save(a.out, "actions", rep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("overlap", "cluster_auroc", "matched_coverage", "actions"):
        p = sp.add_parser(nm)
        p.add_argument("--out", default="runs/main/revision")
        p.add_argument("--states_dir", nargs="+", default=["data/states", "data/techqa/states"])
        p.add_argument("--techqa_dir", default="data/techqa")
        p.add_argument("--variant", default="runs/main/full")
        p.add_argument("--eval_dir", default="runs/main/eval/techqa_test")
        p.add_argument("--systems", nargs="+", default=["UACT", "A7", "B6", "B8", "B9"])
        p.add_argument("--main", default="UACT")
        p.add_argument("--coverages", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
        p.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    {"overlap": cmd_overlap, "cluster_auroc": cmd_cluster_auroc, "matched_coverage": cmd_matched_coverage, "actions": cmd_actions}[a.cmd](a)
