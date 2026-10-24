"""Document-grouped, multi-seed re-evaluation of the evidence-state estimator (RQ2).

Why: the original splits separate QUESTIONS but not DOCUMENTS (many test passages also occur in training text), and the
estimator was trained once (one seed). This script pools every labelled state (train + val + cal + test, synthetic and real
TechQA), groups the states so that no document or question is shared between a training and a held-out group, and runs
K-fold cross-validation over the groups, repeated for several seeds (new group-to-fold assignment AND new network
initialisation per seed). Every state is scored once per seed by a model that never saw its group.

  python scripts/docsplit_seeds.py --states_dir data/states data/techqa/states --techqa_dir data/techqa \
         --variant runs/main/full --seeds 5 --folds 5 --out runs/main/revision

Needs the cached feature files written by train_estimator.py (<variant>/feat_{train,val,cal,test}.npy). No LLM, no GPU.
Outputs <out>/docsplit_seeds.json and prints a summary. Nothing in runs/main is modified.

Grouping. States are linked (union-find) when they share a question text, an ORIGINAL passage text (context-swapped states
are excluded, because their passages belong to another question), or, for real TechQA states, a gold technote id.
Passages that occur under more than --max_df distinct questions are ignored as hubs. The script prints the number of
groups and the share of the largest group; if one group holds more than --max_group_share of the states it stops, because
folds would then be meaningless (raise --max_group_share or lower --max_df and read the printout).
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uact.utils import load_config, read_jsonl  # noqa: E402

SYNTH_SKIP = {"context_swapped", "techqa_retrieved"}


def norm_q(s):
    return " ".join(s.lower().split())


def load_pooled(states_dirs, variant):
    """Rows and features in exactly the order train_estimator.py cached them."""
    from train_estimator import load_split

    rows, X, split = [], [], []
    for sp in ("train", "val", "cal", "test"):
        r = [x for x in load_split(states_dirs, sp) if x.get("reviewed", True)]
        f = np.load(os.path.join(variant, f"feat_{sp}.npy"))
        assert len(r) == len(f), f"{sp}: {len(r)} rows but {len(f)} cached feature rows; re-run train_estimator.py"
        rows += r
        X.append(f)
        split += [sp] * len(r)
    return rows, np.vstack(X), np.array(split)


def passage_texts(r):
    out = []
    for p in r.get("passages") or []:
        out.append(p if isinstance(p, str) else p.get("text", ""))
    return out


def ph(text):
    w = text.lower().split()
    return hashlib.md5(" ".join(w[:60]).encode()).hexdigest() if len(w) >= 15 else None


def build_groups(rows, techqa_dir, max_df):
    gold = {}
    if techqa_dir:
        for fn in ("train", "val", "dev", "test"):
            p = os.path.join(techqa_dir, f"{fn}.jsonl")
            if os.path.exists(p):
                for q in read_jsonl(p):
                    gold[norm_q(q["question"])] = list(q.get("gold_evidence_ids") or [])
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    qs_of = defaultdict(set)  # passage hash -> distinct questions
    keys = []
    for r in rows:
        q = norm_q(r["question"])
        hs = []
        if r.get("source") not in SYNTH_SKIP:
            hs = [h for h in (ph(t) for t in passage_texts(r)) if h]
        for h in hs:
            qs_of[h].add(q)
        keys.append((q, hs))
    for r, (q, hs) in zip(rows, keys):
        union(("q", q), ("q", q))
        for h in hs:
            if len(qs_of[h]) <= max_df:
                union(("q", q), ("p", h))
        if r.get("source") == "techqa_retrieved":
            for g in gold.get(q, []):
                union(("q", q), ("g", g))
    root = {}
    gid = np.empty(len(rows), int)
    for i, (q, _) in enumerate(keys):
        gid[i] = root.setdefault(find(("q", q)), len(root))
    return gid


def assign_folds(gid, k, rng):
    """Greedy balanced assignment of whole groups to k folds."""
    groups, counts = np.unique(gid, return_counts=True)
    order = rng.permutation(len(groups))
    load = np.zeros(k)
    fold_of = {}
    for j in sorted(order, key=lambda j: -counts[j] + rng.random() * 0.5):  # large groups first, random tie-break
        f = int(np.argmin(load))
        fold_of[groups[j]] = f
        load[f] += counts[j]
    return np.array([fold_of[g] for g in gid])


def split_rest(gid, rest_mask, rng, fr_val=0.1, fr_cal=0.1):
    """Split the remaining groups into train / val / cal by whole groups."""
    groups = np.unique(gid[rest_mask])
    rng.shuffle(groups)
    n = rest_mask.sum()
    sizes = {g: int((gid[rest_mask] == g).sum()) for g in groups}
    tag, acc = {}, 0
    for g in groups:
        frac = acc / n
        tag[g] = "val" if frac < fr_val else ("cal" if frac < fr_val + fr_cal else "train")
        acc += sizes[g]
    t = np.array([tag.get(g, "") for g in gid])
    return (t == "train") & rest_mask, (t == "val") & rest_mask, (t == "cal") & rest_mask


def auroc(y, s):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--states_dir", nargs="+", required=True)
    ap.add_argument("--techqa_dir", default=None)
    ap.add_argument("--variant", default="runs/main/full")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--max_df", type=int, default=20)
    ap.add_argument("--max_group_share", type=float, default=0.35)
    ap.add_argument("--out", default="runs/main/revision")
    a = ap.parse_args()
    cfg = load_config(a.config)
    from uact.estimator import Estimator, bad_indices, map_label, p_bad, state_names
    from uact.features import FEATURE_NAMES
    from uact.metrics import ece

    rows, X, orig_split = load_pooled(a.states_dir, a.variant)
    N = len(rows)
    names = state_names("five")
    y = np.array([names.index(map_label(r["state"], "five")) for r in rows])
    bad = np.isin(y, bad_indices("five")).astype(int)
    qid = np.array([norm_q(r["question"]) for r in rows])
    src = np.array([r.get("source", "") for r in rows])
    gid = build_groups(rows, a.techqa_dir, a.max_df)
    sizes = np.bincount(gid)
    share = float(sizes.max() / N)
    print(f"{N} states, {len(sizes)} groups, largest group = {share:.1%} of states, "
          f"{int((sizes == 1).sum())} singleton groups")
    if share > a.max_group_share:
        sys.exit(f"The largest group holds {share:.0%} of all states (> {a.max_group_share:.0%}); folds would be meaningless. "
                 "Lower --max_df (hub passages) or raise --max_group_share, and report what you did.")
    fi = {n: i for i, n in enumerate(FEATURE_NAMES)}
    base = {"top1_similarity": 1 - X[:, fi["rerank_top1"]], "llm_autorater": 1 - X[:, fi["llm_suff"]]}
    slices = {"all": np.ones(N, bool), "synthetic": src != "techqa_retrieved", "techqa_real": src == "techqa_retrieved"}
    ds = np.array(["real_techqa_retrieval" if s == "techqa_retrieved" else "constructed_" + str(r["id"]).split("_")[0].split(":")[0]
                   for r, s in zip(rows, src)])
    for d in sorted(set(ds) - {"real_techqa_retrieval"}):
        slices[d] = ds == d

    oof = np.zeros((a.seeds, N))
    ece_seed = {s: [] for s in slices}
    for seed in range(a.seeds):
        rng = np.random.default_rng(1000 + seed)
        fold = assign_folds(gid, a.folds, rng)
        for f in range(a.folds):
            te = fold == f
            tr, va, ca = split_rest(gid, ~te, rng)
            est = Estimator("five", cfg["estimator"]["hidden"], None)
            est.fit(X[tr], y[tr], X[va], y[va], X[ca], y[ca], epochs=cfg["estimator"]["epochs"], lr=cfg["estimator"]["lr"],
                    wd=cfg["estimator"]["weight_decay"], seed=seed * 100 + f)
            oof[seed, te] = p_bad(est.predict_proba(X[te]), "five")
        for s, m in slices.items():
            pb = oof[seed, m]
            ece_seed[s].append(ece(pb, bad[m], cfg["eval"]["ece_bins"]))
        print(f"seed {seed}: pooled AUROC {auroc(bad, oof[seed]):.3f}", flush=True)

    rng = np.random.default_rng(0)
    rep = {"n_states": int(N), "n_groups": int(len(sizes)), "largest_group_share": round(share, 4),
           "seeds": a.seeds, "folds": a.folds, "max_df": a.max_df, "slices": {}}
    for s, m in slices.items():
        idx = np.where(m)[0]
        qs = np.unique(qid[idx])
        by_q = {q: idx[qid[idx] == q] for q in qs}
        per_seed = [auroc(bad[idx], oof[k, idx]) for k in range(a.seeds)]
        pt = {"estimator": float(np.mean(per_seed))}
        for nm, sc in base.items():
            pt[nm] = auroc(bad[idx], sc[idx])
        boots = {"estimator": [], "top1_similarity": [], "llm_autorater": [], "d_top1": [], "d_autorater": []}
        for _ in range(a.boot):
            ii = np.concatenate([by_q[q] for q in rng.choice(qs, len(qs))])
            if not 0 < bad[ii].sum() < len(ii):
                continue
            e = float(np.mean([auroc(bad[ii], oof[k, ii]) for k in range(a.seeds)]))
            t1, au = auroc(bad[ii], base["top1_similarity"][ii]), auroc(bad[ii], base["llm_autorater"][ii])
            boots["estimator"].append(e); boots["top1_similarity"].append(t1); boots["llm_autorater"].append(au)
            boots["d_top1"].append(e - t1); boots["d_autorater"].append(e - au)

        def ci(v):
            if len(v) == 0:
                return [None, None]
            return [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]

        rep["slices"][s] = {
            "states": int(len(idx)), "questions": int(len(qs)),
            "estimator_auroc_mean_over_seeds": round(pt["estimator"], 3), "estimator_auroc_sd_over_seeds": round(float(np.std(per_seed, ddof=1)) if a.seeds > 1 else 0.0, 3),
            "estimator_ci95_question_bootstrap": ci(boots["estimator"]),
            "top1_similarity_auroc": round(pt["top1_similarity"], 3), "top1_similarity_ci95": ci(boots["top1_similarity"]),
            "llm_autorater_auroc": round(pt["llm_autorater"], 3), "llm_autorater_ci95": ci(boots["llm_autorater"]),
            "estimator_minus_top1_ci95": ci(boots["d_top1"]), "estimator_minus_autorater_ci95": ci(boots["d_autorater"]),
            "ece_P_insufficient_mean_over_seeds": round(float(np.mean(ece_seed[s])), 3),
            "ece_P_insufficient_sd_over_seeds": round(float(np.std(ece_seed[s], ddof=1)) if a.seeds > 1 else 0.0, 3),
            "per_seed_auroc": [round(v, 3) for v in per_seed]}
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep, open(os.path.join(a.out, "docsplit_seeds.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))
    print("\nSUMMARY (document-grouped CV, mean over seeds; question-level bootstrap 95% intervals)")
    for s, v in rep["slices"].items():
        print(f"{s:28s} n={v['states']:5d} est {v['estimator_auroc_mean_over_seeds']:.3f} +-{v['estimator_auroc_sd_over_seeds']:.3f} {v['estimator_ci95_question_bootstrap']} | "
              f"top1 {v['top1_similarity_auroc']:.3f} | autorater {v['llm_autorater_auroc']:.3f} | "
              f"est-top1 {v['estimator_minus_top1_ci95']} est-auto {v['estimator_minus_autorater_ci95']}")


if __name__ == "__main__":
    main()
