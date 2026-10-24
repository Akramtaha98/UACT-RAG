"""Cross-validated choice of the controller's operating point, and a test of the five-state taxonomy against simpler states.

Why: lambda, mu and the abstention margin delta were chosen on only 100 validation questions (lambda ended at the grid edge).
Here the 500 training and 100 validation questions are pooled (they have cached rollouts) and the whole grid is scored by
K-fold cross-validation over QUESTIONS: for each fold the controller is fitted on the other folds and run on the held-out
fold, so every reported number comes from questions the controller did not see. The same folds are used to compare four
state representations, to test whether the five-state taxonomy matters:

  five    P(sufficient, partial, irrelevant, missing, contradictory), entropy, P(insufficient), t, remaining budget, features
  three   P(sufficient, partial, insufficient), entropy, P(insufficient), t, remaining budget, features
  binary  P(sufficient), P(not sufficient), entropy, t, remaining budget, features
  none    t, remaining budget and the 17 evidence features only (no estimator output at all)

All four use the SAME stored rollouts (answers, correctness, support, token counts do not depend on the estimator); only the
state vector given to the controller differs, and 'three', 'binary' and 'none' are derived from the stored five-state
probabilities. No LLM or GPU is needed.

  python scripts/cv_controller.py --variant runs/main/full --folds 5 --repeats 3 --out runs/main/revision
  # optional: also write the CV-selected five-state controller for a (second, documented) test run
  python scripts/cv_controller.py ... --save_controllers runs/cv_select

Selection rule (same as the paper): lowest mean cost subject to cross-validated selective risk <= --risk_target and
coverage >= --min_coverage; if infeasible, lowest risk among points with coverage >= --min_coverage.
Risk and coverage use the grounded-success proxy stored in the rollouts. Outputs <out>/cv_controller.json.
"""
import argparse
import itertools
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_controller import utilities  # noqa: E402
from uact.controller import Controller  # noqa: E402
from uact.estimator import norm_entropy  # noqa: E402
from uact.utils import load_config  # noqa: E402

VARIANTS = ("five", "three", "binary", "none")


def load_roll(variant_dir):
    Rt = pickle.load(open(os.path.join(variant_dir, "rollout_train.pkl"), "rb"))
    Rv = pickle.load(open(os.path.join(variant_dir, "rollout_val.pkl"), "rb"))
    assert Rt["B"] == Rv["B"] and Rt["kind"] == Rv["kind"] == "five", "this script expects five-state rollouts (runs/main/full)"
    R = {"B": Rt["B"], "kind": "five", "mask": Rt["mask"], "qids": list(Rt["qids"]) + list(Rv["qids"]), "states": []}
    for st, sv in zip(Rt["states"], Rv["states"]):
        R["states"].append({k: (np.concatenate([st[k], sv[k]]) if k in sv else st[k]) for k in st})
    return R


def svec(variant, probs, t, B, X):
    n = len(probs)
    tt = np.full((n, 1), float(t))
    rem = np.full((n, 1), float(B - t))
    X = np.asarray(X, dtype="float64")
    if variant == "none":
        return np.hstack([tt, rem, X])
    if variant == "five":
        P, pb = probs, probs[:, 2:5].sum(1, keepdims=True)
    elif variant == "three":
        P = np.stack([probs[:, 0], probs[:, 1], probs[:, 2:5].sum(1)], 1)
        pb = P[:, 2:3]
    else:  # binary
        P = np.stack([probs[:, 0], 1 - probs[:, 0]], 1)
        pb = P[:, 1:2]
    return np.hstack([P, norm_entropy(P)[:, None], pb, tt, rem, X])


def stack(R, variant, V_ans, V_ret, V_abs, idx):
    B, mask = R["B"], R["mask"]
    Ss, va, vr, vb, mr = [], [], [], [], []
    for t, s in enumerate(R["states"]):
        Xm = s["X"][:, mask] if mask is not None else s["X"]
        Ss.append(svec(variant, s["probs"][idx], t, B, Xm[idx]))
        va.append(V_ans[t][idx]); vb.append(V_abs[t][idx])
        vr.append(V_ret[t][idx] if t < B else np.zeros(len(idx)))
        mr.append(np.full(len(idx), t < B))
    ones = np.ones(sum(len(x) for x in va), bool)
    return np.vstack(Ss), {"ANSWER": (np.concatenate(va), ones), "RETRIEVE": (np.concatenate(vr), np.concatenate(mr)),
                           "ABSTAIN": (np.concatenate(vb), ones)}


def simulate(ctl, R, variant, idx, cfg):
    """Per-question outcome of the policy on the questions in idx (always-retrieve evidence path, as in train_controller)."""
    B, mask = R["B"], R["mask"]
    per1k, per_ret = cfg["control"]["cost"]["per_1k_tokens"], cfg["control"]["cost"]["per_retrieval"]
    n = len(idx)
    active = np.ones(n, bool)
    answered = np.zeros(n, bool); error = np.zeros(n); cost = np.zeros(n)
    for t in range(B + 1):
        s = R["states"][t]
        a_i = np.where(active)[0]
        if len(a_i) == 0:
            break
        q = idx[a_i]
        Xm = s["X"][:, mask] if mask is not None else s["X"]
        S = svec(variant, s["probs"][q], t, B, Xm[q])
        acts = ctl.decide(S, s["probs"][q], t, B)
        for k, i in enumerate(a_i):
            if acts[k] == "ANSWER":
                active[i] = False
                cost[i] += s["ans_tokens"][q[k]] / 1000.0 * per1k
                if s["ans_answered"][q[k]]:
                    answered[i] = True
                    error[i] = float(not s["ans_correct"][q[k]])
            elif acts[k] == "RETRIEVE":
                cost[i] += s["ret_tokens"][q[k]] / 1000.0 * per1k + per_ret
            else:
                active[i] = False
    return answered, error, cost


def stats_of(ans, err, cost):
    cov = float(ans.mean())
    return {"coverage": cov, "selective_risk": float((err * ans).sum() / ans.sum()) if ans.sum() > 0 else float("nan"),
            "mean_cost": float(cost.mean())}


def pick(grid_stats, risk_target, min_cov):
    ok = [g for g in grid_stats if g["coverage"] >= min_cov and not np.isnan(g["selective_risk"]) and g["selective_risk"] <= risk_target]
    if ok:
        return min(ok, key=lambda g: g["mean_cost"])
    cand = [g for g in grid_stats if g["coverage"] >= min_cov] or grid_stats
    return min(cand, key=lambda g: (np.nan_to_num(g["selective_risk"], nan=9), g["mean_cost"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--variant", default="runs/main/full")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--mus", type=float, nargs="+", default=[0.0, 0.05, 0.2])
    ap.add_argument("--deltas", type=float, nargs="+", default=[-0.6, -0.4, -0.2, 0.0, 0.2, 0.4])
    ap.add_argument("--risk_target", type=float, default=0.40)
    ap.add_argument("--min_coverage", type=float, default=0.20)
    ap.add_argument("--targets", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--max_q", type=int, default=None, help="debug only")
    ap.add_argument("--out", default="runs/main/revision")
    ap.add_argument("--save_controllers", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    R = load_roll(a.variant)
    N = len(R["qids"]) if not a.max_q else min(a.max_q, len(R["qids"]))
    if a.max_q:
        for s in R["states"]:
            for k, v in s.items():
                s[k] = v[:N]
        R["qids"] = R["qids"][:N]
    print(f"{N} questions pooled (train + val), B={R['B']}, folds={a.folds}, repeats={a.repeats}")
    grid = list(itertools.product(a.lambdas, a.mus, a.deltas))
    U = {(lam, mu): utilities(R, lam, mu, cfg) for lam, mu in itertools.product(a.lambdas, a.mus)}
    # per variant, per grid point: mean over repeats of the held-out answered / error*answered / cost per question
    acc = {v: {g: [np.zeros(N), np.zeros(N), np.zeros(N)] for g in grid} for v in a.variants}
    for rep in range(a.repeats):
        perm = np.random.default_rng(rep).permutation(N)
        folds = np.array_split(perm, a.folds)
        for f in range(a.folds):
            te = np.sort(folds[f])
            tr = np.sort(np.concatenate([folds[j] for j in range(a.folds) if j != f]))
            for v in a.variants:
                for lam, mu in itertools.product(a.lambdas, a.mus):
                    S, V = stack(R, v, *U[(lam, mu)], tr)
                    ctl = Controller("five", "learned").fit(S, V)
                    for d in a.deltas:
                        ctl.delta = d
                        ans, err, cost = simulate(ctl, R, v, te, cfg)
                        z = acc[v][(lam, mu, d)]
                        z[0][te] += ans / a.repeats; z[1][te] += err * ans / a.repeats; z[2][te] += cost / a.repeats
            print(f"repeat {rep + 1}/{a.repeats} fold {f + 1}/{a.folds} done", flush=True)

    rng = np.random.default_rng(0)
    boots = [rng.integers(0, N, N) for _ in range(a.boot)]
    rep_out = {"questions": N, "folds": a.folds, "repeats": a.repeats, "grid": {"lambda": a.lambdas, "mu": a.mus, "delta": a.deltas},
               "risk_target": a.risk_target, "min_coverage": a.min_coverage, "variants": {}}

    def pooled(v, g):
        ans, ea, cost = acc[v][g]
        cov = ans.mean()
        return {"lambda": g[0], "mu": g[1], "delta": g[2], "coverage": float(cov),
                "selective_risk": float(ea.sum() / ans.sum()) if ans.sum() > 0 else float("nan"), "mean_cost": float(cost.mean())}

    chosen = {}
    for v in a.variants:
        rows = [pooled(v, g) for g in grid]
        best = pick(rows, a.risk_target, a.min_coverage)
        chosen[v] = best
        ans, ea, _ = acc[v][(best["lambda"], best["mu"], best["delta"])]
        rr = np.array([ea[b].sum() / ans[b].sum() if ans[b].sum() > 0 else np.nan for b in boots])
        frontier = {}
        for tgt in a.targets:
            near = [r for r in rows if abs(r["coverage"] - tgt) <= 0.05 and not np.isnan(r["selective_risk"])]
            if near:
                p = min(near, key=lambda r: r["selective_risk"])
                aa, ee, _ = acc[v][(p["lambda"], p["mu"], p["delta"])]
                rb = np.array([ee[b].sum() / aa[b].sum() if aa[b].sum() > 0 else np.nan for b in boots])
                frontier[str(tgt)] = {"coverage": round(p["coverage"], 3), "risk": round(p["selective_risk"], 3),
                                      "risk_ci95": [round(float(np.nanpercentile(rb, 2.5)), 3), round(float(np.nanpercentile(rb, 97.5)), 3)],
                                      "lambda": p["lambda"], "mu": p["mu"], "delta": p["delta"]}
        mean_front = float(np.mean([f_["risk"] for f_ in frontier.values()])) if frontier else float("nan")
        rep_out["variants"][v] = {
            "selected": {k: round(x, 4) if isinstance(x, float) else x for k, x in best.items()},
            "selected_risk_ci95": [round(float(np.nanpercentile(rr, 2.5)), 3), round(float(np.nanpercentile(rr, 97.5)), 3)],
            "lambda_at_grid_edge": bool(best["lambda"] in (min(a.lambdas), max(a.lambdas))),
            "delta_at_grid_edge": bool(best["delta"] in (min(a.deltas), max(a.deltas))),
            "frontier_best_risk_near_coverage": frontier,
            "mean_frontier_risk_over_targets": round(mean_front, 3),
            "all_points": [{k: round(x, 4) if isinstance(x, float) else x for k, x in r.items()} for r in rows]}
    # paired differences between variants at the same coverage targets (same bootstrap resamples)
    pair = {}
    ref = "five" if "five" in a.variants else a.variants[0]
    for v in a.variants:
        if v == ref:
            continue
        pair[f"{v}_minus_{ref}"] = {}
        for tgt in a.targets:
            fa, fb = rep_out["variants"][ref]["frontier_best_risk_near_coverage"].get(str(tgt)), rep_out["variants"][v]["frontier_best_risk_near_coverage"].get(str(tgt))
            if not (fa and fb):
                continue
            ga = acc[ref][(fa["lambda"], fa["mu"], fa["delta"])]; gb = acc[v][(fb["lambda"], fb["mu"], fb["delta"])]
            d = []
            for b in boots:
                ra = ga[1][b].sum() / ga[0][b].sum() if ga[0][b].sum() > 0 else np.nan
                rb_ = gb[1][b].sum() / gb[0][b].sum() if gb[0][b].sum() > 0 else np.nan
                d.append(rb_ - ra)
            d = np.array(d)
            pair[f"{v}_minus_{ref}"][str(tgt)] = {"risk_difference": round(fb["risk"] - fa["risk"], 3),
                                                  "ci95": [round(float(np.nanpercentile(d, 2.5)), 3), round(float(np.nanpercentile(d, 97.5)), 3)]}
    rep_out["paired_difference_at_matched_coverage"] = pair
    os.makedirs(a.out, exist_ok=True)
    json.dump(rep_out, open(os.path.join(a.out, "cv_controller.json"), "w"), indent=2, default=float)

    print("\nCROSS-VALIDATED OPERATING POINT (held-out questions only)")
    for v in a.variants:
        r = rep_out["variants"][v]; s = r["selected"]
        print(f"{v:7s} lambda={s['lambda']} mu={s['mu']} delta={s['delta']}  coverage {s['coverage']:.3f}  risk {s['selective_risk']:.3f} "
              f"{r['selected_risk_ci95']}  cost {s['mean_cost']:.2f}  grid-edge lambda={r['lambda_at_grid_edge']} delta={r['delta_at_grid_edge']}")
    print("\nBEST HELD-OUT RISK NEAR EACH COVERAGE (within +-0.05)")
    for v in a.variants:
        print(f"{v:7s}", {k: (x['coverage'], x['risk'], x['risk_ci95']) for k, x in rep_out['variants'][v]['frontier_best_risk_near_coverage'].items()},
              "mean", rep_out["variants"][v]["mean_frontier_risk_over_targets"])
    print("\nPAIRED DIFFERENCE (variant minus five-state), risk at matched coverage")
    for k, d in pair.items():
        print(k, d)

    if a.save_controllers:
        os.makedirs(a.save_controllers, exist_ok=True)
        for v in a.variants:
            b = chosen[v]
            S, V = stack(R, v, *U[(b["lambda"], b["mu"])], np.arange(N))
            ctl = Controller("five", "learned").fit(S, V)
            ctl.delta = b["delta"]
            ctl.save(os.path.join(a.save_controllers, f"controller_cv_{v}.pkl"))
        print("saved controller_cv_*.pkl to", a.save_controllers, "(five = drop-in replacement for controller_best.pkl; see README)")


if __name__ == "__main__":
    main()
