"""Train the cost-sensitive controller and the simple controllers used in ablations; tune on VALIDATION rollouts.

  python scripts/train_controller.py --train runs/main/rollout_train.pkl --val runs/main/rollout_val.pkl

Selection rule (pre-register in the paper): among grid points, choose the lowest mean cost subject to
validation selective risk <= --risk_target and coverage >= --min_coverage; if none is feasible choose the lowest risk.
Outputs: controller_best.pkl (UACT-RAG), controller_nocost.pkl (A6, mu = 0), controller_threshold.pkl (A5),
controller_fixedbudget.pkl (A2), and controller_grid.json (every grid point, for the appendix).
"""
import argparse
import itertools
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uact.controller import Controller, state_vector  # noqa: E402
from uact.estimator import state_names  # noqa: E402
from uact.utils import load_config  # noqa: E402


def utilities(R, lam, mu, cfg):
    B, kind = R["B"], R["kind"]
    c = cfg["control"]
    per1k, per_ret, ac, tau = c["cost"]["per_1k_tokens"], c["cost"]["per_retrieval"], c["abstain_credit"], c["tau_ref"]
    isuff = state_names(kind).index("sufficient")
    V_ans, V_abs, V_ret = [], [], []
    for s in R["states"]:
        u_abs = np.where(s["answerable"], 0.0, ac)
        good = s["ans_answered"] & s["ans_correct"] & (s["ans_support"] >= 0.8)
        u_ans = np.where(s["ans_answered"], np.where(good, 1.0, -lam), u_abs)
        V_ans.append(u_ans - mu * s["ans_tokens"] / 1000.0 * per1k)
        V_abs.append(u_abs)
    cont = [None] * (B + 1)
    V_ret = [None] * (B + 1)
    cont[B] = np.where(R["states"][B]["probs"][:, isuff] >= tau, V_ans[B], V_abs[B])
    for t in range(B - 1, -1, -1):
        s = R["states"][t]
        V_ret[t] = -mu * (s["ret_tokens"] / 1000.0 * per1k + per_ret) + cont[t + 1]
        cont[t] = np.where(s["probs"][:, isuff] >= tau, V_ans[t], V_ret[t])
    return V_ans, V_ret, V_abs


def stack(R, V_ans, V_ret, V_abs):
    B, kind, mask = R["B"], R["kind"], R["mask"]
    Ss, va, vr, vb, mr = [], [], [], [], []
    for t, s in enumerate(R["states"]):
        Xm = s["X"][:, mask] if mask is not None else s["X"]
        Ss.append(state_vector(s["probs"], kind, t, B, Xm))
        va.append(V_ans[t]); vb.append(V_abs[t])
        vr.append(V_ret[t] if t < B else np.zeros(len(V_ans[t])))
        mr.append(np.full(len(V_ans[t]), t < B))
    ones = np.ones(sum(len(x) for x in va), bool)
    return np.vstack(Ss), {"ANSWER": (np.concatenate(va), ones), "RETRIEVE": (np.concatenate(vr), np.concatenate(mr)),
                           "ABSTAIN": (np.concatenate(vb), ones)}


def simulate(ctl, R, cfg):
    """Run the policy along the always-retrieve evidence path (exact, because RETRIEVE is deterministic)."""
    B, kind, mask = R["B"], R["kind"], R["mask"]
    per1k, per_ret = cfg["control"]["cost"]["per_1k_tokens"], cfg["control"]["cost"]["per_retrieval"]
    N = len(R["states"][0]["probs"])
    active = np.ones(N, bool)
    answered = np.zeros(N, bool); error = np.zeros(N); cost = np.zeros(N); calls = np.zeros(N)
    for t in range(B + 1):
        s = R["states"][t]
        idx = np.where(active)[0]
        if len(idx) == 0:
            break
        Xm = s["X"][:, mask] if mask is not None else s["X"]
        S = state_vector(s["probs"][idx], kind, t, B, Xm[idx])
        acts = ctl.decide(S, s["probs"][idx], t, B)
        for k, i in enumerate(idx):
            if acts[k] == "ANSWER":
                active[i] = False
                cost[i] += s["ans_tokens"][i] / 1000.0 * per1k
                if s["ans_answered"][i]:
                    answered[i] = True
                    error[i] = float(not (s["ans_correct"][i]))
            elif acts[k] == "RETRIEVE":
                cost[i] += s["ret_tokens"][i] / 1000.0 * per1k + per_ret
                calls[i] += 1
            else:
                active[i] = False
    return {"coverage": float(answered.mean()),
            "selective_risk": float(error[answered].mean()) if answered.any() else float("nan"),
            "wrong_answer_rate": float((error * answered).mean()), "mean_cost": float(cost.mean()), "mean_calls": float(calls.mean())}


def feasible_rank(rows, risk_target, min_cov):
    ok = [r for r in rows if r["stats"]["coverage"] >= min_cov and not np.isnan(r["stats"]["selective_risk"]) and r["stats"]["selective_risk"] <= risk_target]
    if ok:
        return min(ok, key=lambda r: r["stats"]["mean_cost"])
    return min(rows, key=lambda r: (np.nan_to_num(r["stats"]["selective_risk"], nan=9), r["stats"]["mean_cost"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--risk_target", type=float, default=0.10)
    ap.add_argument("--min_coverage", type=float, default=0.50)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    Rt, Rv = pickle.load(open(a.train, "rb")), pickle.load(open(a.val, "rb"))
    outdir = a.outdir or os.path.dirname(a.train)
    kind = Rt["kind"]
    deltas = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.5]
    grid = []
    for lam, mu in itertools.product(cfg["control"]["lambda_grid"], cfg["control"]["mu_grid"]):
        S, V = stack(Rt, *utilities(Rt, lam, mu, cfg))
        base = Controller(kind, "learned").fit(S, V)
        for d in deltas:
            base.delta = d
            grid.append({"lambda": lam, "mu": mu, "delta": d, "stats": simulate(base, Rv, cfg), "ctl": pickle.loads(pickle.dumps(base))})
    best = feasible_rank(grid, a.risk_target, a.min_coverage)
    best["ctl"].save(os.path.join(outdir, "controller_best.pkl"))
    nocost = feasible_rank([g for g in grid if g["mu"] == 0.0], a.risk_target, a.min_coverage)
    nocost["ctl"].save(os.path.join(outdir, "controller_nocost.pkl"))

    # simple controllers for ablations A5 (thresholds) and A2 (fixed retrieval budget), tuned on validation
    simple = {"threshold": [], "fixed_budget": []}
    for mode in simple:
        for ts, tb in itertools.product([0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0.3, 0.5, 0.7]):
            c = Controller(kind, mode); c.tau_suff, c.tau_bad = ts, tb
            simple[mode].append({"tau_suff": ts, "tau_bad": tb, "stats": simulate(c, Rv, cfg), "ctl": c})
        feasible_rank(simple[mode], a.risk_target, a.min_coverage)["ctl"].save(os.path.join(outdir, f"controller_{mode.replace('_', '')}.pkl"))
    summary = {"best": {k: v for k, v in best.items() if k != "ctl"}, "nocost": {k: v for k, v in nocost.items() if k != "ctl"},
               "grid": [{k: v for k, v in g.items() if k != "ctl"} for g in grid],
               "threshold_grid": [{k: v for k, v in g.items() if k != "ctl"} for g in simple["threshold"]],
               "risk_target": a.risk_target, "min_coverage": a.min_coverage}
    json.dump(summary, open(os.path.join(outdir, "controller_grid.json"), "w"), indent=2, default=float)
    print("selected:", summary["best"])
