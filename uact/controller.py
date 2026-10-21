"""Cost-sensitive controller: predicts the utility of each action and picks the maximiser.

Training targets are built on the TRAIN split by executing the feasible actions (scripts/build_controller_data.py).
RETRIEVE targets use a reference continuation policy, which is an approximation (see the paper's limitations).
"""
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ACTIONS = ["ANSWER", "RETRIEVE", "ABSTAIN"]


def state_vector(probs, kind, t, budget, X):
    from .estimator import norm_entropy, p_bad

    probs = np.asarray(probs)
    U = norm_entropy(probs)[:, None]
    pb = p_bad(probs, kind)[:, None]
    tt = np.full((len(probs), 1), float(t))
    rem = np.full((len(probs), 1), float(budget - t))
    return np.hstack([probs, U, pb, tt, rem, np.asarray(X, dtype="float64")])


class Controller:
    """mode: 'learned' | 'threshold' (A5) | 'fixed_budget' (A2)."""

    def __init__(self, kind="five", mode="learned"):
        self.kind, self.mode = kind, mode
        self.models = {}
        self.delta = 0.0              # abstention margin: ANSWER only if V_ans - V_abs > delta
        self.tau_suff = 0.5           # thresholds for the fixed-rule controllers
        self.tau_bad = 0.5

    def fit(self, S, V):
        """S: (n, d) state vectors; V: dict action -> (values, mask of rows where the action was feasible)."""
        for a in ACTIONS:
            vals, mask = V[a]
            if mask.sum() < 20:
                continue
            m = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, random_state=0)
            m.fit(S[mask], vals[mask])
            self.models[a] = m
        return self

    def values(self, S):
        out = np.full((len(S), 3), -1e9)
        for j, a in enumerate(ACTIONS):
            if a in self.models:
                out[:, j] = self.models[a].predict(S)
        return out

    def confidence(self, S):
        """Margin V_answer - V_abstain: the score used for risk-coverage curves."""
        v = self.values(S)
        return v[:, 0] - v[:, 2]

    def decide(self, S, probs, t, budget):
        """Returns list of action names."""
        from .estimator import p_bad, state_names

        n = len(S)
        can_retrieve = t < budget
        if self.mode == "threshold":
            ps = np.asarray(probs)[:, state_names(self.kind).index("sufficient")]
            pb = p_bad(probs, self.kind)
            acts = []
            for i in range(n):
                if ps[i] >= self.tau_suff:
                    acts.append("ANSWER")
                elif can_retrieve and pb[i] < self.tau_bad + 0.3:
                    acts.append("RETRIEVE")
                else:
                    acts.append("ABSTAIN")
            return acts
        if self.mode == "fixed_budget":
            ps = np.asarray(probs)[:, state_names(self.kind).index("sufficient")]
            return ["RETRIEVE" if can_retrieve else ("ANSWER" if ps[i] >= self.tau_suff else "ABSTAIN") for i in range(n)]
        v = self.values(S)
        acts = []
        for i in range(n):
            va, vr, vb = v[i]
            if not can_retrieve:
                vr = -1e9
            best = ACTIONS[int(np.argmax([va - self.delta, vr, vb]))]
            acts.append(best)
        return acts

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
