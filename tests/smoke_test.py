"""CPU smoke test with stub models: checks control flow, shapes and bookkeeping without a GPU or any download.
   python tests/smoke_test.py
It does NOT validate model quality; it only catches wiring bugs before you spend GPU time."""
import os
import pickle
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from uact import metrics  # noqa: E402
from uact.controller import Controller, state_vector  # noqa: E402
from uact.features import FEATURE_NAMES, FeatureExtractor  # noqa: E402
from uact.report import summarize  # noqa: E402
from uact.systems import (UACT, BudgetMatched, CalibConfController, ConfModel, Ctx, HybridRAG, NoRAG,  # noqa: E402
                          score_records)
from uact.utils import load_config  # noqa: E402

rng = np.random.default_rng(0)
cfg = load_config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"))
cfg["retrieval"]["n_ctx"] = 3


class StubReranker:
    def score(self, q, texts):
        qs = set(q.lower().split())
        return np.array([min(1.0, len(qs & set(t.lower().split())) / max(1, len(qs)) * 2) for t in texts])


class StubNLI:
    def probs(self, p, h, **kw):
        out = rng.dirichlet([1, 2, 2], size=len(p))
        return out


class StubLLM:
    def generate(self, batch, max_tokens=400, temperature=0.0):
        outs = []
        for m in batch:
            last = m[-1]["content"]
            if "search query" in last:
                text = "pump vibration after maintenance procedure"
            elif "Classify the question" in last:
                text = "B"
            else:
                text = "INSUFFICIENT_EVIDENCE" if rng.random() < 0.2 else "Check the coupling alignment [P1]. Inspect the bearings [P2]."
            outs.append({"text": text, "n_prompt": 300, "n_gen": 20, "mean_logprob": float(-rng.random())})
        return outs

    def yes_prob(self, batch):
        return list(rng.random(len(batch)))


CORPUS = [{"pid": f"d{i}#0", "doc_id": f"d{i}", "text": f"pump vibration bearing coupling alignment document {i} " + " ".join(rng.choice(
    ["motor", "temperature", "seal", "valve", "shaft", "service", "torque"], 20)), "revision": "2025-01"} for i in range(40)]


class StubRetriever:
    def __init__(self):
        self.calls = 0

    def search(self, query, rerank_query=None, k=3, mode="hybrid", rerank=True, exclude=None):
        exclude = exclude or set()
        rr = StubReranker()
        c = [dict(p) for p in CORPUS if p["pid"] not in exclude]
        s = rr.score(rerank_query or query, [p["text"] for p in c])
        for p, v in zip(c, s):
            p["rerank"], p["rrf"] = float(v) + rng.random() * 0.01, 0.0
        c.sort(key=lambda p: -p["rerank"])
        self.calls += 1
        return c[:k]


class StubVerifier:
    def verify(self, answer, passages):
        ok = rng.random() > 0.3
        return {"claims": ["a", "b"], "support_rate": 1.0 if ok else 0.5, "all_supported": ok, "unsupported": [] if ok else ["b"]}


class StubEstimator:
    kind, mask = "five", None

    def predict_proba(self, X, calibrated=True):
        z = X[:, :5] * 3 + rng.normal(size=(len(X), 5))
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


QS = [{"qid": f"q{i}", "question": "pump vibration after maintenance bearing", "answerable": bool(i % 4), "reference_answer": "check coupling alignment",
       "gold_evidence_ids": ["d1"]} for i in range(24)]

# 1. metrics
conf, err = rng.random(200), rng.integers(0, 2, 200)
assert 0 <= metrics.ece(conf, 1 - err) <= 1 and metrics.aurc(conf, err) >= 0 and metrics.e_aurc(conf, err) >= -1e-9
assert abs(metrics.mcnemar_exact(err == 0, err == 0)["p_value"] - 1.0) < 1e-9
print("metrics ok")

# 2. features
rr = StubReranker()
fx = FeatureExtractor(rr, StubNLI(), StubLLM(), n_feat=3)
X = fx.featurize([{"question": q["question"], "passages": [dict(p) for p in CORPUS[:4]]} for q in QS[:5]])
assert X.shape == (5, len(FEATURE_NAMES)), X.shape
print("features ok", X.shape)

# 3. systems + controller
ctx = Ctx(cfg, StubRetriever(), StubLLM(), fx, StubEstimator(), StubVerifier(), None, None)
for cls in (NoRAG, HybridRAG, BudgetMatched):
    recs = cls(ctx).run(QS)
    score_records(ctx, QS, recs, use_judge=False)
    s = summarize(recs, ctx.cost)
    print(cls.__name__, {k: round(v, 3) for k, v in s.items() if k in ("coverage", "selective_risk", "retrieval_calls_mean")})
hy = HybridRAG(ctx).run(QS)
score_records(ctx, QS, hy, use_judge=False)
cm = ConfModel().fit(np.array([r["feat"] for r in hy]), np.array([int(r["correct"]) for r in hy]) | (np.arange(len(hy)) % 2),
                     np.array([r["feat"] for r in hy]), np.array([int(r["correct"]) for r in hy]) | (np.arange(len(hy)) % 2))
ctx.conf_model = cm
recs = CalibConfController(ctx).run(QS)
print("B8 ok", len(recs))

import train_controller as tc  # noqa: E402

B, N = cfg["control"]["max_rounds"], 120
states = []
for t in range(B + 1):
    probs = rng.dirichlet(np.ones(5), size=N)
    states.append({"X": rng.random((N, len(FEATURE_NAMES))).astype("float32"), "probs": probs,
                   "ans_answered": rng.random(N) > 0.1, "ans_correct": rng.random(N) < 0.3 + 0.2 * probs[:, 0] + 0.1 * t,
                   "ans_support": rng.random(N), "ans_tokens": np.full(N, 400.0), "ret_tokens": np.full(N, 80.0),
                   "answerable": rng.random(N) > 0.3})
R = {"states": states, "B": B, "kind": "five", "mask": None}
S, V = tc.stack(R, *tc.utilities(R, 3.0, 0.05, cfg))
ctl = Controller("five", "learned").fit(S, V)
print("controller sim:", {k: round(v, 3) for k, v in tc.simulate(ctl, R, cfg).items()})
for mode in ("threshold", "fixed_budget"):
    c = Controller("five", mode)
    print(mode, {k: round(v, 3) for k, v in tc.simulate(c, R, cfg).items()})
ctx.controller = ctl
for ver, ab in ((True, True), (False, True), (True, False)):
    recs = UACT(ctx, use_verifier=ver, allow_abstain=ab).run(QS)
    score_records(ctx, QS, recs, use_judge=False)
    s = summarize(recs, ctx.cost)
    assert all(len(r["actions"]) >= 1 for r in recs)
    print("UACT", ver, ab, {k: round(v, 3) for k, v in s.items() if k in ("coverage", "selective_risk", "retrieval_calls_mean", "faithfulness")})
with tempfile.TemporaryDirectory() as d:
    ctl.save(os.path.join(d, "c.pkl"))
    assert Controller.load(os.path.join(d, "c.pkl")).models.keys() == ctl.models.keys()
print("SMOKE TEST PASSED")
