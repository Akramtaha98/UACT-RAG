"""All evaluated systems share one interface: system.run(list_of_query_dicts) -> list_of_record_dicts.

Query dict: {"qid", "question", "equipment_model"?, "revision"?, "answerable", "reference_answer"?, "gold_evidence_ids"?}
Queries are processed as a batch (round-synchronous) so vLLM is used efficiently. For latency measurement run
with eval.batch_size = 1 (see scripts/run_eval.py).
"""
import pickle
import re

import numpy as np

from .controller import state_vector
from .estimator import p_bad, state_names
from .features import FEATURE_NAMES
from .llm import COMPLEXITY_PROMPT, JUDGE_PROMPT, REWRITE_PROMPT, answer_messages
from .utils import keywords, token_f1


class Ctx:
    """Bundle of loaded components."""

    def __init__(self, cfg, retriever, llm, fx, estimator=None, verifier=None, controller=None, conf_model=None):
        self.cfg, self.retriever, self.llm, self.fx = cfg, retriever, llm, fx
        self.estimator, self.verifier, self.controller, self.conf_model = estimator, verifier, controller, conf_model

    @property
    def B(self):
        return self.cfg["control"]["max_rounds"]

    def cost(self, prompt_tokens, gen_tokens, retrievals):
        c = self.cfg["control"]["cost"]
        return (prompt_tokens + gen_tokens) / 1000.0 * c["per_1k_tokens"] + retrievals * c["per_retrieval"]


def new_rec(q, system):
    return {"qid": q["qid"], "system": system, "answered": False, "answer": "", "actions": [], "retrieval_calls": 0,
            "prompt_tokens": 0, "gen_tokens": 0, "conf": 0.0, "evidence": [], "trace": []}


def _ctx_passages(ev, n):
    return sorted(ev, key=lambda p: -p.get("rerank", 0))[:n]


def _is_abstain_text(t):
    return "INSUFFICIENT_EVIDENCE" in t[:60]


def gen_answers(ctx, qs, evs, recs, idx, extra=None):
    """Generate answers for queries at positions idx using evidence evs[i]; update records. evs[i] None = no retrieval."""
    n_ctx = ctx.cfg["retrieval"]["n_ctx"]
    msgs = []
    for i in idx:
        ps = None if evs[i] is None else _ctx_passages(evs[i], n_ctx)
        m = answer_messages(qs[i]["question"], qs[i].get("equipment_model"), ps)
        if extra and i in extra:
            m = m + extra[i]
        msgs.append(m)
    outs = ctx.llm.generate(msgs, max_tokens=ctx.cfg["generation"]["max_new_tokens"], temperature=ctx.cfg["generation"]["temperature"])
    res = {}
    for i, o in zip(idx, outs):
        recs[i]["prompt_tokens"] += o["n_prompt"]
        recs[i]["gen_tokens"] += o["n_gen"]
        o["abstained"] = _is_abstain_text(o["text"])
        res[i] = o
    return res


def finalize_answer(rec, out, ev, ctx, conf=None):
    rec["answer"] = out["text"]
    rec["answered"] = not out["abstained"]
    rec["mean_logprob"] = out["mean_logprob"]
    rec["evidence"] = [{k: p.get(k) for k in ("pid", "doc_id", "text", "revision", "rerank")} for p in _ctx_passages(ev or [], ctx.cfg["retrieval"]["n_ctx"])]
    rec["conf"] = conf if conf is not None else out["mean_logprob"]


def gaps_for(q, ev):
    text = " ".join(p["text"].lower() for p in ev) if ev else ""
    g = [s for s in keywords(q["question"]) if s not in text][:8]
    em = q.get("equipment_model")
    if em and em.lower() not in text:
        g.append(f"equipment model {em}")
    return g or ["the specific procedure or cause"]


def retrieve_more(ctx, qs, evs, recs, idx):
    """RETRIEVE action: rewrite the query from unresolved gaps, search excluding seen passages, merge."""
    from .retrieval import merge_evidence

    msgs = [[{"role": "user", "content": REWRITE_PROMPT.format(q=qs[i]["question"], gaps=", ".join(gaps_for(qs[i], evs[i])))}] for i in idx]
    outs = ctx.llm.generate(msgs, max_tokens=48)
    for i, o in zip(idx, outs):
        recs[i]["prompt_tokens"] += o["n_prompt"]
        recs[i]["gen_tokens"] += o["n_gen"]
        query = o["text"].strip().strip('"') or qs[i]["question"]
        seen = {p["pid"] for p in evs[i]}
        new = ctx.retriever.search(query, rerank_query=qs[i]["question"], k=ctx.cfg["retrieval"]["n_ctx"], exclude=seen)
        recs[i]["retrieval_calls"] += 1
        evs[i] = merge_evidence(evs[i], new, ctx.cfg["retrieval"]["max_evidence"])


def initial_retrieval(ctx, qs, recs, mode="hybrid", rerank=True):
    evs = []
    for q, r in zip(qs, recs):
        evs.append(ctx.retriever.search(q["question"], k=ctx.cfg["retrieval"]["n_ctx"], mode=mode, rerank=rerank))
        r["retrieval_calls"] += 1
    return evs


class NoRAG:
    name = "B0_no_retrieval"

    def __init__(self, ctx):
        self.ctx = ctx

    def run(self, qs):
        recs = [new_rec(q, self.name) for q in qs]
        outs = gen_answers(self.ctx, qs, [None] * len(qs), recs, range(len(qs)))
        for i, o in outs.items():
            finalize_answer(recs[i], o, [], self.ctx)
            recs[i]["answered"] = True  # B0 has no evidence to abstain on
            recs[i]["actions"] = ["ANSWER"]
        return recs


class StandardRAG:
    name = "B1_standard_rag"

    def __init__(self, ctx):
        self.ctx = ctx

    def run(self, qs):
        recs = [new_rec(q, self.name) for q in qs]
        evs = initial_retrieval(self.ctx, qs, recs, mode="dense", rerank=False)
        outs = gen_answers(self.ctx, qs, evs, recs, range(len(qs)))
        for i, o in outs.items():
            finalize_answer(recs[i], o, evs[i], self.ctx, conf=evs[i][0]["rerank"] if evs[i] else 0.0)
            recs[i]["actions"] = ["ANSWER"]
        return recs


class HybridRAG:
    name = "B2_hybrid_rag"

    def __init__(self, ctx):
        self.ctx = ctx

    def run(self, qs):
        recs = [new_rec(q, self.name) for q in qs]
        evs = initial_retrieval(self.ctx, qs, recs)
        outs = gen_answers(self.ctx, qs, evs, recs, range(len(qs)))
        for i, o in outs.items():
            finalize_answer(recs[i], o, evs[i], self.ctx, conf=evs[i][0]["rerank"] if evs[i] else 0.0)
            recs[i]["actions"] = ["ANSWER"]
            recs[i]["feat"] = conf_features(qs[i], evs[i], o)
        return recs


def conf_features(q, ev, out):
    ps = _ctx_passages(ev, 5)
    r = [p["rerank"] for p in ps] + [0.0, 0.0, 0.0]
    text = " ".join(p["text"].lower() for p in ps)
    slots = keywords(q["question"])
    cov = float(np.mean([s in text for s in slots])) if slots else 0.0
    return [r[0], float(np.mean(r[:3])), out["mean_logprob"], cov]


class ConfModel:
    """Calibrated P(correct) from simple confidence features (used by B6/B8)."""

    def __init__(self):
        self.lr = self.iso = None

    def fit(self, F, y, Fv, yv):
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        self.lr = LogisticRegression(max_iter=1000).fit(F, y)
        self.iso = IsotonicRegression(out_of_bounds="clip").fit(self.lr.predict_proba(Fv)[:, 1], yv)
        return self

    def predict(self, F):
        return self.iso.predict(self.lr.predict_proba(np.asarray(F))[:, 1])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


class FixedAbstain:
    """B6: hybrid RAG + calibrated-confidence threshold abstention."""

    name = "B6_fixed_rag_abstain"

    def __init__(self, ctx, tau=0.5):
        self.ctx, self.tau = ctx, tau

    def run(self, qs):
        recs = HybridRAG(self.ctx).run(qs)
        conf = self.ctx.conf_model.predict([r["feat"] for r in recs])
        for r, c in zip(recs, conf):
            r["system"], r["conf"] = self.name, float(c)
            if c < self.tau:
                r["answered"], r["actions"] = False, ["ABSTAIN"]
        return recs


class SufficiencyAbstain:
    """B9: abstain when an LLM sufficiency autorater says the context is insufficient (Joren et al., 2025 style)."""

    name = "B9_sufficiency_autorater"

    def __init__(self, ctx, tau=0.5):
        self.ctx, self.tau = ctx, tau

    def run(self, qs):
        from .llm import SUFF_PROMPT, format_context

        recs = HybridRAG(self.ctx).run(qs)
        msgs = [[{"role": "user", "content": SUFF_PROMPT.format(q=q["question"], ctx=format_context(r["evidence"]))}] for q, r in zip(qs, recs)]
        ps = self.ctx.llm.yes_prob(msgs)
        for r, p in zip(recs, ps):
            r["system"], r["conf"] = self.name, float(p)
            if p < self.tau:
                r["answered"], r["actions"] = False, ["ABSTAIN"]
        return recs


class BudgetMatched:
    """B7: always perform `rounds` extra retrieval rounds (no decision), then answer."""

    name = "B7_budget_matched_multiretrieval"

    def __init__(self, ctx, rounds=None):
        self.ctx, self.rounds = ctx, rounds

    def run(self, qs):
        ctx = self.ctx
        rounds = self.rounds if self.rounds is not None else ctx.B
        recs = [new_rec(q, self.name) for q in qs]
        evs = initial_retrieval(ctx, qs, recs)
        for _ in range(rounds):
            retrieve_more(ctx, qs, evs, recs, list(range(len(qs))))
        outs = gen_answers(ctx, qs, evs, recs, range(len(qs)))
        for i, o in outs.items():
            finalize_answer(recs[i], o, evs[i], ctx, conf=evs[i][0]["rerank"] if evs[i] else 0.0)
            recs[i]["actions"] = ["RETRIEVE"] * rounds + ["ANSWER"]
        return recs


class CalibConfController:
    """B8: calibrated-confidence controller (answer / retrieve / abstain from a draft-answer confidence)."""

    name = "B8_calibrated_confidence_controller"

    def __init__(self, ctx, tau_hi=0.6, tau_lo=0.15):
        self.ctx, self.tau_hi, self.tau_lo = ctx, tau_hi, tau_lo

    def run(self, qs):
        ctx = self.ctx
        recs = [new_rec(q, self.name) for q in qs]
        evs = initial_retrieval(ctx, qs, recs)
        done = [False] * len(qs)
        for t in range(ctx.B + 1):
            idx = [i for i in range(len(qs)) if not done[i]]
            if not idx:
                break
            outs = gen_answers(ctx, qs, evs, recs, idx)  # draft answers; their cost is charged
            conf = ctx.conf_model.predict([conf_features(qs[i], evs[i], outs[i]) for i in idx])
            more = []
            for i, c in zip(idx, conf):
                if c >= self.tau_hi or (t == ctx.B and c >= self.tau_lo):
                    finalize_answer(recs[i], outs[i], evs[i], ctx, conf=float(c))
                    recs[i]["actions"].append("ANSWER")
                    done[i] = True
                elif t < ctx.B:
                    recs[i]["actions"].append("RETRIEVE")
                    more.append(i)
                else:
                    recs[i]["conf"] = float(c)
                    recs[i]["actions"].append("ABSTAIN")
                    done[i] = True
            if more:
                retrieve_more(ctx, qs, evs, recs, more)
        return recs


class CragStyle:
    """B4 approximation: CRAG-style retrieval evaluator (reranker score) -> correct / ambiguous / incorrect actions.
    NOT the original CRAG: no web search, no decompose-recompose. Label as an approximation in the paper."""

    name = "B4_crag_style_approx"

    def __init__(self, ctx, hi=0.7, lo=0.3):
        self.ctx, self.hi, self.lo = ctx, hi, lo

    def run(self, qs):
        from .retrieval import merge_evidence

        ctx = self.ctx
        recs = [new_rec(q, self.name) for q in qs]
        evs = initial_retrieval(ctx, qs, recs)
        for i in range(len(qs)):
            top = evs[i][0]["rerank"] if evs[i] else 0.0
            if top >= self.hi:
                recs[i]["actions"].append("CORRECT")
            else:
                before = list(evs[i])
                retrieve_more(ctx, qs, evs, recs, [i])
                if top < self.lo:  # incorrect: keep only the new evidence
                    old = {p["pid"] for p in before}
                    evs[i] = [p for p in evs[i] if p["pid"] not in old] or before
                    recs[i]["actions"].append("INCORRECT")
                else:
                    recs[i]["actions"].append("AMBIGUOUS")
        outs = gen_answers(ctx, qs, evs, recs, range(len(qs)))
        for i, o in outs.items():
            finalize_answer(recs[i], o, evs[i], ctx, conf=evs[i][0]["rerank"] if evs[i] else 0.0)
        return recs


class AdaptiveStyle:
    """B3 approximation: zero-shot LLM complexity classifier routes to no-retrieval / single / iterative retrieval.
    NOT the original Adaptive-RAG (which trains a classifier on automatically collected labels)."""

    name = "B3_adaptive_rag_style_approx"

    def __init__(self, ctx):
        self.ctx = ctx

    def run(self, qs):
        ctx = self.ctx
        recs = [new_rec(q, self.name) for q in qs]
        outs = ctx.llm.generate([[{"role": "user", "content": COMPLEXITY_PROMPT.format(q=q["question"])}] for q in qs], max_tokens=2)
        route = []
        for r, o in zip(recs, outs):
            r["prompt_tokens"] += o["n_prompt"]
            r["gen_tokens"] += o["n_gen"]
            route.append("A" if o["text"].strip().upper().startswith("A") else ("C" if o["text"].strip().upper().startswith("C") else "B"))
        evs = [None] * len(qs)
        for i, q in enumerate(qs):
            if route[i] != "A":
                evs[i] = ctx.retriever.search(q["question"], k=ctx.cfg["retrieval"]["n_ctx"])
                recs[i]["retrieval_calls"] += 1
        for _ in range(ctx.B):
            idx = [i for i in range(len(qs)) if route[i] == "C"]
            if idx:
                retrieve_more(ctx, qs, evs, recs, idx)
        outs = gen_answers(ctx, qs, evs, recs, range(len(qs)))
        for i, o in outs.items():
            finalize_answer(recs[i], o, evs[i] or [], ctx, conf=(evs[i][0]["rerank"] if evs[i] else 0.0))
            recs[i]["actions"] = [f"ROUTE_{route[i]}"]
        return recs


class UACT:
    """The proposed system. Flags implement the ablations:
    use_verifier=False -> A3;  allow_abstain=False -> A4 (offline benchmark only);
    controller.mode = 'threshold' -> A5, 'fixed_budget' -> A2;
    a controller trained with mu=0 -> A6; an estimator trained with states=three / relevance-only mask -> A7 / A1."""

    def __init__(self, ctx, name="UACT-RAG", use_verifier=True, allow_abstain=True):
        self.ctx, self.name, self.use_verifier, self.allow_abstain = ctx, name, use_verifier, allow_abstain

    def _items(self, qs, evs, idx):
        return [{"question": qs[i]["question"], "equipment_model": qs[i].get("equipment_model"),
                 "revision": qs[i].get("revision"), "passages": evs[i]} for i in idx]

    def run(self, qs):
        ctx = self.ctx
        B, kind = ctx.B, ctx.estimator.kind
        recs = [new_rec(q, self.name) for q in qs]
        evs = initial_retrieval(ctx, qs, recs)
        done = [False] * len(qs)
        for t in range(B + 1):
            idx = [i for i in range(len(qs)) if not done[i]]
            if not idx:
                break
            X = ctx.fx.featurize(self._items(qs, evs, idx))
            Xm = X[:, ctx.estimator.mask] if ctx.estimator.mask is not None else X
            probs = ctx.estimator.predict_proba(X)
            S = state_vector(probs, kind, t, B, Xm)
            acts = ctx.controller.decide(S, probs, t, B)
            margin = ctx.controller.confidence(S) if ctx.controller.mode == "learned" else probs[:, state_names(kind).index("sufficient")]
            ans_idx, ret_idx = [], []
            for k, i in enumerate(idx):
                a = acts[k]
                if a == "ABSTAIN" and not self.allow_abstain:
                    a = "ANSWER"
                recs[i]["trace"].append({"t": t, "action": a, "probs": [round(float(p), 4) for p in probs[k]]})
                recs[i]["actions"].append(a)
                recs[i]["conf"] = float(margin[k])
                if a == "ANSWER":
                    ans_idx.append(i)
                elif a == "RETRIEVE":
                    ret_idx.append(i)
                else:
                    done[i] = True
            if ans_idx:
                outs = gen_answers(ctx, qs, evs, recs, ans_idx)
                redo = []
                for i in ans_idx:
                    o = outs[i]
                    if o["abstained"] or not self.use_verifier:
                        finalize_answer(recs[i], o, evs[i], ctx, conf=recs[i]["conf"])
                        done[i] = True
                        continue
                    v = ctx.verifier.verify(o["text"], _ctx_passages(evs[i], ctx.cfg["retrieval"]["n_ctx"]))
                    if v["all_supported"]:
                        finalize_answer(recs[i], o, evs[i], ctx, conf=recs[i]["conf"])
                        done[i] = True
                    else:
                        redo.append((i, o, v))
                if redo:  # one regeneration with the unsupported claims flagged, then verify again
                    extra = {i: [{"role": "assistant", "content": o["text"]},
                                 {"role": "user", "content": "These statements are NOT supported by the passages:\n- " + "\n- ".join(v["unsupported"]) +
                                  "\nRewrite the answer using only supported statements, or reply INSUFFICIENT_EVIDENCE."}]
                             for i, o, v in redo}
                    outs2 = gen_answers(ctx, qs, evs, recs, [i for i, _, _ in redo], extra=extra)
                    for i, o, v in redo:
                        o2 = outs2[i]
                        v2 = ctx.verifier.verify(o2["text"], _ctx_passages(evs[i], ctx.cfg["retrieval"]["n_ctx"])) if not o2["abstained"] else None
                        if v2 is not None and (v2["all_supported"] or not self.allow_abstain):
                            finalize_answer(recs[i], o2, evs[i], ctx, conf=recs[i]["conf"])
                        else:
                            recs[i]["answer"], recs[i]["answered"] = "", False
                            recs[i]["actions"].append("ABSTAIN_UNVERIFIED")
                        done[i] = True
            if ret_idx:
                retrieve_more(ctx, qs, evs, recs, ret_idx)
        for i, r in enumerate(recs):
            if not done[i]:
                r["actions"].append("ABSTAIN_BUDGET")
        return recs


def parse_verdict(text):
    """'VERDICT: CORRECT' / 'VERDICT: INCORRECT' (last occurrence wins); anything unparseable counts as incorrect."""
    m = re.findall(r"VERDICT:\s*(INCORRECT|CORRECT)", text.upper())
    return bool(m) and m[-1] == "CORRECT"


def score_records(ctx, qs, recs, use_judge=None):
    """Add correctness / support fields. Not included in latency or cost."""
    cfg = ctx.cfg["eval"]
    use_judge = cfg["use_llm_judge"] if use_judge is None else use_judge
    ans_idx = [i for i, r in enumerate(recs) if r["answered"]]
    judge, judge_text = {}, {}
    if use_judge and ans_idx:
        keys = [i for i in ans_idx if qs[i]["answerable"]]
        msgs = [[{"role": "user", "content": JUDGE_PROMPT.format(q=qs[i]["question"], ref=qs[i].get("reference_answer", ""), ans=recs[i]["answer"])}] for i in keys]
        for i, o in zip(keys, ctx.llm.generate(msgs, max_tokens=120, temperature=0.0)):
            judge[i], judge_text[i] = parse_verdict(o["text"]), o["text"]
    grounded = cfg.get("grounded_correctness", True)
    for i, (q, r) in enumerate(zip(qs, recs)):
        f1 = token_f1(r["answer"], q.get("reference_answer", "")) if (r["answered"] and q["answerable"]) else 0.0
        r["f1"] = f1
        gold = set(q.get("gold_evidence_ids") or [])
        ev = r.get("evidence") or []
        gold_hit = bool(gold & ({p.get("doc_id") for p in ev} | {p.get("pid") for p in ev}))
        if not r["answered"]:
            r["correct"], r["error"] = False, 0
        elif not q["answerable"]:
            r["correct"], r["error"] = False, 1
        else:
            if grounded and gold:  # primary rule, validated against a human audit
                r["correct"] = bool(gold_hit and f1 >= cfg["f1_correct_threshold"])
            elif use_judge:
                r["correct"] = bool(judge[i])
            else:
                r["correct"] = f1 >= cfg["f1_correct_threshold"]
            if i in judge:  # LLM judge logged as a secondary metric
                r["judge_correct"], r["judge_text"] = bool(judge[i]), judge_text[i]
            r["error"] = int(not r["correct"])
        if r["answered"] and ctx.verifier is not None and r["evidence"]:
            v = ctx.verifier.verify(r["answer"], r["evidence"])
            r["support_rate"], r["all_supported"] = v["support_rate"], v["all_supported"]
            r["n_unsupported_claims"], r["n_claims"] = len(v["unsupported"]), len(v["claims"])
        if gold:
            seen = {p["pid"] for p in r["evidence"]} | {p.get("doc_id") for p in r["evidence"]}
            r["gold_recall"] = float(len(gold & seen) / len(gold))
    return recs
