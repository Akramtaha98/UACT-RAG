"""Evidence features z(q, D) = [r, c, x, a] (relevance, coverage, contradiction, answerability/agreement).

An 'item' is a dict: {"question", "equipment_model" (optional), "revision" (optional, query-specified),
"passages": [ {pid, doc_id, text, revision?, equipment_model?, rerank?}, ... ] }.
Passages may come from retrieval (rerank already set) or from a fixed RAGBench context (rerank computed here).
"""
import itertools

import numpy as np

from .llm import SUFF_PROMPT, format_context
from .utils import keywords

R_FEATS = ["rerank_top1", "rerank_top2", "rerank_mean3", "rerank_gap12", "n_rerank_gt50", "n_rerank_gt20"]
C_FEATS = ["kw_coverage", "kw_coverage_top1", "equip_match", "n_slots"]
X_FEATS = ["nli_contra_max", "nli_contra_mean", "rev_conflict"]
A_FEATS = ["nli_entail_max", "n_distinct_docs", "llm_suff", "log_evidence_len"]
FEATURE_NAMES = R_FEATS + C_FEATS + X_FEATS + A_FEATS
BASELINE_FEATURES = R_FEATS  # ablation A1: relevance-only ("baseline") features


class FeatureExtractor:
    def __init__(self, reranker, nli, llm=None, n_feat=5, use_llm_suff=True):
        self.reranker, self.nli, self.llm = reranker, nli, llm
        self.n_feat, self.use_llm_suff = n_feat, use_llm_suff

    def ensure_rerank(self, items):
        for it in items:
            ps = it["passages"]
            if ps and any("rerank" not in p for p in ps):
                s = self.reranker.score(it["question"], [p["text"] for p in ps])
                for p, v in zip(ps, s):
                    p["rerank"] = float(v)

    def _contradiction(self, items):
        """Max/mean contradiction and max entailment over passage pairs that address the same topic."""
        prem, hyp, owner = [], [], []
        for n, it in enumerate(items):
            ps = sorted(it["passages"], key=lambda p: -p.get("rerank", 0))[:4]
            kws = [set(keywords(p["text"])) for p in ps]
            for i, j in itertools.permutations(range(len(ps)), 2):
                inter, union = len(kws[i] & kws[j]), max(1, len(kws[i] | kws[j]))
                if inter / union >= 0.05:  # skip pairs about unrelated topics
                    prem.append(" ".join(ps[i]["text"].split()[:180]))
                    hyp.append(" ".join(ps[j]["text"].split()[:180]))
                    owner.append(n)
        contra = [[] for _ in items]
        entail = [[] for _ in items]
        if prem:
            probs = self.nli.probs(prem, hyp)
            for o, p in zip(owner, probs):
                contra[o].append(p[0])
                entail[o].append(p[2])
        return contra, entail

    def featurize(self, items):
        self.ensure_rerank(items)
        contra, entail = self._contradiction(items)
        suff = [0.5] * len(items)
        if self.llm is not None and self.use_llm_suff:
            msgs = []
            for it in items:
                ctx = format_context(sorted(it["passages"], key=lambda p: -p.get("rerank", 0))[: self.n_feat])
                q = it["question"] + (f" (equipment: {it['equipment_model']})" if it.get("equipment_model") else "")
                msgs.append([{"role": "user", "content": SUFF_PROMPT.format(q=q, ctx=ctx)}])
            suff = self.llm.yes_prob(msgs)
        X = []
        for n, it in enumerate(items):
            ps = sorted(it["passages"], key=lambda p: -p.get("rerank", 0))[: self.n_feat]
            r = [p.get("rerank", 0.0) for p in ps] + [0.0, 0.0, 0.0]
            slots = keywords(it["question"])
            text_all = " ".join(p["text"].lower() for p in ps)
            text_top = ps[0]["text"].lower() if ps else ""
            cov = np.mean([s in text_all for s in slots]) if slots else 0.0
            cov1 = np.mean([s in text_top for s in slots]) if slots else 0.0
            em = it.get("equipment_model")
            equip = 1.0 if not em else float(any((p.get("equipment_model") == em) or (em.lower() in p["text"].lower()) for p in ps))
            revs = {p.get("revision") for p in ps if p.get("revision")}
            rev_conflict = float(len(revs) > 1 and not it.get("revision"))
            c = contra[n] or [0.0]
            e = entail[n] or [0.0]
            X.append(
                [
                    r[0], r[1], float(np.mean(r[:3])), r[0] - r[1],
                    float(sum(v > 0.5 for v in r[: len(ps)])), float(sum(v > 0.2 for v in r[: len(ps)])),
                    cov, cov1, equip, float(len(slots)),
                    float(max(c)), float(np.mean(c)), rev_conflict,
                    float(max(e)), float(len({p.get("doc_id") for p in ps})), float(suff[n]),
                    float(np.log1p(sum(len(p["text"].split()) for p in ps))),
                ]
            )
        return np.asarray(X, dtype="float32")


def feature_mask(names):
    return np.array([f in names for f in FEATURE_NAMES])
