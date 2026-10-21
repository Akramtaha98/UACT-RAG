"""Claim-level verifier: each answer sentence must be entailed by its cited passages (NLI).

This is an automated proxy. Validate it against the human audit subset before reporting faithfulness numbers.
"""
import re

import numpy as np

from .utils import split_sentences

CITE = re.compile(r"\[P(\d+)\]")


def _windows(text, size=220, stride=180):
    w = text.split()
    if len(w) <= size:
        return [text]
    return [" ".join(w[i : i + size]) for i in range(0, max(1, len(w) - size + stride), stride)]


class Verifier:
    def __init__(self, nli, threshold=0.5):
        self.nli, self.threshold = nli, threshold

    def verify(self, answer, passages):
        """Returns dict(claims=[...], support_rate, all_supported, unsupported=[sentences])."""
        claims = [s for s in split_sentences(answer) if len(s.split()) >= 4]
        if not claims:
            return {"claims": [], "support_rate": 1.0, "all_supported": True, "unsupported": []}
        prem, hyp, owner = [], [], []
        for ci, c in enumerate(claims):
            cited = [int(m) - 1 for m in CITE.findall(c) if 0 < int(m) <= len(passages)]
            pool = [passages[i] for i in cited] if cited else passages  # uncited claims are checked against all evidence
            clean = CITE.sub("", c).replace("DOCUMENTED", "").replace("INFERENCE", "").strip(" :-")
            for p in pool:
                for w in _windows(p["text"]):
                    prem.append(w)
                    hyp.append(clean)
                    owner.append(ci)
        probs = self.nli.probs(prem, hyp) if prem else np.zeros((0, 3))
        best = np.zeros(len(claims))
        for o, p in zip(owner, probs):
            best[o] = max(best[o], p[2])
        sup = best >= self.threshold
        return {
            "claims": claims,
            "claim_scores": best.tolist(),
            "support_rate": float(sup.mean()),
            "all_supported": bool(sup.all()),
            "unsupported": [c for c, s in zip(claims, sup) if not s],
        }
