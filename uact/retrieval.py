"""Hybrid retrieval: BM25 (bm25s) + dense (BGE-M3 + FAISS) + reciprocal rank fusion + cross-encoder reranking.

Corpus format (JSONL, one passage per line):
  {"pid": "...", "doc_id": "...", "text": "...", "revision": "2025-01", "equipment_model": "Model-A"}
"""
import os

import numpy as np

from .utils import read_jsonl


def rrf_fuse(rank_lists, k=60, top=50):
    """rank_lists: list of lists of passage indices (best first). Returns [(idx, score)] best first."""
    scores = {}
    for lst in rank_lists:
        for r, idx in enumerate(lst):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + r + 1)
    return sorted(scores.items(), key=lambda t: -t[1])[:top]


def build_index(cfg, corpus_path, index_dir, device="cuda"):
    import bm25s
    import faiss

    from .models import Embedder

    os.makedirs(index_dir, exist_ok=True)
    rows = read_jsonl(corpus_path)
    texts = [r["text"] for r in rows]
    print(f"[index] {len(texts)} passages")
    tokens = bm25s.tokenize(texts, stopwords="en")
    bm = bm25s.BM25()
    bm.index(tokens)
    bm.save(os.path.join(index_dir, "bm25"))
    emb = Embedder(cfg["models"]["embedder"], device).encode(texts)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, os.path.join(index_dir, "dense.faiss"))
    print("[index] done")


class Retriever:
    def __init__(self, cfg, corpus_path, index_dir, reranker, embedder=None, device="cuda"):
        import bm25s
        import faiss

        from .models import Embedder

        self.cfg = cfg["retrieval"]
        self.passages = read_jsonl(corpus_path)
        self.bm25 = bm25s.BM25.load(os.path.join(index_dir, "bm25"))
        self.bm25s = bm25s
        self.index = faiss.read_index(os.path.join(index_dir, "dense.faiss"))
        self.embedder = embedder or Embedder(cfg["models"]["embedder"], device)
        self.reranker = reranker
        self.calls = 0

    def _bm25(self, query, k):
        res, _ = self.bm25.retrieve(self.bm25s.tokenize([query], stopwords="en"), k=min(k, len(self.passages)))
        return [int(i) for i in res[0]]

    def _dense(self, query, k):
        q = self.embedder.encode([query])
        _, idx = self.index.search(q, min(k, len(self.passages)))
        return [int(i) for i in idx[0] if i >= 0]

    def search(self, query, rerank_query=None, k=None, mode="hybrid", rerank=True, exclude=None):
        """Return up to k passage dicts sorted by reranker score (or fused score).

        rerank_query: scores are computed against this query (use the ORIGINAL question so scores stay
        comparable across rounds even when the search query was reformulated).
        exclude: set of pids already in the evidence set (so a RETRIEVE action brings new evidence).
        """
        c = self.cfg
        k = k or c["n_ctx"]
        exclude = exclude or set()
        self.calls += 1
        lists = []
        if mode in ("hybrid", "sparse"):
            lists.append(self._bm25(query, c["bm25_k"] + len(exclude)))
        if mode in ("hybrid", "dense"):
            lists.append(self._dense(query, c["dense_k"] + len(exclude)))
        fused = rrf_fuse(lists, c["rrf_k"], c["fuse_k"] + len(exclude))
        fused = [(i, s) for i, s in fused if self.passages[i]["pid"] not in exclude][: c["fuse_k"]]
        cands = []
        for i, s in fused:
            p = dict(self.passages[i])
            p["rrf"] = float(s)
            cands.append(p)
        if rerank and cands:
            rs = self.reranker.score(rerank_query or query, [p["text"] for p in cands])
            for p, r in zip(cands, rs):
                p["rerank"] = float(r)
            cands.sort(key=lambda p: -p["rerank"])
        else:
            for p in cands:
                p["rerank"] = p["rrf"]
        return cands[:k]


def merge_evidence(existing, new, cap):
    """Union by pid, keep the best `cap` by reranker score."""
    seen = {p["pid"]: p for p in existing}
    for p in new:
        if p["pid"] not in seen or p["rerank"] > seen[p["pid"]]["rerank"]:
            seen[p["pid"]] = p
    return sorted(seen.values(), key=lambda p: -p["rerank"])[:cap]
