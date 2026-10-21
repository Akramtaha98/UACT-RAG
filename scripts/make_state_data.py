"""Create evidence-state training data (sufficient / partial / irrelevant / missing / contradictory).

Stages
  candidates        LLM drafts contradiction candidates from RAGBench relevant sentences -> a file a HUMAN REVIEWS
                    (set "reviewed": true/false per line). Unreviewed contradictions are never used unless
                    --allow_unreviewed is passed (development only; do not report results trained that way).
  build             builds state items from RAGBench with transparent rules:
                      sufficient   completeness >= 0.8 and the context has relevant sentences
                      partial      relevant sentences exist but completeness < 0.8
                      irrelevant   context swapped for another question's context  (or no relevant sentences)
                      missing      all relevant sentences removed from a sufficient context
                      contradictory reviewed contradicting passage appended to a sufficient context
  techqa_retrieved  labels REAL retrieval outcomes on TechQA: unanswerable -> missing; answerable & gold doc retrieved
                    -> sufficient; answerable & gold absent -> irrelevant (repairable retrieval failure).

  python scripts/make_state_data.py candidates --subsets emanual techqa --n 600
  python scripts/make_state_data.py build --subsets emanual techqa
  python scripts/make_state_data.py techqa_retrieved --data data/techqa
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uact.utils import load_config, read_jsonl, write_jsonl  # noqa: E402

LABELS = ["sufficient", "partial", "irrelevant", "missing", "contradictory"]


def _docs_text(e):
    return [d if isinstance(d, str) else str(d) for d in e["documents"]]


def _relevant_sentences(e):
    keys = set(e.get("all_relevant_sentence_keys") or [])
    out = []
    for doc in e.get("documents_sentences") or []:
        for key, sent in doc:
            if key in keys:
                out.append(sent)
    return out


def _without_relevant(e):
    keys = set(e.get("all_relevant_sentence_keys") or [])
    docs = []
    for doc in e.get("documents_sentences") or []:
        kept = " ".join(s for k, s in doc if k not in keys)
        if len(kept.split()) >= 15:
            docs.append(kept)
    return docs


def _load(subsets, data, split):
    rows = []
    for s in subsets:
        p = os.path.join(data, "ragbench", f"{s}_{split}.jsonl")
        if os.path.exists(p):
            rows += read_jsonl(p)
    return rows


def _passages(texts, prefix):
    return [{"pid": f"{prefix}#{i}", "doc_id": f"{prefix}_d{i}", "text": t} for i, t in enumerate(texts)]


def candidates(a, cfg):
    from uact.llm import LLM

    rng = random.Random(0)
    rows = [e for e in _load(a.subsets, a.data, "train") if _relevant_sentences(e) and (e.get("completeness_score") or 0) >= 0.8]
    rng.shuffle(rows)
    rows = rows[: a.n]
    llm = LLM(cfg["models"]["generator"], cfg["models"]["vllm_gpu_mem_util"], cfg["models"]["max_model_len"], cfg["seed"])
    msgs, srcs = [], []
    for e in rows:
        sent = rng.choice(_relevant_sentences(e))
        msgs.append([{"role": "user", "content": "Rewrite this technical-manual sentence so it plausibly CONTRADICTS the original "
                      "(e.g. a different value, step order, or condition), as a different manual revision might. Keep the style. "
                      f"Output only the rewritten sentence.\n\nSentence: {sent}"}])
        srcs.append((e["id"], sent))
    outs = llm.generate(msgs, max_tokens=100)
    write_jsonl(os.path.join(a.data, "contradiction_candidates.jsonl"),
                [{"source_id": i, "original": s, "contradicting": o["text"], "reviewed": None} for (i, s), o in zip(srcs, outs)])
    print("Wrote candidates. A qualified reviewer must set reviewed=true/false on every line before 'build'.")


def build(a, cfg):
    rng = random.Random(0)
    appr = {}
    cp = os.path.join(a.data, "contradiction_candidates.jsonl")
    if os.path.exists(cp):
        for c in read_jsonl(cp):
            if c["reviewed"] is True or (a.allow_unreviewed and c["reviewed"] is None):
                appr[c["source_id"]] = c
    if a.allow_unreviewed:
        print("WARNING: unreviewed contradictions included -> development use only.")
    out = {"train": [], "val": [], "cal": [], "test": []}
    for split, names in (("train", ["train"]), ("validation", ["val", "cal"]), ("test", ["test"])):
        rows = _load(a.subsets, a.data, split)
        if not rows:
            continue
        pool = list(rows)
        for e in rows:
            tgt = names[rng.randrange(len(names))]
            eid = e["id"]
            rel = _relevant_sentences(e)
            comp = e.get("completeness_score")
            if not rel:
                base = "irrelevant"
            elif comp is not None and comp >= 0.8:
                base = "sufficient"
            else:
                base = "partial"
            items = [(base, _docs_text(e), "original")]
            if base == "sufficient":
                miss = _without_relevant(e)
                if miss:
                    items.append(("missing", miss, "relevant_removed"))
                if eid in appr:
                    items.append(("contradictory", _docs_text(e) + [appr[eid]["contradicting"]], "reviewed_contradiction"))
            other = rng.choice(pool)
            if other["id"] != eid:
                items.append(("irrelevant", _docs_text(other), "context_swapped"))
            for lab, texts, src in items:
                out[tgt].append({"id": f"{eid}:{src}", "question": e["question"], "passages": _passages(texts, f"{eid}:{src}"),
                                 "state": lab, "source": src, "reviewed": src != "reviewed_contradiction" or appr[eid]["reviewed"] is True})
    for k, rows in out.items():
        write_jsonl(os.path.join(a.data, "states", f"{k}.jsonl"), rows)
        print(k, len(rows), {l: sum(r["state"] == l for r in rows) for l in LABELS})


def techqa_retrieved(a, cfg):
    from uact.loading import load_ctx

    ctx = load_ctx(cfg, os.path.join(a.data, "index"), os.path.join(a.data, "corpus.jsonl"), with_llm=False, with_verifier=False)
    for split in ("train", "val", "test"):
        qs = read_jsonl(os.path.join(a.data, f"{split}.jsonl"))
        rows = []
        for q in qs:
            ev = ctx.retriever.search(q["question"], k=cfg["retrieval"]["n_ctx"])
            gold = set(q.get("gold_evidence_ids") or [])
            if not q["answerable"]:
                lab = "missing"
            elif gold & {p["doc_id"] for p in ev}:
                lab = "sufficient"
            else:
                lab = "irrelevant"
            rows.append({"id": f"{q['qid']}:retrieved", "question": q["question"], "passages": ev, "state": lab,
                         "source": "techqa_retrieved", "reviewed": True})
        write_jsonl(os.path.join(a.data, "states", f"{split}_retrieved.jsonl"), rows)
        print(split, len(rows))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["candidates", "build", "techqa_retrieved"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", default="data")
    ap.add_argument("--subsets", nargs="+", default=["emanual", "techqa"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--allow_unreviewed", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    {"candidates": candidates, "build": build, "techqa_retrieved": techqa_retrieved}[a.stage](a, cfg)
