"""Leave-one-document-out end-to-end benchmark on RAGBench engineering-manual (emanual) data.

Answerable questions keep their gold document(s) in the index. For a random half of the questions the gold
document(s) are REMOVED from the corpus, so the question is unanswerable BY CONSTRUCTION while the index still
returns real, topically related manual text. Questions that share a gold document are grouped so a document removed
for one question is never needed by an answerable question. Splits are RAGBench's own train/validation/test, so test
questions were never used to train the estimator. Semi-synthetic: say so in the paper.

  python scripts/make_emanual_loo.py --data data --out data/emanual_loo
"""
import argparse
import hashlib
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uact.utils import chunk_words, read_jsonl, write_jsonl  # noqa: E402


def did(text):
    return "em_" + hashlib.md5(text.encode()).hexdigest()[:12]


def gold_docs(e):
    """doc ids (hash of document text) of documents that contain a relevant sentence."""
    keys = set(e.get("all_relevant_sentence_keys") or [])
    out = []
    for text, sents in zip(e["documents"], e.get("documents_sentences") or []):
        if any(k in keys for k, _ in sents):
            out.append(did(text if isinstance(text, str) else str(text)))
    return out


def rel_text(e):
    keys = set(e.get("all_relevant_sentence_keys") or [])
    return " ".join(s for doc in (e.get("documents_sentences") or []) for k, s in doc if k in keys)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/emanual_loo")
    ap.add_argument("--unanswerable_frac", type=float, default=0.5)
    ap.add_argument("--passage_words", type=int, default=200)
    ap.add_argument("--passage_overlap", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    raw = {}
    for sp, f in (("train", "train"), ("val", "validation"), ("test", "test")):
        raw[sp] = read_jsonl(os.path.join(a.data, "ragbench", f"emanual_{f}.jsonl"))
        print("raw", sp, len(raw[sp]))
    # candidate questions: complete answers with identifiable gold documents; one per distinct question text
    cands, seen = {}, {}
    for sp, rows in raw.items():
        for e in rows:
            g = gold_docs(e)
            if not g or (e.get("completeness_score") or 0) < 0.8 or not (e.get("response") or "").strip():
                continue
            q = e["question"].strip()
            if q in seen and seen[q] != sp:
                cands.pop(q, None)  # same question in two splits -> drop (no leakage)
                seen[q] = "dup"
                continue
            if q in seen:
                continue
            seen[q] = sp
            cands[q] = (sp, e, g)
    cands = {q: v for q, v in cands.items() if seen[q] != "dup"}
    # group questions that share a gold document
    parent = {q: q for q in cands}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_doc = defaultdict(list)
    for q, (_, _, g) in cands.items():
        for d in g:
            by_doc[d].append(q)
    for qs in by_doc.values():
        for q in qs[1:]:
            parent[find(q)] = find(qs[0])
    comps = defaultdict(list)
    for q in cands:
        comps[find(q)].append(q)
    # assign answerability per component (a component = questions sharing gold documents; never split across labels).
    # Greedy toward the requested fraction in EVERY split; components spanning splits are placed first.
    n_sp = defaultdict(int)
    for q, (sp, _, _) in cands.items():
        n_sp[sp] += 1
    cur = defaultdict(int)
    order = list(comps.values())
    rng.shuffle(order)
    order.sort(key=lambda c: -len({cands[q][0] for q in c}))
    unans = set()
    for c in order:
        cnt = defaultdict(int)
        for q in c:
            cnt[cands[q][0]] += 1
        deficit = sum(a.unanswerable_frac * n_sp[sp] - cur[sp] for sp in cnt)
        if deficit > 0:
            unans.update(c)
            for sp, k in cnt.items():
                cur[sp] += k
    mixed = [c for c in comps.values() if len({cands[q][0] for q in c}) > 1]
    removed = set()
    for q in unans:
        removed.update(cands[q][2])
    # a removed document must not be gold for any answerable question
    answerable_gold = {d for q, (_, _, g) in cands.items() if q not in unans for d in g}
    assert not (removed & answerable_gold), "grouping failed"
    out = {"train": [], "val": [], "test": []}
    for i, (q, (sp, e, g)) in enumerate(sorted(cands.items())):
        u = q in unans
        out[sp].append({"qid": f"emloo_{sp}_{i}", "question": q, "answerable": not u,
                        "reference_answer": "" if u else e["response"].strip(),
                        "gold_evidence_ids": [] if u else g, "removed_gold_ids": g if u else [],
                        "gold_sentences": "" if u else rel_text(e)})
    corpus, done = [], set()
    for rows in raw.values():
        for e in rows:
            for text in e["documents"]:
                text = text if isinstance(text, str) else str(text)
                d = did(text)
                if d in done or d in removed:
                    continue
                done.add(d)
                for j, ch in enumerate(chunk_words(text, a.passage_words, a.passage_overlap)):
                    corpus.append({"pid": f"{d}#{j}", "doc_id": d, "text": ch})
    os.makedirs(a.out, exist_ok=True)
    write_jsonl(os.path.join(a.out, "corpus.jsonl"), corpus)
    for sp, rows in out.items():
        write_jsonl(os.path.join(a.out, f"{sp}.jsonl"), rows)
        print(sp, len(rows), "answerable:", sum(r["answerable"] for r in rows), "unanswerable:", sum(not r["answerable"] for r in rows))
    print("corpus passages:", len(corpus), "| documents removed for unanswerable questions:", len(removed), "| mixed-split groups:", len(mixed))
