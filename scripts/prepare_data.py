"""Prepare datasets into the unified formats used by the pipeline.

  python scripts/prepare_data.py ragbench --subsets emanual techqa
  python scripts/prepare_data.py techqa --root /path/to/TechQA --out data/techqa
  python scripts/prepare_data.py engineering --questions my_q.jsonl --corpus my_passages.jsonl --out data/engineering

Unified question record: {qid, question, equipment_model?, revision?, answerable, reference_answer?, gold_evidence_ids?}
Corpus record:           {pid, doc_id, text, revision?, equipment_model?}
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uact.utils import chunk_words, read_jsonl, write_jsonl  # noqa: E402


def ragbench(args):
    from datasets import load_dataset

    for sub in args.subsets:
        ds = load_dataset("rungalileo/ragbench", sub)  # verify the dataset id / split names on the Hugging Face card
        for split in ds.keys():
            rows = []
            for i, e in enumerate(ds[split]):
                rows.append({
                    "id": e.get("id", f"{sub}_{split}_{i}"), "subset": sub, "question": e["question"],
                    "documents": e["documents"], "documents_sentences": e.get("documents_sentences"),
                    "response": e.get("response"),
                    "relevance_score": e.get("relevance_score"), "utilization_score": e.get("utilization_score"),
                    "completeness_score": e.get("completeness_score"), "adherence_score": e.get("adherence_score"),
                    "all_relevant_sentence_keys": e.get("all_relevant_sentence_keys"),
                })
            write_jsonl(os.path.join(args.out, "ragbench", f"{sub}_{split}.jsonl"), rows)
            print(f"ragbench {sub}/{split}: {len(rows)} rows")


def _technote_text(rec):
    for k in ("text", "content", "text_without_boilerplate"):
        if isinstance(rec, dict) and rec.get(k):
            return rec[k]
    return ""


def techqa(args):
    """Expects the files distributed with TechQA after registration. File names/fields differ between releases;
    adjust the three marked lines if your copy differs."""
    root = args.root
    out = args.out
    with open(args.technotes) as f:  # e.g. <root>/training_dev_technotes.json  (dict or list of technotes) [ADJUST]
        notes = json.load(f)
    items = notes.items() if isinstance(notes, dict) else [(n.get("id") or n.get("doc_id"), n) for n in notes]
    corpus = []
    for doc_id, rec in items:
        for j, ch in enumerate(chunk_words(_technote_text(rec), args.passage_words, args.passage_overlap)):
            corpus.append({"pid": f"{doc_id}#{j}", "doc_id": doc_id, "text": ch})
    write_jsonl(os.path.join(out, "corpus.jsonl"), corpus)
    print("corpus passages:", len(corpus))

    def conv(path):
        rows = []
        for r in json.load(open(path)):  # list of question records [ADJUST]
            ans = r.get("ANSWERABLE", "Y") == "Y" and r.get("DOCUMENT", "-") not in ("-", None)
            rows.append({
                "qid": r["QUESTION_ID"], "question": (r.get("QUESTION_TITLE", "") + " " + r.get("QUESTION_TEXT", "")).strip(),
                "answerable": bool(ans), "reference_answer": r.get("ANSWER", "") if ans else "",  # [ADJUST]
                "gold_evidence_ids": [r["DOCUMENT"]] if ans else [],
            })
        return rows

    train = conv(os.path.join(root, args.train))
    dev = conv(os.path.join(root, args.dev))
    if args.test and os.path.exists(os.path.join(root, args.test)):
        test = conv(os.path.join(root, args.test))
        val = dev
    else:  # the held-out evaluation set is not always released: split dev 50/50 and DOCUMENT this deviation in the paper
        random.Random(0).shuffle(dev)
        val, test = dev[: len(dev) // 2], dev[len(dev) // 2 :]
        print("WARNING: no test set found; dev was split 50/50 into val/test. Report this in the paper.")
    for name, rows in (("train", train), ("val", val), ("test", test)):
        write_jsonl(os.path.join(out, f"{name}.jsonl"), rows)
        print(name, len(rows), "answerable:", sum(r["answerable"] for r in rows))


def engineering(args):
    """Validate the engineering collection schema and create document/equipment-family-level splits."""
    qs = read_jsonl(args.questions)
    need = {"question_id", "question", "answerable", "evidence_state"}
    for q in qs:
        miss = need - set(q)
        if miss:
            raise SystemExit(f"{q.get('question_id')}: missing fields {miss}")
    fam = args.family_field
    fams = sorted({q.get(fam) or "unknown" for q in qs})
    random.Random(0).shuffle(fams)
    n = len(fams)
    split_of = {f: ("train" if i < 0.6 * n else "val" if i < 0.75 * n else "cal" if i < 0.85 * n else "test") for i, f in enumerate(fams)}
    by = {"train": [], "val": [], "cal": [], "test": []}
    for q in qs:
        by[split_of[q.get(fam) or "unknown"]].append({
            "qid": q["question_id"], "question": q["question"], "equipment_model": q.get("equipment_model"),
            "revision": q.get("document_revision") if q.get("revision_specified") else None,
            "answerable": q["answerable"], "reference_answer": q.get("reference_answer", ""),
            "gold_evidence_ids": q.get("gold_evidence_ids", []), "evidence_state": q["evidence_state"],
        })
    for k, rows in by.items():
        write_jsonl(os.path.join(args.out, f"{k}.jsonl"), rows)
        print(k, len(rows))
    os.makedirs(args.out, exist_ok=True)
    corpus = read_jsonl(args.corpus)
    write_jsonl(os.path.join(args.out, "corpus.jsonl"), corpus)
    print("splits are by", fam, "- revisions of one family never straddle splits. corpus:", len(corpus))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("ragbench"); a.add_argument("--subsets", nargs="+", default=["emanual", "techqa"]); a.add_argument("--out", default="data")
    b = sp.add_parser("techqa"); b.add_argument("--root", required=True); b.add_argument("--out", default="data/techqa")
    b.add_argument("--technotes", required=True); b.add_argument("--train", default="training_Q_A.json")
    b.add_argument("--dev", default="dev_Q_A.json"); b.add_argument("--test", default="eval_Q_A.json")
    b.add_argument("--passage_words", type=int, default=200); b.add_argument("--passage_overlap", type=int, default=40)
    c = sp.add_parser("engineering"); c.add_argument("--questions", required=True); c.add_argument("--corpus", required=True)
    c.add_argument("--out", default="data/engineering"); c.add_argument("--family_field", default="equipment_model")
    args = ap.parse_args()
    {"ragbench": ragbench, "techqa": techqa, "engineering": engineering}[args.cmd](args)
