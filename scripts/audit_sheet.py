import argparse, csv, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from review_sheet import read_sheet

OUT = "data/review"


def export(a):
    qs = {q["qid"]: q for q in map(json.loads, open(os.path.join(a.data, f"{a.split}.jsonl")))}
    skip = set()
    for p in a.exclude or []:
        skip |= {(v["system"], v["qid"]) for v in json.load(open(p)).values()}
    pool = []
    for s in a.systems:
        for line in open(os.path.join(a.eval_dir, f"{s}.jsonl")):
            r = json.loads(line)
            if r["answered"] and qs[r["qid"]]["answerable"] and (s, r["qid"]) not in skip:
                pool.append((s, r))
    random.Random(a.seed).shuffle(pool)
    pool = pool[: a.n]
    os.makedirs(OUT, exist_ok=True)
    key = {}
    with open(os.path.join(OUT, f"{a.name}.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "reference_answer", "candidate_answer", "decision (y/n)", "notes"])
        for i, (s, r) in enumerate(pool):
            q = qs[r["qid"]]
            w.writerow([i, " ".join(q["question"].split()), " ".join(q["reference_answer"].split()), " ".join(r["answer"].split()), "", ""])
            key[i] = {"system": s, "qid": r["qid"], "judge": r.get("judge_correct", r.get("correct")),
                      "gold": (r.get("gold_recall") or 0) > 0, "f1": r["f1"]}
    json.dump(key, open(os.path.join(OUT, f"{a.name}_key.json"), "w"), indent=1)
    print(f"wrote {OUT}/{a.name}.csv ({len(pool)} items) and {a.name}_key.json (do NOT show the key to the annotator)")


def report(name, ids, H, pred):
    p = [bool(pred(i)) for i in ids]; h = [H[i] for i in ids]; n = len(ids)
    tp = sum(x and y for x, y in zip(p, h)); tn = sum((not x) and (not y) for x, y in zip(p, h))
    fp = sum(x and not y for x, y in zip(p, h)); fn = sum((not x) and y for x, y in zip(p, h))
    po = (tp + tn) / n; pp, ph = (tp + fp) / n, (tp + fn) / n
    pe = pp * ph + (1 - pp) * (1 - ph); k = (po - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"{name:32s} agree={po:.2f} kappa={k:.2f} | too lenient={fp}  too strict={fn}  | rule-correct {pp:.2f} vs human-correct {ph:.2f}")


def do_import(a):
    key = {int(k): v for k, v in json.load(open(a.key)).items()}
    H = read_sheet(a.sheet)
    ids = sorted(set(H) & set(key))
    print(f"n={len(ids)} labelled items; human says correct: {sum(H[i] for i in ids) / len(ids):.2f}")
    report("LLM judge", [i for i in ids if key[i].get("judge") is not None], H, lambda i: key[i]["judge"])
    report("gold doc in evidence", ids, H, lambda i: key[i]["gold"])
    report(f"gold doc AND F1 >= {a.f1}", ids, H, lambda i: key[i]["gold"] and key[i]["f1"] >= a.f1)
    for s in sorted({key[i]["system"] for i in ids}):
        v = [i for i in ids if key[i]["system"] == s]
        report(f"  {s} (n={len(v)}): gold AND F1>={a.f1}", v, H, lambda i: key[i]["gold"] and key[i]["f1"] >= a.f1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("export")
    e.add_argument("--eval_dir", required=True); e.add_argument("--systems", nargs="+", required=True)
    e.add_argument("--n", type=int, default=60); e.add_argument("--seed", type=int, default=0)
    e.add_argument("--data", default="data/techqa"); e.add_argument("--split", default="val")
    e.add_argument("--name", default="judge_audit"); e.add_argument("--exclude", nargs="*")
    i = sp.add_parser("import")
    i.add_argument("--sheet", required=True); i.add_argument("--key", default=os.path.join(OUT, "judge_audit_key.json"))
    i.add_argument("--f1", type=float, default=0.15)
    a = ap.parse_args()
    export(a) if a.cmd == "export" else do_import(a)
