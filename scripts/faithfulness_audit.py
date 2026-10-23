"""Independent (human) check of claim support, to replace the circular NLI faithfulness number.

  # 1. export a blind claim-level sheet (needs the NLI model once, to store its verdict in the key; GPU or CPU)
  python scripts/faithfulness_audit.py export --eval_dir runs/main/eval/techqa_test --systems UACT B9 B6 B2 A3 --per_system 40 --name faith1
  # 2. annotator(s) fill 'decision (y/n)' in data/review/faith1_A.csv: is the claim fully supported by the passages shown (y) or not (n)?
  # 3. report
  python scripts/faithfulness_audit.py import --sheet data/review/faith1_A.csv --key data/review/faith1_key.json [--sheet2 data/review/faith1_B.csv]

Sheet A has every claim; sheet B is a random subset for a second annotator. The sheet hides the system and the NLI verdict.
Reported: human-supported rate per system (Wilson 95% CI), and agreement of the NLI verdict with the human label.
"""
import argparse, csv, json, math, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
OUT = "data/review"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def export(a):
    import yaml
    from uact.models import NLI
    from uact.utils import split_sentences
    from uact.verifier import CITE, Verifier
    cfg = yaml.safe_load(open(a.config))
    ver = Verifier(NLI(cfg["models"]["nli"], a.device), cfg["eval"]["support_threshold"])
    rng = random.Random(a.seed)
    items = []
    for s in a.systems:
        pool = []
        for line in open(os.path.join(a.eval_dir, f"{s}.jsonl")):
            r = json.loads(line)
            if r.get("answered") and r.get("answer") and r.get("evidence"):
                for c in split_sentences(r["answer"]):
                    if len(c.split()) >= 4:
                        pool.append((s, r, c))
        rng.shuffle(pool)
        items += pool[: a.per_system]
    rng.shuffle(items)
    os.makedirs(OUT, exist_ok=True)
    rows, key = [], {}
    for i, (s, r, c) in enumerate(items):
        ev = r["evidence"]
        cited = [int(m) - 1 for m in CITE.findall(c) if 0 < int(m) <= len(ev)]
        show = [ev[j] for j in cited] if cited else ev
        txt = "\n\n".join(f"[P{ev.index(p) + 1}] " + " ".join(p["text"].split()[:a.max_words]) for p in show)
        clean = CITE.sub("", c).replace("DOCUMENTED", "").replace("INFERENCE", "").strip(" :-")
        v = ver.verify(c, ev)
        rows.append([i, clean, txt, "", ""])
        key[i] = {"system": s, "qid": r["qid"], "nli_supported": bool(v["all_supported"]), "nli_score": (v.get("claim_scores") or [None])[0]}
    hdr = ["id", "claim", "passages (cited, or all evidence if none cited)", "decision (y/n)", "notes"]
    second = sorted(rng.sample(range(len(rows)), min(a.second_n, len(rows))))
    for suf, sel in (("A", range(len(rows))), ("B", second)):
        with open(os.path.join(OUT, f"{a.name}_{suf}.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(hdr)
            for i in sel:
                w.writerow(rows[i])
    json.dump(key, open(os.path.join(OUT, f"{a.name}_key.json"), "w"), indent=1)
    print(f"wrote {len(rows)} claims to {OUT}/{a.name}_A.csv (+ {len(second)} in _B.csv) and {a.name}_key.json; keep the key from annotators")


def read(path):
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            d = (row.get("decision (y/n)") or "").strip().lower()
            if d in ("y", "yes", "1", "true"): out[int(row["id"])] = True
            elif d in ("n", "no", "0", "false"): out[int(row["id"])] = False
    return out


def kappa(a, b):
    ids = sorted(set(a) & set(b)); n = len(ids)
    if not n: return None, 0
    po = sum(a[i] == b[i] for i in ids) / n; pa, pb = sum(a[i] for i in ids) / n, sum(b[i] for i in ids) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (1.0 if pe == 1 else (po - pe) / (1 - pe)), n


def do_import(a):
    key = {int(k): v for k, v in json.load(open(a.key)).items()}
    H = read(a.sheet)
    if a.sheet2:
        k, n = kappa(H, read(a.sheet2)); print(f"second annotator: n={n}, Cohen kappa={k:.2f}" if k is not None else "no overlap")
    ids = sorted(set(H) & set(key))
    print(f"{len(ids)} labelled claims")
    for s in sorted({key[i]['system'] for i in ids}):
        v = [i for i in ids if key[i]['system'] == s]; k = sum(H[i] for i in v); lo, hi = wilson(k, len(v))
        nli = [i for i in v if key[i]['nli_supported']]; kn = sum(H[i] for i in nli)
        agree = sum(H[i] == key[i]['nli_supported'] for i in v) / len(v)
        print(f"{s:8s} n={len(v):3d} human-supported {k / len(v):.2f} [{lo:.2f}, {hi:.2f}] | NLI-supported claims: {len(nli)}, of which human-supported {kn} | NLI-human agreement {agree:.2f}")
    k, n = kappa({i: H[i] for i in ids}, {i: key[i]['nli_supported'] for i in ids})
    print(f"pooled NLI vs human kappa={k:.2f} (n={n})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("export")
    e.add_argument("--config", default="configs/default.yaml"); e.add_argument("--eval_dir", default="runs/main/eval/techqa_test")
    e.add_argument("--systems", nargs="+", default=["UACT", "B9", "B6", "B2", "A3"]); e.add_argument("--per_system", type=int, default=40)
    e.add_argument("--second_n", type=int, default=60); e.add_argument("--max_words", type=int, default=150)
    e.add_argument("--name", default="faith1"); e.add_argument("--seed", type=int, default=0); e.add_argument("--device", default="cuda")
    i = sp.add_parser("import"); i.add_argument("--sheet", required=True); i.add_argument("--key", required=True); i.add_argument("--sheet2")
    a = ap.parse_args(); (export if a.cmd == "export" else do_import)(a)
