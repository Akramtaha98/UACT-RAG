"""Independent (human) check of claim support, to replace the circular NLI faithfulness number.

  # 1. export a blind claim-level sheet (needs the NLI model once, to store its verdict in the key; GPU or CPU)
  python scripts/faithfulness_audit.py export --eval_dir runs/main/eval/techqa_test --systems UACT B9 B6 B2 A3 --per_system 40 --name faith1
  # 2. annotator(s) fill 'decision (y/n)' in data/review/faith1_A.csv: is the claim fully supported by the passages shown (y) or not (n)?
  # 3. report
  python scripts/faithfulness_audit.py import --sheet data/review/faith1_A.csv --key data/review/faith1_key.json [--sheet2 data/review/faith1_B.csv]

Sheet A has every claim; sheet B is a random subset for a second annotator. The sheet hides the system and the NLI verdict.
Reported: human-supported rate per system (Wilson 95% CI), and agreement of the NLI verdict with the human label.

Matched-evidence check (the sheet shows passages cut at --max_words words, the verifier saw full passages in 220-word windows):
  # 4. re-run NLI on EXACTLY the claim and the passage text shown on the sheet (needs the NLI model once)
  python scripts/faithfulness_audit.py renli --sheet data/review/faith1_A.csv --out data/review/faith1_nli_trunc.json
  # 5. report with answer-clustered bootstrap intervals and both NLI verdicts (original and matched evidence)
  python scripts/faithfulness_audit.py import --sheet data/review/faith1_final.csv --key data/review/faith1_key.json \
         --nli_trunc data/review/faith1_nli_trunc.json --boot 2000
Claims from the same answer are not independent, so the intervals resample ANSWERS (system + question), not claims.
"""
import argparse, csv, json, math, os, random, re, sys

import numpy as np

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


def renli(a):
    """Re-score every claim with the NLI model against exactly the passage text shown on the annotation sheet."""
    import yaml
    from uact.models import NLI
    from uact.verifier import _windows
    cfg = yaml.safe_load(open(a.config))
    nli, thr = NLI(cfg["models"]["nli"], a.device), cfg["eval"]["support_threshold"]
    out = {}
    with open(a.sheet, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    pcol = [c for c in rows[0] if c.startswith("passages")][0]
    for row in rows:
        shown = [re.sub(r"^\[P\d+\]\s*", "", x.strip()) for x in re.split(r"\n\s*\n", row[pcol]) if x.strip()]
        prem = [w for t in shown for w in _windows(t)]
        probs = nli.probs(prem, [row["claim"]] * len(prem)) if prem else []
        best = max([float(p[2]) for p in probs], default=0.0)
        out[int(row["id"])] = {"nli_trunc_supported": bool(best >= thr), "nli_trunc_score": best}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {len(out)} matched-evidence NLI verdicts to {a.out}; supported {sum(v['nli_trunc_supported'] for v in out.values())}")


def kappa_vec(x, y):
    x, y = np.asarray(x, bool), np.asarray(y, bool)
    po = (x == y).mean(); pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return float("nan") if pe == 1 else float((po - pe) / (1 - pe))


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
    print(f"pooled NLI (original evidence) vs human kappa={k:.2f} (n={n})")
    # ---- answer-clustered bootstrap: claims from one answer (system + question) are resampled together
    T = json.load(open(a.nli_trunc)) if a.nli_trunc else None
    T = {int(i): v["nli_trunc_supported"] for i, v in T.items()} if T else None
    rng = np.random.default_rng(0)
    cl = {i: f"{key[i]['system']}|{key[i]['qid']}" for i in ids}
    systems = sorted({key[i]['system'] for i in ids})
    print(f"\nAnswer-clustered bootstrap ({a.boot} resamples); clusters = answers (system + question)")
    for s_ in systems + ["ALL"]:
        v = [i for i in ids if s_ == "ALL" or key[i]['system'] == s_]
        groups = {}
        for i in v: groups.setdefault(cl[i], []).append(i)
        keys_ = list(groups)
        hs, kn, kt = [], [], []
        for _ in range(a.boot):
            pick = [groups[keys_[j]] for j in rng.integers(0, len(keys_), len(keys_))]
            ii = [i for g in pick for i in g]
            h = [H[i] for i in ii]
            hs.append(np.mean(h))
            if s_ == "ALL":
                kn.append(kappa_vec(h, [key[i]['nli_supported'] for i in ii]))
                if T is not None and all(i in T for i in ii):
                    kt.append(kappa_vec(h, [T[i] for i in ii]))
        pt = np.mean([H[i] for i in v])
        line = f"{s_:8s} answers={len(keys_):3d} claims={len(v):3d} human-supported {pt:.2f} cluster-95% [{np.percentile(hs, 2.5):.2f}, {np.percentile(hs, 97.5):.2f}]"
        if s_ == "ALL":
            line += f" | NLI(original) kappa {kappa_vec([H[i] for i in v], [key[i]['nli_supported'] for i in v]):.2f} [{np.nanpercentile(kn, 2.5):.2f}, {np.nanpercentile(kn, 97.5):.2f}]"
            if T is not None and all(i in T for i in v):
                line += f" | NLI(matched evidence) kappa {kappa_vec([H[i] for i in v], [T[i] for i in v]):.2f} [{np.nanpercentile(kt, 2.5):.2f}, {np.nanpercentile(kt, 97.5):.2f}]"
        print(line)
    if T is not None:
        print("\nMatched-evidence NLI (verifier run on exactly the passage text the annotators saw)")
        for s_ in systems:
            v = [i for i in ids if key[i]['system'] == s_ and i in T]
            nl = [i for i in v if T[i]]
            print(f"{s_:8s} NLI-supported {len(nl):2d} of {len(v)}, of which human-supported {sum(H[i] for i in nl):2d} | agreement {np.mean([H[i] == T[i] for i in v]):.2f}"
                  f" | original-evidence NLI supported {sum(key[i]['nli_supported'] for i in v):2d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("export")
    e.add_argument("--config", default="configs/default.yaml"); e.add_argument("--eval_dir", default="runs/main/eval/techqa_test")
    e.add_argument("--systems", nargs="+", default=["UACT", "B9", "B6", "B2", "A3"]); e.add_argument("--per_system", type=int, default=40)
    e.add_argument("--second_n", type=int, default=60); e.add_argument("--max_words", type=int, default=150)
    e.add_argument("--name", default="faith1"); e.add_argument("--seed", type=int, default=0); e.add_argument("--device", default="cuda")
    i = sp.add_parser("import"); i.add_argument("--sheet", required=True); i.add_argument("--key", required=True); i.add_argument("--sheet2")
    i.add_argument("--nli_trunc", default=None); i.add_argument("--boot", type=int, default=2000)
    r = sp.add_parser("renli"); r.add_argument("--sheet", required=True); r.add_argument("--out", default="data/review/faith1_nli_trunc.json")
    r.add_argument("--config", default="configs/default.yaml"); r.add_argument("--device", default="cuda")
    a = ap.parse_args(); {"export": export, "import": do_import, "renli": renli}[a.cmd](a)
