import argparse, csv, io, json, os, random, shutil, sys


def nli_check(rows, labels, name, min_auc=0.65):
    """Sanity check: human y/n labels should correlate with an NLI contradiction score. Catches misaligned or careless sheets."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    ids = sorted(labels)
    y = np.array([labels[i] for i in ids], dtype=int)
    if len(ids) < 30 or y.sum() in (0, len(y)):
        return True
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    from uact.models import NLI
    nli = NLI("microsoft/deberta-large-mnli", device="cuda" if torch.cuda.is_available() else "cpu")
    p = nli.probs([rows[i]["original"] for i in ids], [rows[i]["contradicting"] for i in ids], batch_size=32)[:, 0]
    auc = roc_auc_score(y, p)
    print(f"sanity check {name}: AUROC of your labels vs NLI contradiction score = {auc:.3f} (n={len(ids)})")
    return auc >= min_auc

CAND = "data/contradiction_candidates.jsonl"
OUT = "data/review"
YES, NO = {"y", "yes", "1", "true", "t"}, {"n", "no", "0", "false", "f"}


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def save(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_sheet(path, rows, idxs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "original", "rewritten_to_contradict", "decision (y/n)", "notes"])
        for i in idxs:
            r = rows[i]
            cur = {True: "y", False: "n", None: ""}[r["reviewed"]]
            w.writerow([i, r["original"], r["contradicting"], cur, ""])


def _rows(path):
    raw = open(path, "rb").read()
    if raw[:2] == b"PK":  # real Excel workbook (whatever the file name says)
        import openpyxl
        ws = openpyxl.load_workbook(io.BytesIO(raw), data_only=True).active
        it = ws.iter_rows(values_only=True)
        head = [str(h or "").strip() for h in next(it)]
        return [dict(zip(head, ["" if v is None else str(v) for v in r])) for r in it]
    text = None
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeError:
            continue
    first = text.splitlines()[0] if text else ""
    delim = ";" if first.count(";") > first.count(",") else ("\t" if first.count("\t") > first.count(",") else ",")
    return list(csv.DictReader(io.StringIO(text, newline=""), delimiter=delim))


def read_sheet(path):
    out = {}
    for row in _rows(path):
        d = str(row.get("decision (y/n)") or row.get("decision") or "").strip().lower()
        try:
            i = int(float(row["id"]))
        except (KeyError, ValueError, TypeError):
            continue
        if d in YES:
            out[i] = True
        elif d in NO:
            out[i] = False
    return out


def kappa(a, b):
    ids = sorted(set(a) & set(b))
    n = len(ids)
    if n == 0:
        return None, 0
    po = sum(a[i] == b[i] for i in ids) / n
    pa, pb = sum(a[i] for i in ids) / n, sum(b[i] for i in ids) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (1.0 if pe == 1 else (po - pe) / (1 - pe)), n


def export(a):
    rows = load(a.cand)
    idxs = [i for i, r in enumerate(rows) if r["reviewed"] is None] if a.pending else list(range(len(rows)))
    write_sheet(os.path.join(OUT, "review_A.csv"), rows, idxs)
    sample = sorted(random.Random(0).sample(idxs, min(a.sample, len(idxs))))
    write_sheet(os.path.join(OUT, "review_B.csv"), rows, sample)
    print(f"wrote {OUT}/review_A.csv ({len(idxs)} items) and review_B.csv ({len(sample)} items)")


def do_import(a):
    rows = load(a.cand)
    shutil.copy(a.cand, a.cand + ".bak")
    A = read_sheet(a.a)
    B = read_sheet(a.b) if a.b else {}
    print(f"read {len(A)} decisions from A" + (f", {len(B)} from B" if a.b else ""))
    if not a.no_check:
        ok = nli_check(rows, A, "A") and (not B or nli_check(rows, B, "B"))
        if not ok:
            print("ABORTED, nothing saved: the labels do not match the content of the rows (AUROC < 0.65). "
                  "Typical causes: the y/n column was sorted or filtered separately, rows were hidden, or labels were typed into another copy. "
                  "Re-do the sheet in its original row order, or pass --no_check to override.")
            return
    disagree = []
    for i, r in enumerate(rows):
        if i in A and i in B and A[i] != B[i]:
            r["reviewed"] = None
            disagree.append(i)
        elif i in A:
            r["reviewed"] = A[i]
        elif i in B:
            r["reviewed"] = B[i]
    save(a.cand, rows)
    if disagree:
        write_sheet(os.path.join(OUT, "disagreements.csv"), rows, disagree)
    k, n = kappa(A, B)
    print(f"accepted {sum(r['reviewed'] is True for r in rows)}, rejected {sum(r['reviewed'] is False for r in rows)}, "
          f"pending {sum(r['reviewed'] is None for r in rows)}")
    if k is not None:
        print(f"double-labelled: {n}, disagreements: {len(disagree)}, Cohen's kappa = {k:.3f}")
    else:
        print("no overlap between sheets A and B, so no agreement statistic")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("export"); e.add_argument("--sample", type=int, default=60); e.add_argument("--cand", default=CAND); e.add_argument("--pending", action="store_true")
    i = sp.add_parser("import"); i.add_argument("--a", required=True); i.add_argument("--b"); i.add_argument("--cand", default=CAND); i.add_argument("--no_check", action="store_true")
    a = ap.parse_args()
    export(a) if a.cmd == "export" else do_import(a)
