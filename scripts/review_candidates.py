"""Interactive review of contradiction candidates.  python scripts/review_candidates.py [--file PATH]
Keys: y = plausible contradiction (keep)  n = reject  s = skip for now  q = save and quit."""
import argparse, json, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="data/contradiction_candidates.jsonl")
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.file) if l.strip()]
shutil.copy(a.file, a.file + ".bak")
todo = [i for i, r in enumerate(rows) if r["reviewed"] is None]
print(f"{len(todo)} unreviewed of {len(rows)}")
for k, i in enumerate(todo, 1):
    r = rows[i]
    print(f"\n[{k}/{len(todo)}]\nORIGINAL:     {r['original']}\nCONTRADICTS?: {r['contradicting']}")
    while True:
        c = input("y / n / s / q > ").strip().lower()
        if c in ("y", "n", "s", "q"):
            break
    if c == "q":
        break
    if c != "s":
        r["reviewed"] = (c == "y")
    with open(a.file, "w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
with open(a.file, "w") as f:
    for x in rows:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
n = lambda v: sum(x["reviewed"] is v for x in rows)
print(f"accepted {n(True)}, rejected {n(False)}, pending {n(None)}")
