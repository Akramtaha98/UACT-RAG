import sys
import pandas as pd

a = pd.read_csv("data/review/faith1_A.csv", dtype=str, keep_default_na=False)
c = pd.read_csv("data/review/faith1_C.csv", dtype=str, keep_default_na=False)
assert list(a["id"]) == list(c["id"]), "row order differs"
dec = [x for x in a.columns if x.lower().startswith("decision")][0]
pas = a.columns[2]
ak = a[dec].str.strip().str.lower()
fa = c[dec].str.strip().str.lower()

if sys.argv[1] == "prep":
    print("yes-rate Akram", round((ak == "y").mean(), 2), "Fanan", round((fa == "y").mean(), 2))
    print("both y", ((ak == "y") & (fa == "y")).sum(), "both n", ((ak == "n") & (fa == "n")).sum())
    print("Akram y/Fanan n", ((ak == "y") & (fa == "n")).sum(), "Akram n/Fanan y", ((ak == "n") & (fa == "y")).sum())
    d = a[ak != fa][["id", "claim", pas]].copy()
    d["akram"] = ak[ak != fa]
    d["fanan"] = fa[ak != fa]
    d["final"] = ""
    d.to_csv("data/review/faith1_disagree200.csv", index=False)
    print("disagreements written:", len(d))

if sys.argv[1] == "merge":
    d = pd.read_csv("data/review/faith1_disagree200.csv", dtype=str, keep_default_na=False)
    fin = dict(zip(d["id"], d["final"].str.strip().str.lower()))
    assert all(v in ("y", "n") for v in fin.values()), "fill every final cell with y or n"
    out = a.copy()
    out[dec] = [fin.get(i, v) for i, v in zip(a["id"], ak)]
    out.to_csv("data/review/faith1_final.csv", index=False)
    print("final labels written; supported rate:", round((out[dec] == "y").mean(), 3))