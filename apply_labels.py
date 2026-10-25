import pandas as pd

f = "data/review/faith1_A.csv"
df = pd.read_csv(f, dtype=str, keep_default_na=False)
idcol = "id"
dec = [c for c in df.columns if c.lower().startswith("decision")][0]

y = {5, 10, 32, 48, 64, 66, 70, 72, 84, 107, 169, 176, 188}
n = {51, 65, 95, 100, 125, 126, 129, 161, 167, 170, 174, 183, 184}
new = {**{i: "y" for i in y}, **{i: "n" for i in n}}

ids = df[idcol].astype(int)
mask = ids.isin(new)
df.loc[mask, dec] = ids[mask].map(new)
df.to_csv(f, index=False)
print("changed rows:", mask.sum(), "of", len(new))