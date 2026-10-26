"""Reader for annotation sheets (.csv or .xlsx with sheet 'Items'): returns {claim id: True/False} for rows with a y/n decision."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from faithfulness_audit import read as read_csv  # noqa: E402


def read_any(p):
    if not p.lower().endswith(".xlsx"):
        return read_csv(p)
    import openpyxl
    ws = openpyxl.load_workbook(p, data_only=True)["Items"]
    rows = list(ws.iter_rows(values_only=True))
    h = [str(x or "").strip().lower() for x in rows[0]]
    di = next(k for k, x in enumerate(h) if x.startswith("decision"))
    out = {}
    for r in rows[1:]:
        d = str(r[di] or "").strip().lower()
        if d in ("y", "yes", "1", "true"):
            out[int(r[h.index("id")])] = True
        elif d in ("n", "no", "0", "false"):
            out[int(r[h.index("id")])] = False
    return out
