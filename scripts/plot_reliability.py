"""Figure 3: reliability diagrams of the evidence-state estimator on the state-test split.
  python scripts/plot_reliability.py --states_dir data/states data/techqa/states --variant runs/main/full --out figures
(a) top-label confidence, before vs after temperature scaling; (b) P(insufficient evidence), before vs after.
Read-only: no refitting. Also writes figures/fig3_reliability.json with ECE values and bin counts."""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

_av = {f.name for f in fm.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in ("Arial", "Liberation Sans", "DejaVu Sans") if f in _av), "sans-serif")
C_RAW, C_CAL = "#C0504D", "#1F5FA8"


def bins(conf, hit, n=15):
    edges = np.linspace(0, 1, n + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            out.append((float(conf[m].mean()), float(hit[m].mean()), int(m.sum())))
    return out


def ece(b):
    tot = sum(c for _, _, c in b)
    return sum(c * abs(a - f) for f, a, c in b) / tot if tot else float("nan")


def panel(ax, series, title, xlabel, ylabel):
    ax.plot([0, 1], [0, 1], ls="--", c="#777777", lw=0.9, label="perfect calibration")
    res = {}
    for name, (conf, hit), col, mk in series:
        b = bins(conf, hit)
        e = ece(b)
        res[name] = {"ece": round(e, 4), "bins": [[round(x, 4), round(y, 4), n] for x, y, n in b]}
        ax.plot([x for x, _, _ in b], [y for _, y, _ in b], marker=mk, ms=4, lw=1.2, c=col, label=f"{name} (ECE {e:.3f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=9); ax.set_xlabel(xlabel, fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(alpha=0.25, lw=0.5); ax.set_aspect("equal")
    ax.legend(fontsize=6.5, loc="upper left", frameon=False)
    return res


def draw(P_raw, P_cal, y, bad_idx, out):
    bad = np.isin(y, bad_idx).astype(float)
    res = {}
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.6))
    ser_a = [(n, (P.max(1), (P.argmax(1) == y).astype(float)), c, m) for n, P, c, m in
             (("before scaling", P_raw, C_RAW, "o"), ("after scaling", P_cal, C_CAL, "s"))]
    res["top_label"] = panel(axs[0], ser_a, "(a) Five-state prediction", "Top-label confidence", "Observed accuracy")
    ser_b = [(n, (P[:, bad_idx].sum(1), bad), c, m) for n, P, c, m in
             (("before scaling", P_raw, C_RAW, "o"), ("after scaling", P_cal, C_CAL, "s"))]
    res["p_insufficient"] = panel(axs[1], ser_b, "(b) Insufficient-evidence probability", "Predicted P(insufficient)", "Observed fraction insufficient")
    fig.tight_layout()
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig3_reliability.{ext}"), dpi=300, bbox_inches="tight")
    json.dump(res, open(os.path.join(out, "fig3_reliability.json"), "w"), indent=2)
    print(json.dumps({k: {n: v["ece"] for n, v in d.items()} for k, d in res.items()}, indent=2))


def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from uact.estimator import Estimator, bad_indices, map_label, state_names
    from uact.utils import read_jsonl
    ap = argparse.ArgumentParser()
    ap.add_argument("--states_dir", nargs="+", required=True)
    ap.add_argument("--variant", default="runs/main/full")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    rows = []
    for d in a.states_dir:
        for suf in ("", "_retrieved"):
            p = os.path.join(d, f"test{suf}.jsonl")
            if os.path.exists(p):
                rows += read_jsonl(p)
    X = np.load(os.path.join(a.variant, "feat_test.npy"))
    assert len(rows) == len(X), (len(rows), len(X))
    est = Estimator.load(os.path.join(a.variant, "estimator.pt"))
    names = state_names(est.kind)
    y = np.array([names.index(map_label(r["state"], est.kind)) for r in rows])
    draw(est.predict_proba(X, calibrated=False), est.predict_proba(X, calibrated=True), y, np.array(bad_indices(est.kind)), a.out)


if __name__ == "__main__":
    main()
