"""Figure 2: risk-coverage curves (panel a, automatic proxy) and human-audited operating points (panel b).

  python scripts/plot_risk_coverage.py --eval_dir runs/main/eval/techqa_test
Writes fig2_risk_coverage.pdf / .png (300 dpi) into --eval_dir (or --out). Curves come from the FORCED runs (every
question answered, ranked by the system's confidence); B2 has no abstention so its plain run is used. Where a forced run
leaves some questions unanswered, only answered ones are ranked and the curve ends at the coverage reached.
Panel (b) uses the human-audit estimates of Sections 5.11 / 6.1 (override with --human_json).
"""
import argparse
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from matplotlib import font_manager as _fm  # noqa: E402

_avail = {f.name for f in _fm.fontManager.ttflist}
_font = next((f for f in ("Arial", "Liberation Sans", "DejaVu Sans") if f in _avail), "sans-serif")
plt.rcParams.update({"font.family": _font, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
COL = {"B2": "#0072B2", "B6": "#999999", "B8": "#E69F00", "B9": "#009E73", "UACT": "#D55E00"}
NAME = {"B2": "B2 hybrid RAG", "B6": "B6 fixed + threshold", "B8": "B8 confidence controller", "B9": "B9 sufficiency autorater", "UACT": "UACT-RAG"}
# (coverage, risk, lo, hi): intervals are the simulation-based 95% intervals of Table 7 (audit sampling + question sampling), not Wilson intervals
HUMAN = {"B2": (1.0, 0.732, 0.65, 0.81), "B8": (0.535, 0.668, 0.57, 0.77), "B9": (0.068, 0.571, 0.38, 0.76), "UACT": (0.065, 0.450, 0.25, 0.65)}
MARK = {"UACT": "*", "B9": "s", "B8": "D", "B6": "^", "B2": "o"}


def load(p):
    return [json.loads(x) for x in open(p) if x.strip()]


def curve(recs, n_total):
    rr = [r for r in recs if r.get("answered", True) and r.get("conf") is not None and math.isfinite(float(r["conf"]))]
    rr.sort(key=lambda r: -float(r["conf"]))
    err = np.array([float(r.get("error", 1)) for r in rr])
    k = np.arange(1, len(rr) + 1)
    return k / n_total, np.cumsum(err) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--systems", nargs="+", default=["B2", "B6", "B8", "B9", "UACT"])
    ap.add_argument("--human_json", default=None, help='{"B2": [coverage, risk, lo, hi], ...}')
    a = ap.parse_args()
    out = a.out or a.eval_dir
    human = {k: tuple(v) for k, v in json.load(open(a.human_json)).items()} if a.human_json else HUMAN
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.25, 1]})
    for s in a.systems:
        p = os.path.join(a.eval_dir, f"{s}_forced.jsonl")
        if not os.path.exists(p):
            p = os.path.join(a.eval_dir, f"{s}.jsonl")
        if not os.path.exists(p):
            continue
        recs = load(p)
        cov, risk = curve(recs, len(recs))
        ax[0].plot(cov, risk, color=COL.get(s), lw=1.4, label=NAME.get(s, s))
        q = os.path.join(a.eval_dir, f"{s}.jsonl")  # operating point of the real (abstaining) run
        if os.path.exists(q):
            r2 = load(q)
            ans = [r for r in r2 if r["answered"]]
            if ans:
                ax[0].scatter([len(ans) / len(r2)], [np.mean([r.get("error", 1) for r in ans])], marker=MARK.get(s, "o"), s=42 if s != "UACT" else 90,
                              color=COL.get(s), edgecolor="black", linewidth=0.6, zorder=5)
    ax[0].set_xlabel("Coverage (fraction of 310 test questions answered)")
    ax[0].set_ylabel("Selective risk (automatic proxy)")
    ax[0].set_xlim(0, 1.02); ax[0].set_ylim(0, 1.0)
    ax[0].grid(alpha=0.25, lw=0.5)
    ax[0].legend(fontsize=6.5, frameon=False, loc="lower right")
    ax[0].text(0.0, 1.03, "(a)", transform=ax[0].transAxes, fontweight="bold")
    for s, (c, r, lo, hi) in human.items():
        ax[1].errorbar([c + {"B9": -0.015, "UACT": 0.015}.get(s, 0.0)], [r], yerr=[[r - lo], [hi - r]], fmt=MARK.get(s, "o"), color=COL.get(s), ecolor=COL.get(s), capsize=2.5,
                       markersize=7 if s != "UACT" else 11, markeredgecolor="black", markeredgewidth=0.6, label=NAME.get(s, s))
    ax[1].set_xlabel("Coverage (fraction answered)")
    ax[1].set_ylabel("Selective risk (human audit, simulated 95% interval)")
    ax[1].set_xlim(-0.03, 1.06); ax[1].set_ylim(0, 1.0)
    ax[1].grid(alpha=0.25, lw=0.5)
    ax[1].text(0.0, 1.03, "(b)", transform=ax[1].transAxes, fontweight="bold")
    plt.tight_layout()
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig2_risk_coverage.{ext}"), dpi=300)
    print("wrote", os.path.join(out, "fig2_risk_coverage.pdf"), "and .png")


if __name__ == "__main__":
    main()
