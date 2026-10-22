"""Figure 1: UACT-RAG architecture (vector PDF + 300 dpi PNG).   python scripts/draw_fig1.py --out figures"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

_av = {f.name for f in fm.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in ("Arial", "Liberation Sans", "DejaVu Sans") if f in _av), "sans-serif")

BLUE, ORANGE, GREEN, GREY, RED = "#DCEBF7", "#FCE8CF", "#DDF1E7", "#EEEEEE", "#F8DCD3"


def box(ax, x, y, w, h, title, body="", fc=GREY, fs=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", fc=fc, ec="#333333", lw=0.9, clip_on=False))
    if body:
        ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top", fontsize=fs, fontweight="bold")
        ax.text(x + w / 2, y + h - 0.105, body, ha="center", va="top", fontsize=fs - 0.9, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs, fontweight="bold")


def arrow(ax, p, q, text="", rad=0.0, tx=0.0, ty=0.0, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=9, lw=1.0, color="#222222", connectionstyle=f"arc3,rad={rad}", linestyle=ls))
    if text:
        ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2 + ty, text, fontsize=6.4, ha="center", va="center", style="italic",
                bbox=dict(fc="white", ec="none", pad=0.6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1); ax.axis("off")
    # row 1: retrieval -> estimator
    box(ax, 0.01, 0.74, 0.13, 0.24, "Question q", "technical-support\nquery", fc=GREY)
    box(ax, 0.19, 0.74, 0.28, 0.24, "1  Hybrid retrieval", "BM25 + BGE-M3 dense (FAISS)\nreciprocal rank fusion\nbge-reranker-v2-m3\nevidence set D$_t$ (top 5)", fc=BLUE)
    box(ax, 0.52, 0.74, 0.47, 0.24, "2  Evidence-state estimator", "features z = [r, c, x, a]  (relevance, coverage,\ncontradiction, agreement)\nMLP + temperature scaling\np over 5 states,  uncertainty U,  P$_{bad}$", fc=ORANGE)
    arrow(ax, (0.14, 0.86), (0.19, 0.86))
    arrow(ax, (0.47, 0.86), (0.52, 0.86))
    # row 2: controller and query rewrite
    box(ax, 0.60, 0.42, 0.39, 0.22, "3  Cost-sensitive controller", "predicts utility V(a) for ANSWER,\nRETRIEVE, ABSTAIN; retrieval-cost\npenalty \u03bc, abstention margin \u03b4", fc=ORANGE)
    box(ax, 0.19, 0.42, 0.28, 0.22, "Query rewrite", "from unresolved gaps;\nsearch unseen passages", fc=BLUE)
    arrow(ax, (0.80, 0.74), (0.80, 0.64))
    arrow(ax, (0.60, 0.53), (0.47, 0.53), "RETRIEVE\n(\u2264 B rounds)", ty=0.075)
    arrow(ax, (0.33, 0.64), (0.33, 0.74), "new evidence\nmerged into D$_t$", tx=0.10)
    # row 3: answer path
    box(ax, 0.72, 0.04, 0.27, 0.22, "4  Generator", "Qwen3-8B answers\nfrom the evidence", fc=GREEN)
    box(ax, 0.38, 0.04, 0.29, 0.22, "5  Claim-level verifier", "DeBERTa-large-MNLI checks each\nclaim; regenerate once, then\nanswer or abstain", fc=GREEN)
    box(ax, 0.01, 0.04, 0.32, 0.22, "Output", "answer with cited passages,\nor ABSTAIN stating what\nis missing", fc=RED)
    arrow(ax, (0.86, 0.42), (0.86, 0.26), "ANSWER", tx=0.07)
    arrow(ax, (0.72, 0.15), (0.67, 0.15))
    arrow(ax, (0.38, 0.15), (0.33, 0.15))
    arrow(ax, (0.62, 0.42), (0.27, 0.265), "ABSTAIN", tx=0.0, ty=0.02)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"fig1_architecture.{ext}"), dpi=300, bbox_inches="tight")
    print("wrote", os.path.join(a.out, "fig1_architecture.pdf"))


if __name__ == "__main__":
    main()
