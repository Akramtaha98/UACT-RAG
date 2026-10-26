"""Run systems on the (frozen) test split and write per-question records + summaries.

  python scripts/run_eval.py --data data/techqa --systems B0 B1 B2 B3 B4 B6 B7 B8 B9 UACT A1 A2 A3 A4 A5 A6 A7
  python scripts/run_eval.py ... --forced            # forced-answer runs used for risk-coverage curves / AURC
  python scripts/run_eval.py ... --batch_size 1      # latency runs (true per-query latency)
  python scripts/run_eval.py ... --split val         # use val while developing; test is for the final run only

Work-dir layout expected (created by run_all.sh):
  runs/main/full/{estimator.pt, controller_best.pkl, controller_nocost.pkl, controller_threshold.pkl, controller_fixedbudget.pkl}
  runs/main/A1/{estimator.pt, controller_best.pkl}      (relevance-only features)
  runs/main/A7/{estimator.pt, controller_best.pkl}      (three-state estimator)
  runs/main/{conf_model.pkl, baseline_thresholds.json}
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uact.controller import Controller  # noqa: E402
from uact.estimator import Estimator  # noqa: E402
from uact.loading import load_ctx  # noqa: E402
from uact.report import summarize  # noqa: E402
from uact.systems import (UACT, AdaptiveStyle, BudgetMatched, CalibConfController, CragStyle, FixedAbstain,  # noqa: E402
                          HybridRAG, NoRAG, StandardRAG, SufficiencyAbstain, score_records)
from uact.utils import load_config, read_jsonl, set_seed, write_jsonl  # noqa: E402

ABL = {  # system -> (variant dir, controller file, use_verifier, allow_abstain)
    "UACT": ("full", "controller_best.pkl", True, True),
    "A1": ("A1", "controller_best.pkl", True, True),
    "A2": ("full", "controller_fixedbudget.pkl", True, True),
    "A3": ("full", "controller_best.pkl", False, True),
    "A4": ("full", "controller_best.pkl", True, False),
    "A5": ("full", "controller_threshold.pkl", True, True),
    "A6": ("full", "controller_nocost.pkl", True, True),
    "A7": ("A7", "controller_best.pkl", True, True),
    "UACTG": ("full_gated", "controller_best.pkl", True, True),  # controller selected with the verifier gate in the rollouts (post hoc)
}


def gpu_mem():
    try:
        import torch

        peak = torch.cuda.max_memory_allocated() / 2**30
    except Exception:
        peak = float("nan")
    try:
        used = float(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode().split()[0]) / 1024
    except Exception:
        used = float("nan")
    return {"torch_peak_gb_this_process": peak, "nvidia_smi_used_gb": used}


def build(name, ctx, wd, thr, forced):
    if name in ABL:
        d, cfile, ver, ab = ABL[name]
        ctx.estimator = Estimator.load(os.path.join(wd, d, "estimator.pt"))
        ctx.controller = Controller.load(os.path.join(wd, d, cfile))
        return UACT(ctx, name=name, use_verifier=ver, allow_abstain=ab and not forced)
    inf = -1e9 if forced else None
    return {
        "B0": lambda: NoRAG(ctx), "B1": lambda: StandardRAG(ctx), "B2": lambda: HybridRAG(ctx),
        "B3": lambda: AdaptiveStyle(ctx), "B4": lambda: CragStyle(ctx), "B7": lambda: BudgetMatched(ctx),
        "B6": lambda: FixedAbstain(ctx, inf if forced else thr["B6_tau"]),
        "B9": lambda: SufficiencyAbstain(ctx, inf if forced else thr["B9_tau"]),
        "B8": lambda: CalibConfController(ctx, thr["B8_tau_hi"], inf if forced else thr["B8_tau_lo"]),
    }[name]()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--forced", action="store_true")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--max_q", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. control.max_rounds=2")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    ov = {}
    for kv in a.set:
        k, v = kv.split("=")
        try:
            ov[k] = json.loads(v)
        except ValueError:
            ov[k] = v
    cfg = load_config(a.config, ov)
    set_seed(cfg["seed"])
    wd = cfg["paths"]["work_dir"]
    bs = a.batch_size or cfg["eval"]["batch_size"]
    thr_path = os.path.join(wd, "baseline_thresholds.json")
    thr = json.load(open(thr_path)) if os.path.exists(thr_path) else {}
    ctx = load_ctx(cfg, os.path.join(a.data, "index"), os.path.join(a.data, "corpus.jsonl"),
                   conf_model=os.path.join(wd, "conf_model.pkl"))
    qs = read_jsonl(os.path.join(a.data, f"{a.split}.jsonl"))[: a.max_q]
    outdir = os.path.join(wd, "eval", os.path.basename(a.data.rstrip("/")) + f"_{a.split}{a.tag}")
    os.makedirs(outdir, exist_ok=True)
    for name in a.systems:
        sysobj = build(name, ctx, wd, thr, a.forced)
        sysobj.run(qs[: min(2, len(qs))])  # warm-up (kernel compilation, caches); results discarded
        recs = []
        for i in range(0, len(qs), bs):
            chunk = qs[i : i + bs]
            t0 = time.perf_counter()
            r = sysobj.run(chunk)
            dt = (time.perf_counter() - t0) / len(chunk)  # per-query latency; exact only when batch_size == 1
            for x in r:
                x["latency"] = dt
            recs += r
        score_records(ctx, qs, recs)
        suffix = "_forced" if a.forced else ""
        write_jsonl(os.path.join(outdir, f"{name}{suffix}.jsonl"), recs)
        s = summarize(recs, ctx.cost)
        s.update({"system": name, "forced": a.forced, "batch_size": bs, "split": a.split, "max_rounds": ctx.B, **gpu_mem()})
        json.dump(s, open(os.path.join(outdir, f"summary_{name}{suffix}.json"), "w"), indent=2)
        print(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items() if k in
                     ("accuracy_all", "coverage", "selective_risk", "faithfulness", "retrieval_calls_mean", "latency_mean")})
