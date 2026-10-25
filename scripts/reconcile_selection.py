"""Print the values the reported runs ACTUALLY used, to reconcile the manuscript with code and configuration.
  python scripts/reconcile_selection.py --variant full --eval_dir runs/main/eval/techqa_test
Reads <work>/<variant>/controller_grid.json (written by train_controller.py), configs/default.yaml and the test records."""
import argparse, collections, json, os
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--work", default="runs/main"); ap.add_argument("--variant", default="full")
ap.add_argument("--eval_dir", default="runs/main/eval/techqa_test"); ap.add_argument("--config", default="configs/default.yaml")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
print("== config ==")
print("lambda_grid", cfg["control"]["lambda_grid"], "mu_grid", cfg["control"]["mu_grid"], "support_floor", cfg["control"].get("support_floor"),
      "n_ctx", cfg["retrieval"]["n_ctx"], "support_threshold", cfg["eval"]["support_threshold"], "f1_thr", cfg["eval"]["f1_correct_threshold"])
p = os.path.join(a.work, a.variant, "controller_grid.json")
if os.path.exists(p):
    g = json.load(open(p))
    print("\n== controller selection ==")
    print("risk_target", g.get("risk_target"), "min_coverage", g.get("min_coverage"))
    print("selected:", g["best"])
    feas = [r for r in g["grid"] if r["stats"]["coverage"] >= g["min_coverage"] and r["stats"]["selective_risk"] == r["stats"]["selective_risk"]
            and r["stats"]["selective_risk"] <= g["risk_target"]]
    print("grid points:", len(g["grid"]), "| feasible under the stated target:", len(feas),
          "-> fallback (lowest risk) was used" if not feas else "-> feasible rule was used")
    print("lambdas in grid:", sorted({r["lambda"] for r in g["grid"]}), "mus:", sorted({r["mu"] for r in g["grid"]}))
else:
    print("missing", p)
print("\n== test actions (UACT) ==")
c = collections.Counter(); n = ans = 0
for line in open(os.path.join(a.eval_dir, "UACT.jsonl")):
    r = json.loads(line); n += 1; ans += bool(r.get("answered"))
    for x in r.get("actions", []):
        c[x if x.startswith("ABSTAIN_") else x.split(":")[0]] += 1
print("questions", n, "answered", ans, dict(c))
