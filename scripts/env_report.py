"""Print the software, hardware and model-revision details a reviewer may ask for (paste the output to the authors).

  python scripts/env_report.py > logs/env_report.txt
"""
import glob
import importlib
import os
import platform
import subprocess
import sys

print("python", sys.version.split()[0], platform.platform())
for m in ("torch", "vllm", "transformers", "sentence_transformers", "FlagEmbedding", "faiss", "bm25s", "sklearn", "numpy", "pandas", "scipy", "yaml"):
    try:
        v = getattr(importlib.import_module(m), "__version__", "installed")
    except Exception:
        v = "not installed"
    print(f"{m:24s}{v}")
try:
    print(subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]).decode().strip())
except Exception as e:
    print("nvidia-smi unavailable:", e)
try:
    print("git commit", subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip())
except Exception:
    pass
try:
    import yaml

    cfg = yaml.safe_load(open("configs/default.yaml"))
    print("seed", cfg["seed"], "| models", cfg["models"])
except Exception as e:
    print("config unreadable:", e)
roots = [os.environ.get("HF_HOME", ""), os.path.expanduser("~/.cache/huggingface")]
for r in roots:
    for d in glob.glob(os.path.join(r, "hub", "models--*", "snapshots", "*")):
        print("model revision", d.split("models--")[1].replace("--", "/").split(os.sep)[0], os.path.basename(d))
