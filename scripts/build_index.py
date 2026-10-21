"""Build BM25 + dense (FAISS) indexes for a corpus.   python scripts/build_index.py --data data/techqa"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from uact.retrieval import build_index  # noqa: E402
from uact.utils import load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--data", required=True)
a = ap.parse_args()
cfg = load_config(a.config)
build_index(cfg, os.path.join(a.data, "corpus.jsonl"), os.path.join(a.data, "index"))
