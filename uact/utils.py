import json
import os
import random
import re
import string
from collections import Counter

import numpy as np
import yaml

STOP = set(
    "a an the of to in on for and or is are was were be been it this that with after before when what which who "
    "how why do does did can could should would my our your from by at as not no yes if then than into over under "
    "while still please need get gets getting have has had i we you they there here its also any some about out up".split()
)
CODE_RE = re.compile(r"\b[A-Za-z]{1,5}[-_]?\d{2,}[A-Za-z0-9\-]*\b")


def load_config(path, overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in (overrides or {}).items():
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = v
    return cfg


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def normalize_text(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred, gold):
    p, g = normalize_text(pred).split(), normalize_text(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec, rec = n / len(p), n / len(g)
    return 2 * prec * rec / (prec + rec)


def keywords(text):
    """Query 'slots': content words plus error/part codes."""
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_]*", text.lower())
    kws = [t for t in toks if t not in STOP and len(t) > 2]
    kws += [c.lower() for c in CODE_RE.findall(text)]
    return sorted(set(kws))


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 3]


def chunk_words(text, size=200, overlap=40):
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    out, step = [], max(1, size - overlap)
    for i in range(0, len(words), step):
        out.append(" ".join(words[i : i + size]))
        if i + size >= len(words):
            break
    return out
