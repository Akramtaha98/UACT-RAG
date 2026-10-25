"""Small neural helpers: cross-encoder reranker, NLI classifier, embedder."""
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Reranker:
    def __init__(self, name, device="cuda"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(name, max_length=512, device=device)

    def score(self, query, texts, batch_size=64):
        if not texts:
            return np.zeros(0)
        s = np.asarray(self.model.predict([(query, t) for t in texts], batch_size=batch_size), dtype=float)
        if s.min() < 0.0 or s.max() > 1.0:  # raw logits depending on library version
            s = _sigmoid(s)
        return s

    def score_pairs(self, pairs, batch_size=64):
        if not pairs:
            return np.zeros(0)
        s = np.asarray(self.model.predict(list(pairs), batch_size=batch_size), dtype=float)
        if s.min() < 0.0 or s.max() > 1.0:
            s = _sigmoid(s)
        return s


class NLI:
    """Returns probabilities in the fixed order [contradiction, neutral, entailment]."""

    def __init__(self, name, device="cuda"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        self.device = device
        lab = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        inv = {v: k for k, v in lab.items()}
        self.order = [inv["contradiction"], inv["neutral"], inv["entailment"]]

    def probs(self, premises, hypotheses, batch_size=32, max_length=512):
        out = []
        torch = self.torch
        for i in range(0, len(premises), batch_size):
            enc = self.tok(
                premises[i : i + batch_size],
                hypotheses[i : i + batch_size],
                truncation=True,  # longest_first: trims whichever side is longer (only_first can fail on long pairs)
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                p = torch.softmax(self.model(**enc).logits.float(), -1).cpu().numpy()
            out.append(p[:, self.order])
        return np.concatenate(out) if out else np.zeros((0, 3))


class Embedder:
    def __init__(self, name, device="cuda"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, device=device)
        self.model.half()
        self.model.max_seq_length = 512

    def encode(self, texts, batch_size=256):
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=len(texts) > 1000
        ).astype("float32")
