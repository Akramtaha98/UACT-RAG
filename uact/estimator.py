"""Evidence-state estimator: small MLP + temperature scaling."""
import numpy as np

STATES_FIVE = ["sufficient", "partial", "irrelevant", "missing", "contradictory"]
STATES_THREE = ["sufficient", "partial", "insufficient"]
_TO_THREE = {"sufficient": "sufficient", "partial": "partial", "irrelevant": "insufficient",
             "missing": "insufficient", "contradictory": "insufficient"}


def state_names(kind):
    return STATES_FIVE if kind == "five" else STATES_THREE


def map_label(label, kind):
    return label if kind == "five" else _TO_THREE[label]


def bad_indices(kind):
    names = state_names(kind)
    bad = ["irrelevant", "missing", "contradictory"] if kind == "five" else ["insufficient"]
    return [names.index(b) for b in bad]


def p_bad(probs, kind):
    return np.asarray(probs)[:, bad_indices(kind)].sum(1)


def norm_entropy(probs):
    p = np.clip(np.asarray(probs), 1e-12, 1.0)
    return -(p * np.log(p)).sum(1) / np.log(p.shape[1])


class Estimator:
    def __init__(self, kind="five", hidden=128, mask=None):
        self.kind, self.hidden, self.mask = kind, hidden, mask
        self.k = len(state_names(kind))
        self.T = 1.0
        self.net = None
        self.mu = self.sd = None

    def _prep(self, X):
        X = np.asarray(X, dtype="float32")
        if self.mask is not None:
            X = X[:, self.mask]
        return (X - self.mu) / self.sd

    def fit(self, X, y, Xv, yv, Xc=None, yc=None, epochs=200, lr=1e-3, wd=1e-4, device="cpu", seed=0):
        """(Xv, yv) = validation split (early stopping); (Xc, yc) = separate calibration split (temperature scaling)."""
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        Xs = np.asarray(X, dtype="float32")
        if self.mask is not None:
            Xs = Xs[:, self.mask]
        self.mu, self.sd = Xs.mean(0), Xs.std(0) + 1e-6
        Xt = torch.tensor(self._prep(X), device=device)
        yt = torch.tensor(np.asarray(y), device=device)
        Xvt = torch.tensor(self._prep(Xv), device=device)
        yvt = torch.tensor(np.asarray(yv), device=device)
        d = Xt.shape[1]
        self.net = nn.Sequential(nn.Linear(d, self.hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Linear(self.hidden, self.k)).to(device)
        # class-balanced loss: engineering collections are usually imbalanced
        counts = np.bincount(np.asarray(y), minlength=self.k).astype("float32") + 1.0
        w = torch.tensor(counts.sum() / (self.k * counts), device=device)
        lossf = nn.CrossEntropyLoss(weight=w)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=wd)
        best, best_state, patience = 1e9, None, 0
        for ep in range(epochs):
            self.net.train()
            perm = torch.randperm(len(Xt), device=device)
            for i in range(0, len(Xt), 256):
                idx = perm[i : i + 256]
                opt.zero_grad()
                lossf(self.net(Xt[idx]), yt[idx]).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = nn.functional.cross_entropy(self.net(Xvt), yvt).item()
            if vl < best - 1e-4:
                best, patience = vl, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                patience += 1
                if patience >= 20:
                    break
        self.net.load_state_dict(best_state)
        if Xc is not None:
            self._fit_temperature(torch.tensor(self._prep(Xc), device=device), torch.tensor(np.asarray(yc), device=device))
        else:
            self._fit_temperature(Xvt, yvt)
        return self

    def _fit_temperature(self, Xvt, yvt):
        """Temperature scaling on the CALIBRATION split (Guo et al., 2017)."""
        import torch

        with torch.no_grad():
            logits = self.net(Xvt)
        logT = torch.zeros(1, requires_grad=True, device=logits.device)
        opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(logits / logT.exp(), yvt)
            loss.backward()
            return loss

        opt.step(closure)
        self.T = float(logT.exp().item())

    def predict_proba(self, X, calibrated=True):
        import torch

        dev = next(self.net.parameters()).device
        with torch.no_grad():
            logits = self.net(torch.tensor(self._prep(X), device=dev))
            if calibrated:
                logits = logits / self.T
            return torch.softmax(logits, -1).cpu().numpy()

    def save(self, path):
        import torch

        torch.save({"kind": self.kind, "hidden": self.hidden, "mask": self.mask, "T": self.T, "mu": self.mu,
                    "sd": self.sd, "state": self.net.state_dict(), "d": len(self.mu)}, path)

    @staticmethod
    def load(path, device="cpu"):
        import torch
        import torch.nn as nn

        ck = torch.load(path, map_location=device, weights_only=False)
        e = Estimator(ck["kind"], ck["hidden"], ck["mask"])
        e.T, e.mu, e.sd = ck["T"], ck["mu"], ck["sd"]
        e.net = nn.Sequential(nn.Linear(ck["d"], e.hidden), nn.ReLU(), nn.Dropout(0.1),
                              nn.Linear(e.hidden, e.hidden), nn.ReLU(), nn.Linear(e.hidden, e.k)).to(device)
        e.net.load_state_dict(ck["state"])
        e.net.eval()
        return e
