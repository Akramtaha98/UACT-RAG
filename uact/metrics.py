"""Evaluation metrics: calibration, selective prediction, paired statistics."""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def nll(probs, y):
    probs = np.clip(np.asarray(probs), 1e-12, 1.0)
    return float(-np.mean(np.log(probs[np.arange(len(y)), y])))


def brier_binary(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def ece(conf, correct, n_bins=15):
    """Expected calibration error with equal-width bins (Guo et al., 2017)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total, n = 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability_bins(conf, correct, n_bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            rows.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return rows


def estimator_report(probs, y, bad_idx, n_bins=15):
    """probs: (n,K); y: int labels; bad_idx: indices of states that mean 'insufficient evidence'."""
    probs, y = np.asarray(probs), np.asarray(y)
    pred = probs.argmax(1)
    conf = probs.max(1)
    pbad = probs[:, bad_idx].sum(1)
    ybad = np.isin(y, bad_idx).astype(int)
    rep = {
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float((pred == y).mean()),
        "nll": nll(probs, y),
        "ece_state": ece(conf, pred == y, n_bins),
        "brier_insufficient": brier_binary(pbad, ybad),
        "ece_insufficient": ece(np.maximum(pbad, 1 - pbad), (pbad > 0.5) == ybad, n_bins),
    }
    if 0 < ybad.sum() < len(ybad):
        rep["auroc_insufficient"] = float(roc_auc_score(ybad, pbad))
        rep["auprc_insufficient"] = float(average_precision_score(ybad, pbad))
    return rep


def risk_coverage(conf, error):
    """Sort by confidence descending; return coverage[], risk[] (risk undefined -> nan never occurs after first item)."""
    conf, error = np.asarray(conf, float), np.asarray(error, float)
    order = np.argsort(-conf, kind="stable")
    err = error[order]
    n = len(err)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(err) / np.arange(1, n + 1)
    return cov, risk


def aurc(conf, error):
    cov, risk = risk_coverage(conf, error)
    return float(np.mean(risk))


def e_aurc(conf, error):
    """AURC minus the AURC of the optimal-in-hindsight ranking (Geifman et al., 2019)."""
    error = np.asarray(error, float)
    return aurc(conf, error) - aurc(-error, error)


def selective_stats(answered, error):
    """answered: bool array; error: 1 if the answer is wrong (only meaningful where answered)."""
    answered, error = np.asarray(answered, bool), np.asarray(error, float)
    cov = float(answered.mean())
    risk = float(error[answered].mean()) if answered.any() else float("nan")  # undefined when nothing answered
    wrong_rate = float((error * answered).sum() / len(answered))
    return {"coverage": cov, "selective_risk": risk, "wrong_answer_rate": wrong_rate}


def bootstrap_ci(values, stat=np.mean, n=2000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    stats = np.array([stat(values[i]) for i in idx])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def paired_bootstrap_diff(a, b, n=2000, alpha=0.05, seed=0):
    """CI for mean(a - b) over paired per-question values."""
    d = np.asarray(a, float) - np.asarray(b, float)
    lo, hi = bootstrap_ci(d, np.mean, n, alpha, seed)
    return float(d.mean()), lo, hi


def mcnemar_exact(correct_a, correct_b):
    """Exact two-sided McNemar test on paired binary correctness."""
    from scipy.stats import binomtest

    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b01, b10 = int((~a & b).sum()), int((a & ~b).sum())
    n = b01 + b10
    p = 1.0 if n == 0 else float(binomtest(min(b01, b10), n, 0.5).pvalue)
    return {"a_only": b10, "b_only": b01, "p_value": p}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def percentile(x, q):
    return float(np.percentile(np.asarray(x, float), q))
