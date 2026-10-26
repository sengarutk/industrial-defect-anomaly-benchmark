from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats


def bootstrap_ci(values: np.ndarray, n_bootstraps: int = 2000, ci: float = 0.95, seed: int = 0):
    """Percentile bootstrap CI of the mean. Returns (mean, low, high)."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    if values.size == 1:
        v = float(values[0])
        return v, v, v
    rng = np.random.RandomState(seed)
    means = values[rng.randint(0, values.size, size=(n_bootstraps, values.size))].mean(axis=1)
    a = (1.0 - ci) / 2.0
    return float(values.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def validate_bootstrap_ci_coverage(n_simulations: int = 500, sample_size: int = 100, true_mean: float = 5.0,
                                   true_std: float = 2.0, ci: float = 0.95, n_bootstraps: int = 500,
                                   seed: int = 0) -> float:
    """Monte Carlo coverage of bootstrap_ci on Gaussian samples with known mean."""
    rng = np.random.RandomState(seed)
    hits = 0
    for i in range(n_simulations):
        data = rng.normal(true_mean, true_std, sample_size)
        _, lo, hi = bootstrap_ci(data, n_bootstraps=n_bootstraps, ci=ci, seed=seed + i + 1)
        hits += lo <= true_mean <= hi
    return hits / n_simulations


def hierarchical_bootstrap_ci(records: Sequence[Dict[str, Any]], metric_fn: Callable[[List[Dict[str, Any]]], float],
                              item_keys: Sequence[str], n_resamples: int = 2000, ci: float = 0.95,
                              seed: int = 0, strata_key: Optional[str] = None) -> Dict[str, float]:
    """Two-stage bootstrap: resample runs (seeds) with replacement, then items within each run.

    `item_keys` names the per-item arrays in each record that are resampled
    jointly (same indices). If `strata_key` is given (e.g. "labels"), items are
    resampled within each stratum so class counts stay fixed and per-class
    rates are always defined. Other entries are passed through unchanged.
    """
    if len(records) == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "std_error": float("nan")}
    rng = np.random.RandomState(seed)
    estimate = float(metric_fn(list(records)))
    boots = np.empty(n_resamples)
    for b in range(n_resamples):
        sample = []
        for r in rng.randint(0, len(records), size=len(records)):
            rec = dict(records[r])
            if strata_key is None:
                n = len(rec[item_keys[0]])
                idx = rng.randint(0, n, size=n)
            else:
                strata = np.asarray(rec[strata_key])
                idx = np.concatenate([rng.choice(np.flatnonzero(strata == v), size=int(np.sum(strata == v)))
                                      for v in np.unique(strata)])
            for k in item_keys:
                rec[k] = np.asarray(rec[k])[idx]
            sample.append(rec)
        boots[b] = metric_fn(sample)
    a = (1.0 - ci) / 2.0
    return {"estimate": estimate, "ci_low": float(np.nanquantile(boots, a)),
            "ci_high": float(np.nanquantile(boots, 1 - a)), "std_error": float(np.nanstd(boots))}


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Two-sided Wilcoxon signed-rank test on paired values (one pair per independent unit).

    Zero differences are discarded; the exact null distribution is used unless |differences| tie.
    Also reports the Hodges-Lehmann estimate and the matched-pairs rank-biserial correlation.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = a - b
    n = int(d.size)
    walsh = [(d[i] + d[j]) / 2.0 for i in range(n) for j in range(i, n)]
    nz = d[d != 0]
    out = {"n_pairs": n, "n_nonzero": int(nz.size), "mean_diff": float(np.mean(d)) if n else float("nan"),
           "hodges_lehmann": float(np.median(walsh)) if n else float("nan")}
    if nz.size == 0:
        out.update(statistic=float("nan"), p_value=1.0, rank_biserial=0.0, min_attainable_p=1.0)
        return out
    # Zeros are dropped (Wilcoxon's original rule) and the exact null distribution is used on the
    # remaining differences unless their absolute values contain ties.
    exact = len(np.unique(np.abs(nz))) == nz.size
    res = stats.wilcoxon(nz, alternative="two-sided", method="exact" if exact else "approx")
    ranks = stats.rankdata(np.abs(nz))
    w_plus, w_minus = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    out.update(statistic=float(res.statistic), p_value=float(res.pvalue),
               rank_biserial=float((w_plus - w_minus) / (w_plus + w_minus)),
               min_attainable_p=float(2.0 / 2 ** nz.size))
    return out


def holm_bonferroni(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
    """Holm step-down adjustment; returns adjusted p-values and rejection decisions."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, Dict[str, Any]] = {}
    running = 0.0
    still_rejecting = True
    for rank, (key, p) in enumerate(items):
        adj = min(1.0, max(running, p * (m - rank)))
        running = adj
        still_rejecting = still_rejecting and p <= alpha / (m - rank)
        out[key] = {"raw_p": float(p), "adjusted_p": float(adj), "rank": rank + 1, "reject": bool(still_rejecting)}
    return out
