import warnings
from typing import Tuple, Dict, Any, Union
import numpy as np
from scipy import stats


def bootstrap_ci(
    values: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Performs non-parametric bootstrap resampling on metric arrays, returning (mean, lower_bound, upper_bound).
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]

    if len(arr) == 0:
        return 0.0, 0.0, 0.0

    if len(arr) == 1:
        val = float(arr[0])
        return val, val, val

    mean_val = float(np.mean(arr))
    rng = np.random.default_rng(seed)

    boot_means = np.zeros(n_bootstraps, dtype=float)
    n = len(arr)
    for b in range(n_bootstraps):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[b] = np.mean(sample)

    alpha = (1.0 - ci) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    lower_bound = float(np.percentile(boot_means, lower_pct))
    upper_bound = float(np.percentile(boot_means, upper_pct))

    return mean_val, lower_bound, upper_bound


def compute_wilcoxon_significance(
    method_a_scores: np.ndarray,
    method_b_scores: np.ndarray
) -> Dict[str, float]:
    """
    Executes two-sided Wilcoxon signed-rank test across matched category/seed splits using scipy.stats.wilcoxon,
    returning test statistic W and p-value.
    """
    a = np.asarray(method_a_scores, dtype=float)
    b = np.asarray(method_b_scores, dtype=float)

    if len(a) != len(b):
        raise ValueError(f"Arrays must have matching lengths, got {len(a)} and {len(b)}")

    if len(a) == 0:
        return {"statistic": 0.0, "p_value": 1.0}

    diff = a - b
    # Check if all differences are zero
    if np.all(diff == 0.0) or np.count_nonzero(diff) < 2:
        return {"statistic": 0.0, "p_value": 1.0}

    try:
        res = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        stat = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception as e:
        warnings.warn(f"Wilcoxon test encountered exception: {e}. Defaulting to p=1.0.")
        stat = 0.0
        p_val = 1.0

    return {
        "statistic": stat,
        "p_value": p_val
    }
