import numpy as np
import pytest
from scipy.ndimage import label

from src.metrics.pixel_metrics import compute_aupro, compute_pixel_auroc, pro_curve


def brute_force_aupro(masks, maps, max_fpr=0.3):
    """Independent reference: explicit loop over every distinct threshold (slow, small inputs only)."""
    gt = masks > 0.5
    comps = []
    for i in range(len(gt)):
        lab, n = label(gt[i])
        comps += [(i, lab == k) for k in range(1, n + 1)]
    normal = ~gt
    fprs, pros = [0.0], [0.0]
    for t in np.unique(maps)[::-1]:
        pred = maps >= t
        fprs.append((pred & normal).sum() / normal.sum())
        pros.append(np.mean([(pred[i] & c).sum() / c.sum() for i, c in comps]))
    fprs, pros = np.array(fprs), np.array(pros)
    keep = fprs <= max_fpr
    x, y = list(fprs[keep]), list(pros[keep])
    j = np.argmax(fprs > max_fpr)
    if fprs[j] > max_fpr and x[-1] < max_fpr:
        y.append(pros[j - 1] + (pros[j] - pros[j - 1]) * (max_fpr - fprs[j - 1]) / (fprs[j] - fprs[j - 1]))
        x.append(max_fpr)
    return float(np.trapz(y, x) / max_fpr)


def random_case(seed):
    rng = np.random.RandomState(seed)
    masks = np.zeros((6, 24, 24), dtype=np.uint8)
    masks[0, 2:8, 2:8] = 1
    masks[0, 15:20, 15:22] = 1
    masks[2, 5:12, 9:14] = 1
    masks[4, 0:3, 0:24] = 1
    maps = rng.rand(6, 24, 24) + 0.6 * masks
    maps = np.round(maps, 2)  # force ties
    return masks, maps


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_aupro_matches_brute_force_reference(seed):
    masks, maps = random_case(seed)
    assert compute_aupro(masks, maps) == pytest.approx(brute_force_aupro(masks, maps), abs=1e-9)


def test_aupro_extremes():
    masks = np.zeros((3, 32, 32), dtype=np.uint8)
    masks[0, 5:15, 5:15] = 1
    masks[1, 20:30, 1:4] = 1
    assert compute_aupro(masks, masks * 5.0 + 0.0) == pytest.approx(1.0, abs=1e-6)
    assert compute_aupro(masks, 1.0 - masks) < 0.01


def test_pro_curve_monotone_and_bounded():
    masks, maps = random_case(5)
    fpr, pro = pro_curve(masks, maps)
    assert np.all(np.diff(fpr) >= -1e-12) and np.all(np.diff(pro) >= -1e-12)
    assert fpr[0] == 0 and pro[0] == 0 and fpr[-1] == pytest.approx(1) and pro[-1] == pytest.approx(1)


def test_no_defects_returns_nan():
    with pytest.warns(UserWarning):
        assert np.isnan(compute_aupro(np.zeros((2, 8, 8)), np.random.rand(2, 8, 8)))
    with pytest.warns(UserWarning):
        assert np.isnan(compute_pixel_auroc(np.zeros((2, 8, 8)), np.random.rand(2, 8, 8)))
