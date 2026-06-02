"""Per-(metric × view) error statistics: RMSE, MAE, Bland-Altman, ICC(2,1), Pearson r, CMC."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedStats:
    n: int
    bias: float                # mean signed error (test - ref)
    rmse: float
    mae: float
    sd_diff: float             # SD of differences
    loa_lower: float           # bias - 1.96 * sd_diff
    loa_upper: float           # bias + 1.96 * sd_diff
    pearson_r: float
    icc_2_1: float
    cmc: float                 # coefficient of multiple correlation (waveform)


def _drop_nans(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _icc_2_1(test: np.ndarray, ref: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single rater.

    Treats each paired observation as one subject rated by two methods.
    Returns ICC in [-?, 1]; near 0 if no agreement, 1 if perfect.
    """
    if test.size < 2:
        return float("nan")
    n = test.size
    Y = np.stack([test, ref], axis=1)  # (n, 2)
    grand_mean = Y.mean()
    row_means = Y.mean(axis=1)
    col_means = Y.mean(axis=0)
    ss_total = ((Y - grand_mean) ** 2).sum()
    ss_rows = 2 * ((row_means - grand_mean) ** 2).sum()
    ss_cols = n * ((col_means - grand_mean) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / 1
    ms_err = ss_err / max(1, (n - 1))
    denom = ms_rows + ms_cols + (2 - 1) * ms_err + (2 / n) * (ms_cols - ms_err)
    if denom <= 0:
        return float("nan")
    return float((ms_rows - ms_err) / denom)


def _cmc(test: np.ndarray, ref: np.ndarray) -> float:
    """Coefficient of multiple correlation (waveform similarity).

    Standard in biomechanics (Pose2Sim, OpenCap papers). Ranges [0,1].
    """
    if test.size < 3:
        return float("nan")
    Y = np.stack([test, ref], axis=1)
    Yp_mean = Y.mean(axis=1)         # per-frame mean across the two waveforms
    Y_mean = Y.mean()
    num = ((Y - Yp_mean[:, None]) ** 2).sum()
    den = ((Y - Y_mean) ** 2).sum()
    if den <= 0:
        return float("nan")
    val = 1.0 - num / den
    if val < 0:
        return 0.0
    return float(np.sqrt(val))


def compute_paired_stats(test: np.ndarray, ref: np.ndarray) -> PairedStats:
    """Compute paired-sample agreement statistics. Inputs must be same length."""
    test, ref = _drop_nans(np.asarray(test, float), np.asarray(ref, float))
    if test.size == 0:
        nan = float("nan")
        return PairedStats(0, nan, nan, nan, nan, nan, nan, nan, nan, nan)
    diff = test - ref
    bias = float(diff.mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    mae = float(np.abs(diff).mean())
    sd_diff = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    loa_lower = bias - 1.96 * sd_diff
    loa_upper = bias + 1.96 * sd_diff
    if test.size > 1 and test.std() > 0 and ref.std() > 0:
        r = float(np.corrcoef(test, ref)[0, 1])
    else:
        r = float("nan")
    return PairedStats(
        n=int(test.size),
        bias=bias,
        rmse=rmse,
        mae=mae,
        sd_diff=sd_diff,
        loa_lower=loa_lower,
        loa_upper=loa_upper,
        pearson_r=r,
        icc_2_1=_icc_2_1(test, ref),
        cmc=_cmc(test, ref),
    )


def quality_tier(rmse: float) -> str:
    """Standard biomech quality tier per Theia3D systematic review / OpenCap conventions."""
    if not np.isfinite(rmse):
        return "n/a"
    if rmse < 5:
        return "excellent"
    if rmse < 10:
        return "good"
    if rmse < 15:
        return "moderate"
    return "poor"
