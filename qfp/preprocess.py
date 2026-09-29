"""Cross-sectional preprocessing: universe filter, winsorization, standardization,
and neutralization. All operations act on one date at a time, so no information
leaks across dates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Panel


def tradable_universe(p: Panel, min_history: int = 60, point_in_time: bool = True) -> pd.DataFrame:
    """Boolean mask of stocks eligible to be ranked on each date.

    A stock is eligible at date t if it traded on t, has at least
    ``min_history`` days of prices (excludes new listings, whose early days are
    dominated by IPO effects in A-shares), and, when ``point_in_time`` and
    membership data are available, had already joined the index by t.
    """
    traded = p.volume.fillna(0) > 0
    history = p.close.notna().cumsum() >= min_history
    mask = traded & history
    if point_in_time and p.membership is not None:
        mask &= p.membership.reindex_like(mask).fillna(False).astype(bool)
    return mask


def entry_blocked(p: Panel) -> pd.DataFrame:
    """Mask (indexed by signal date t) of stocks that cannot be bought at open t+1.

    Blocked when suspended on t+1 or opening at/above the daily limit-up price.
    Limits: 10% main board; 20% for ChiNext (300xxx) after 2020-08-24 and for
    STAR (688xxx). ST stocks (5%) are rare in CSI 300 and not modeled.
    """
    limit = pd.DataFrame(0.10, index=p.dates, columns=p.codes)
    chinext = [c for c in p.codes if c.startswith("30")]
    star = [c for c in p.codes if c.startswith("68")]
    limit.loc[limit.index >= "2020-08-24", chinext] = 0.20
    limit.loc[:, star] = 0.20
    open_gap = p.open / p.close.shift(1) - 1.0
    limit_up_open = open_gap >= limit - 0.002
    suspended = p.volume.fillna(0) <= 0
    return (limit_up_open | suspended).shift(-1, fill_value=True).astype(bool)


def winsorize_mad(x: pd.DataFrame, k: float = 5.0) -> pd.DataFrame:
    """Clip each row to median +/- k * 1.4826 * MAD."""
    med = x.median(axis=1)
    mad = x.sub(med, axis=0).abs().median(axis=1) * 1.4826
    return x.clip(lower=med - k * mad, upper=med + k * mad, axis=0)


def zscore(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)


def neutralize(x: pd.DataFrame, exposures: list[pd.DataFrame]) -> pd.DataFrame:
    """Per-date OLS residual of x on the given exposures (plus intercept)."""
    out = pd.DataFrame(np.nan, index=x.index, columns=x.columns)
    xv = x.to_numpy()
    ev = [e.reindex_like(x).to_numpy() for e in exposures]
    for i in range(len(x.index)):
        y = xv[i]
        cols = [np.ones_like(y)] + [e[i] for e in ev]
        design = np.column_stack(cols)
        ok = np.isfinite(y) & np.isfinite(design).all(axis=1)
        if ok.sum() <= design.shape[1] + 5:
            continue
        beta, *_ = np.linalg.lstsq(design[ok], y[ok], rcond=None)
        resid = np.full_like(y, np.nan)
        resid[ok] = y[ok] - design[ok] @ beta
        out.iloc[i] = resid
    return out


def standardize(
    raw: pd.DataFrame,
    universe: pd.DataFrame,
    size_exposure: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """universe mask -> MAD winsorize -> z-score -> (optional) size-neutralize -> z-score."""
    x = raw.where(universe)
    x = zscore(winsorize_mad(x))
    if size_exposure is not None:
        x = zscore(neutralize(x, [zscore(size_exposure.where(universe))]))
    return x
