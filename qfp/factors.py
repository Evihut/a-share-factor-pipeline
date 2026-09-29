"""Factor library.

Every factor is a function ``Panel -> DataFrame`` (date x code) that only uses
information available at the close of each date. Values are raw: the sign
convention is *not* flipped here; direction is learned in-sample by the
pipeline (see ``pipeline.evaluate_factor``) so that nothing is hand-tuned on the
test period.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .data import Panel

FactorFn = Callable[[Panel], pd.DataFrame]


def _ret(p: Panel) -> pd.DataFrame:
    return p.close.pct_change(fill_method=None)


def momentum_20(p: Panel) -> pd.DataFrame:
    """1-month price momentum (in A-shares this usually behaves as reversal)."""
    return p.close / p.close.shift(20) - 1.0


def momentum_120_skip20(p: Panel) -> pd.DataFrame:
    """6-month momentum skipping the most recent month (Jegadeesh-Titman style)."""
    return p.close.shift(20) / p.close.shift(120) - 1.0


def reversal_5(p: Panel) -> pd.DataFrame:
    """Negative 1-week return."""
    return -(p.close / p.close.shift(5) - 1.0)


def volatility_20(p: Panel) -> pd.DataFrame:
    """20-day realized volatility of daily returns (low-volatility anomaly)."""
    return _ret(p).rolling(20, min_periods=15).std()


def turnover_20(p: Panel) -> pd.DataFrame:
    """20-day average turnover rate (attention / liquidity)."""
    return p.turnover.rolling(20, min_periods=15).mean()


def abnormal_turnover(p: Panel) -> pd.DataFrame:
    """Recent 5-day turnover relative to its 60-day average."""
    return p.turnover.rolling(5, min_periods=4).mean() / p.turnover.rolling(60, min_periods=40).mean() - 1.0


def amihud_20(p: Panel) -> pd.DataFrame:
    """Amihud illiquidity: mean |return| per unit of traded value (per 1e8 CNY)."""
    illiq = _ret(p).abs() / (p.amount / 1e8)
    return illiq.replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=15).mean()


def price_volume_corr_20(p: Panel) -> pd.DataFrame:
    """Rolling correlation of close price and volume."""
    return p.close.rolling(20, min_periods=15).corr(p.volume)


def intraday_range_20(p: Panel) -> pd.DataFrame:
    """Average daily high-low range relative to close."""
    return ((p.high - p.low) / p.close).rolling(20, min_periods=15).mean()


def size(p: Panel) -> pd.DataFrame:
    """Log float market cap."""
    return np.log(p.float_mcap)


FACTORS: dict[str, FactorFn] = {
    "mom_20": momentum_20,
    "mom_120_skip20": momentum_120_skip20,
    "rev_5": reversal_5,
    "vol_20": volatility_20,
    "turn_20": turnover_20,
    "abn_turn": abnormal_turnover,
    "amihud_20": amihud_20,
    "pv_corr_20": price_volume_corr_20,
    "range_20": intraday_range_20,
    "size": size,
}


def compute_all(p: Panel, names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    names = names or list(FACTORS)
    return {n: FACTORS[n](p) for n in names}
