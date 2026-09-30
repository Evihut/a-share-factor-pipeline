"""Factor evaluation: forward returns, IC / RankIC / ICIR, IC decay, and a
quantile (layered) backtest with turnover and transaction costs.

Timing convention (no look-ahead): a factor value at date t uses data up to the
close of t. The position is entered at the *open of t+1* and exited at the open
of t+1+h, so ``forward_returns(h)[t] = open[t+1+h] / open[t+1] - 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def forward_returns(open_: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return open_.shift(-(1 + horizon)) / open_.shift(-1) - 1.0


# --------------------------------------------------------------------------- IC


def ic_series(factor: pd.DataFrame, fwd: pd.DataFrame, method: str = "spearman", min_names: int = 30) -> pd.Series:
    """Cross-sectional correlation between factor and forward return, per date."""
    f = factor.reindex_like(fwd)
    ok = f.notna() & fwd.notna()
    if method == "spearman":
        f = f.where(ok).rank(axis=1)
        r = fwd.where(ok).rank(axis=1)
    else:
        f, r = f.where(ok), fwd.where(ok)
    fc = f.sub(f.mean(axis=1), axis=0)
    rc = r.sub(r.mean(axis=1), axis=0)
    cov = (fc * rc).sum(axis=1)
    denom = np.sqrt((fc**2).sum(axis=1) * (rc**2).sum(axis=1))
    ic = cov / denom
    return ic.where(ok.sum(axis=1) >= min_names).dropna()


@dataclass
class ICStats:
    mean: float
    std: float
    icir: float  # mean / std of the per-period IC
    icir_annual: float  # ICIR scaled to one year of non-overlapping periods
    t_stat: float  # on non-overlapping periods only
    positive_ratio: float
    n_periods: int

    def as_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def ic_stats(ic: pd.Series, horizon: int) -> ICStats:
    """Summary statistics. For h > 1 consecutive daily ICs overlap, which would
    inflate the t-statistic, so the t-stat uses every h-th observation."""
    mean, std = ic.mean(), ic.std()
    icir = mean / std if std > 0 else np.nan
    sampled = ic.iloc[::horizon]
    t = stats.ttest_1samp(sampled, 0.0).statistic if len(sampled) > 2 else np.nan
    return ICStats(
        mean=float(mean),
        std=float(std),
        icir=float(icir),
        icir_annual=float(icir * np.sqrt(TRADING_DAYS / horizon)),
        t_stat=float(t),
        positive_ratio=float((ic > 0).mean()),
        n_periods=int(len(sampled)),
    )


def ic_decay(factor: pd.DataFrame, open_: pd.DataFrame, horizons=(1, 2, 5, 10, 20)) -> pd.Series:
    return pd.Series({h: ic_series(factor, forward_returns(open_, h)).mean() for h in horizons}, name="rank_ic")


# --------------------------------------------------------------------------- quantile backtest


@dataclass
class QuantileResult:
    nav: pd.DataFrame  # daily NAV per quantile + long_short + universe (equal-weight)
    period_returns: pd.DataFrame  # per-rebalance returns, gross
    turnover: pd.DataFrame  # one-way turnover 0.5 * sum|w_target - w_drifted| per rebalance per quantile
    cost_bps: float

    def gross_nav(self, column: str) -> pd.Series:
        """NAV compounded per rebalance period, before costs (same basis as ``net_nav``)."""
        return (1 + self.period_returns[column]).cumprod()

    def net_nav(self, column: str) -> pd.Series:
        """NAV after costs, charging cost_bps per unit of one-way turnover at each rebalance."""
        gross = self.period_returns[column]
        if column == "long_short":
            to = self.turnover[self.top] + self.turnover[self.bottom]
        else:
            to = self.turnover[column] if column in self.turnover else 0.0
        net = gross - to * self.cost_bps / 1e4
        return (1 + net).cumprod()

    @property
    def top(self) -> str:
        return f"Q{len(self.turnover.columns)}"

    @property
    def bottom(self) -> str:
        return "Q1"


def quantile_backtest(
    factor: pd.DataFrame,
    open_: pd.DataFrame,
    n_groups: int = 5,
    rebalance: int = 5,
    blocked: pd.DataFrame | None = None,
    cost_bps: float = 15.0,
    min_names: int = 50,
) -> QuantileResult:
    """Equal-weight quantile portfolios, rebalanced every ``rebalance`` days.

    On each rebalance date t, eligible stocks (factor available and not blocked
    at the next open) are sorted into ``n_groups`` buckets (Q1 = lowest factor).
    Each bucket is bought equal-weight at open t+1 and held buy-and-hold until
    the next rebalance open, so the daily NAV reflects weight drift correctly.
    The long-short leg is Q_top minus Q1 (both legs rebalanced to equal weight).

    Turnover at a rebalance is ``0.5 * sum_i |w_target_i - w_drifted_i|``, where
    the drifted weights are the previous equal weights after a period of
    buy-and-hold price moves. This counts both replaced names and the trades
    needed to restore equal weights among names that stay. The first period
    (building from cash) counts as 1.0.
    """
    dates = factor.index
    rel = open_ / open_.shift(1)  # open-to-open price relative, indexed by the later open
    labels = [f"Q{i + 1}" for i in range(n_groups)]
    daily = {c: [] for c in [*labels, "universe"]}
    period_rows, turnover_rows, period_index = [], [], []
    drifted: dict[str, pd.Series] = {}  # weights held just before the next rebalance

    rebal_dates = dates[: len(dates) - rebalance - 1 : rebalance]
    for t in rebal_dates:
        i = dates.get_loc(t)
        f = factor.loc[t]
        if blocked is not None:
            f = f[~blocked.loc[t].reindex(f.index, fill_value=True)]
        f = f.dropna()
        # holdings must have a price at the entry open
        f = f[open_.iloc[i + 1].reindex(f.index).notna()]
        if len(f) < min_names:
            continue
        buckets = pd.qcut(f.rank(method="first"), n_groups, labels=labels)
        path_dates = dates[i + 2 : i + 2 + rebalance]  # opens t+2 .. t+1+rebalance
        path = rel.loc[path_dates].fillna(1.0).cumprod()  # suspended names carry flat

        row, to_row = {}, {}
        for q in [*labels, "universe"]:
            members = f.index if q == "universe" else f.index[buckets == q]
            value = path[members].mean(axis=1)  # buy-and-hold equal-weight value path
            daily[q].append(value / value.shift(1, fill_value=1.0) - 1.0)
            row[q] = value.iloc[-1] - 1.0
            if q != "universe":
                target = pd.Series(1.0 / len(members), index=members)
                old = drifted.get(q)
                to_row[q] = 1.0 if old is None else 0.5 * target.sub(old, fill_value=0.0).abs().sum()
                growth = path[members].iloc[-1]
                drifted[q] = growth / growth.sum()
        row["long_short"] = row[labels[-1]] - row[labels[0]]
        period_rows.append(row)
        turnover_rows.append(to_row)
        period_index.append(t)

    daily_ret = pd.DataFrame({q: pd.concat(v) for q, v in daily.items()})
    daily_ret["long_short"] = daily_ret[labels[-1]] - daily_ret[labels[0]]
    return QuantileResult(
        nav=(1 + daily_ret).cumprod(),
        period_returns=pd.DataFrame(period_rows, index=pd.DatetimeIndex(period_index)),
        turnover=pd.DataFrame(turnover_rows, index=pd.DatetimeIndex(period_index)),
        cost_bps=cost_bps,
    )


# --------------------------------------------------------------------------- performance


def perf_stats(nav: pd.Series, periods_per_year: float = TRADING_DAYS) -> dict[str, float]:
    ret = nav.pct_change().fillna(nav.iloc[0] - 1.0)
    years = len(ret) / periods_per_year
    ann_ret = nav.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    ann_vol = ret.std() * np.sqrt(periods_per_year)
    dd = nav / nav.cummax() - 1.0
    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(ret.mean() / ret.std() * np.sqrt(periods_per_year)) if ret.std() > 0 else np.nan,
        "max_drawdown": float(dd.min()),
    }


def monotonicity(period_returns: pd.DataFrame, n_groups: int) -> float:
    """Spearman correlation between quantile rank and mean quantile return (1 = perfectly monotone)."""
    means = period_returns[[f"Q{i + 1}" for i in range(n_groups)]].mean()
    return float(stats.spearmanr(np.arange(n_groups), means.to_numpy()).statistic)
