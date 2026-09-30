import numpy as np
import pandas as pd
import pytest

from qfp import evaluate as ev
from qfp.data import synthetic_panel
from qfp.pipeline import Config, run
from qfp.preprocess import entry_blocked, neutralize, standardize, tradable_universe, winsorize_mad, zscore


@pytest.fixture(scope="module")
def planted():
    return synthetic_panel(n_days=400, n_codes=200, signal_strength=0.3, seed=1)


@pytest.fixture(scope="module")
def null():
    return synthetic_panel(n_days=400, n_codes=200, signal_strength=0.0, seed=2)


# ---------------------------------------------------------------- forward returns / timing


def test_forward_return_enters_next_open():
    idx = pd.bdate_range("2024-01-01", periods=6)
    open_ = pd.DataFrame({"a": [10, 11, 12, 13, 14, 15]}, index=idx, dtype=float)
    fwd = ev.forward_returns(open_, 2)
    # signal on day 0 -> buy open day 1 (11) -> sell open day 3 (13)
    assert fwd.iloc[0, 0] == pytest.approx(13 / 11 - 1)
    assert fwd.iloc[-3:].isna().all().all()


def test_perfect_foresight_factor_has_ic_one(planted):
    panel, _ = planted
    fwd = ev.forward_returns(panel.open, 5)
    ic = ev.ic_series(fwd, fwd)
    assert ic.min() == pytest.approx(1.0)


def test_same_day_return_is_not_forward_return(null):
    """A factor built from information up to t must show ~zero IC on pure noise:
    guards against off-by-one alignment that would leak the future."""
    panel, _ = null
    today_ret = panel.close.pct_change(fill_method=None)
    ic = ev.ic_series(today_ret, ev.forward_returns(panel.open, 1))
    assert abs(ic.mean()) < 0.02


# ---------------------------------------------------------------- IC statistics


def test_planted_signal_is_recovered(planted):
    panel, signal = planted
    ic = ev.ic_series(signal, ev.forward_returns(panel.open, 1))
    s = ev.ic_stats(ic, 1)
    assert s.mean > 0.2
    assert s.t_stat > 10
    assert s.positive_ratio > 0.95


def test_noise_signal_has_no_ic(null):
    panel, signal = null
    s = ev.ic_stats(ev.ic_series(signal, ev.forward_returns(panel.open, 1)), 1)
    assert abs(s.mean) < 0.02
    assert abs(s.t_stat) < 3


def test_ic_stats_t_uses_non_overlapping_periods():
    ic = pd.Series(np.random.default_rng(0).normal(0.05, 0.1, 500))
    assert ev.ic_stats(ic, 5).n_periods == 100
    assert ev.ic_stats(ic, 1).n_periods == 500


def test_ic_decays_for_one_day_signal(planted):
    panel, signal = planted
    decay = ev.ic_decay(signal, panel.open, horizons=(1, 5, 20))
    assert decay[1] > decay[5] > decay[20]


# ---------------------------------------------------------------- quantile backtest


def test_quantiles_monotone_with_planted_signal(planted):
    panel, signal = planted
    q = ev.quantile_backtest(signal, panel.open, n_groups=5, rebalance=1, cost_bps=0)
    means = q.period_returns[[f"Q{i}" for i in range(1, 6)]].mean()
    assert means.is_monotonic_increasing
    assert ev.monotonicity(q.period_returns, 5) == pytest.approx(1.0)
    assert q.nav["long_short"].iloc[-1] > 1.5


def test_costs_reduce_returns(planted):
    panel, signal = planted
    q = ev.quantile_backtest(signal, panel.open, rebalance=1, cost_bps=20)
    assert q.net_nav("long_short").iloc[-1] < (1 + q.period_returns["long_short"]).cumprod().iloc[-1]
    # a random daily signal replaces most of each bucket every day
    assert q.turnover["Q5"].iloc[1:].mean() > 0.6


def test_turnover_counts_reweighting_of_unchanged_names():
    """Same names every period: name-based turnover would be 0, but price drift
    still forces trades to restore equal weight."""
    dates = pd.bdate_range("2024-01-01", periods=40)
    codes = [f"{i:06d}" for i in range(60)]
    # half the names rise 1%/day, half are flat -> weights drift apart
    growth = np.where(np.arange(60) < 30, 1.01, 1.0)
    open_ = pd.DataFrame(np.cumprod(np.tile(growth, (40, 1)), axis=0), index=dates, columns=codes)
    factor = pd.DataFrame(np.tile(np.arange(60.0), (40, 1)), index=dates, columns=codes)  # constant ranks
    q = ev.quantile_backtest(factor, open_, n_groups=1, rebalance=5, min_names=10)
    to = q.turnover["Q1"]
    assert to.iloc[0] == pytest.approx(1.0)
    # after 5 days the risers weigh g/(30g+30) and the flat names 1/(30g+30); every name is
    # off its 1/60 target by the same amount, so turnover = 0.5 * 60 * |1/60 - g/(30g+30)|
    g = 1.01**5
    expected = 0.5 * 60 * abs(1 / 60 - (g / (30 * g + 30)))
    assert to.iloc[1:].to_numpy() == pytest.approx(expected, rel=1e-9)
    assert expected > 0.01


def test_turnover_zero_without_drift_or_name_changes():
    dates = pd.bdate_range("2024-01-01", periods=30)
    codes = [f"{i:06d}" for i in range(60)]
    open_ = pd.DataFrame(100.0, index=dates, columns=codes)
    factor = pd.DataFrame(np.tile(np.arange(60.0), (30, 1)), index=dates, columns=codes)
    q = ev.quantile_backtest(factor, open_, n_groups=5, rebalance=5, min_names=10)
    assert (q.turnover.iloc[1:] == 0).all().all()


def test_daily_nav_consistent_with_period_returns(planted):
    panel, signal = planted
    q = ev.quantile_backtest(signal, panel.open, rebalance=5, cost_bps=0)
    for col in ["Q1", "Q5", "universe"]:
        assert q.nav[col].iloc[-1] == pytest.approx((1 + q.period_returns[col]).prod(), rel=1e-9)


def test_blocked_names_are_excluded(planted):
    panel, signal = planted
    blocked = pd.DataFrame(False, index=signal.index, columns=signal.columns)
    blocked.iloc[:, :100] = True
    q = ev.quantile_backtest(signal, panel.open, n_groups=5, rebalance=5, blocked=blocked, min_names=50)
    q_all = ev.quantile_backtest(signal, panel.open, n_groups=5, rebalance=5, min_names=50)
    assert not q.nav["universe"].equals(q_all.nav["universe"])


# ---------------------------------------------------------------- preprocessing


def test_winsorize_and_zscore():
    x = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 1000.0]])
    w = winsorize_mad(x, k=3)
    assert w.iloc[0, -1] < 1000
    z = zscore(w)
    assert z.iloc[0].mean() == pytest.approx(0.0, abs=1e-12)
    assert z.iloc[0].std() == pytest.approx(1.0)


def test_neutralize_removes_exposure():
    rng = np.random.default_rng(0)
    size = pd.DataFrame(rng.normal(size=(3, 300)))
    x = 2.0 * size + pd.DataFrame(rng.normal(size=(3, 300)))
    resid = neutralize(x, [size])
    for i in range(3):
        assert np.corrcoef(resid.iloc[i], size.iloc[i])[0, 1] == pytest.approx(0.0, abs=1e-10)


def test_entry_blocked_limit_up_and_suspension():
    idx = pd.bdate_range("2021-01-04", periods=4)
    close = pd.DataFrame({"600000": [10.0, 10.0, 11.0, 11.0], "300001": [10.0, 10.0, 11.0, 11.0]}, index=idx)
    open_ = pd.DataFrame({"600000": [10.0, 10.0, 11.0, 11.0], "300001": [10.0, 10.0, 11.0, 11.0]}, index=idx)
    vol = pd.DataFrame(1.0, index=idx, columns=close.columns)
    vol.iloc[3, 0] = 0.0
    panel = synthetic_panel(n_days=4, n_codes=2)[0]
    panel.close, panel.open, panel.volume = close, open_, vol
    blocked = entry_blocked(panel)
    # day 2 opens +10%: main board is limit-up (blocked on signal day 1); ChiNext (20% limit) is not
    assert blocked.iloc[1]["600000"] and not blocked.iloc[1]["300001"]
    # 600000 suspended on day 3 -> blocked on signal day 2; last day always blocked
    assert blocked.iloc[2]["600000"]
    assert blocked.iloc[3].all()


def test_universe_requires_history(planted):
    panel, _ = planted
    u = tradable_universe(panel, min_history=60)
    assert not u.iloc[:59].any().any()
    assert u.iloc[60:].all().all()


def test_standardize_is_cross_sectional(planted):
    panel, signal = planted
    u = tradable_universe(panel, 1)
    s = standardize(signal, u, np.log(panel.float_mcap))
    row = s.iloc[100].dropna()
    assert row.mean() == pytest.approx(0.0, abs=1e-9)
    assert row.std() == pytest.approx(1.0)


# ---------------------------------------------------------------- end to end


def test_pipeline_end_to_end_on_synthetic(tmp_path):
    panel, _ = synthetic_panel(n_days=320, n_codes=150, signal_strength=0.0, seed=3)
    cfg = Config(
        out_dir=tmp_path,
        start="2020-01-01",
        split_date="2020-09-01",
        composite_min_abs_t=0.0,
        factors=["mom_20", "rev_5", "vol_20", "turn_20"],
    )
    out = run(cfg, panel=panel)
    assert (tmp_path / "REPORT.md").exists()
    assert (tmp_path / "figures" / "composite_oos.png").exists()
    assert set(out["summary"]["factor"]) == {"mom_20", "rev_5", "vol_20", "turn_20", "composite"}
    assert np.isfinite(out["headline"]["composite_oos_rank_ic"])


def test_point_in_time_membership_from_snapshots():
    from qfp.data import membership_mask

    panel = synthetic_panel(n_days=60, n_codes=3, seed=1)[0]
    a, b, c = panel.codes
    snaps = pd.DataFrame(
        {
            "date": ["2020-01-06", "2020-01-06", "2020-02-03", "2020-02-03"],
            "code": [a, b, a, c],  # b removed and c added on 2020-02-03
        }
    )
    panel.membership = membership_mask(panel.dates, panel.codes, snaps)
    u = tradable_universe(panel, min_history=1)
    assert not u.loc["2020-01-01":"2020-01-03"].any().any()  # before the first snapshot
    assert u.loc["2020-01-06":"2020-01-31", [a, b]].all().all()
    assert not u.loc["2020-01-06":"2020-01-31", c].any()
    assert u.loc["2020-02-03":, [a, c]].all().all()
    assert not u.loc["2020-02-03":, b].any()
    assert tradable_universe(panel, min_history=1, point_in_time=False).all().all()
