"""End-to-end research pipeline: data -> factors -> preprocessing -> evaluation
-> composite -> report. Driven by a YAML config so every run is reproducible."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import evaluate as ev
from .data import Panel, load_panel
from .factors import compute_all
from .preprocess import entry_blocked, standardize, tradable_universe


@dataclass
class Config:
    data_dir: Path = Path("data")
    out_dir: Path = Path("reports")
    start: str = "2018-01-01"
    end: str | None = None
    split_date: str = "2022-01-01"  # in-sample before, out-of-sample from this date
    horizon: int = 5
    rebalance: int = 5
    n_groups: int = 5
    cost_bps: float = 15.0  # per unit of one-way turnover
    neutralize_size: bool = True
    point_in_time_universe: bool = True  # only rank stocks after their index inclusion date
    composite_min_abs_t: float = 2.0  # in-sample |t| needed to enter the composite
    factors: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for key in ("data_dir", "out_dir"):
            if key in raw:
                raw[key] = Path(raw[key])
        return cls(**raw)


@dataclass
class FactorReport:
    name: str
    direction: int  # +1 / -1, fitted in-sample
    ic: pd.Series  # daily RankIC of the raw (unflipped) standardized factor
    ic_pearson: pd.Series
    stats_full: ev.ICStats
    stats_is: ev.ICStats
    stats_oos: ev.ICStats
    pearson_full: ev.ICStats
    decay: pd.Series
    quantiles: ev.QuantileResult  # on the direction-aligned factor: top group = expected best


def slice_panel(p: Panel, start: str, end: str | None) -> Panel:
    sl = slice(start, end)
    fields = {k: getattr(p, k).loc[sl] for k in p.__dataclass_fields__ if k not in ("benchmark", "membership")}
    bench = p.benchmark.loc[sl] if p.benchmark is not None else None
    member = p.membership.loc[sl] if p.membership is not None else None
    return Panel(**fields, benchmark=bench, membership=member)


def evaluate_factor(
    name: str,
    x: pd.DataFrame,
    p: Panel,
    fwd: pd.DataFrame,
    blocked: pd.DataFrame,
    cfg: Config,
    direction: int | None = None,
) -> FactorReport:
    split = pd.Timestamp(cfg.split_date)
    ic = ev.ic_series(x, fwd, "spearman")
    ic_p = ev.ic_series(x, fwd, "pearson")
    is_ic, oos_ic = ic[ic.index < split], ic[ic.index >= split]
    if direction is None:
        direction = 1 if is_ic.mean() >= 0 else -1
    q = ev.quantile_backtest(direction * x, p.open, cfg.n_groups, cfg.rebalance, blocked=blocked, cost_bps=cfg.cost_bps)
    return FactorReport(
        name=name,
        direction=direction,
        ic=ic,
        ic_pearson=ic_p,
        stats_full=ev.ic_stats(ic, cfg.horizon),
        stats_is=ev.ic_stats(is_ic, cfg.horizon),
        stats_oos=ev.ic_stats(oos_ic, cfg.horizon),
        pearson_full=ev.ic_stats(ic_p, cfg.horizon),
        decay=ev.ic_decay(x, p.open),
        quantiles=q,
    )


def build_composite(
    std: dict[str, pd.DataFrame], reports: dict[str, FactorReport], cfg: Config
) -> tuple[pd.DataFrame, list[str]]:
    """Equal-weight, direction-aligned average of factors that were significant in-sample.

    Selection and signs use only in-sample ICs, so the out-of-sample composite
    evaluation is not contaminated by the test period.
    """
    chosen = [n for n, r in reports.items() if abs(r.stats_is.t_stat) >= cfg.composite_min_abs_t]
    if not chosen:
        raise ValueError("no factor passed the in-sample significance filter")
    stacked = sum(reports[n].direction * std[n].fillna(0.0) for n in chosen)
    coverage = sum(std[n].notna().astype(int) for n in chosen)
    composite = (stacked / coverage.replace(0, np.nan)).where(coverage >= max(1, len(chosen) // 2))
    return ev_zscore(composite), chosen


def ev_zscore(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)


def factor_correlation(std: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Time-average of the cross-sectional Spearman correlation between factors."""
    names = list(std)
    ranks = {n: std[n].rank(axis=1) for n in names}
    out = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ra, rb = ranks[a], ranks[b]
            ok = ra.notna() & rb.notna()
            ra, rb = ra.where(ok), rb.where(ok)
            ca, cb = ra.sub(ra.mean(axis=1), axis=0), rb.sub(rb.mean(axis=1), axis=0)
            c = (ca * cb).sum(axis=1) / np.sqrt((ca**2).sum(axis=1) * (cb**2).sum(axis=1))
            out.loc[a, b] = out.loc[b, a] = c.mean()
    return out


def run(cfg: Config, panel: Panel | None = None) -> dict:
    p = panel if panel is not None else load_panel(cfg.data_dir)
    p = slice_panel(p, cfg.start, cfg.end)
    universe = tradable_universe(p, point_in_time=cfg.point_in_time_universe)
    blocked = entry_blocked(p)
    fwd = ev.forward_returns(p.open, cfg.horizon)

    raw = compute_all(p, cfg.factors or None)
    size_exp = np.log(p.float_mcap) if cfg.neutralize_size else None
    std = {n: standardize(f, universe, None if n == "size" else size_exp) for n, f in raw.items()}

    reports = {n: evaluate_factor(n, x, p, fwd, blocked, cfg) for n, x in std.items()}
    composite, chosen = build_composite(std, reports, cfg)
    comp_report = evaluate_factor("composite", composite, p, fwd, blocked, cfg, direction=1)

    # Out-of-sample-only backtest of the composite: the headline number.
    split = pd.Timestamp(cfg.split_date)
    oos = composite.loc[split:]
    comp_oos_q = ev.quantile_backtest(
        oos, p.open.loc[split:], cfg.n_groups, cfg.rebalance, blocked=blocked.loc[split:], cost_bps=cfg.cost_bps
    )

    from .report import write_report  # matplotlib import deferred until needed

    return write_report(
        cfg=cfg,
        panel=p,
        reports=reports,
        composite=comp_report,
        composite_members=chosen,
        composite_oos=comp_oos_q,
        corr=factor_correlation(std),
        universe=universe,
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run the factor research pipeline")
    ap.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = ap.parse_args(argv)
    summary = run(Config.from_yaml(args.config))
    print(json.dumps(summary["headline"], indent=2))


if __name__ == "__main__":
    main()
