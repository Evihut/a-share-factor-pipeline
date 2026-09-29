"""Tear-sheet style report: summary tables (CSV/Markdown/JSON) and figures."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import evaluate as ev  # noqa: E402

if TYPE_CHECKING:
    from .data import Panel
    from .pipeline import Config, FactorReport

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
    }
)
QCOLORS = plt.cm.RdYlGn(np.linspace(0.1, 0.9, 10))


def _quantile_metrics(q: ev.QuantileResult, cfg: Config) -> dict[str, float]:
    ppy = ev.TRADING_DAYS / cfg.rebalance
    top, bottom = q.top, q.bottom
    ls_gross = ev.perf_stats(q.nav["long_short"])
    ls_net = ev.perf_stats(q.net_nav("long_short"), ppy)
    excess = (
        1
        + q.nav[top].pct_change().fillna(q.nav[top].iloc[0] - 1)
        - q.nav["universe"].pct_change().fillna(q.nav["universe"].iloc[0] - 1)
    ).cumprod()
    top_excess = ev.perf_stats(excess)
    top_net = ev.perf_stats(q.net_nav(top), ppy)
    return {
        "ls_ann_return": ls_gross["ann_return"],
        "ls_sharpe": ls_gross["sharpe"],
        "ls_max_dd": ls_gross["max_drawdown"],
        "ls_ann_return_net": ls_net["ann_return"],
        "ls_sharpe_net": ls_net["sharpe"],
        "top_ann_return_net": top_net["ann_return"],
        "top_excess_ann": top_excess["ann_return"],
        "top_excess_ir": top_excess["sharpe"],
        "top_turnover": float(q.turnover[top].iloc[1:].mean()),
        "bottom_turnover": float(q.turnover[bottom].iloc[1:].mean()),
        "monotonicity": ev.monotonicity(q.period_returns, len(q.turnover.columns)),
    }


def _row(r: FactorReport, cfg: Config) -> dict[str, float]:
    row = {
        "factor": r.name,
        "direction": r.direction,
        "rank_ic": r.stats_full.mean,
        "rank_icir": r.stats_full.icir,
        "rank_icir_ann": r.stats_full.icir_annual,
        "t_stat": r.stats_full.t_stat,
        "ic_pos_ratio": r.stats_full.positive_ratio,
        "pearson_ic": r.pearson_full.mean,
        "rank_ic_is": r.stats_is.mean,
        "rank_ic_oos": r.stats_oos.mean,
        "rank_icir_oos": r.stats_oos.icir,
    }
    row.update(_quantile_metrics(r.quantiles, cfg))
    return row


# --------------------------------------------------------------------------- figures


def _factor_figure(r: FactorReport, cfg: Config, path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    split = pd.Timestamp(cfg.split_date)

    ax = axes[0, 0]
    ax.bar(r.ic.index, r.ic.values, width=1.0, color="#9db4d0", alpha=0.6, label="daily RankIC")
    ax2 = ax.twinx()
    ax2.plot(r.ic.cumsum(), color="#1f4e79", lw=1.4, label="cumulative")
    ax2.grid(False)
    ax.axvline(split, color="k", ls="--", lw=0.8)
    ax.set_title(f"RankIC ({cfg.horizon}d fwd)  mean={r.stats_full.mean:.4f}  ICIR={r.stats_full.icir:.3f}")

    ax = axes[0, 1]
    ax.bar([str(h) for h in r.decay.index], r.decay.values, color="#1f4e79")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("IC decay: mean RankIC by horizon (days)")

    ax = axes[1, 0]
    nav = r.quantiles.nav
    n = len(r.quantiles.turnover.columns)
    for i in range(n):
        ax.plot(nav[f"Q{i + 1}"], color=QCOLORS[int(i * 9 / max(n - 1, 1))], lw=1.1, label=f"Q{i + 1}")
    ax.plot(nav["universe"], color="grey", ls=":", lw=1.0, label="universe EW")
    ax.axvline(split, color="k", ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.legend(fontsize=7, ncol=3)
    sign = "" if r.direction > 0 else " (sign flipped)"
    ax.set_title(f"Quantile NAV, rebalance every {cfg.rebalance}d{sign}")

    ax = axes[1, 1]
    ann = (1 + r.quantiles.period_returns[[f"Q{i + 1}" for i in range(n)]].mean()) ** (
        ev.TRADING_DAYS / cfg.rebalance
    ) - 1
    ax.bar(ann.index, ann.values * 100, color=[QCOLORS[int(i * 9 / max(n - 1, 1))] for i in range(n)])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("annualized return, %")
    ax.set_title("Annualized return by quantile (gross)")

    fig.suptitle(r.name, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _corr_figure(corr: pd.DataFrame, path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Mean cross-sectional rank correlation")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _composite_oos_figure(q: ev.QuantileResult, cfg: Config, path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    n = len(q.turnover.columns)
    ax = axes[0]
    for i in range(n):
        ax.plot(q.nav[f"Q{i + 1}"], color=QCOLORS[int(i * 9 / max(n - 1, 1))], lw=1.2, label=f"Q{i + 1}")
    ax.plot(q.nav["universe"], color="grey", ls=":", label="universe EW")
    ax.legend(fontsize=7, ncol=3)
    ax.set_title(f"Composite, out-of-sample from {cfg.split_date}: quantile NAV")
    ax = axes[1]
    ax.plot(q.nav["long_short"], color="#1f4e79", label="long-short gross (daily)")
    ax.plot(q.net_nav("long_short"), color="#c0392b", label=f"long-short net of {cfg.cost_bps:.0f} bps/turnover")
    ax.legend(fontsize=7)
    ax.set_title("Composite long-short NAV (OOS)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- report


def _fmt_table(df: pd.DataFrame, cols: dict[str, str], pct: set[str]) -> str:
    head = "| " + " | ".join(cols.values()) + " |\n|" + "---|" * len(cols) + "\n"
    lines = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, str):
                cells.append(v)
            elif c in pct:
                cells.append(f"{v * 100:.1f}%")
            elif c == "direction":
                cells.append("+" if v > 0 else "−")
            else:
                cells.append(f"{v:.3f}" if abs(v) >= 0.1 or v == 0 else f"{v:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"


def write_report(
    *, cfg, panel: Panel, reports, composite, composite_members, composite_oos, corr, universe: pd.DataFrame
) -> dict:
    out = cfg.out_dir
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    rows = [_row(r, cfg) for r in reports.values()] + [_row(composite, cfg)]
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False, float_format="%.6f")

    for r in [*reports.values(), composite]:
        _factor_figure(r, cfg, figs / f"{r.name}.png")
    _corr_figure(corr, figs / "factor_correlation.png")
    _composite_oos_figure(composite_oos, cfg, figs / "composite_oos.png")

    oos_metrics = _quantile_metrics(composite_oos, cfg)
    oos_ic = composite.stats_oos
    headline = {
        "period": f"{panel.dates[0].date()} to {panel.dates[-1].date()}",
        "split_date": cfg.split_date,
        "n_stocks": int(universe.any().sum()),  # stocks eligible on at least one date
        "avg_universe_size": float(universe.sum(axis=1).mean()),
        "composite_members": composite_members,
        "composite_oos_rank_ic": oos_ic.mean,
        "composite_oos_rank_icir": oos_ic.icir,
        "composite_oos_rank_icir_annual": oos_ic.icir_annual,
        "composite_oos_t_stat": oos_ic.t_stat,
        **{f"composite_oos_{k}": v for k, v in oos_metrics.items()},
    }
    (out / "metrics.json").write_text(json.dumps(headline, indent=2, default=float))

    main_cols = {
        "factor": "Factor",
        "direction": "Dir",
        "rank_ic": "RankIC",
        "rank_icir": "ICIR",
        "rank_icir_ann": "ICIR (ann.)",
        "t_stat": "t",
        "ic_pos_ratio": "IC>0",
        "rank_ic_is": "RankIC IS",
        "rank_ic_oos": "RankIC OOS",
        "monotonicity": "Monotonic",
    }
    bt_cols = {
        "factor": "Factor",
        "ls_ann_return": "L/S ann. (gross)",
        "ls_sharpe": "L/S Sharpe",
        "ls_max_dd": "L/S MaxDD",
        "ls_ann_return_net": "L/S ann. (net)",
        "ls_sharpe_net": "L/S Sharpe (net)",
        "top_excess_ann": "Top excess ann.",
        "top_excess_ir": "Top IR",
        "top_turnover": "Top turnover",
    }
    pct = {"ic_pos_ratio", "ls_ann_return", "ls_max_dd", "ls_ann_return_net", "top_excess_ann", "top_turnover"}

    md = [
        "# Factor research report",
        "",
        f"Universe: {'point-in-time CSI 300 members' if cfg.point_in_time_universe else 'CSI 300 (static list)'} "
        f"({headline['n_stocks']} distinct stocks, "
        f"avg. {headline['avg_universe_size']:.0f} eligible per day), {headline['period']}. "
        f"Signals at close *t*, traded at open *t+1*; {cfg.horizon}-day forward open-to-open returns; "
        f"{cfg.n_groups} equal-weight quantiles rebalanced every {cfg.rebalance} days; costs "
        f"{cfg.cost_bps:.0f} bps per unit one-way turnover. Factors are MAD-winsorized, z-scored"
        + (" and size-neutralized" if cfg.neutralize_size else "")
        + f". In-sample (IS) before {cfg.split_date}, out-of-sample (OOS) after.",
        "",
        "## Out-of-sample composite (headline)",
        "",
        f"Members (selected by in-sample |t| ≥ {cfg.composite_min_abs_t}, signs from IS): "
        + ", ".join(f"`{m}`" for m in composite_members),
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| RankIC (OOS) | {oos_ic.mean:.4f} |",
        f"| ICIR per period / annualized | {oos_ic.icir:.3f} / {oos_ic.icir_annual:.2f} |",
        f"| t-stat (non-overlapping) | {oos_ic.t_stat:.2f} |",
        f"| Long-short ann. return gross / net | {oos_metrics['ls_ann_return'] * 100:.1f}% / {oos_metrics['ls_ann_return_net'] * 100:.1f}% |",
        f"| Long-short Sharpe gross / net | {oos_metrics['ls_sharpe']:.2f} / {oos_metrics['ls_sharpe_net']:.2f} |",
        f"| Long-short max drawdown | {oos_metrics['ls_max_dd'] * 100:.1f}% |",
        f"| Top quantile excess vs universe (ann.) | {oos_metrics['top_excess_ann'] * 100:.1f}% (IR {oos_metrics['top_excess_ir']:.2f}) |",
        f"| Quantile monotonicity | {oos_metrics['monotonicity']:.2f} |",
        "",
        "![composite OOS](figures/composite_oos.png)",
        "",
        "## IC / ICIR (full sample)",
        "",
        "`Dir` is the sign fitted in-sample; quantile backtests use the sign-aligned factor so the top quantile is "
        "always the expected winner. ICIR = mean / std of the per-date RankIC; annualized ICIR scales by "
        f"√(252/{cfg.horizon}); t-stat uses non-overlapping periods only.",
        "",
        _fmt_table(summary, main_cols, pct),
        "## Quantile backtest (full sample)",
        "",
        _fmt_table(summary, bt_cols, pct),
        "## Factor correlation",
        "",
        "![correlation](figures/factor_correlation.png)",
        "",
        "## Per-factor tear sheets",
        "",
    ]
    for r in [*reports.values(), composite]:
        md += [f"### {r.name}", "", f"![{r.name}](figures/{r.name}.png)", ""]
    (out / "REPORT.md").write_text("\n".join(md))
    return {"summary": summary, "headline": headline}
