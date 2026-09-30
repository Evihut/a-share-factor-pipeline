"""Build the frozen sample dataset (data/sample/) from the full download cache.

A fixed random subset of the stocks that were CSI 300 members, over a shorter
window, in one long-format parquet file, with its membership snapshots,
benchmark, and a checksum manifest. Prices are rounded to 4 decimals, volumes
and amounts to whole units, and float market cap to 10k CNY, so the file
compresses to a size that can be committed. CI runs the real pipeline end to
end on this frozen data.

Usage:
    python scripts/make_sample.py [--n-stocks 100] [--seed 7] [--start 2019-01-01] [--end 2023-12-31]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qfp.data import write_manifest  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data")
    ap.add_argument("--dst", type=Path, default=ROOT / "data/sample")
    ap.add_argument("--n-stocks", type=int, default=100)
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    members = pd.read_csv(args.src / "membership_weekly.csv", dtype={"code": str})
    members = members[(members["date"] >= args.start) & (members["date"] <= args.end)]
    candidates = sorted(c for c in members["code"].unique() if (args.src / "raw" / f"{c}.parquet").exists())
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(candidates, size=args.n_stocks, replace=False))

    args.dst.mkdir(parents=True, exist_ok=True)
    bars = pd.concat([pd.read_parquet(args.src / "raw" / f"{c}.parquet") for c in chosen], ignore_index=True)
    bars = bars[(bars["date"] >= args.start) & (bars["date"] <= args.end)]
    bars = bars.sort_values(["code", "date"]).reset_index(drop=True)
    for col in ["open", "close", "high", "low", "turnover"]:
        bars[col] = bars[col].round(4)
    bars["pct_chg"] = bars["pct_chg"].round(2)
    bars["volume"] = bars["volume"].round(0)
    bars["amount"] = bars["amount"].round(0)
    bars["float_mcap"] = bars["float_mcap"].round(-4)  # nearest 10k CNY; only used for size ranks
    bars.to_parquet(args.dst / "bars.parquet", index=False, compression="zstd")
    members[members["code"].isin(chosen)].to_csv(args.dst / "membership_weekly.csv", index=False)
    pd.read_parquet(args.src / "benchmark.parquet").to_parquet(args.dst / "benchmark.parquet")
    write_manifest(
        args.dst, note=f"sample of {args.n_stocks} stocks, seed {args.seed}, built by scripts/make_sample.py"
    )
    print(f"{len(chosen)} stocks, {len(bars)} rows -> {args.dst}")
