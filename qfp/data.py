"""Data loading: point-in-time index membership (baostock), daily A-share bars
(Sina via akshare), parquet caching, and reshaping into wide (date x code) panels.

Survivorship bias: the universe is every stock that was a CSI 300 member at any
point in the sample, including those later removed or delisted, and each stock
is only eligible while it was actually in the index (weekly snapshots).

The raw cache is one parquet file per stock under ``data/raw`` so that an
interrupted download resumes where it stopped. ``load_panel`` is the only entry
point the rest of the pipeline uses.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["open", "close", "high", "low", "volume", "amount", "pct_chg", "turnover", "float_mcap"]


@dataclass
class Panel:
    """Wide-format market data. Every frame is indexed by date with one column per code."""

    open: pd.DataFrame
    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    amount: pd.DataFrame
    pct_chg: pd.DataFrame
    turnover: pd.DataFrame  # percent of float shares traded
    float_mcap: pd.DataFrame  # float market cap in CNY (unadjusted), for size neutralization
    benchmark: pd.Series | None = None
    # True while a stock was an index member (point in time). None = every stock is a member.
    membership: pd.DataFrame | None = None

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def codes(self) -> pd.Index:
        return self.close.columns


# --------------------------------------------------------------------------- download


def fetch_membership(start: str, end: str, freq: str = "W-MON") -> pd.DataFrame:
    """Point-in-time CSI 300 constituents from baostock, one snapshot per ``freq`` period.

    Each row is (snapshot date, code): the code was a member on that date.
    """
    import baostock as bs

    bs.login()
    rows = []
    try:
        for d in pd.bdate_range(start, end, freq=freq):
            rs = bs.query_hs300_stocks(date=d.strftime("%Y-%m-%d"))
            while rs.next():
                _, code, name = rs.get_row_data()
                rows.append((d, code.split(".")[1], name))
    finally:
        bs.logout()
    return pd.DataFrame(rows, columns=["date", "code", "name"])


def _sina_symbol(code: str) -> str:
    return ("sh" if code.startswith(("6", "9")) else "bj" if code.startswith(("4", "8")) else "sz") + code


def fetch_one(code: str, start: str, end: str, retries: int = 2) -> pd.DataFrame:
    """Daily hfq-adjusted bars from Sina. Returns long format with FIELDS columns."""
    import akshare as ak

    for attempt in range(retries):
        try:
            df = ak.stock_zh_a_daily(symbol=_sina_symbol(code), start_date=start, end_date=end, adjust="hfq")
            break
        except Exception:  # throttling, or no data (Sina drops delisted stocks)
            if attempt == retries - 1:
                raise
            time.sleep(5.0 * (attempt + 1))
    return _finish(df, code)


def fetch_one_baostock(code: str, start: str, end: str) -> pd.DataFrame:
    """Fallback for stocks Sina no longer serves (mostly delisted). Needs an active baostock login."""
    import baostock as bs

    iso = lambda d: pd.Timestamp(d).strftime("%Y-%m-%d")  # noqa: E731
    rs = bs.query_history_k_data_plus(
        f"{_sina_symbol(code)[:2]}.{code}",
        "date,open,high,low,close,volume,amount,turn",
        start_date=iso(start),
        end_date=iso(end),
        frequency="d",
        adjustflag="1",  # backward adjusted (hfq), same convention as the Sina path
    )
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        raise ValueError(f"baostock returned no rows for {code}: {rs.error_msg}")
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount", "turn"])
    df = df.replace("", np.nan)
    for c in df.columns[1:]:
        df[c] = df[c].astype(float)
    df["turnover"] = df.pop("turn") / 100.0  # same fraction-of-float units as Sina
    return _finish(df, code)


def _finish(df: pd.DataFrame, code: str) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    # Sina's turnover is a fraction of float shares; amount / turnover recovers
    # the unadjusted float market cap (≈ VWAP x float shares).
    df["float_mcap"] = df["amount"] / df["turnover"].replace(0.0, np.nan)
    df["turnover"] = df["turnover"] * 100.0
    df["pct_chg"] = df["close"].pct_change() * 100.0
    df["code"] = code
    return df[["date", "code", *FIELDS]]


def fetch_benchmark(symbol: str = "sh000300") -> pd.Series:
    import akshare as ak

    df = ak.stock_zh_index_daily(symbol=symbol)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["close"].rename("benchmark")


def download(data_dir: Path, start: str, end: str, pause: float = 0.5) -> None:
    raw = data_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    member_path = data_dir / "membership_weekly.csv"
    if not member_path.exists():
        fetch_membership(start, end).to_csv(member_path, index=False)
    members = pd.read_csv(member_path, dtype={"code": str})
    codes = sorted(members["code"].unique())
    print(f"{len(codes)} stocks were CSI 300 members at some point in {start}-{end}", flush=True)
    failed, from_baostock = [], []
    bs_session = None
    try:
        for i, code in enumerate(codes):
            path = raw / f"{code}.parquet"
            if path.exists():
                continue
            try:
                fetch_one(code, start, end).to_parquet(path, index=False)
            except Exception:
                try:
                    if bs_session is None:
                        import baostock as bs

                        bs.login()
                        bs_session = bs
                    fetch_one_baostock(code, start, end).to_parquet(path, index=False)
                    from_baostock.append(code)
                except Exception as exc:
                    failed.append((code, repr(exc)))
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(codes)} processed", flush=True)
            time.sleep(pause)
    finally:
        if bs_session is not None:
            # baostock's logout can hang indefinitely; don't let it block the run.
            t = threading.Thread(target=bs_session.logout, daemon=True)
            t.start()
            t.join(timeout=10)
    fetch_benchmark().loc[pd.Timestamp(start) : pd.Timestamp(end)].to_frame().to_parquet(data_dir / "benchmark.parquet")
    print(f"{len(from_baostock)} stocks came from the baostock fallback: {from_baostock}")
    # Keep the committed reference manifest so a fresh download can be compared
    # with the data behind the published report. Updating it is explicit.
    if not (data_dir / MANIFEST).exists():
        write_manifest(data_dir, note=f"download {start}..{end}")
    if failed:
        print(f"{len(failed)} codes failed: {failed}")


# --------------------------------------------------------------------------- load


def _to_panel(long: pd.DataFrame, benchmark: pd.Series | None) -> Panel:
    frames = {}
    for field in FIELDS:
        wide = long.pivot(index="date", columns="code", values=field).sort_index()
        frames[field] = wide.astype(float)
    return Panel(**frames, benchmark=benchmark)


def membership_mask(dates: pd.DatetimeIndex, codes: pd.Index, snapshots: pd.DataFrame) -> pd.DataFrame:
    """Daily membership from periodic snapshots: each snapshot holds until the next one."""
    wide = pd.crosstab(pd.to_datetime(snapshots["date"]), snapshots["code"]).gt(0)
    wide = wide.reindex(columns=codes, fill_value=False)
    return wide.reindex(dates, method="ffill", fill_value=False).astype(bool)


def load_panel(data_dir: Path) -> Panel:
    """Load a data directory: either one parquet per stock under ``raw/`` (the
    download cache) or a single long-format ``bars.parquet`` (the frozen sample)."""
    bars = data_dir / "bars.parquet"
    if bars.exists():
        long = pd.read_parquet(bars)
    else:
        files = sorted((data_dir / "raw").glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no cached data in {data_dir}; run scripts/download_data.py first")
        long = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    long["code"] = long["code"].astype(str).str.zfill(6)
    bench_path = data_dir / "benchmark.parquet"
    bench = pd.read_parquet(bench_path)["benchmark"] if bench_path.exists() else None
    panel = _to_panel(long, bench)
    member_path = data_dir / "membership_weekly.csv"
    if member_path.exists():
        snapshots = pd.read_csv(member_path, dtype={"code": str})
        panel.membership = membership_mask(panel.dates, panel.codes, snapshots)
    return panel


# --------------------------------------------------------------------------- manifest


MANIFEST = "manifest.json"


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _data_files(data_dir: Path) -> list[Path]:
    patterns = ["bars.parquet", "benchmark.parquet", "membership_weekly.csv", "raw/*.parquet"]
    return sorted(p for pat in patterns for p in data_dir.glob(pat))


def write_manifest(data_dir: Path, note: str = "") -> dict:
    """Record what the data directory contains: SHA-256 per file plus the source
    library versions, so a later run can tell whether it sees the same data."""
    import datetime
    import importlib.metadata as md
    import json
    import platform

    def version(pkg: str) -> str | None:
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "note": note,
        "python": platform.python_version(),
        "sources": {"akshare": version("akshare"), "baostock": version("baostock"), "pandas": version("pandas")},
        "files": {str(p.relative_to(data_dir)): _sha256(p) for p in _data_files(data_dir)},
    }
    (data_dir / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def verify_manifest(data_dir: Path) -> list[str]:
    """Return a list of problems (empty if every file matches the manifest)."""
    import json

    manifest = json.loads((data_dir / MANIFEST).read_text())
    problems = []
    for rel, digest in manifest["files"].items():
        path = data_dir / rel
        if not path.exists():
            problems.append(f"missing: {rel}")
        elif _sha256(path) != digest:
            problems.append(f"checksum mismatch: {rel}")
    extra = {str(p.relative_to(data_dir)) for p in _data_files(data_dir)} - set(manifest["files"])
    problems += [f"not in manifest: {rel}" for rel in sorted(extra)]
    return problems


# --------------------------------------------------------------------------- synthetic


def synthetic_panel(
    n_days: int = 500,
    n_codes: int = 200,
    signal_strength: float = 0.0,
    seed: int = 0,
) -> tuple[Panel, pd.DataFrame]:
    """Generate a panel with a known planted signal, for tests and offline demos.

    Returns the panel and the planted signal. Next-day returns are
    ``signal_strength * signal + noise`` cross-sectionally, so a correct
    evaluator must recover a positive IC when ``signal_strength > 0`` and an IC
    near zero otherwise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    codes = [f"{i:06d}" for i in range(n_codes)]
    signal = pd.DataFrame(rng.standard_normal((n_days, n_codes)), index=dates, columns=codes)
    noise = rng.standard_normal((n_days, n_codes)) * 0.02
    market = rng.standard_normal((n_days, 1)) * 0.01
    # Synthetic opens equal the previous close, so the signal known at close t
    # drives the open(t+1) -> open(t+2) return: exactly what the pipeline trades.
    ret = market + noise + signal_strength * 0.02 * signal.shift(1).fillna(0.0).to_numpy()
    ret = np.clip(ret, -0.095, 0.095)
    close = pd.DataFrame(100.0 * np.cumprod(1.0 + ret, axis=0), index=dates, columns=codes)
    open_ = close.shift(1).fillna(100.0)
    volume = pd.DataFrame(rng.uniform(1e6, 5e6, (n_days, n_codes)), index=dates, columns=codes)
    panel = Panel(
        open=open_,
        close=close,
        high=np.maximum(open_, close) * 1.005,
        low=np.minimum(open_, close) * 0.995,
        volume=volume,
        amount=volume * close,
        pct_chg=pd.DataFrame(ret * 100.0, index=dates, columns=codes),
        turnover=volume / 1e7 * 100.0,
        float_mcap=pd.DataFrame(rng.lognormal(23.0, 1.0, (1, n_codes)).repeat(n_days, 0), index=dates, columns=codes),
        benchmark=pd.Series(close.mean(axis=1), name="benchmark"),
    )
    return panel, signal
