"""Download point-in-time CSI 300 membership and daily bars (hfq-adjusted) of every
stock that was a member during the period into data/.

Usage:
    python scripts/download_data.py --start 20180101 --end 20250630
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qfp.data import download  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20180101")
    ap.add_argument("--end", default="20250630")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = ap.parse_args()
    download(args.data_dir, args.start, args.end)
    print("done")
