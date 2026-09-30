"""Write or verify data/manifest.json (SHA-256 of every data file + source versions).

Usage:
    python scripts/data_manifest.py write  [--data-dir data] [--note "..."]
    python scripts/data_manifest.py verify [--data-dir data]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qfp.data import verify_manifest, write_manifest  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["write", "verify"])
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    if args.action == "write":
        m = write_manifest(args.data_dir, args.note)
        print(f"wrote {len(m['files'])} checksums to {args.data_dir / 'manifest.json'}")
    else:
        problems = verify_manifest(args.data_dir)
        print("\n".join(problems) if problems else "all files match the manifest")
        sys.exit(1 if problems else 0)
