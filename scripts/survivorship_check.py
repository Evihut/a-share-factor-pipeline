"""Quantify survivorship bias: rerun the pipeline on the common but biased setup
(the end-of-sample constituent list used for the whole history) and compare with
the point-in-time universe. Writes reports/survivorship_bias.md.

Usage (after the main pipeline has produced reports/summary.csv):
    python scripts/survivorship_check.py
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qfp.data import load_panel  # noqa: E402
from qfp.pipeline import Config, run  # noqa: E402

COLS = {"direction": "Dir", "rank_ic": "RankIC", "t_stat": "t", "ls_ann_return": "L/S ann.", "ls_sharpe": "L/S Sharpe"}

if __name__ == "__main__":
    cfg = Config.from_yaml(ROOT / "configs/default.yaml")
    cfg.data_dir = ROOT / cfg.data_dir
    panel = load_panel(cfg.data_dir)

    members = pd.read_csv(cfg.data_dir / "membership_weekly.csv", dtype={"code": str})
    last = set(members.loc[members["date"] == members["date"].max(), "code"])
    keep = [c for c in panel.codes if c in last]
    for name in panel.__dataclass_fields__:
        value = getattr(panel, name)
        if isinstance(value, pd.DataFrame):
            setattr(panel, name, value[keep])

    with tempfile.TemporaryDirectory() as tmp:
        cfg.out_dir = Path(tmp)
        cfg.point_in_time_universe = False
        biased = run(cfg, panel=panel)["summary"].set_index("factor")
    pit = pd.read_csv(ROOT / "reports/summary.csv").set_index("factor")

    lines = [
        "# Survivorship bias check",
        "",
        f"*Biased*: the {len(keep)} constituents at the end of the sample, treated as the universe for the whole "
        "period. *Point-in-time*: every stock only while it was actually a CSI 300 member (default pipeline). "
        "Same factors, preprocessing, costs and in-sample sign fitting. L/S = Q5 − Q1, gross, annualized.",
        "",
        "| Factor | " + " | ".join(f"{v} biased | {v} PIT" for v in COLS.values()) + " |",
        "|---|" + "---:|" * (2 * len(COLS)),
    ]
    for f in pit.index:
        cells = []
        for c in COLS:
            for df in (biased, pit):
                v = df.loc[f, c]
                if c == "direction":
                    cells.append("+" if v > 0 else "−")
                elif c == "ls_ann_return":
                    cells.append(f"{v * 100:.1f}%")
                else:
                    cells.append(f"{v:.3f}")
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    out = ROOT / "reports/survivorship_bias.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())
