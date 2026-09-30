"""End-to-end regression on the committed frozen sample (data/sample).

The sample's checksums must match its manifest, and the full pipeline must
reproduce the committed metrics. If a code change moves these numbers on
purpose, regenerate the expected files:

    python -m qfp.pipeline --config configs/sample.yaml
    cp reports_sample/metrics.json tests/expected/sample_metrics.json
    cp reports_sample/summary.csv tests/expected/sample_summary.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qfp.data import verify_manifest
from qfp.pipeline import Config, run

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "tests" / "expected"
RTOL = 1e-6  # tolerant of BLAS / platform rounding, not of behavior changes


def test_sample_data_matches_manifest():
    assert verify_manifest(ROOT / "data" / "sample") == []


@pytest.fixture(scope="module")
def sample_run(tmp_path_factory):
    cfg = Config.from_yaml(ROOT / "configs" / "sample.yaml")
    cfg.data_dir = ROOT / cfg.data_dir
    cfg.out_dir = tmp_path_factory.mktemp("sample_report")
    return cfg.out_dir, run(cfg)


def test_sample_headline_reproduces(sample_run):
    _, out = sample_run
    expected = json.loads((EXPECTED / "sample_metrics.json").read_text())
    got = json.loads(json.dumps(out["headline"], default=float))
    assert got.keys() == expected.keys()
    for key, want in expected.items():
        if isinstance(want, float):
            assert got[key] == pytest.approx(want, rel=RTOL, abs=1e-12), key
        else:
            assert got[key] == want, key


def test_sample_factor_table_reproduces(sample_run):
    out_dir, _ = sample_run
    got = pd.read_csv(out_dir / "summary.csv").set_index("factor")
    want = pd.read_csv(EXPECTED / "sample_summary.csv").set_index("factor")
    pd.testing.assert_index_equal(got.index, want.index)
    pd.testing.assert_index_equal(got.columns, want.columns)
    # summary.csv is written with 6 decimals, so allow a last-digit rounding flip
    np.testing.assert_allclose(got.to_numpy(float), want.to_numpy(float), rtol=RTOL, atol=2e-6)
    assert (out_dir / "REPORT.md").exists()
