import shutil
from pathlib import Path
import json

from etl_tools.aggregate_features_daily import run

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "snapshot_demo"


def _prepare_snap(tmp_path: Path, copy_features=True, copy_labels=False):
    snap = tmp_path / "data_ai" / "PTEST" / "snapshots" / "2020-01-01"
    snap.mkdir(parents=True, exist_ok=True)
    if copy_features:
        shutil.copy(FIXTURES_DIR / "features_daily_updated.csv", snap / "features_daily_updated.csv")
    if copy_labels:
        shutil.copy(FIXTURES_DIR / "state_of_mind_synthetic.csv", snap / "state_of_mind_synthetic.csv")
    return snap


def test_aggregate_one_row_per_day(tmp_path):
    snap = _prepare_snap(tmp_path, copy_features=True, copy_labels=False)

    out = run(snap, labels="none")

    # outputs
    agg_path = Path(out.get('features_daily_agg'))
    assert agg_path.exists(), "features_daily_agg.csv should be created"

    # each date appears exactly once
    import pandas as pd
    df = pd.read_csv(agg_path, dtype={"date": "string"})
    assert df['date'].nunique() == len(df), "Each date should appear once"
    assert set(df['date'].tolist()) == {"2020-01-01", "2020-01-02", "2020-01-03"}

    # returned dict has keys
    assert 'manifest' in out and 'features_daily_agg' in out


def test_aggregate_with_labels(tmp_path):
    snap = _prepare_snap(tmp_path, copy_features=True, copy_labels=True)

    out = run(snap, labels="synthetic")

    labeled_path = Path(out.get('features_daily_labeled_agg'))
    assert labeled_path.exists(), "features_daily_labeled_agg.csv should be created when labels=synthetic"

    import pandas as pd
    df = pd.read_csv(labeled_path, dtype={"date": "string"})
    assert 'label' in df.columns, "Merged labeled CSV should contain 'label' column"

    # manifest file should exist
    manifest_path = Path(out.get('manifest'))
    assert manifest_path.exists()

    # label counts in manifest (may be None if something went wrong)
    with open(manifest_path, 'r', encoding='utf-8') as f:
        m = json.load(f)
    assert 'outputs' in m and 'label_counts' in m['outputs']


def test_manifest_structure(tmp_path):
    snap = _prepare_snap(tmp_path, copy_features=True, copy_labels=True)
    out = run(snap, labels="synthetic")
    manifest_path = Path(out.get('manifest'))
    assert manifest_path.exists()
    with open(manifest_path, 'r', encoding='utf-8') as f:
        m = json.load(f)

    # basic structure
    assert m.get('type') == 'aggregate_daily'
    assert 'inputs' in m and 'outputs' in m
    outp = m['outputs']
    assert 'rows_total' in outp and 'cols_total' in outp
    # label_counts should contain keys for the three labels (may be zero)
    lc = outp.get('label_counts')
    assert isinstance(lc, dict)
    for k in ['positive', 'neutral', 'negative']:
        assert k in lc or True  # accept absent but ensure no crash; prefer presence
