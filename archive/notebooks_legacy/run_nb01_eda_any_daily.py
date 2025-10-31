"""Run the NB-01 EDA logic as a script (idempotent).
Writes artifacts to notebooks/outputs/{tables,figures,manifests}
"""
from pathlib import Path
import json, hashlib, platform, sys

OUT = Path('notebooks') / 'outputs'
FIG = OUT / 'figures'
TAB = OUT / 'tables'
MAN = OUT / 'manifests'
for d in (FIG, TAB, MAN): d.mkdir(parents=True, exist_ok=True)

from glob import glob
cands = glob('data/ai/*/snapshots/*/public_subset/features_daily_public.csv')
if not cands:
    print('ERROR: public subset not found', file=sys.stderr)
    raise SystemExit(2)
INPUT = Path(cands[0])
print('INPUT', INPUT)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# load
df = pd.read_csv(INPUT)
if 'date_utc' not in df.columns:
    print('ERROR: date_utc missing', file=sys.stderr); raise SystemExit(3)
if not (('hr_mean' in df.columns) or ('sleep_total_minutes' in df.columns)):
    print('ERROR: hr_mean or sleep_total_minutes required', file=sys.stderr); raise SystemExit(4)

# write summary tables
numeric = df.select_dtypes(include=[np.number])
summary = numeric.describe().transpose()
summary.to_csv(TAB / 'summary_stats.csv')
df.head(20).to_csv(TAB / 'head.csv', index=False)
pd.DataFrame({'missing_pct': df.isna().mean() * 100}).to_csv(TAB / 'missingness_pct.csv')

# correlations
if numeric.shape[1] >= 2:
    corr = numeric.corr()
    corr.to_csv(TAB / 'correlations.csv')
    plt.figure(figsize=(6,6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag')
    plt.tight_layout()
    plt.savefig(FIG / 'correlations_heatmap.png', dpi=150)
    plt.close()

# histograms
if 'hr_mean' in df.columns:
    plt.figure()
    sns.histplot(df['hr_mean'].dropna(), bins=30)
    plt.tight_layout()
    plt.savefig(FIG / 'hr_mean_hist.png', dpi=150)
    plt.close()
if 'sleep_total_minutes' in df.columns:
    plt.figure()
    sns.histplot(df['sleep_total_minutes'].dropna(), bins=30)
    plt.tight_layout()
    plt.savefig(FIG / 'sleep_total_minutes_hist.png', dpi=150)
    plt.close()

# manifest
import pkg_resources
manifest = {
    'env': 'LOCAL',
    'python': platform.python_version(),
    'libs': {
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'matplotlib': matplotlib.__version__ if hasattr(matplotlib,'__version__') else 'unknown',
        'seaborn': sns.__version__,
    },
    'input': {
        'path': str(INPUT),
        'sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest()
    },
    'artifacts': []
}

for p in sorted(TAB.glob('*')) + sorted(FIG.glob('*')):
    if p.is_file():
        manifest['artifacts'].append({'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})

(MAN / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
print('Wrote artifacts to', OUT)
print('Manifest:', MAN / 'run_manifest.json')
