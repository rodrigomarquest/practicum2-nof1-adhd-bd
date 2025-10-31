import argparse
import pandas as pd
import sys

p = argparse.ArgumentParser()
p.add_argument('--path', required=True)
args = p.parse_args()

try:
    df = pd.read_csv(args.path)
except Exception as e:
    sys.exit(f"[ERROR] Cannot read {args.path}: {e}")

cols = set(df.columns)
needed = {'date', 'label'}
missing = needed - cols
if missing:
    sys.exit(f"[ERROR] Missing columns: {missing}")

n = len(df)
n_labeled = int(df['label'].notna().sum())
print(f"Rows: {n}, labeled: {n_labeled} ({n_labeled/n*100:.1f}%)")
if n_labeled == 0:
    sys.exit("[ERROR] No labeled rows found.")

num_cols = [c for c in df.columns if str(df[c].dtype) in ('float64','int64')]
print(f"Numeric features: {len(num_cols)}")
print("[OK] Dataset ready for NB2.")
