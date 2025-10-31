"""Runner for NB-03 ML baselines (6-fold temporal CV).
Saves artifacts under notebooks/outputs/{tables,figures,manifests}.
"""
from pathlib import Path
import json, hashlib, platform, sys
OUT = Path('notebooks') / 'outputs'
FIG = OUT / 'figures'
TAB = OUT / 'tables'
MAN = OUT / 'manifests'
for d in (FIG, TAB, MAN): d.mkdir(parents=True, exist_ok=True)

from glob import glob
feat = glob('data/ai/*/snapshots/*/public_subset/features_daily_public.csv')
if not feat:
    print('ERROR: features_daily_public.csv not found', file=sys.stderr); raise SystemExit(2)
FEATURES = Path(feat[0])
labels = glob('data/ai/*/snapshots/*/public_subset/features_daily_labeled_public.csv')
LABELS = Path(labels[0]) if labels else None
print('FEATURES', FEATURES, 'LABELS', LABELS)

import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, roc_auc_score

SEED = 20251023

# load
DF = pd.read_csv(FEATURES)
if 'date_utc' not in DF.columns:
    print('ERROR: date_utc missing', file=sys.stderr); raise SystemExit(3)
try:
    DF['date_utc'] = pd.to_datetime(DF['date_utc'])
except Exception:
    pass
DF = DF.sort_values('date_utc').reset_index(drop=True)
DF['month'] = DF['date_utc'].dt.to_period('M')

# prepare folds monthly
months = DF['month'].drop_duplicates().sort_values().tolist()
train_months = 4; val_months = 2
folds = []
for start in range(0, max(0, len(months) - (train_months + val_months) + 1)):
    tr = months[start:start+train_months]
    va = months[start+train_months:start+train_months+val_months]
    folds.append((tr,va))
folds = folds[:6]
print('prepared folds', len(folds))

if LABELS and LABELS.exists():
    print('supervised path')
    LAB = pd.read_csv(LABELS)
    try:
        LAB['date_utc'] = pd.to_datetime(LAB['date_utc'])
    except Exception:
        pass
    M = DF.merge(LAB, on='date_utc', how='inner')
    if M.empty:
        print('no overlap, doing unsupervised')
        supervised = False
    else:
        supervised = True
else:
    print('no labels found, unsupervised path')
    supervised = False

if supervised:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(M['label'])
    X = M.drop(columns=['date_utc','month','label']).select_dtypes(include=[np.number]).fillna(0)
    models = {'logreg': LogisticRegression(max_iter=1000, random_state=SEED), 'rf': RandomForestClassifier(n_estimators=100, random_state=SEED)}
    try:
        import xgboost as xgb
        models['xgb'] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=SEED)
    except Exception:
        pass
    results = []
    last_preds = None
    for i,(tr,va) in enumerate(folds, start=1):
        tr_mask = M['month'].isin(tr)
        va_mask = M['month'].isin(va)
        Xtr = X[tr_mask]; ytr = y[tr_mask]
        Xva = X[va_mask]; yva = y[va_mask]
        if len(ytr)<10 or len(yva)<5: continue
        row = {'fold': i}
        for name, model in models.items():
            try:
                model.fit(Xtr, ytr)
                p = model.predict(Xva)
                row[name+'_f1_macro'] = f1_score(yva, p, average='macro')
                row[name+'_f1_weighted'] = f1_score(yva, p, average='weighted')
                row[name+'_bal_acc'] = balanced_accuracy_score(yva, p)
                row[name+'_kappa'] = cohen_kappa_score(yva, p)
                if hasattr(model,'predict_proba'):
                    try:
                        pr = model.predict_proba(Xva)
                        row[name+'_auroc_ovr'] = roc_auc_score(yva, pr, multi_class='ovr')
                    except Exception:
                        row[name+'_auroc_ovr'] = None
                last_preds = p
            except Exception as e:
                row[name+'_error'] = str(e)
        results.append(row)
    pd.DataFrame(results).to_csv(TAB / 'results_table.csv', index=False)
    if last_preds is not None:
        cm = confusion_matrix(yva, last_preds)
        import seaborn as sns; import matplotlib.pyplot as plt
        plt.figure(figsize=(5,4)); sns.heatmap(cm, annot=True, fmt='d'); plt.savefig(FIG / 'confusion.png'); plt.close()
    metrics = {'folds': len(results), 'seed': SEED}
    (MAN / 'metrics.json').write_text(json.dumps(metrics, indent=2))
else:
    print('running unsupervised sketch')
    X = DF.select_dtypes(include=[np.number]).fillna(0)
    if X.shape[1]==0:
        print('no numeric features; exiting')
    else:
        pca = PCA(n_components=min(10,X.shape[1]), random_state=SEED)
        Xr = pca.fit_transform(X)
        kmeans = KMeans(n_clusters=2, random_state=SEED)
        labs = kmeans.fit_predict(Xr)
        pd.DataFrame({'cluster': labs}).to_csv(TAB / 'clusters.csv', index=False)
        sil = silhouette_score(Xr, labs) if len(set(labs))>1 else None
        (MAN / 'metrics.json').write_text(json.dumps({'silhouette': sil, 'seed': SEED}, indent=2))

manifest = {'env': 'LOCAL', 'python': platform.python_version(), 'input': str(FEATURES), 'input_sha256': hashlib.sha256(FEATURES.read_bytes()).hexdigest(), 'seed': SEED}
(MAN / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
print('done')
