#!/usr/bin/env bash
# Quick start script for complete biomarkers pipeline

set -e

PARTICIPANT="${1:-P000001}"
SNAPSHOT="${2:-2025-11-07}"
ZEPP_PASSWORD="${3:-}"

echo "======================================"
echo "Clinical Biomarkers Pipeline"
echo "======================================"
echo "Participant: $PARTICIPANT"
echo "Snapshot: $SNAPSHOT"
echo ""

# Step 1: Prepare Zepp data
echo "[1/5] Preparing Zepp data from raw directory..."
python scripts/prepare_zepp_data.py \
    --participant "$PARTICIPANT" \
    --snapshot "$SNAPSHOT" \
    --zepp-source "data/raw/$PARTICIPANT/zepp" \
    --target-base "data/etl" \
    --symlink

# Step 2: Extract biomarkers
echo ""
echo "[2/5] Extracting clinical biomarkers (Tier 1+2+X)..."
python scripts/extract_biomarkers.py \
    --participant "$PARTICIPANT" \
    --snapshot "$SNAPSHOT" \
    --data-dir "data/etl/$PARTICIPANT/$SNAPSHOT/extracted" \
    --output-dir "data/etl" \
    --cutoff-months 30 \
    --verbose 1

# Step 3: Generate heuristic labels
echo ""
echo "[3/5] Generating clinical heuristic labels..."
python build_heuristic_labels.py \
    --participant "$PARTICIPANT" \
    --snapshot "$SNAPSHOT"

# Step 4: Run NB2 baselines
echo ""
echo "[4/5] Running NB2 baseline models (temporal folds)..."
python run_nb2_engage7.py \
    --pid "$PARTICIPANT" \
    --snapshot "$SNAPSHOT" \
    --n-folds 6 \
    --train-days 120 \
    --val-days 60 \
    --class-weight balanced \
    --seed 42 \
    --plots 1 \
    --save-all 1 \
    --verbose 1

# Step 5: Summary
echo ""
echo "[5/5] Pipeline complete!"
echo ""
echo "Output files:"
echo "  - Biomarkers: data/etl/$PARTICIPANT/$SNAPSHOT/joined/joined_features_daily_biomarkers.csv"
echo "  - Labels: data/etl/$PARTICIPANT/$SNAPSHOT/joined/heuristic_labels.csv"
echo "  - NB2 results: reports/nb2_beiwe_$PARTICIPANT.html"
echo ""
echo "Next steps:"
echo "  1. Review biomarkers in data/etl/$PARTICIPANT/$SNAPSHOT/joined/"
echo "  2. Check label distribution in heuristic_labels.csv"
echo "  3. Analyze model performance in reports/nb2_beiwe_$PARTICIPANT.html"
echo ""
