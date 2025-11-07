"""
Unify Apple + Zepp daily data into canonical schema.

Purpose:
    Merge daily measurements from Apple Watch and Zepp into a single,
    canonical daily feature matrix with consistent column names and QC flags.

Canonical schema:
    date, sleep_total_h, sleep_efficiency, apple_hr_mean, apple_hr_max,
    apple_hrv_rmssd, steps, exercise_min, stand_hours, move_kcal,
    missing_sleep, missing_cardio, missing_activity,
    source_sleep, source_cardio, source_activity
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Canonical column names
CANONICAL_SCHEMA = {
    'date': 'date',
    'sleep_total_h': float,
    'sleep_efficiency': float,
    'apple_hr_mean': float,
    'apple_hr_max': float,
    'apple_hrv_rmssd': float,
    'steps': float,
    'exercise_min': float,
    'stand_hours': float,
    'move_kcal': float,
    'missing_sleep': int,
    'missing_cardio': int,
    'missing_activity': int,
    'source_sleep': str,
    'source_cardio': str,
    'source_activity': str,
}

# Column name mappings
SLEEP_MAPPINGS = {
    'total': ['sleep_total_h', 'apple_sleep_total_h', 'zepp_slp_total_h', 'sleep_hours'],
    'efficiency': ['sleep_eff', 'sleep_efficiency', 'apple_sleep_eff', 'zepp_slp_eff'],
}

CARDIO_MAPPINGS = {
    'hr_mean': ['apple_hr_mean', 'zepp_hr_mean', 'hr_mean'],
    'hr_max': ['apple_hr_max', 'hr_max', 'zepp_hr_max'],
    'hrv_rmssd': ['apple_hrv_rmssd', 'zepp_hrv_rmssd', 'hrv_rmssd'],
}

ACTIVITY_MAPPINGS = {
    'steps': ['steps', 'apple_steps', 'zepp_steps'],
    'exercise_min': ['exercise_min', 'apple_exercise_min', 'zepp_exercise_min'],
    'stand_hours': ['stand_hours', 'apple_stand_hours', 'zepp_stand_hours'],
    'move_kcal': ['move_kcal', 'apple_move_kcal', 'zepp_kcal'],
}


def find_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Find first column that exists in dataframe."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_efficiency(val: float) -> float:
    """Normalize sleep efficiency to 0..1 range."""
    if pd.isna(val):
        return np.nan
    if val > 1.5:  # Assume it's 0-100 scale
        return val / 100.0
    return val


def load_apple_data(apple_dir: Path) -> Optional[pd.DataFrame]:
    """Load Apple Watch daily aggregates."""
    logger.info(f"Loading Apple data from {apple_dir}")
    
    if not apple_dir.exists():
        logger.warning(f"Apple directory not found: {apple_dir}")
        return None
    
    csv_files = list(apple_dir.glob('*.csv'))
    if not csv_files:
        logger.warning(f"No CSV files in {apple_dir}")
        return None
    
    # Try to load all CSVs and merge
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'date' in df.columns:
                dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not load {csv_file}: {e}")
    
    if not dfs:
        return None
    
    # Merge on date
    apple_df = dfs[0]
    for df in dfs[1:]:
        apple_df = apple_df.merge(df, on='date', how='outer')
    
    apple_df['date'] = pd.to_datetime(apple_df['date'])
    logger.info(f"✓ Loaded Apple data: {len(apple_df)} days")
    return apple_df


def load_zepp_data(zepp_dir: Path) -> Optional[pd.DataFrame]:
    """Load Zepp daily aggregates."""
    logger.info(f"Loading Zepp data from {zepp_dir}")
    
    if not zepp_dir.exists():
        logger.warning(f"Zepp directory not found: {zepp_dir}")
        return None
    
    csv_files = list(zepp_dir.glob('*.csv'))
    if not csv_files:
        logger.warning(f"No CSV files in {zepp_dir}")
        return None
    
    # Try to load all CSVs and merge
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'date' in df.columns:
                dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not load {csv_file}: {e}")
    
    if not dfs:
        return None
    
    # Merge on date
    zepp_df = dfs[0]
    for df in dfs[1:]:
        zepp_df = zepp_df.merge(df, on='date', how='outer')
    
    zepp_df['date'] = pd.to_datetime(zepp_df['date'])
    logger.info(f"✓ Loaded Zepp data: {len(zepp_df)} days")
    return zepp_df


def extract_canonical_row(row: pd.Series, source: str) -> Dict:
    """Extract canonical columns from a single row."""
    result = {'source_sleep': None, 'source_cardio': None, 'source_activity': None}
    
    # Sleep
    sleep_total_col = find_column(row.index.to_frame().T, SLEEP_MAPPINGS['total'])
    sleep_eff_col = find_column(row.index.to_frame().T, SLEEP_MAPPINGS['efficiency'])
    
    if sleep_total_col and not pd.isna(row[sleep_total_col]):
        result['sleep_total_h'] = float(row[sleep_total_col])
        result['source_sleep'] = source
        result['missing_sleep'] = 0
    else:
        result['sleep_total_h'] = np.nan
        result['missing_sleep'] = 1
    
    if sleep_eff_col and not pd.isna(row[sleep_eff_col]):
        result['sleep_efficiency'] = normalize_efficiency(float(row[sleep_eff_col]))
        if not result['source_sleep']:
            result['source_sleep'] = source
        result['missing_sleep'] = 0
    else:
        result['sleep_efficiency'] = np.nan
    
    # Cardio
    hr_mean_col = find_column(row.index.to_frame().T, CARDIO_MAPPINGS['hr_mean'])
    hr_max_col = find_column(row.index.to_frame().T, CARDIO_MAPPINGS['hr_max'])
    hrv_col = find_column(row.index.to_frame().T, CARDIO_MAPPINGS['hrv_rmssd'])
    
    cardio_count = 0
    if hr_mean_col and not pd.isna(row[hr_mean_col]):
        result['apple_hr_mean'] = float(row[hr_mean_col])
        result['source_cardio'] = source
        cardio_count += 1
    else:
        result['apple_hr_mean'] = np.nan
    
    if hr_max_col and not pd.isna(row[hr_max_col]):
        result['apple_hr_max'] = float(row[hr_max_col])
        if not result['source_cardio']:
            result['source_cardio'] = source
        cardio_count += 1
    else:
        result['apple_hr_max'] = np.nan
    
    if hrv_col and not pd.isna(row[hrv_col]):
        result['apple_hrv_rmssd'] = float(row[hrv_col])
        if not result['source_cardio']:
            result['source_cardio'] = source
        cardio_count += 1
    else:
        result['apple_hrv_rmssd'] = np.nan
    
    result['missing_cardio'] = 1 if cardio_count == 0 else 0
    
    # Activity
    steps_col = find_column(row.index.to_frame().T, ACTIVITY_MAPPINGS['steps'])
    exercise_col = find_column(row.index.to_frame().T, ACTIVITY_MAPPINGS['exercise_min'])
    stand_col = find_column(row.index.to_frame().T, ACTIVITY_MAPPINGS['stand_hours'])
    kcal_col = find_column(row.index.to_frame().T, ACTIVITY_MAPPINGS['move_kcal'])
    
    activity_count = 0
    if steps_col and not pd.isna(row[steps_col]):
        result['steps'] = float(row[steps_col])
        result['source_activity'] = source
        activity_count += 1
    else:
        result['steps'] = np.nan
    
    if exercise_col and not pd.isna(row[exercise_col]):
        result['exercise_min'] = float(row[exercise_col])
        if not result['source_activity']:
            result['source_activity'] = source
        activity_count += 1
    else:
        result['exercise_min'] = np.nan
    
    if stand_col and not pd.isna(row[stand_col]):
        result['stand_hours'] = float(row[stand_col])
        if not result['source_activity']:
            result['source_activity'] = source
        activity_count += 1
    else:
        result['stand_hours'] = np.nan
    
    if kcal_col and not pd.isna(row[kcal_col]):
        result['move_kcal'] = float(row[kcal_col])
        if not result['source_activity']:
            result['source_activity'] = source
        activity_count += 1
    else:
        result['move_kcal'] = np.nan
    
    result['missing_activity'] = 1 if activity_count == 0 else 0
    
    return result


def merge_apple_zepp(
    apple_df: Optional[pd.DataFrame],
    zepp_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge Apple and Zepp data with preference: Apple > Zepp."""
    
    logger.info("Merging Apple and Zepp data (preference: Apple > Zepp)")
    
    if apple_df is None and zepp_df is None:
        raise ValueError("No Apple or Zepp data available")
    
    # Collect all dates
    all_dates = set()
    if apple_df is not None:
        all_dates.update(apple_df['date'].unique())
    if zepp_df is not None:
        all_dates.update(zepp_df['date'].unique())
    
    all_dates = sorted(list(all_dates))
    
    # Process each date
    rows = []
    for date in all_dates:
        row_data = {'date': date}
        
        # Try Apple first
        apple_row = None
        if apple_df is not None:
            mask = apple_df['date'] == date
            if mask.any():
                apple_row = apple_df[mask].iloc[0]
        
        # Try Zepp
        zepp_row = None
        if zepp_df is not None:
            mask = zepp_df['date'] == date
            if mask.any():
                zepp_row = zepp_df[mask].iloc[0]
        
        # Extract and merge
        if apple_row is not None:
            apple_data = extract_canonical_row(apple_row, 'apple')
            row_data.update(apple_data)
        
        if zepp_row is not None:
            zepp_data = extract_canonical_row(zepp_row, 'zepp')
            # Only fill in missing Apple values with Zepp
            for key, val in zepp_data.items():
                if key not in row_data or pd.isna(row_data.get(key)):
                    row_data[key] = val
                elif key.startswith('source_') and row_data[key] is None:
                    row_data[key] = val
        
        # Fill any remaining NaNs in source columns
        if row_data.get('source_sleep') is None:
            row_data['source_sleep'] = 'none'
        if row_data.get('source_cardio') is None:
            row_data['source_cardio'] = 'none'
        if row_data.get('source_activity') is None:
            row_data['source_activity'] = 'none'
        
        rows.append(row_data)
    
    unified_df = pd.DataFrame(rows)
    unified_df = unified_df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"✓ Merged data: {len(unified_df)} unique dates")
    
    return unified_df


def validate_and_report(df: pd.DataFrame) -> None:
    """Print QC report."""
    logger.info("\n" + "="*80)
    logger.info("UNIFICATION REPORT")
    logger.info("="*80)
    
    logger.info(f"\nDate span: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Total days: {len(df)}")
    logger.info(f"Unique dates: {df['date'].nunique()}")
    
    assert df['date'].nunique() == len(df), "Duplicate dates found!"
    
    # Missing data report
    logger.info(f"\nMissing data:")
    logger.info(f"  missing_sleep: {df['missing_sleep'].sum()} ({100*df['missing_sleep'].mean():.1f}%)")
    logger.info(f"  missing_cardio: {df['missing_cardio'].sum()} ({100*df['missing_cardio'].mean():.1f}%)")
    logger.info(f"  missing_activity: {df['missing_activity'].sum()} ({100*df['missing_activity'].mean():.1f}%)")
    
    # Source breakdown
    logger.info(f"\nData sources:")
    for col in ['source_sleep', 'source_cardio', 'source_activity']:
        logger.info(f"  {col}:")
        for source, count in df[col].value_counts().items():
            pct = 100 * count / len(df)
            logger.info(f"    {source}: {count} ({pct:.1f}%)")
    
    # Zepp date warning
    if 'source_sleep' in df.columns:
        zepp_sleep_mask = df['source_sleep'] == 'zepp'
        if zepp_sleep_mask.any():
            max_zepp_sleep = df[zepp_sleep_mask]['date'].max()
            if pd.Timestamp('2024-06-30') < max_zepp_sleep:
                logger.warning(f"⚠️  Zepp sleep data found after 2024-06: {max_zepp_sleep}")
    
    logger.info(f"\nLast 5 rows:")
    logger.info(df.tail(5).to_string())
    
    logger.info("="*80 + "\n")


def unify_daily(
    apple_dir: Path,
    zepp_dir: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Main function: load, merge, validate, save."""
    
    logger.info("Starting daily unification")
    
    # Load data
    apple_df = load_apple_data(apple_dir)
    zepp_df = load_zepp_data(zepp_dir)
    
    # Merge
    unified_df = merge_apple_zepp(apple_df, zepp_df)
    
    # Validate
    validate_and_report(unified_df)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unified_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved unified data to {output_path}")
    
    return unified_df
