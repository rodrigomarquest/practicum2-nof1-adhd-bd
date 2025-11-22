"""
CLI Runner for ML6/ML7 Extended Models.

Usage:
    python scripts/run_extended_models.py --which all --models all
    python scripts/run_extended_models.py --which ml6 --models rf,xgb,lgbm
    python scripts/run_extended_models.py --which ml7 --models gru,tcn
    
    # Custom participant/snapshot
    python scripts/run_extended_models.py --which all --models all \\
        --participant P000001 --snapshot 2025-11-07
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ml6_extended import run_ml6_extended
from src.models.ml7_extended import run_ml7_extended


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run ML6/ML7 extended models with temporal instability regularization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all models
    python scripts/run_extended_models.py --which all --models all
    
    # Run only ML6 models
    python scripts/run_extended_models.py --which ml6 --models rf,xgb,lgbm,svm
    
    # Run only ML7 models
    python scripts/run_extended_models.py --which ml7 --models gru,tcn,mlp
    
    # Run specific models
    python scripts/run_extended_models.py --which ml6 --models xgb,lgbm
    python scripts/run_extended_models.py --which ml7 --models gru
        """
    )
    
    parser.add_argument(
        '--which',
        choices=['ml6', 'ml7', 'all'],
        required=True,
        help='Which extended models to run (ml6, ml7, or all)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        default='all',
        help='Comma-separated list of models (e.g., "rf,xgb,lgbm" or "gru,tcn"). Use "all" for all models.'
    )
    
    parser.add_argument(
        '--participant',
        type=str,
        default='P000001',
        help='Participant ID (default: P000001)'
    )
    
    parser.add_argument(
        '--snapshot',
        type=str,
        default='2025-11-07',
        help='Snapshot date (default: 2025-11-07)'
    )
    
    parser.add_argument(
        '--no-saliency',
        action='store_true',
        help='Skip gradient saliency computation for ML7 (faster)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # Construct paths
    participant = args.participant
    snapshot = args.snapshot
    
    ml6_csv = f'data/ai/{participant}/{snapshot}/ml6/features_daily_ml6.csv'
    ml7_csv = f'data/etl/{participant}/{snapshot}/joined/features_daily_labeled.csv'
    cv_summary = f'data/ai/{participant}/{snapshot}/ml6/cv_summary.json'
    segments_csv = f'data/etl/{participant}/{snapshot}/segment_autolog.csv'
    
    ml6_output = f'data/ai/{participant}/{snapshot}/ml6_ext'
    ml7_output = f'data/ai/{participant}/{snapshot}/ml7_ext'
    
    # Validate paths
    for path, name in [
        (ml6_csv, 'ML6 features'),
        (ml7_csv, 'ML7 features'),
        (cv_summary, 'CV summary'),
        (segments_csv, 'Segments')
    ]:
        if not Path(path).exists():
            logger.error(f"{name} file not found: {path}")
            logger.error("Please run ML6 baseline first or check participant/snapshot IDs")
            return 1
    
    # Parse model list
    if args.models == 'all':
        ml6_models = ['rf', 'xgb', 'lgbm', 'svm']
        ml7_models = ['gru', 'tcn', 'mlp']
    else:
        model_list = [m.strip().lower() for m in args.models.split(',')]
        
        # Validate models
        ml6_valid = {'rf', 'xgb', 'lgbm', 'svm'}
        ml7_valid = {'gru', 'tcn', 'mlp'}
        
        ml6_models = [m for m in model_list if m in ml6_valid]
        ml7_models = [m for m in model_list if m in ml7_valid]
        
        invalid = [m for m in model_list if m not in ml6_valid and m not in ml7_valid]
        if invalid:
            logger.error(f"Invalid model names: {invalid}")
            logger.error(f"Valid ML6 models: {ml6_valid}")
            logger.error(f"Valid ML7 models: {ml7_valid}")
            return 1
    
    # Run ML6 extended
    if args.which in ['ml6', 'all'] and ml6_models:
        logger.info("\n" + "="*80)
        logger.info(f"RUNNING ML6 EXTENDED MODELS: {', '.join(ml6_models).upper()}")
        logger.info("="*80 + "\n")
        
        try:
            run_ml6_extended(
                ml6_csv=ml6_csv,
                cv_summary_json=cv_summary,
                segments_csv=segments_csv,
                output_dir=ml6_output,
                models=ml6_models
            )
            logger.info(f"[OK] ML6 extended models complete. Results: {ml6_output}")
        except Exception as e:
            logger.error(f"❌ ML6 extended models failed: {e}", exc_info=True)
            return 1
    
    # Run ML7 extended
    if args.which in ['ml7', 'all'] and ml7_models:
        logger.info("\n" + "="*80)
        logger.info(f"RUNNING ML7 EXTENDED MODELS: {', '.join(ml7_models).upper()}")
        logger.info("="*80 + "\n")
        
        try:
            run_ml7_extended(
                features_csv=ml7_csv,
                cv_summary_json=cv_summary,
                output_dir=ml7_output,
                models=ml7_models,
                compute_saliency=not args.no_saliency
            )
            logger.info(f"[OK] ML7 extended models complete. Results: {ml7_output}")
        except Exception as e:
            logger.error(f"❌ ML7 extended models failed: {e}", exc_info=True)
            return 1
    
    logger.info("\n" + "="*80)
    logger.info("🎉 ALL EXTENDED MODELS COMPLETE")
    logger.info("="*80)
    logger.info(f"\nResults:")
    if args.which in ['ml6', 'all'] and ml6_models:
        logger.info(f"  ML6: {ml6_output}")
    if args.which in ['ml7', 'all'] and ml7_models:
        logger.info(f"  ML7: {ml7_output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
