#!/usr/bin/env python
"""
Prepare raw Zepp data for biomarkers extraction.

Este script:
1. Localiza arquivos Zepp em data/raw/P000001/zepp
2. Cria symlinks ou copia para data/etl/P000001/SNAPSHOT/extracted/zepp/cloud/
3. Mantém estrutura de diretórios esperada pelo pipeline

Uso:
    python -m src.cli.prepare_zepp_data --participant P000001 --snapshot 2025-11-07 \
        --zepp-source data/raw/P000001/zepp --symlink

"""

import argparse
import sys
from pathlib import Path
import shutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def prepare_zepp_data(
    participant: str,
    snapshot: str,
    zepp_source: Path,
    target_base: Path = Path("data/etl"),
    use_symlink: bool = False,
) -> Path:
    """
    Prepare Zepp data by linking/copying to ETL structure.

    Parameters
    ----------
    participant : str
        Participant ID (e.g., P000001)
    snapshot : str
        Snapshot date (e.g., 2025-11-07)
    zepp_source : Path
        Source directory with Zepp extracted data
    target_base : Path
        Base directory for ETL (data/etl)
    use_symlink : bool
        If True, create symlinks; else copy files

    Returns
    -------
    Path
        Path to prepared zepp/cloud directory
    """
    # Target directory
    target_dir = target_base / participant / snapshot / "extracted" / "zepp" / "cloud"
    target_dir.mkdir(parents=True, exist_ok=True)

    if not zepp_source.exists():
        logger.error(f"Source not found: {zepp_source}")
        return target_dir

    logger.info(f"Source: {zepp_source}")
    logger.info(f"Target: {target_dir}")

    # Process each subdirectory
    _process_zepp_subdirs(zepp_source, target_dir, use_symlink)

    # Summary
    csv_count = sum(1 for _ in target_dir.glob("*/*.csv"))
    logger.info(f"Total files prepared: {csv_count}")

    return target_dir


def _process_zepp_subdirs(
    zepp_source: Path,
    target_dir: Path,
    use_symlink: bool,
) -> None:
    """Process each subdirectory in Zepp source."""
    for source_subdir in zepp_source.iterdir():
        if not source_subdir.is_dir():
            continue

        target_subdir = target_dir / source_subdir.name
        target_subdir.mkdir(exist_ok=True)

        _copy_or_link_files(source_subdir, target_subdir, use_symlink)


def _copy_or_link_files(
    source_subdir: Path,
    target_subdir: Path,
    use_symlink: bool,
) -> None:
    """Copy or symlink files from source to target."""
    for source_file in source_subdir.glob("*"):
        if not source_file.is_file() or source_file.suffix not in [".csv", ".json"]:
            continue

        target_file = target_subdir / source_file.name

        if target_file.exists():
            logger.debug(f"Already exists, skipping: {target_file.name}")
            continue

        if use_symlink:
            _create_symlink_or_copy(source_file, target_file, target_subdir)
        else:
            shutil.copy2(source_file, target_file)
            logger.info(f"Copy: {source_file.name} → {target_subdir.name}/")


def _create_symlink_or_copy(
    source_file: Path,
    target_file: Path,
    target_subdir: Path,
) -> None:
    """Try to create symlink, fall back to copy if not supported."""
    try:
        target_file.symlink_to(source_file)
        logger.info(f"Symlink: {source_file.name} → {target_subdir.name}/")
    except (OSError, NotImplementedError) as e:
        logger.warning(f"Symlink failed ({e}), copying instead...")
        shutil.copy2(source_file, target_file)
        logger.info(f"Copy: {source_file.name} → {target_subdir.name}/")


def main():
    parser = argparse.ArgumentParser(description="Prepare raw Zepp data for biomarkers extraction")
    parser.add_argument("--participant", default="P000001", help="Participant ID")
    parser.add_argument("--snapshot", default="2025-11-07", help="Snapshot date")
    parser.add_argument(
        "--zepp-source",
        default="data/raw/P000001/zepp",
        help="Source directory with Zepp data",
    )
    parser.add_argument(
        "--target-base",
        default="data/etl",
        help="Base directory for ETL structure",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copying files",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    zepp_source = Path(args.zepp_source)

    try:
        prepared_dir = prepare_zepp_data(
            participant=args.participant,
            snapshot=args.snapshot,
            zepp_source=zepp_source,
            target_base=Path(args.target_base),
            use_symlink=args.symlink,
        )

        print("\n=== Data Preparation Complete ===")
        print("Zepp data prepared at:")
        print(f"  {prepared_dir}")
        print("\nNext step: Run biomarkers extraction")
        print(f"  make biomarkers PID={args.participant} SNAPSHOT={args.snapshot}")

        return 0

    except Exception as e:
        logger.exception(f"Error during data preparation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
