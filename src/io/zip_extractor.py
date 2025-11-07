"""
ZIP Discovery & Extraction Utility

Purpose:
    Discover ZIP files in data/raw/ recursively, classify by vendor (Apple/Zepp),
    extract to data/extracted/ with progress bar, handle AES-encrypted Zepp archives.

Reuses patterns from src/etl_pipeline.py (pyzipper, progress_bar).
"""

import os
import sys
import zipfile
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Try to import pyzipper (for AES-encrypted Zepp ZIPs)
try:
    import pyzipper
    HAS_PYZIPPER = True
except ImportError:
    HAS_PYZIPPER = False

logger = logging.getLogger(__name__)


def progress_bar(total: int = 100, desc: str = "", unit: str = "it", leave: bool = True):
    """Simple progress bar context manager (compatible with tqdm API)."""
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit=unit, leave=leave, ncols=80)
    except ImportError:
        # Fallback: no-op context manager
        class NoOpBar:
            def update(self, n=1):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return NoOpBar()


def classify_vendor(zip_path: Path) -> str:
    """Classify ZIP as 'apple' or 'zepp' based on path or filename."""
    path_str = str(zip_path).lower()
    
    # Heuristics
    if "apple" in path_str:
        return "apple"
    elif any(x in path_str for x in ["zepp", "gtr", "helio"]):
        return "zepp"
    
    # Fallback: check filename patterns
    name_lower = zip_path.name.lower()
    if name_lower.startswith("apple") or "health" in name_lower:
        return "apple"
    
    # Default to zepp if ambiguous
    return "zepp"


def discover_zips(raw_dir: Path = None) -> List[Dict]:
    """
    Discover all ZIPs in data/raw/ recursively.
    
    Returns:
        List of dicts with keys: path, vendor, participant, size_mb
    """
    if raw_dir is None:
        raw_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    
    if not raw_dir.exists():
        logger.warning(f"Raw directory not found: {raw_dir}")
        return []
    
    zips = []
    logger.info(f"🔍 Scanning for ZIPs in {raw_dir}...")
    
    for zip_path in raw_dir.rglob("*.zip"):
        vendor = classify_vendor(zip_path)
        # Extract participant from path structure: /data/raw/P000001/vendor/subfolder/file.zip
        # So we need parent.parent.parent.name
        participant = zip_path.parent.parent.parent.name if zip_path.parent.parent.parent.name.startswith("P") else "unknown"
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        
        zips.append({
            "path": zip_path,
            "vendor": vendor,
            "participant": participant,
            "size_mb": round(size_mb, 2),
        })
        logger.info(f"  Found: {vendor:5s} / {participant:8s} / {zip_path.name:50s} ({size_mb:6.2f} MB)")
    
    logger.info(f"✓ Discovered {len(zips)} ZIP files")
    return zips


def extract_apple_zip(zip_path: Path, target_dir: Path, progress: bool = True) -> Tuple[int, List[str]]:
    """
    Extract Apple ZIP (unencrypted, standard format).
    
    Returns:
        (num_files, warnings)
    """
    warnings = []
    num_files = 0
    
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            desc = f"Apple {zip_path.name[:30]:30s}"
            
            with progress_bar(total=len(members), desc=desc, unit="files") as bar:
                for member in members:
                    target = target_dir / Path(member)
                    
                    # Create directories
                    if member.endswith('/'):
                        target.mkdir(parents=True, exist_ok=True)
                        bar.update(1)
                        continue
                    
                    # Extract file
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        data = zf.read(member)
                        with open(target, "wb") as f:
                            f.write(data)
                        num_files += 1
                    except Exception as e:
                        warnings.append(f"Failed to extract {member}: {e}")
                    
                    bar.update(1)
    except Exception as e:
        warnings.append(f"Failed to extract Apple ZIP {zip_path}: {e}")
        logger.error(f"❌ Apple extraction error: {e}")
    
    return num_files, warnings


def extract_zepp_zip(
    zip_path: Path,
    target_dir: Path,
    password: Optional[str] = None,
    progress: bool = True
) -> Tuple[int, List[str]]:
    """
    Extract Zepp ZIP (may be AES-encrypted).
    
    Returns:
        (num_files, warnings)
    """
    warnings = []
    num_files = 0
    
    # Get password from env if not provided
    if password is None:
        password = os.environ.get("ZEPP_ZIP_PASSWORD")
    
    if not password:
        msg = f"Zepp ZIP password not provided and ZEPP_ZIP_PASSWORD not set: {zip_path}"
        logger.warning(f"⚠️  {msg}")
        return 0, [msg]
    
    try:
        if HAS_PYZIPPER:
            # Use pyzipper for AES support
            with pyzipper.AESZipFile(str(zip_path)) as zf:
                zf.pwd = password.encode("utf-8") if isinstance(password, str) else password
                members = zf.namelist()
                desc = f"Zepp {zip_path.name[:30]:30s}"
                
                with progress_bar(total=len(members), desc=desc, unit="files") as bar:
                    for member in members:
                        target = target_dir / Path(member)
                        
                        if member.endswith('/'):
                            target.mkdir(parents=True, exist_ok=True)
                            bar.update(1)
                            continue
                        
                        target.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            data = zf.read(member)
                            with open(target, "wb") as f:
                                f.write(data)
                            num_files += 1
                        except Exception as e:
                            warnings.append(f"Failed to extract {member}: {e}")
                        
                        bar.update(1)
        else:
            # Fallback to standard zipfile (won't work for AES)
            logger.warning("⚠️  pyzipper not available; trying standard zipfile (may fail for encrypted archives)")
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = zf.namelist()
                desc = f"Zepp {zip_path.name[:30]:30s}"
                
                with progress_bar(total=len(members), desc=desc, unit="files") as bar:
                    for member in members:
                        target = target_dir / Path(member)
                        
                        if member.endswith('/'):
                            target.mkdir(parents=True, exist_ok=True)
                            bar.update(1)
                            continue
                        
                        target.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            data = zf.read(member)
                            with open(target, "wb") as f:
                                f.write(data)
                            num_files += 1
                        except Exception as e:
                            warnings.append(f"Failed to extract {member}: {e}")
                        
                        bar.update(1)
    except RuntimeError as e:
        msg = f"Zepp ZIP password error (bad password?): {e}"
        logger.error(f"❌ {msg}")
        warnings.append(msg)
    except Exception as e:
        msg = f"Failed to extract Zepp ZIP {zip_path}: {e}"
        logger.error(f"❌ {msg}")
        warnings.append(msg)
    
    return num_files, warnings


def extract_all_zips(
    raw_dir: Path = None,
    extracted_dir: Path = None,
    participant: str = None,
    zepp_password: Optional[str] = None,
    dry_run: bool = False,
) -> Dict:
    """
    Discover and extract all ZIPs.
    
    Args:
        raw_dir: Source directory (default: data/raw)
        extracted_dir: Target directory (default: data/extracted)
        participant: Filter to specific participant (e.g., "P000001")
        zepp_password: Override ZEPP_ZIP_PASSWORD
        dry_run: Don't extract, just report
    
    Returns:
        Summary dict with stats
    """
    if raw_dir is None:
        raw_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    
    if extracted_dir is None:
        extracted_dir = Path(__file__).parent.parent.parent / "data" / "extracted"
    
    # Discover ZIPs
    all_zips = discover_zips(raw_dir)
    
    # Filter by participant if specified
    if participant:
        all_zips = [z for z in all_zips if z["participant"] == participant or z["participant"] == "unknown"]
    
    if dry_run:
        print(f"\nDRY RUN: Would extract {len(all_zips)} ZIPs to {extracted_dir}")
        for z in all_zips:
            print(f"  - {z['vendor']:5s} {z['participant']:8s} {z['path'].name:40s} ({z['size_mb']:6.2f} MB)")
        return {"dry_run": True, "num_zips": len(all_zips), "num_extracted": 0, "extracted_files": 0}
    
    # Extract
    extracted_dir.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "raw_dir": str(raw_dir),
        "extracted_dir": str(extracted_dir),
        "num_zips_discovered": len(all_zips),
        "num_extracted": 0,
        "total_files": 0,
        "by_vendor": {},
        "by_participant": {},
        "warnings": [],
    }
    
    for zip_info in all_zips:
        zip_path = zip_info["path"]
        vendor = zip_info["vendor"]
        participant_id = zip_info["participant"]
        
        # Create target dir: data/extracted/{vendor}/{participant}/
        target_dir = extracted_dir / vendor / participant_id
        target_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n📦 Extracting {vendor} → {target_dir}...")
        
        if vendor == "apple":
            num_files, warnings = extract_apple_zip(zip_path, target_dir)
        else:  # zepp
            num_files, warnings = extract_zepp_zip(zip_path, target_dir, password=zepp_password)
        
        stats["num_extracted"] += 1 if num_files > 0 else 0
        stats["total_files"] += num_files
        
        if vendor not in stats["by_vendor"]:
            stats["by_vendor"][vendor] = {"count": 0, "files": 0}
        stats["by_vendor"][vendor]["count"] += 1
        stats["by_vendor"][vendor]["files"] += num_files
        
        if participant_id not in stats["by_participant"]:
            stats["by_participant"][participant_id] = {"count": 0, "files": 0}
        stats["by_participant"][participant_id]["count"] += 1
        stats["by_participant"][participant_id]["files"] += num_files
        
        if warnings:
            logger.warning(f"⚠️  Warnings: {len(warnings)} issues during extraction")
            stats["warnings"].extend(warnings)
        
        logger.info(f"✓ Extracted {num_files} files to {target_dir}")
    
    logger.info(f"\n✅ Extraction complete: {stats['num_extracted']}/{stats['num_zips_discovered']} ZIPs, {stats['total_files']} files total")
    
    return stats


if __name__ == "__main__":
    import json
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    
    # Parse args
    import argparse
    parser = argparse.ArgumentParser(description="Extract all ZIPs from data/raw to data/extracted")
    parser.add_argument("--participant", help="Filter to specific participant (e.g., P000001)")
    parser.add_argument("--zepp-password", help="Zepp ZIP password (overrides env)")
    parser.add_argument("--dry-run", action="store_true", help="Don't extract, just report")
    args = parser.parse_args()
    
    stats = extract_all_zips(
        participant=args.participant,
        zepp_password=args.zepp_password,
        dry_run=args.dry_run,
    )
    
    print("\n" + "=" * 80)
    print("ZIP Extraction Summary")
    print("=" * 80)
    print(json.dumps(stats, indent=2, default=str))
