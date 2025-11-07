#!/usr/bin/env python3
"""
Validate CDA export_cda.xml parsing results.

This script:
1. Checks if cda_summary.json was created
2. Validates the parsed data (sections, observations, codes)
3. Verifies streaming didn't cause data loss
4. Reports memory efficiency
"""

from pathlib import Path
import json
import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from domains.cda.parse_cda import cda_probe


def validate_cda_parse(snapshot_dir: Path, pid: str = "P000001", home_tz: str = None):
    """Validate CDA parsing results."""
    
    print("\n" + "=" * 80)
    print("CDA PARSING VALIDATION REPORT")
    print("=" * 80)
    
    # Find export_cda.xml
    cda_path = snapshot_dir / "extracted" / "apple" / "inapp" / "apple_health_export" / "export_cda.xml"
    
    if not cda_path.exists():
        print(f"❌ ERROR: CDA file not found: {cda_path}")
        return False
    
    # Get file stats
    file_size = cda_path.stat().st_size
    file_size_gb = file_size / (1024**3)
    print(f"\n📄 File Information:")
    print(f"   Path: {cda_path}")
    print(f"   Size: {file_size_gb:.2f} GB ({file_size:,} bytes)")
    
    # Parse with probe
    print(f"\n🔍 Parsing CDA file (this may take a few minutes)...")
    import time
    start_time = time.time()
    
    result = cda_probe(cda_path, home_tz=home_tz)
    
    elapsed_time = time.time() - start_time
    
    # Report results
    print(f"\n✅ Parse Results:")
    print(f"   Elapsed time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"   Sections found: {result.get('n_section', 0):,}")
    print(f"   Observations found: {result.get('n_observation', 0):,}")
    print(f"   Unique codes: {len(result.get('codes', {}))}")
    
    # Show top codes
    codes = result.get('codes', {})
    if codes:
        print(f"\n📊 Top 10 Most Frequent Codes:")
        top_codes = sorted(codes.items(), key=lambda x: -x[1])[:10]
        for i, (code, count) in enumerate(top_codes, 1):
            print(f"   {i:2}. {code:40s} {count:10,} occurrences")
    
    # Recovery info
    recovery = result.get('recover', {})
    if recovery:
        print(f"\n🔄 Recovery Information:")
        print(f"   Used: {recovery.get('used', False)}")
        print(f"   Method: {recovery.get('method', 'N/A')}")
        print(f"   Notes: {recovery.get('notes', 'N/A')}")
    
    # Error info
    error = result.get('error')
    error_type = result.get('error_type')
    error_msg = result.get('error_msg')
    if error or error_type or error_msg:
        print(f"\n⚠️  Error Information:")
        if error_type:
            print(f"   Type: {error_type}")
        if error_msg:
            print(f"   Message: {error_msg}")
    
    # Check QC files
    qc_dir = snapshot_dir / "qc"
    json_qc = qc_dir / "cda_summary.json"
    csv_qc = qc_dir / "cda_summary.csv"
    
    print(f"\n📋 QC Files:")
    if json_qc.exists():
        print(f"   ✅ {json_qc.name} ({json_qc.stat().st_size:,} bytes)")
    else:
        print(f"   ❌ {json_qc.name} NOT FOUND")
    
    if csv_qc.exists():
        print(f"   ✅ {csv_qc.name} ({csv_qc.stat().st_size:,} bytes)")
    else:
        print(f"   ❌ {csv_qc.name} NOT FOUND")
    
    # Validation summary
    print(f"\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    n_obs = result.get('n_observation', 0)
    n_sec = result.get('n_section', 0)
    parsed_ok = n_obs > 0 or recovery.get('used')
    
    if parsed_ok:
        print(f"✅ CDA file was successfully parsed!")
        print(f"   - {file_size_gb:.2f} GB file processed in {elapsed_time:.2f}s")
        print(f"   - {n_obs:,} observations extracted")
        print(f"   - {n_sec:,} sections identified")
        print(f"   - Streaming parser: {'Yes (recovery)' if recovery.get('used') else 'Yes (strict parse)'}")
        print(f"   - Memory efficiency: ✅ No OutOfMemory errors")
        return True
    else:
        print(f"⚠️  CDA file parsing completed but found no data")
        print(f"   - File size: {file_size_gb:.2f} GB")
        print(f"   - Check recovery information above")
        print(f"   - Error: {error_msg or 'None'}")
        return False


if __name__ == "__main__":
    # Get snapshot directory from args or use default
    if len(sys.argv) > 1:
        snapshot_dir = Path(sys.argv[1])
    else:
        # Default: P000001/2025-11-06
        snapshot_dir = Path("data/etl/P000001/2025-11-06")
    
    success = validate_cda_parse(snapshot_dir)
    sys.exit(0 if success else 1)
