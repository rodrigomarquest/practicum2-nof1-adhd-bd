#!/usr/bin/env python3
"""
Truncate export.xml to N records for testing purposes.

Usage:
    python truncate_export_xml.py <input.xml> <output.xml> <max_records>
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

def truncate_export_xml(input_path: Path, output_path: Path, max_records: int):
    """Truncate export.xml to max_records Records."""
    
    print(f"Truncating {input_path.name} to {max_records} records...")
    
    # Parse input
    tree = ET.parse(input_path)
    root = tree.getroot()
    
    # Count and remove Records beyond max_records
    records = root.findall('.//Record')
    total = len(records)
    
    print(f"  Total records in source: {total}")
    
    # Remove records beyond max_records
    for record in records[max_records:]:
        root.remove(record)
    
    kept = len(root.findall('.//Record'))
    print(f"  Records kept: {kept}")
    
    # Write output
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"  Written to: {output_path}")
    print(f"  Saved {output_path.stat().st_size / (1024*1024):.2f} MB")
    
    return kept

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    max_recs = int(sys.argv[3])
    
    if not input_file.exists():
        print(f"ERROR: {input_file} not found")
        sys.exit(1)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    truncate_export_xml(input_file, output_file, max_recs)
