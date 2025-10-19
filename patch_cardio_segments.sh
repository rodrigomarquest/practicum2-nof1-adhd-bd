#!/usr/bin/env bash
set -euo pipefail

FILE="etl_modules/cardiovascular/cardio_etl.py"
cp -f "$FILE" "$FILE.bak.$(date +%Y%m%d%H%M%S)"

python - <<'PY'
from pathlib import Path
p = Path("etl_modules/cardiovascular/cardio_etl.py")
src = p.read_text(encoding="utf-8")

needle = "    segments = read_csv_if_exists(os.path.join(snapshot_dir, \"version_log_enriched.csv\"))"
if needle not in src:
    print("⚠️ Trecho esperado não encontrado; nada alterado.")
    raise SystemExit(0)

patch = '''
    segments = read_csv_if_exists(os.path.join(snapshot_dir, "version_log_enriched.csv"))
    # --- Normalize segments: accept either daily 'date' or ranged 'start'/'end' ---
    if segments is not None and not segments.empty:
        import pandas as pd
        cols = {c.lower(): c for c in segments.columns}
        if "date" in [c.lower() for c in segments.columns]:
            # Case A: already daily
            segments["date"] = pd.to_datetime(segments[cols.get("date")]).dt.date
        elif ("start" in cols) and ("end" in cols) and ("segment_id" in cols):
            # Case B: ranges -> expand to daily
            seg = segments.rename(columns={cols["start"]: "start", cols["end"]: "end", cols["segment_id"]: "segment_id"}).copy()
            seg["start"] = pd.to_datetime(seg["start"])
            seg["end"] = pd.to_datetime(seg["end"])
            rows = []
            for _, r in seg.iterrows():
                if pd.isna(r["start"]) or pd.isna(r["end"]):
                    continue
                # inclusive range
                days = pd.date_range(r["start"].normalize(), r["end"].normalize(), freq="D")
                for d in days:
                    rows.append({"date": d.date(), "segment_id": int(r["segment_id"]) if pd.notna(r["segment_id"]) else None})
            segments = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date","segment_id"])
        else:
            # Unknown format -> drop segments gracefully
            segments = None
    else:
        segments = None
'''
src = src.replace(needle, patch)

p.write_text(src, encoding="utf-8")
print("✅ cardio_etl.py patched to support start/end segment ranges.")
PY

echo "Done."
