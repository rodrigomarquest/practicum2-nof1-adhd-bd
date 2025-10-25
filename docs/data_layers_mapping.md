# 🗂️ Data Layers Mapping — ETL → Modeling

**Project:** N-of-1 ADHD + BD (Practicum Part 2)
**Participant:** `P000001` (extends to `P00000n`)
**Timezone:** `Europe/Dublin`
**Snapshot root:**
`data/etl/<PARTICIPANT>/snapshots/<YYYY-MM-DD>/`

> **Note:** For snapshot `2025-10-22`, **`fused/` is an alias of `joined/`**.
> Future releases may diverge, with `fused/` holding feature-complete, model-ready tables.

---

## 1️⃣ Layer definitions

| Layer           | Path                                          | Purpose                                                           | Produced by                 | Write policy              |
| :-------------- | :-------------------------------------------- | :---------------------------------------------------------------- | :-------------------------- | :------------------------ |
| **raw/**        | `raw/`                                        | Untouched exports/backups from devices & apps (Apple, iOS, Zepp). | Manual exports / collectors | **Append-only**           |
| **extracted/**  | `extracted/`                                  | Parsed / intermediate CSV or SQLite from raw.                     | Extract scripts             | Overwrite per snapshot    |
| **normalized/** | `normalized/`                                 | Time-zone & unit harmonized (long→tidy).                          | ETL A6                      | Overwrite; version-locked |
| **processed/**  | `processed/`                                  | Daily aggregates per domain (HR, HRV, Sleep, Usage).              | ETL A7                      | Overwrite; idempotent     |
| **joined/**     | `joined/`                                     | Cross-source daily join (current model input).                    | ETL A8                      | Overwrite; current input  |
| **fused/**      | `fused/`                                      | (Future) Feature-complete and policy-checked tables.              | Feature pipeline (future)   | Overwrite; future input   |
| **ai/**         | `ai/`                                         | Model artifacts (splits, metrics, exports).                       | NB2 / NB3                   | Overwrite allowed         |
| **manifest/**   | `manifest/` or `etl_qc_summary_manifest.json` | SHA256 + size QC summary per snapshot.                            | A8 QC                       | Overwrite by A8 only      |
| **reports/**    | `reports/` (repo root)                        | Human/CI summaries (PX, NB, metrics).                             | PX series & NBs             | Overwrite per run         |

---

## 2️⃣ Minimal inventory per layer (examples)

- **processed/**

  - `apple/health_hr_daily.csv`
  - `apple/health_hrv_sdnn_daily.csv`
  - `apple/health_sleep_daily.csv`
  - `ios/ios_usage_daily.csv`
  - `zepp/zepp_hr_daily.csv`
  - `zepp/zepp_hrv_daily.csv`
  - `zepp/zepp_sleep_daily.csv`

- **joined/** _(current fused alias)_

  - `features_daily.csv` (or `features_daily_updated.csv`)
  - `version_log_enriched.csv` (segments S1–S6)

- **manifest (QC)**

  - `etl_qc_summary_manifest.json` (filename → sha256, size_bytes, mtime, notes)

- **reports/** (repo root)

  - `etl_provenance_report.csv` (PX8/QA)
  - `drift_hint_summary.{csv,md}` (PX8-Lite)
  - `nb1_eda_summary.md` (NB1)
  - `metrics_nb2.csv` (NB2)

---

## 3️⃣ Lifecycle & triggers

```text
raw → extracted → normalized → processed → joined → fused → ai
         (A6)             (A7)          (A8)     (future)   (NBs)
```

- **A6:** Normalization (TZ = Europe/Dublin, units) — idempotent.
- **A7:** Per-domain daily aggregates.
- **A8:** Cross-domain join → `joined/` + QC manifest.
- **PX8-Lite:** Provenance + drift-hint audit.
- **NB1/NB2:** EDA + baselines using `joined/features_daily*.csv`.
- **Future:** `fused/` holds curated, feature-complete tables.

**Rebuild triggers:** new raw/extracted files, firmware change, TZ policy update, schema change, drift alerts.

---

## 4️⃣ Keys · Time · Segmentation

- **Primary key:** `date` (naive date; TZ alignment upstream).
- **Segments:** `version_log_enriched.csv` → `segment_id, start, end, src, firmware_version, app_version, notes`
- **Acceptance rules:**

  - No duplicate `date`.
  - No gap > 6 h (HR/HRV) or > 24 h (sleep).
  - Overlaps < 5 %.
  - Column changes documented in `etl_provenance_report.csv`.

---

## 5️⃣ Publishing policy (Kaggle & papers)

- **Public v1:** `joined/features_daily.csv` + `version_log_enriched.csv` + `LICENSE`, `README`, metrics.
- **Future:** use `fused/` for stable schema & policy-checked features.

---

## 6️⃣ Joined vs Fused — rule of thumb

| Today                                       | Tomorrow                                                 |
| :------------------------------------------ | :------------------------------------------------------- |
| `joined/` is the model input (alias fused). | `fused/` becomes the model input after feature curation. |

> “Joined os dados estão — mas _fused_, em harmonia, estarão.”
> — _Yoda, PhD in ETL Pipelines (2025)_

---

## 7️⃣ Do / Don’t

✅ Do keep manifests updated every A8 run.
✅ Do log SHA/size/schema changes to `etl_provenance_report.csv`.
✅ Do segment analyses by S1–S6.
🚫 Don’t write into `raw/`.
🚫 Don’t mix TZ conversions below `normalized/`.
🚫 Don’t rename columns silently — document changes in reports.

---

## 8️⃣ Quick path cheatsheet

```bash
SNAPSHOT_DIR="data/etl/P000001/snapshots/2025-10-22"
JOINED_DAILY="${SNAPSHOT_DIR}/joined/features_daily.csv"
VERSION_LOG="${SNAPSHOT_DIR}/joined/version_log_enriched.csv"
MANIFEST="${SNAPSHOT_DIR}/etl_qc_summary_manifest.json"
FUSED_DIR="${SNAPSHOT_DIR}/joined"   # alias for this snapshot
```

---

## 9️⃣ Acceptance checklist (per snapshot)

- [ ] Manifest present and parsable (100 % files have SHA/size).
- [ ] Joined daily exists with `date` + ≥ 5 features.
- [ ] Version log (S1–S6) valid date ranges.
- [ ] `etl_provenance_report.csv` present (or re-generated).
- [ ] `drift_hint_summary.csv` present (zero flags OK).
- [ ] NB1/NB2 run without extra dependencies.

---

## 🔧 Optional config for code (`config/layers.yml`)

```yaml
timezone: Europe/Dublin
layers:
  raw: raw
  extracted: extracted
  normalized: normalized
  processed: processed
  joined: joined
  fused: joined # alias for 2025-10-22; change when fused/ exists
  ai: ai
snapshots:
  pattern: "data/etl/{participant}/snapshots/{date}"
files:
  features_daily:
    search_order:
      - "{snapshot}/joined/features_daily.csv"
      - "{snapshot}/joined/features_daily_updated.csv"
      - "{snapshot}/processed/features_daily.csv"
  version_log: "{snapshot}/joined/version_log_enriched.csv"
  manifest: "{snapshot}/etl_qc_summary_manifest.json"
policies:
  gaps:
    hrv_hr_max_gap_hours: 6
    sleep_max_gap_hours: 24
  overlap_pct_max: 5
  drift_hint:
    delta_mean_pct: 20
    ks_pvalue: 0.05
```
