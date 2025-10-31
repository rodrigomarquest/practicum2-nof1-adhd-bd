from __future__ import annotations
import os
import argparse
import io
import csv
import codecs
from pathlib import Path
from hashlib import sha256

import pandas as pd
from dateutil import tz

from etl_modules.io_utils import _open_zip_any, _norm_posix_lower


# -------------------- Utils: safe write & export id --------------------
def _concat_nonempty(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatena apenas DataFrames não vazios e com pelo menos 1 coluna válida."""
    clean = [
        d for d in dfs if isinstance(d, pd.DataFrame) and not d.empty and d.shape[1] > 0
    ]
    if not clean:
        return pd.DataFrame()
    return pd.concat(clean, ignore_index=True)


def safe_write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)  # atomic rename on same FS


def derive_export_id(p: Path) -> str:
    # ZIP com padrão 3088..._1760...zip -> "1760..."
    if p.is_file():
        stem = p.stem
        if "_" in stem:
            cand = stem.split("_")[-1]
            if cand.isdigit():
                return cand
        return sha256(stem.encode("utf-8")).hexdigest()[:12]
    # diretório -> hash curto do caminho absoluto
    return "dir_" + sha256(str(p.resolve()).encode("utf-8")).hexdigest()[:12]


# -------------------- Robust CSV readers --------------------


def _sniff_sep(sample: str) -> str | None:
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except Exception:
        return None


def _read_csv_robust_text(text: str) -> pd.DataFrame:
    sep = _sniff_sep(text.splitlines()[0] if text else ",")
    try:
        return pd.read_csv(
            io.StringIO(text), sep=(sep or None), engine="python", on_bad_lines="skip"
        )
    except Exception:
        return pd.read_csv(
            io.StringIO(text), sep="\t", engine="python", on_bad_lines="skip"
        )


def _read_csv_robust_from_zip(
    zf, member_name: str, password: str | None
) -> pd.DataFrame:
    # tenta sem senha e depois com pwd=
    try:
        with zf.open(member_name, "r") as fh:
            data = fh.read()
    except Exception:
        with zf.open(
            member_name, "r", pwd=(password.encode("utf-8") if password else None)
        ) as fh:
            data = fh.read()
    text = codecs.decode(data, "utf-8", errors="replace")
    if text and text[0] == "\ufeff":
        text = text.lstrip("\ufeff")
    return _read_csv_robust_text(text)


def _read_zip_table(
    zip_path: Path, password: str | None, folder_prefix: str
) -> pd.DataFrame:
    """Concatena todos CSVs dentro de uma pasta (ex.: 'SLEEP/') de um ZIP."""
    zf = _open_zip_any(zip_path, password)
    try:
        names = zf.namelist()
        sel = [
            n
            for n in names
            if _norm_posix_lower(n).startswith(_norm_posix_lower(folder_prefix))
        ]
        sel = [n for n in sel if _norm_posix_lower(n).endswith(".csv")]
        if not sel:
            return pd.DataFrame()
        dfs = []
        for n in sel:
            df = _read_csv_robust_from_zip(zf, n, password)
            df["_source"] = n
            dfs.append(df)
        return _concat_nonempty(dfs)
    finally:
        zf.close()


def _read_dir_table(dir_path: Path, folder_prefix: str) -> pd.DataFrame:
    base = dir_path / folder_prefix.strip("/\\")
    if not base.exists():
        return pd.DataFrame()
    dfs = []
    for p in sorted(base.rglob("*.csv")):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            df = _read_csv_robust_text(text)
        except Exception:
            data = p.read_bytes()
            text = codecs.decode(data, "utf-8", errors="replace")
            df = _read_csv_robust_text(text)
        df["_source"] = str(p.relative_to(dir_path))
        dfs.append(df)
    return _concat_nonempty(dfs)


# -------------------- Date handling --------------------

from datetime import date as _date


def _to_local_date(
    series_utc, cutover_str: str, tz_before: str, tz_after: str
) -> pd.Series:
    """
    Converte série de timestamps para data local (string), aplicando cutover:
      - datas UTC < cutover  -> tz_before
      - datas UTC >= cutover -> tz_after
    Detecta epoch ms/s e formatos ISO comuns.
    """
    import numpy as np

    s = pd.Series(series_utc)

    # 1) parse para UTC (tz-aware)
    if pd.api.types.is_numeric_dtype(s):
        m = s.astype("float64")
        if np.isfinite(m).any():
            if (m > 1e12).mean() > 0.5:
                dt = pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
            elif (m > 1e9).mean() > 0.5:
                dt = pd.to_datetime(s, unit="s", utc=True, errors="coerce")
            else:
                dt = pd.to_datetime(s, utc=True, errors="coerce")
        else:
            dt = pd.to_datetime(s, utc=True, errors="coerce")
    elif pd.api.types.is_string_dtype(s):
        fmt_try = [
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d",
            "%Y/%m/%d",
        ]
        dt = None
        for fmt in fmt_try:
            try:
                dt = pd.to_datetime(s, format=fmt, utc=True, errors="raise")
                break
            except Exception:
                dt = None
        if dt is None:
            dt = pd.to_datetime(s, utc=True, errors="coerce")
    else:
        dt = pd.to_datetime(s, utc=True, errors="coerce")

    # 2) aplica cutover: duas TZs -> calcula datas locais separadas e combina
    y, m, d = map(int, cutover_str.split("-"))
    cutover = _date(y, m, d)
    tz_b = tz.gettz(tz_before)
    tz_a = tz.gettz(tz_after)

    mask_after = dt.dt.tz_convert(tz.gettz("UTC")).dt.date >= cutover
    dates_before = dt.dt.tz_convert(tz_b).dt.date.astype("string")
    dates_after = dt.dt.tz_convert(tz_a).dt.date.astype("string")

    out = pd.Series(index=dt.index, dtype="string")
    out.loc[mask_after] = dates_after.loc[mask_after]
    out.loc[~mask_after] = dates_before.loc[~mask_after]
    return out


def _maybe_col(df: pd.DataFrame, *candidates: str) -> str | None:
    cols_l = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_l:
            return cols_l[c.lower()]
    for c in candidates:
        for k, orig in cols_l.items():
            if c.lower() in k:
                return orig
    return None


# -------------------- Latest rebuild & manifest --------------------


def rebuild_latest(processed_base: Path, filename: str) -> bool:
    dfs = []
    for sub in sorted(processed_base.iterdir()):
        if not sub.is_dir() or sub.name == "_latest":
            continue
        f = sub / filename
        if f.exists():
            d = pd.read_csv(f, dtype={"date": "string"})
            d["export_id"] = sub.name
            dfs.append(d)
    if not dfs:
        return False
    all_df = _concat_nonempty(dfs)
    if all_df.empty:
        return False
    all_df = (
        all_df.sort_values(["date", "export_id"])
        .drop_duplicates(subset=["date"], keep="last")
        .drop(columns=["export_id"])
    )
    safe_write_csv(all_df, processed_base / "_latest" / filename)
    return True


def append_manifest(
    pid: str,
    outroot: Path,
    zip_path: Path,
    export_id: str,
    categories: list[str],
    df_coverage: dict,
):
    import csv as _csv

    man_dir = outroot.parent / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    man_csv = man_dir / "ZEPP_EXPORTS.csv"

    row = {
        "participant_id": pid,
        "export_id": export_id,
        "zip_name": zip_path.name if zip_path.is_file() else str(zip_path),
        "sha256": "",
        "source": "zepp_portal",
        "from_date": df_coverage.get("from_date", ""),
        "to_date": df_coverage.get("to_date", ""),
        "categories": ",".join(categories),
        "rows": df_coverage.get("rows", 0),
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    try:
        row["sha256"] = sha256(zip_path.read_bytes()).hexdigest()
    except Exception:
        pass

    new = not man_csv.exists()
    with man_csv.open("a", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)


# -------------------- CLI --------------------


def parse_args():
    ap = argparse.ArgumentParser("parse-zepp-export")
    ap.add_argument("--input", required=True, help="Path to Zepp export ZIP (or dir)")
    ap.add_argument(
        "--outdir-root",
        required=True,
        help="Root dir for processed outputs (per export_id)",
    )
    ap.add_argument(
        "--password", default=None, help="ZIP password (fallback ZEPP_ZIP_PASSWORD)"
    )
    # cutover Brazil -> Ireland
    ap.add_argument("--cutover", required=True, help="Cutover date YYYY-MM-DD (BR→IE)")
    ap.add_argument(
        "--tz_before",
        required=True,
        help="IANA TZ before cutover, e.g., America/Sao_Paulo",
    )
    ap.add_argument(
        "--tz_after", required=True, help="IANA TZ after cutover, e.g., Europe/Dublin"
    )
    ap.add_argument(
        "--export-id",
        default=None,
        help="Override export_id (default: derived from input)",
    )
    ap.add_argument(
        "--participant", default="P000001", help="Participant ID (for manifests)"
    )
    return ap.parse_args()


# -------------------- Main --------------------


def main():
    args = parse_args()
    zip_password = args.password or os.getenv("ZEPP_ZIP_PASSWORD") or None
    src = Path(args.input)

    export_id = args.export_id or derive_export_id(src)

    processed_root = Path(args.outdir_root) / export_id
    latest_root = Path(args.outdir_root) / "_latest"
    processed_root.mkdir(parents=True, exist_ok=True)
    latest_root.mkdir(parents=True, exist_ok=True)

    # leitor conforme tipo de input (dir/zip)
    is_dir = src.is_dir()
    read_table = (
        (lambda prefix: _read_dir_table(src, prefix))
        if is_dir
        else (lambda prefix: _read_zip_table(src, zip_password, prefix))
    )

    # ---------------- HEARTRATE + HEARTRATE_AUTO ----------------
    hr = read_table("HEARTRATE/")
    hra = read_table("HEARTRATE_AUTO/")
    hr_all = (
        pd.concat([hr, hra], ignore_index=True)
        if (not hr.empty or not hra.empty)
        else pd.DataFrame()
    )

    hr_daily = pd.DataFrame(
        columns=["date", "zepp_hr_mean", "zepp_hr_median", "zepp_hr_p95"]
    )
    if not hr_all.empty:
        ts_col = _maybe_col(hr_all, "timestamp", "time", "dateTime", "startTime")
        val_col = _maybe_col(hr_all, "heart_rate", "heartrate", "hr", "bpm", "value")
        if ts_col and val_col:
            hr_all["date"] = _to_local_date(
                hr_all[ts_col], args.cutover, args.tz_before, args.tz_after
            )
            hr_all[val_col] = pd.to_numeric(hr_all[val_col], errors="coerce")
            g = hr_all.groupby("date", dropna=False)[val_col]
            hr_daily = g.agg(
                zepp_hr_mean="mean",
                zepp_hr_median="median",
                zepp_hr_p95=lambda s: s.quantile(0.95),
            ).reset_index()
    safe_write_csv(hr_daily, processed_root / "zepp_hr_daily.csv")

    # ---------------- SLEEP ----------------
    sleep = read_table("SLEEP/")
    sleep_daily = pd.DataFrame(columns=["date", "zepp_sleep_minutes"])
    if not sleep.empty:
        cols = {c.lower(): c for c in sleep.columns}
        # Preferimos a coluna 'date' já fornecida pelo Zepp summary (não recalculamos)
        if all(k in cols for k in ["date", "deepsleeptime", "shallowsleeptime"]) and (
            "remtime" in cols or "naps" in cols
        ):
            date_col = cols["date"]
            dcol = cols["deepsleeptime"]
            scol = cols["shallowsleeptime"]
            rcol = cols.get("remtime")
            ncol = cols.get("naps")
            for c in [dcol, scol, rcol, ncol]:
                if c and c in sleep.columns:
                    sleep[c] = pd.to_numeric(sleep[c], errors="coerce").fillna(0)
            total_min = sum(
                [sleep[dcol], sleep[scol]]
                + ([sleep[rcol]] if rcol else [])
                + ([sleep[ncol]] if ncol else [])
            )
            sleep_daily = (
                pd.DataFrame(
                    {
                        "date": sleep[date_col].astype("string"),
                        "zepp_sleep_minutes": total_min,
                    }
                )
                .groupby("date", dropna=False)["zepp_sleep_minutes"]
                .sum(min_count=1)
                .reset_index()
            )
        else:
            ts_col = _maybe_col(
                sleep, "timestamp", "time", "dateTime", "start", "startTime", "date"
            )
            dur_col = _maybe_col(
                sleep, "duration", "sleep_duration", "minutes", "sleepMinutes"
            )
            if ts_col and dur_col:
                sleep["date"] = _to_local_date(
                    sleep[ts_col], args.cutover, args.tz_before, args.tz_after
                )
                mins = pd.to_numeric(sleep[dur_col], errors="coerce")
                sleep_daily = (
                    sleep.groupby("date", dropna=False)[mins.name].sum().reset_index()
                )
                sleep_daily = sleep_daily.rename(
                    columns={mins.name: "zepp_sleep_minutes"}
                )
    safe_write_csv(sleep_daily, processed_root / "zepp_sleep_daily.csv")

    # ---------------- ACTIVITY + ACTIVITY_MINUTE ----------------
    act = read_table("ACTIVITY/")
    actm = read_table("ACTIVITY_MINUTE/")
    activity_daily = pd.DataFrame(
        columns=["date", "zepp_steps", "zepp_calories", "zepp_active_minutes"]
    )
    if not act.empty:
        ts_col = _maybe_col(act, "timestamp", "time", "dateTime", "startTime", "date")
        steps_col = _maybe_col(act, "steps", "step_count")
        cal_col = _maybe_col(act, "calorie", "calories", "kcal")
        if ts_col and steps_col:
            act["date"] = _to_local_date(
                act[ts_col], args.cutover, args.tz_before, args.tz_after
            )
            daily = (
                act.groupby("date", dropna=False)
                .agg({steps_col: "sum", **({cal_col: "sum"} if cal_col else {})})
                .reset_index()
            )
            daily = daily.rename(columns={steps_col: "zepp_steps"})
            if cal_col:
                daily = daily.rename(columns={cal_col: "zepp_calories"})
            else:
                daily["zepp_calories"] = 0
            activity_daily = daily
    if not actm.empty:
        ts_col = _maybe_col(actm, "timestamp", "time", "dateTime", "startTime", "date")
        dur_col = _maybe_col(
            actm, "duration", "minutes", "active_minutes", "activityMinutes"
        )
        if ts_col and dur_col:
            actm["date"] = _to_local_date(
                actm[ts_col], args.cutover, args.tz_before, args.tz_after
            )
            tmp = (
                actm.groupby("date", dropna=False)[dur_col]
                .sum()
                .reset_index()
                .rename(columns={dur_col: "zepp_active_minutes"})
            )
            activity_daily = (
                activity_daily.merge(tmp, on="date", how="outer")
                if not activity_daily.empty
                else tmp
            )
    safe_write_csv(activity_daily, processed_root / "zepp_activity_daily.csv")

    # ---------------- BODY ----------------
    body = read_table("BODY/")
    body_daily = pd.DataFrame(columns=["date", "zepp_weight_kg", "zepp_bodyfat_pct"])
    if not body.empty:
        ts_col = _maybe_col(
            body, "timestamp", "time", "dateTime", "measureTime", "startTime", "date"
        )
        w_col = _maybe_col(body, "weight", "weight_kg", "body_weight")
        bf_col = _maybe_col(body, "bodyfat", "body_fat", "bodyfat_pct", "fat_rate")
        if ts_col and (w_col or bf_col):
            body["date"] = _to_local_date(
                body[ts_col], args.cutover, args.tz_before, args.tz_after
            )
            agg = {}
            if w_col:
                agg[w_col] = "mean"
            if bf_col:
                agg[bf_col] = "mean"
            bd = body.groupby("date", dropna=False).agg(agg).reset_index()
            if w_col:
                bd = bd.rename(columns={w_col: "zepp_weight_kg"})
            if bf_col:
                bd = bd.rename(columns={bf_col: "zepp_bodyfat_pct"})
            body_daily = bd
    safe_write_csv(body_daily, processed_root / "zepp_body_daily.csv")

    # ---------------- HEALTH_DATA ----------------
    hdata = read_table("HEALTH_DATA/")
    health_daily = pd.DataFrame(
        columns=["date", "zepp_spo2_mean", "zepp_temp_mean", "zepp_stress_mean"]
    )
    if not hdata.empty:
        ts_col = _maybe_col(
            hdata, "timestamp", "time", "dateTime", "startTime", "measureTime", "date"
        )
        spo2 = _maybe_col(
            hdata, "spo2", "blood_oxygen", "oxygensaturation", "saturation"
        )
        temp = _maybe_col(
            hdata,
            "temp",
            "temperature",
            "skin_temp",
            "skin_temperature",
            "body_temperature",
        )
        stress = _maybe_col(hdata, "stress", "stress_score", "mental_stress")
        if ts_col:
            hdata["date"] = _to_local_date(
                hdata[ts_col], args.cutover, args.tz_before, args.tz_after
            )
            pieces = []
            if spo2:
                pieces.append(
                    hdata.groupby("date", dropna=False)[spo2]
                    .mean(numeric_only=True)
                    .reset_index()
                    .rename(columns={spo2: "zepp_spo2_mean"})
                )
            if temp:
                pieces.append(
                    hdata.groupby("date", dropna=False)[temp]
                    .mean(numeric_only=True)
                    .reset_index()
                    .rename(columns={temp: "zepp_temp_mean"})
                )
            if stress:
                pieces.append(
                    hdata.groupby("date", dropna=False)[stress]
                    .mean(numeric_only=True)
                    .reset_index()
                    .rename(columns={stress: "zepp_stress_mean"})
                )
            if pieces:
                base = pieces[0]
                for p in pieces[1:]:
                    base = base.merge(p, on="date", how="outer")
                health_daily = base
    safe_write_csv(health_daily, processed_root / "zepp_health_daily.csv")

    # ---------------- Consolidado diário (somente Zepp) ----------------
    dfs = [
        d
        for d in (hr_daily, sleep_daily, activity_daily, body_daily, health_daily)
        if (d is not None and not d.empty)
    ]
    if dfs:
        daily = dfs[0]
        for d in dfs[1:]:
            daily = daily.merge(d, on="date", how="outer")
        daily = daily.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    else:
        daily = pd.DataFrame(columns=["date"])
    safe_write_csv(daily, processed_root / "zepp_daily_features.csv")

    # ---------- rebuild _latest ----------
    for fname in [
        "zepp_sleep_daily.csv",
        "zepp_hr_daily.csv",
        "zepp_health_daily.csv",
        "zepp_body_daily.csv",
        "zepp_activity_daily.csv",
        "zepp_daily_features.csv",
    ]:
        rebuild_latest(Path(args.outdir_root), fname)

    # ---------- manifest append-only ----------
    cov, cats, rows = {}, [], 0
    p = processed_root / "zepp_sleep_daily.csv"
    if p.exists():
        d = pd.read_csv(p, dtype={"date": "string"})
        if not d.empty:
            cov["from_date"] = d["date"].min()
            cov["to_date"] = d["date"].max()
            rows += len(d)
            cats.append("sleep")
    cov["rows"] = rows
    append_manifest(args.participant, processed_root, src, export_id, cats, cov)

    print(f"✅ Zepp parsed (export_id={export_id}) → {processed_root}")
    print(f"↪️  _latest updated under: {latest_root}")


if __name__ == "__main__":
    main()
