"""Zepp extraction discovery helpers.

Provides a small discovery helper that lists CSV files per known Zepp
domain under `extracted/zepp/cloud`.

This module intentionally only discovers files and does not parse them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger("etl.extract")


ZEPP_DOMAINS = (
    "ACTIVITY",
    "ACTIVITY_MINUTE",
    "ACTIVITY_STAGE",
    "HEARTRATE",
    "HEARTRATE_AUTO",
    "SLEEP",
    "SPORT",
    "HEALTH_DATA",
    "BODY",
    "USER",
)


def discover_zepp_tables(root: Path) -> Dict[str, List[Path]]:
    """
    root = .../extracted/zepp/cloud
    Return mapping domain -> list[Path] of CSVs found under that domain.

    Only include keys that have at least one file.
    """
    res: Dict[str, List[Path]] = {}
    try:
        r = Path(root)
        if not r.exists():
            return {}

        for d in ZEPP_DOMAINS:
            domain_dir = r / d
            files: List[Path] = []
            if domain_dir.exists() and domain_dir.is_dir():
                # recursive glob to capture nested dirs
                files = [p for p in domain_dir.rglob("*.csv") if p.is_file()]
            # also accept some vendors that place files directly under cloud with prefix
            if not files:
                cand = [p for p in r.rglob(f"{d}*.csv") if p.is_file()]
                if cand:
                    files.extend(cand)

            if files:
                res[d] = sorted(files)

        # log concise summary
        if res:
            parts = [f"{k}={len(v)}" for k, v in sorted(res.items())]
            logger.info("zepp tables discovered: %s", ", ".join(parts))
        else:
            logger.info("zepp tables discovered: none")

    except Exception as exc:
        logger.warning("discover_zepp_tables failure: %s", exc)
    return res

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
