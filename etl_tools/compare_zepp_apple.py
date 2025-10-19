from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def read_apple(apple_dir: Path) -> pd.DataFrame:
    f = Path(apple_dir) / "features_daily.csv"
    if not f.exists():
        raise FileNotFoundError(f"Apple features_daily.csv not found at: {f}")
    df = pd.read_csv(f, parse_dates=["date"])
    # tenta detectar a coluna de sono em minutos (ajuste se seu ETL usa outro nome)
    cand = [c for c in df.columns if c.endswith("sleep_mean") or c == "sleep_minutes" or c == "sleep_sum_min" or c == "sleep_sum_h"]
    if "sleep_sum_h" in df.columns and "sleep_minutes" not in df.columns:
        df["apple_sleep_minutes"] = (df["sleep_sum_h"] * 60).astype(float)
    elif "sleep_minutes" in df.columns:
        df["apple_sleep_minutes"] = df["sleep_minutes"].astype(float)
    elif cand:
        # fallback: tratar *_mean como minutos
        df["apple_sleep_minutes"] = df[cand[0]].astype(float)
    else:
        # sem sono na Apple → coluna vazia
        df["apple_sleep_minutes"] = np.nan
    return df[["date","apple_sleep_minutes"]].copy()

def read_zepp(zepp_root: Path) -> pd.DataFrame:
    # usa _latest por padrão
    f = Path(zepp_root) / "_latest" / "zepp_sleep_daily.csv"
    if not f.exists():
        # fallback: tenta um único subdir
        subs = [p for p in Path(zepp_root).iterdir() if p.is_dir() and p.name != "_latest"]
        for s in subs:
            cand = s / "zepp_sleep_daily.csv"
            if cand.exists():
                f = cand; break
    if not f.exists():
        # sem Zepp → vazio com schema
        return pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]"),
                             "zepp_sleep_minutes": pd.Series([], dtype="float")})
    df = pd.read_csv(f, parse_dates=["date"])
    if "zepp_sleep_minutes" not in df.columns:
        # tenta inferir
        zc = [c for c in df.columns if "sleep" in c.lower() and "min" in c.lower()]
        if zc:
            df = df.rename(columns={zc[0]:"zepp_sleep_minutes"})
        else:
            df["zepp_sleep_minutes"] = np.nan
    return df[["date","zepp_sleep_minutes"]].copy()

def build_hybrid(z: pd.DataFrame, a: pd.DataFrame, policy: str) -> pd.DataFrame:
    df = pd.merge(a, z, on="date", how="outer")
    has_a = df["apple_sleep_minutes"].notna()
    has_z = df["zepp_sleep_minutes"].notna()

    if policy == "apple_first":
        source = np.select([has_a, (~has_a & has_z)], ["apple","zepp"], default=None)
        hybrid = np.where(has_a, df["apple_sleep_minutes"],
                          np.where(has_z, df["zepp_sleep_minutes"], np.nan))
    elif policy == "zepp_first":
        source = np.select([has_z, (~has_z & has_a)], ["zepp","apple"], default=None)
        hybrid = np.where(has_z, df["zepp_sleep_minutes"],
                          np.where(has_a, df["apple_sleep_minutes"], np.nan))
    else:  # best_of_day
        def score(v):
            # válido e dentro de 2h–15h
            return (v.notna()).astype(int) * (v.between(120, 900))
        sa = score(df["apple_sleep_minutes"])
        sz = score(df["zepp_sleep_minutes"])
        pick_zepp  = (sz > sa)
        pick_apple = (sa >= sz) & has_a
        source = np.where(pick_zepp, "zepp", np.where(pick_apple, "apple", None))
        hybrid = np.where(pick_zepp, df["zepp_sleep_minutes"],
                 np.where(pick_apple, df["apple_sleep_minutes"], np.nan))

    df["sleep_source"] = source
    df["sleep_minutes_hybrid"] = hybrid
    return df.sort_values("date").reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser("compare_zepp_apple")
    ap.add_argument("--pid", required=True)
    ap.add_argument("--zepp-root", required=True, help="data_etl/<PID>/zepp_processed")
    ap.add_argument("--apple-dir", required=True, help="data_ai/<PID>/snapshots/<SNAP>")
    ap.add_argument("--out-dir", required=True, help=".../hybrid_join")
    ap.add_argument("--sleep-policy", choices=["apple_first","zepp_first","best_of_day"], default="apple_first")
    args = ap.parse_args()

    a = read_apple(Path(args.apple_dir))
    z = read_zepp(Path(args.zepp_root))
    hybrid = build_hybrid(z, a, args.sleep_policy)

    # embute policy e grava EM SUBPASTA da policy
    hybrid["policy"] = args.sleep_policy
    out_dir = Path(args.out_dir) / args.sleep_policy
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "join_hybrid_daily.csv"
    hybrid.to_csv(out_csv, index=False)
    (out_dir / "POLICY.txt").write_text(args.sleep_policy + "\n", encoding="utf-8")

    print(f"✅ hybrid join → {out_dir}")

if __name__ == "__main__":
    main()
