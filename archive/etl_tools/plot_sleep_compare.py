# etl_tools/plot_sleep_compare.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pick_col(df: pd.DataFrame, must_have: list[str]) -> str | None:
    cols = list(df.columns)
    for c in cols:
        low = c.lower()
        if all(tok in low for tok in must_have):
            return c
    return None

def main():
    ap = argparse.ArgumentParser("plot-sleep-compare")
    ap.add_argument("--join", required=True, help="CSV do join híbrido (daily)")
    ap.add_argument("--outdir", required=True, help="Pasta de saída para figuras")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.join)
    # normaliza data
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        # tenta achar uma coluna que pareça data
        poss = [c for c in df.columns if "date" in c.lower()]
        if poss:
            df.rename(columns={poss[0]:"date"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # detectar colunas de sono
    apple_col = (pick_col(df, ["sleep", "apple"]) or
                 pick_col(df, ["apple", "minutes"]) or
                 pick_col(df, ["sleep", "hk"]) )
    zepp_col  = (pick_col(df, ["sleep", "zepp"]) or
                 pick_col(df, ["zepp", "minutes"]))
    join_col  = (pick_col(df, ["sleep", "join"]) or
                 pick_col(df, ["sleep", "hybrid"]) or
                 pick_col(df, ["zepp_sleep_minutes"]) or  # fallback
                 pick_col(df, ["sleep_minutes"]))         # fallback bem genérico

    # converte candidatos para numérico
    cand_cols = [c for c in [apple_col, zepp_col, join_col] if c]
    for c in cand_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # limita aos registros com data válida
    if "date" in df.columns:
        df = df.sort_values("date")

    # ----- TIME SERIES -----
    plt.figure()
    plotted = False
    if "date" in df.columns and apple_col:
        plt.plot(df["date"], df[apple_col], label=f"Apple ({apple_col})")
        plotted = True
    if "date" in df.columns and zepp_col:
        plt.plot(df["date"], df[zepp_col], label=f"Zepp ({zepp_col})")
        plotted = True
    if "date" in df.columns and join_col and join_col not in (apple_col, zepp_col):
        plt.plot(df["date"], df[join_col], label=f"Join ({join_col})")
        plotted = True

    if plotted:
        plt.title("Sleep minutes — time series")
        plt.xlabel("Date")
        plt.ylabel("Minutes")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "sleep_timeseries.png", dpi=140)
    plt.close()

    # ----- SCATTER Apple × Zepp -----
    if apple_col and zepp_col:
        sub = df[[apple_col, zepp_col]].dropna(how="any")
        if len(sub) > 0:
            x = sub[apple_col].to_numpy(dtype=float)
            y = sub[zepp_col].to_numpy(dtype=float)
            # limites numéricos seguros
            lo = float(np.nanmin([0, np.nanmin(x), np.nanmin(y)]))
            hi = float(np.nanmax([900, np.nanmax(x), np.nanmax(y)]))
            plt.figure()
            plt.scatter(x, y, s=12, alpha=0.6)
            plt.plot([lo, hi], [lo, hi])  # y=x
            plt.xlim(lo, hi); plt.ylim(lo, hi)
            plt.xlabel(f"Apple ({apple_col})")
            plt.ylabel(f"Zepp ({zepp_col})")
            plt.title("Sleep minutes — Apple vs Zepp")
            plt.tight_layout()
            plt.savefig(outdir / "sleep_scatter_apple_vs_zepp.png", dpi=140)
            plt.close()

    # ----- HIST diff (Join − Apple) se existir join e apple -----
    if join_col and apple_col:
        diff = (df[join_col] - df[apple_col]).dropna()
        if len(diff) > 0:
            plt.figure()
            plt.hist(diff.to_numpy(dtype=float), bins=30)
            plt.title("Distribution: (Join − Apple) minutes")
            plt.xlabel("Minutes")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(outdir / "sleep_diff_join_minus_apple_hist.png", dpi=140)
            plt.close()

    print(f"✅ plots → {outdir}")

if __name__ == "__main__":
    main()
