#!/usr/bin/env python3
"""Generate the weekly report markdown for a snapshot.

This is the extracted version of the Makefile heredoc used by the weekly-report target.
"""
from __future__ import annotations
import pathlib as P
import pandas as pd
from make_scripts.common import parse_pid_snap


def span(p: P.Path):
    if not p.exists():
        return ("–", "–", 0)
    df = pd.read_csv(p, parse_dates=["date"])
    return (str(df["date"].min().date()), str(df["date"].max().date()), len(df))


def main(argv=None):
    pid, snap, policy = parse_pid_snap(argv)
    base = P.Path(f"data_ai/{pid}/snapshots/{snap}")
    feat = base / "features_daily.csv"
    join = base / f"hybrid_join/{policy}/join_hybrid_daily.csv"
    qc = base / "etl_qc_summary.csv"
    outd = P.Path("docs_build")
    outd.mkdir(parents=True, exist_ok=True)
    md = outd / f"weekly_report_{pid}_{snap}.md"

    fmin, fmax, fn = span(feat)
    jmin, jmax, jn = span(join)

    qcrows = []
    if qc.exists():
        q = pd.read_csv(qc)
        qcrows = [f"- {r['metric']}: {int(r.get('value', r.get('days_present',0)))}" for _, r in q.iterrows()]

    md.write_text(f"""# Weekly Report — {pid} / {snap}

## Snapshot
- Features: **{fn}** linhas · período **{fmin} → {fmax}**
- Join (policy=`{policy}`): **{jn}** linhas · período **{jmin} → {jmax}**

## Diretores/artefatos
- Features: `{feat.as_posix()}`
- Join: `{join.as_posix()}`
- Plots: `{(base/f'hybrid_join/{policy}/plots').as_posix()}`

## QC (ETL Apple)
{chr(10).join(qcrows) if qcrows else "- (sem métricas adicionais)"}

## Observações
- Outliers de sono no Zepp (> 16h) aparecem como colunas verticais nos scatters.
- Política recomendada: **best_of_day** (para notebooks iniciais).

""", encoding="utf-8")
    print(f"✅ weekly report → {md}")


if __name__ == "__main__":
    main()
