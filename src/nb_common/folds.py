import pandas as pd


def build_temporal_folds(
    dates, train_days=120, gap_days=10, val_days=60, max_train_days=240, min_classes=2
):
    ser_dates = pd.to_datetime(dates)
    df = pd.DataFrame({"date": ser_dates}).sort_values("date")
    start, end = df["date"].min(), df["date"].max()
    folds = []
    anchor = start
    while True:
        tr_start = anchor
        tr_end = tr_start + pd.Timedelta(days=train_days - 1)
        te_start = tr_end + pd.Timedelta(days=gap_days)
        te_end = te_start + pd.Timedelta(days=val_days - 1)
        if te_end > end:
            break
        dtr = df[(df["date"] >= tr_start) & (df["date"] <= tr_end)]
        dte = df[(df["date"] >= te_start) & (df["date"] <= te_end)]
        tr_span = train_days
        # if class info not provided here, caller should check class diversity
        while (dtr.empty or dte.empty) and tr_span < max_train_days:
            tr_span += 30
            tr_end = tr_start + pd.Timedelta(days=tr_span - 1)
            te_start = tr_end + pd.Timedelta(days=gap_days)
            te_end = te_start + pd.Timedelta(days=val_days - 1)
            if te_end > end:
                break
            dtr = df[(df["date"] >= tr_start) & (df["date"] <= tr_end)]
            dte = df[(df["date"] >= te_start) & (df["date"] <= te_end)]
        if dtr.empty or dte.empty:
            anchor = anchor + pd.Timedelta(days=30)
            continue
        folds.append(((tr_start, tr_end), (te_start, te_end)))
        anchor = te_end + pd.Timedelta(days=1)
    return folds
