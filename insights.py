import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, List

def _numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=["number"]).columns.tolist()

def detect_outliers_iqr(df: pd.DataFrame, col: str, thresh=1.5):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - thresh * iqr
    upper = q3 + thresh * iqr
    return df[(df[col] < lower) | (df[col] > upper)][col]

def detect_outliers_isolation_forest(df: pd.DataFrame, cols=None):
    if cols is None:
        cols = _numeric_columns(df)
    if not cols:
        return []
    iso = IsolationForest(contamination=0.01, random_state=42)
    clean_df = df[cols].dropna()
    if clean_df.shape[0] < 10:
        return []
    preds = iso.fit_predict(clean_df)
    mask = preds == -1
    outliers = clean_df[mask]
    return outliers

def simple_trend_detection(df: pd.DataFrame):
    # detect time-like columns
    trends = []
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                s = pd.to_datetime(df[col], errors="coerce")
                if s.dropna().shape[0] > 10:
                    # find numeric columns correlated with this time
                    numeric = _numeric_columns(df)
                    if numeric:
                        # pick first numeric column for a simple trend example
                        ncol = numeric[0]
                        tmp = pd.DataFrame({"t": s, "y": df[ncol]}).dropna()
                        tmp = tmp.sort_values("t")
                        if tmp.shape[0] > 5:
                            start = tmp["y"].iloc[0]
                            end = tmp["y"].iloc[-1]
                            pct = (end - start) / max(abs(start), 1e-9) * 100.0
                            trends.append({"time_col": col, "metric": ncol, "pct_change": pct})
            except Exception:
                continue
    return trends

def generate_proactive_insights(df: pd.DataFrame, max_findings=6) -> Dict[str, Any]:
    findings = []
    followups = []
    # basic stats
    try:
        n_rows, n_cols = df.shape
        findings.append(f"Dataset has {n_rows} rows and {n_cols} columns.")
    except Exception:
        findings.append("Unable to determine dataset shape.")

    # missingness
    try:
        miss = df.isna().mean() * 100
        top_missing = miss.sort_values(ascending=False).head(3)
        for col, pct in top_missing.items():
            findings.append(f"Column `{col}` has {pct:.1f}% missing values.")
    except Exception:
        pass

    # numeric summaries
    try:
        numeric = _numeric_columns(df)
        for col in numeric[:3]:
            mean = df[col].mean()
            median = df[col].median()
            findings.append(f"Column `{col}` — mean: {mean:.2f}, median: {median:.2f}.")
            followups.append(f"Show distribution of `{col}`")
    except Exception:
        pass

    # outliers
    try:
        if numeric:
            out = detect_outliers_iqr(df, numeric[0])
            if not out.empty:
                findings.append(f"Detected {out.shape[0]} outliers in `{numeric[0]}` using IQR method.")
                followups.append(f"Show top outliers in `{numeric[0]}`")
    except Exception:
        pass

    # trend detection
    try:
        trends = simple_trend_detection(df)
        for t in trends:
            findings.append(f"Detected trend on `{t['metric']}` over `{t['time_col']}`: change {t['pct_change']:.1f}% from start to end.")
            followups.append(f"Plot `{t['metric']}` over `{t['time_col']}`")
    except Exception:
        pass

    # cap findings
    findings = findings[:max_findings]
    # generate a couple generic followups
    followups = list(dict.fromkeys(followups))[:6]

    return {"bullets": findings, "followups": followups}
