"""CSV Auto-Report Generator
Usage:
    python auto_report.py data.csv
    python auto_report.py data.csv --out report.html --target price
"""
import argparse
import base64
import io
import html
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# ---------- 1. CLEANING ----------
def clean(df: pd.DataFrame):
    log = []
    df = df.copy()

    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(r"[^\w]+", "_", regex=True).str.strip("_"))
    log.append("Standardized column names (lowercase, underscores).")

    before = len(df)
    df = df.drop_duplicates()
    if before - len(df):
        log.append(f"Removed {before - len(df)} duplicate rows.")

    # Strip whitespace in text columns, try numeric/date conversion
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype(str).str.strip().replace({"nan": None, "": None})
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.9:
            df[col] = converted
            log.append(f"Converted '{col}' to numeric.")
            continue
        if any(k in col for k in ("date", "time", "day")):
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().mean() > 0.9:
                df[col] = dt
                log.append(f"Converted '{col}' to datetime.")

    # Impute missing values
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if not n_missing:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
            log.append(f"Filled {n_missing} missing in '{col}' with median.")
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        else:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if len(mode) else "unknown")
            log.append(f"Filled {n_missing} missing in '{col}' with mode.")
    return df, log


def outlier_counts(df: pd.DataFrame) -> pd.Series:
    out = {}
    for col in df.select_dtypes("number"):
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        out[col] = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
    return pd.Series(out, name="outliers (IQR rule)")


# ---------- 2. VISUALIZATION ----------
def fig_to_html(fig, title):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<h3>{html.escape(title)}</h3><img src="data:image/png;base64,{b64}">'


def make_plots(df: pd.DataFrame, target=None, max_cols=6):
    plots = []
    num = list(df.select_dtypes("number").columns)
    cat = [c for c in df.select_dtypes(exclude=["number", "datetime"]).columns
           if df[c].nunique() <= 15]

    for col in num[:max_cols]:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        sns.histplot(df[col], kde=True, ax=ax[0])
        sns.boxplot(x=df[col], ax=ax[1])
        ax[0].set_title(f"Distribution of {col}")
        ax[1].set_title(f"Boxplot of {col}")
        plots.append(fig_to_html(fig, f"Numeric: {col}"))

    for col in cat[:max_cols]:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        sns.countplot(y=df[col], order=df[col].value_counts().index, ax=ax)
        plots.append(fig_to_html(fig, f"Categorical: {col}"))

    if len(num) >= 2:
        fig, ax = plt.subplots(figsize=(6.5, 5))
        sns.heatmap(df[num].corr(), annot=len(num) <= 10, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax)
        plots.append(fig_to_html(fig, "Correlation heatmap"))

    if target and target in num:
        for col in [c for c in num if c != target][:3]:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.scatterplot(data=df, x=col, y=target, ax=ax)
            plots.append(fig_to_html(fig, f"{col} vs {target}"))
    return plots


# ---------- 3. REPORT ----------
def build_html(name, raw, df, log, plots):
    missing = raw.isna().sum()
    missing = missing[missing > 0].to_frame("missing values")
    css = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:auto;"
           "padding:24px;color:#222}table{border-collapse:collapse;margin:8px 0}"
           "td,th{border:1px solid #ccc;padding:4px 10px;font-size:13px}"
           "img{max-width:100%}h1,h2{border-bottom:1px solid #eee;padding-bottom:4px}")
    parts = [
        f"<html><head><meta charset='utf-8'><title>Report - {html.escape(name)}</title>"
        f"<style>{css}</style></head><body>",
        f"<h1>Auto Report: {html.escape(name)}</h1>",
        f"<p><b>Raw shape:</b> {raw.shape[0]} rows × {raw.shape[1]} columns &nbsp;|&nbsp; "
        f"<b>Clean shape:</b> {df.shape[0]} × {df.shape[1]}</p>",
        "<h2>Cleaning steps</h2><ul>"
        + ("".join(f"<li>{html.escape(l)}</li>" for l in log) or "<li>Nothing to clean.</li>")
        + "</ul>",
        "<h2>Missing values (before cleaning)</h2>",
        missing.to_html() if len(missing) else "<p>None.</p>",
        "<h2>Column types</h2>",
        df.dtypes.astype(str).to_frame("dtype").to_html(),
        "<h2>Summary statistics</h2>",
        df.describe(include="all").T.round(2).to_html(),
        "<h2>Outliers</h2>",
        outlier_counts(df).to_frame().to_html(),
        "<h2>Visualizations</h2>",
        *plots,
        "</body></html>",
    ]
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Auto-generate an EDA report from a CSV.")
    ap.add_argument("csv", help="path to CSV file")
    ap.add_argument("--out", default="report.html", help="output HTML file")
    ap.add_argument("--target", help="optional numeric target column for scatter plots")
    ap.add_argument("--clean-csv", help="also save the cleaned CSV here")
    args = ap.parse_args()

    raw = pd.read_csv(args.csv, sep=None, engine="python")  # auto-detect delimiter
    df, log = clean(raw)
    target = args.target.strip().lower() if args.target else None
    plots = make_plots(df, target)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(build_html(args.csv, raw, df, log, plots))
    if args.clean_csv:
        df.to_csv(args.clean_csv, index=False)
    print(f"Report saved to {args.out}")


if __name__ == "__main__":
    main()
