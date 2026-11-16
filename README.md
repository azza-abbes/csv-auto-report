# CSV Auto-Report Generator

Pipeline that takes a raw CSV, cleans it, runs exploratory data analysis, and
generates a self-contained HTML report with visualizations.

## What it does
- **Cleaning:** standardizes column names, removes duplicates, fixes types
  (numeric/date), imputes missing values (median / mode).
- **EDA:** shape, dtypes, missing values, summary statistics, outlier counts (IQR rule).
- **Visuals:** histograms + boxplots, category counts, correlation heatmap,
  scatter plots against an optional target column.
- **Output:** one `report.html` file (images embedded, easy to share).

## Install & run
```bash
pip install -r requirements.txt
python auto_report.py data.csv
python auto_report.py data.csv --out report.html --target price --clean-csv clean.csv
```

Built with Pandas, Seaborn, Matplotlib.
