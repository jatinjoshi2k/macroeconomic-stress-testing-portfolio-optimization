import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Configuration ──────────────────────────────────────────────────────────────
TICKERS = {
    "SPY":     "Stocks (S&P 500 ETF)",
    "IEAC.AS": "Euro Bonds (iShares EUR Corp Bond ETF)",
    "GLD":     "Gold (SPDR Gold Shares ETF)",
}
END_DATE   = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
TRADING_DAYS = 252

# ── Download adjusted close prices ────────────────────────────────────────────
print(f"\nDownloading data for {list(TICKERS.keys())} from {START_DATE} to {END_DATE} …\n")
raw = yf.download(
    tickers=list(TICKERS.keys()),
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,   # gives 'Close' already adjusted; avoids deprecated Adj Close
    progress=False,
)

# yfinance returns a MultiIndex when >1 ticker; pull the Close level
prices = raw["Close"][list(TICKERS.keys())]
prices.dropna(how="all", inplace=True)

print(f"Price data shape : {prices.shape}  ({prices.index[0].date()} to {prices.index[-1].date()})")
print(f"Missing values   :\n{prices.isna().sum()}\n")

# ── Daily returns ──────────────────────────────────────────────────────────────
returns = prices.pct_change(fill_method=None).dropna()
print(f"Returns data shape: {returns.shape}\n")

# ── Annualised statistics ──────────────────────────────────────────────────────
ann_returns = returns.mean() * TRADING_DAYS
ann_cov     = returns.cov()  * TRADING_DAYS

# ── Pretty print ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  ANNUALISED EXPECTED RETURNS")
print("=" * 60)
for ticker, label in TICKERS.items():
    pct = ann_returns[ticker] * 100
    print(f"  {ticker:<10} {label}")
    print(f"             => {pct:+.2f}% per year\n")

print("=" * 60)
print("  ANNUALISED COVARIANCE MATRIX")
print("=" * 60)
print(f"\n{ann_cov.to_string()}\n")

print("=" * 60)
print("  ANNUALISED CORRELATION MATRIX  (sanity check)")
print("=" * 60)
ann_corr = returns.corr()
print(f"\n{ann_corr.to_string()}\n")

# ── Save returns to CSV ────────────────────────────────────────────────────────
out_path = "portfolio_returns.csv"
returns.to_csv(out_path)
print(f"Daily returns saved to: {out_path}")
print(f"Rows: {len(returns)}  |  Columns: {list(returns.columns)}")
