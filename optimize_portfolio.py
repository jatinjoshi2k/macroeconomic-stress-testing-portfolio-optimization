import pandas as pd
import numpy as np
from scipy.optimize import minimize

# ── Load returns ───────────────────────────────────────────────────────────────
returns = pd.read_csv("portfolio_returns.csv", index_col=0, parse_dates=True)
tickers = list(returns.columns)

TRADING_DAYS  = 252
RISK_FREE_RATE = 0.02   # 2% ECB proxy

# Pre-compute annualised building blocks once
ann_mean = returns.mean() * TRADING_DAYS          # shape (3,)
ann_cov  = returns.cov()  * TRADING_DAYS          # shape (3,3)

print(f"Loaded {len(returns)} daily return observations for: {tickers}")
print(f"Risk-free rate : {RISK_FREE_RATE*100:.1f}%\n")

# ── Portfolio performance ──────────────────────────────────────────────────────
def portfolio_performance(weights: np.ndarray):
    """Return (expected_return, volatility, sharpe_ratio) for a weight vector."""
    weights     = np.asarray(weights)
    exp_return  = weights @ ann_mean
    variance    = weights @ ann_cov @ weights
    volatility  = np.sqrt(variance)
    sharpe      = (exp_return - RISK_FREE_RATE) / volatility
    return exp_return, volatility, sharpe

# ── Objective: minimise –Sharpe ────────────────────────────────────────────────
def neg_sharpe(weights: np.ndarray) -> float:
    return -portfolio_performance(weights)[2]

# ── Constraints and bounds ────────────────────────────────────────────────────
n = len(tickers)
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}   # weights sum to 1
bounds      = [(0.0, 1.0)] * n                                    # no short selling

# Multiple random starting points to avoid local minima
best_result = None
rng = np.random.default_rng(42)
for _ in range(200):
    w0 = rng.dirichlet(np.ones(n))           # random valid starting weights
    res = minimize(
        neg_sharpe,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if res.success and (best_result is None or res.fun < best_result.fun):
        best_result = res

opt_weights    = best_result.x
exp_ret, vol, sharpe = portfolio_performance(opt_weights)

# ── Print results ──────────────────────────────────────────────────────────────
LABELS = {
    "SPY":     "Stocks  (S&P 500 ETF)",
    "IEAC.AS": "Bonds   (iShares EUR Corp Bond ETF)",
    "GLD":     "Gold    (SPDR Gold Shares ETF)",
}

print("=" * 58)
print("  OPTIMAL ASSET ALLOCATION  (Max Sharpe Ratio)")
print("=" * 58)
for ticker, weight in zip(tickers, opt_weights):
    bar = "#" * int(round(weight * 40))
    print(f"  {ticker:<10} {LABELS.get(ticker, ticker)}")
    print(f"             Weight : {weight*100:6.2f}%  [{bar:<40}]")
print()

print("=" * 58)
print("  OPTIMAL PORTFOLIO STATISTICS")
print("=" * 58)
print(f"  Expected Return  : {exp_ret*100:+.2f}% per year")
print(f"  Volatility (Risk): {vol*100:.2f}%  per year")
print(f"  Sharpe Ratio     : {sharpe:.4f}")
print(f"  Risk-Free Rate   : {RISK_FREE_RATE*100:.1f}%")
print()

# ── Sanity check: equally-weighted baseline ────────────────────────────────────
ew = np.ones(n) / n
ew_ret, ew_vol, ew_sharpe = portfolio_performance(ew)
print("=" * 58)
print("  EQUAL-WEIGHT BASELINE  (33/33/33)")
print("=" * 58)
print(f"  Expected Return  : {ew_ret*100:+.2f}% per year")
print(f"  Volatility (Risk): {ew_vol*100:.2f}%  per year")
print(f"  Sharpe Ratio     : {ew_sharpe:.4f}")
print()
print(f"  Sharpe improvement vs equal-weight: "
      f"{((sharpe - ew_sharpe) / abs(ew_sharpe)) * 100:+.1f}%")
