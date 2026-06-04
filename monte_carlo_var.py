import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — saves to file reliably
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── Load returns & rebuild annualised stats ────────────────────────────────────
returns = pd.read_csv("portfolio_returns.csv", index_col=0, parse_dates=True)
tickers = list(returns.columns)

TRADING_DAYS   = 252
RISK_FREE_RATE = 0.02
OPT_WEIGHTS    = np.array([0.4426, 0.0000, 0.5574])   # SPY / IEAC.AS / GLD

daily_mean = returns.mean().values          # shape (3,)
daily_cov  = returns.cov().values           # shape (3,3)

# ── Monte Carlo via Cholesky decomposition ────────────────────────────────────
N_SIMS  = 10_000
N_DAYS  = TRADING_DAYS
rng     = np.random.default_rng(42)

# Cholesky factor for correlated draws
L = np.linalg.cholesky(daily_cov)          # lower-triangular  (3×3)

# Simulate N_SIMS paths × N_DAYS days of correlated asset returns
# z: (N_DAYS, 3, N_SIMS) -> each slice [:, :, i] is one simulation
z            = rng.standard_normal((N_DAYS, 3, N_SIMS))
corr_returns = (L @ z.reshape(3, -1)).reshape(3, N_DAYS, N_SIMS)  # shape (3, N_DAYS, N_SIMS)
#  add daily mean drift to each asset
corr_returns += daily_mean[:, np.newaxis, np.newaxis]

# Weighted portfolio daily returns  →  shape (N_DAYS, N_SIMS)
port_daily = np.einsum("a,adn->dn", OPT_WEIGHTS, corr_returns)

# Compound over the year:  final value = product of (1 + r_t)
port_annual = np.prod(1 + port_daily, axis=0) - 1   # shape (N_SIMS,)

# ── VaR ───────────────────────────────────────────────────────────────────────
var_95 = np.percentile(port_annual, 5)    # 5th percentile = 95% VaR
var_99 = np.percentile(port_annual, 1)    # 1st percentile = 99% VaR
mean_r = port_annual.mean()
med_r  = np.median(port_annual)

print("=" * 52)
print("  MONTE CARLO STRESS TEST  (10,000 simulations)")
print("=" * 52)
print(f"  Optimal weights  : SPY {OPT_WEIGHTS[0]*100:.1f}% | "
      f"IEAC.AS {OPT_WEIGHTS[1]*100:.1f}% | GLD {OPT_WEIGHTS[2]*100:.1f}%")
print(f"  Simulation horizon : {N_DAYS} trading days (1 year)")
print()
print(f"  Mean simulated return  : {mean_r*100:+.2f}%")
print(f"  Median simulated return: {med_r*100:+.2f}%")
print()
print(f"  95% Value at Risk (VaR): {var_95*100:+.2f}%")
print(f"  99% Value at Risk (VaR): {var_99*100:+.2f}%")
print()
print("  Interpretation:")
print(f"    - In 95% of scenarios the portfolio loses less than "
      f"{abs(var_95)*100:.2f}%")
print(f"    - In the worst 5% of scenarios the loss exceeds "
      f"{abs(var_95)*100:.2f}%")
print(f"    - In the worst 1% of scenarios the loss exceeds "
      f"{abs(var_99)*100:.2f}%")
print("=" * 52)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#1a1d27")

# Histogram — split into gain (green) and loss (red) bins
bins = np.linspace(port_annual.min(), port_annual.max(), 80)
gains  = port_annual[port_annual >= 0]
losses = port_annual[port_annual <  0]

ax.hist(gains,  bins=bins, color="#2ecc71", alpha=0.75, label="Gain scenarios")
ax.hist(losses, bins=bins, color="#e74c3c", alpha=0.75, label="Loss scenarios")

# 95% VaR line
ax.axvline(var_95, color="#ff3333", linewidth=2.8, linestyle="--",
           label=f"95% VaR  {var_95*100:+.2f}%")
ax.axvline(var_99, color="#ff9900", linewidth=2.0, linestyle=":",
           label=f"99% VaR  {var_99*100:+.2f}%")

# Mean return line
ax.axvline(mean_r, color="#3498db", linewidth=1.8, linestyle="-",
           label=f"Mean return  {mean_r*100:+.2f}%")

# Shade the tail beyond 95% VaR
tail_mask = port_annual <= var_95
ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] else 800],
                 port_annual.min(), var_95,
                 color="#ff3333", alpha=0.12)

# Labels & styling
ax.set_xlabel("1-Year Simulated Portfolio Return", color="white", fontsize=12)
ax.set_ylabel("Number of Simulations", color="white", fontsize=12)
ax.set_title(
    "Monte Carlo Simulation — Optimal Portfolio (SPY 44.3% / GLD 55.7%)\n"
    "10,000 Paths over 252 Trading Days",
    color="white", fontsize=13, pad=14,
)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")

legend = ax.legend(framealpha=0.25, labelcolor="white",
                   facecolor="#1a1d27", edgecolor="#555", fontsize=10)

# Annotation box for VaR values
bbox_props = dict(boxstyle="round,pad=0.5", facecolor="#1a1d27",
                  edgecolor="#ff3333", alpha=0.85)
ax.text(
    var_95 - 0.004, ax.get_ylim()[1] * 0.78 if ax.get_ylim()[1] else 600,
    f"95% VaR\n{var_95*100:+.2f}%",
    color="#ff4444", fontsize=10, ha="right", va="top", bbox=bbox_props,
)

plt.tight_layout()
out_path = "monte_carlo_var.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\nPlot saved to: {out_path}")
