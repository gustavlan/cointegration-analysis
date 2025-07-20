import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


def _fetch_benchmarks(index):
    """
    Download S&P 500 (^GSPC) and 3M T-bill (^IRX),
    compute daily excess returns, and align to `index`.
    Returns a pd.Series of excess returns named 'spx_exc'.
    """
    # Download data
    start, end = index.min(), index.max()
    spx = yf.download('^GSPC', start=start, end=end, auto_adjust=True)['Close']
    irx = yf.download('^IRX', start=start, end=end, auto_adjust=True)['Close']

    # Compute market returns
    spx_ret = spx.pct_change()
    spx_ret.name = 'spx_ret'

    # Convert annual yield% to daily risk-free rate
    rf_daily = ((1 + irx/100) ** (1/252) - 1)
    rf_daily.name = 'rf'

    # Align and compute excess
    df = pd.concat([spx_ret, rf_daily], axis=1).dropna()
    excess = df.iloc[:, 0] - df.iloc[:, 1]
    excess.name = 'spx_exc'

    # Reindex to strategy dates and drop NA
    return excess.reindex(index).dropna()


def plot_drawdown(strat_ret):
    """
    Plot the drawdown of a strategy return series.
    """
    cum = (1 + strat_ret).cumprod()
    dd = cum - cum.cummax()

    plt.figure()
    plt.plot(dd)
    plt.title("Strategy Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_rolling_sharpe(strat_ret, window=63):
    """
    Plot rolling Sharpe ratio (annualized) over a fixed window.
    """
    roll_mean = strat_ret.rolling(window).mean()
    roll_std = strat_ret.rolling(window).std()
    roll_sh = roll_mean / roll_std * np.sqrt(252)

    plt.figure()
    plt.plot(roll_sh)
    plt.title(f"Rolling Sharpe Ratio ({window}-day)")
    plt.xlabel("Date")
    plt.ylabel("Sharpe")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_rolling_beta(strat_ret, window=126):
    """
    Plot rolling beta of strategy vs S&P 500 excess returns.
    """
    # Fetch benchmark excess returns
    spx_exc = _fetch_benchmarks(strat_ret.index)

    # Combine and drop NA
    df = pd.concat([strat_ret.rename('strat'), spx_exc], axis=1).dropna()

    # Rolling covariance and variance
    cov = df['strat'].rolling(window).cov(df['spx_exc'])
    var = df['spx_exc'].rolling(window).var()
    beta = cov / var

    plt.figure()
    plt.plot(beta)
    plt.title(f"Rolling β vs S&P 500 Excess ({window}-day)")
    plt.xlabel("Date")
    plt.ylabel("β")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_performance(strat_ret, sharpe_window=63, beta_window=126):
    """
    Generate all three standard performance charts for a strategy:
      1) Drawdown
      2) Rolling Sharpe
      3) Rolling β vs SPX excess
    """
    plot_drawdown(strat_ret)
    plot_rolling_sharpe(strat_ret, window=sharpe_window)
    plot_rolling_beta(strat_ret, window=beta_window)
