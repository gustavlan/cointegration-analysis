import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from coint_tests import engle_granger
from threshold_optimization import optimize_thresholds, plot_threshold_tradeoff


def _fetch_benchmarks(index):
    """
    Download S&P 500 (^GSPC) and 3M T-bill (^IRX),
    compute daily excess returns, and align to index.
    """
    start, end = index.min(), index.max()
    spx = yf.download('^GSPC', start=start, end=end, auto_adjust=True)['Close']
    irx = yf.download('^IRX', start=start, end=end, auto_adjust=True)['Close']

    spx_ret = spx.pct_change()
    spx_ret.name = 'spx_ret'
    rf_daily = ((1 + irx/100) ** (1/252) - 1) # Convert annual yield% to daily risk-free rate
    rf_daily.name = 'rf'
    df = pd.concat([spx_ret, rf_daily], axis=1).dropna()
    excess = df.iloc[:, 0] - df.iloc[:, 1]
    excess.name = 'spx_exc'

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
    spx_exc = _fetch_benchmarks(strat_ret.index)
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
    Drawdown, Rolling Sharpe, Rolling β vs SPX excess
    """
    plot_drawdown(strat_ret)
    plot_rolling_sharpe(strat_ret, window=sharpe_window)
    plot_rolling_beta(strat_ret, window=beta_window)


def analyze_pairs_nb(all_data, selected,
                     Z_min=0.5, Z_max=3.0, dZ=0.1, cost=0.0):
    """
    Analysis of selected 2-asset pairs.
    """
    summary = []
    opt_tables = {}

    for pair in selected:
        df = all_data.get(pair)
        if df is None or df.shape[1] != 2:
            print(f"'{pair}': not found or not a 2-asset series.")
            continue

        y_col, x_col = df.columns
        eg = engle_granger(df, y_col, x_col)
        spread, beta = eg['spread'], eg['beta']
        if spread is None:
            print(f"'{pair}' not cointegrated (p={eg['eg_pvalue']:.3f}), skipping.")
            continue

        mu, sigma = spread.mean(), spread.std()
        opt_df = optimize_thresholds(
            spread, mu, sigma, beta,
            y=df[y_col], x=df[x_col],
            Z_min=Z_min, Z_max=Z_max, dZ=dZ, cost=cost
        )
        opt_tables[pair] = opt_df

        # pick best-Z by cum_PnL
        best = opt_df.loc[opt_df['cum_PnL'].idxmax()]
        summary.append({
            'pair':     pair,
            'best_Z':   best['Z'],
            'N_trades': best['N_trades'],
            'cum_PnL':  best['cum_PnL'],
            'avg_PnL':  best['avg_PnL']
        })

        fig = plot_threshold_tradeoff(opt_df)
        fig.suptitle(f"Tradeoff: {pair}", y=1.02)
        plt.show()

    summary_df = pd.DataFrame(summary)

    return summary_df, opt_tables
