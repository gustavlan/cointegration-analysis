import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from coint_tests import engle_granger
from threshold_optimization import optimize_thresholds, plot_threshold_tradeoff

# If available, reuse the normalizer
try:
    from backtest import normalize_returns_for_beta
except Exception:
    def normalize_returns_for_beta(returns, scale_window=63, clip_sigma=5.0):
        r = returns.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        vol = r.rolling(scale_window, min_periods=max(10, scale_window // 3)).std().replace(0, np.nan)
        r_norm = r / (vol + 1e-12)
        mu = r_norm.rolling(scale_window, min_periods=max(10, scale_window // 3)).mean()
        sd = r_norm.rolling(scale_window, min_periods=max(10, scale_window // 3)).std()
        upper = mu + clip_sigma * sd
        lower = mu - clip_sigma * sd
        r_norm = np.minimum(np.maximum(r_norm, lower), upper)
        return r_norm.fillna(0.0)


def _safe_rolling_beta(strategy_returns: pd.Series, bench_returns: pd.Series, window: int) -> pd.Series:
    """Calculate safe rolling beta that prevents blow-ups."""
    sr = normalize_returns_for_beta(strategy_returns, scale_window=max(20, window // 2))
    br = bench_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cov = sr.rolling(window, min_periods=max(10, window // 3)).cov(br)
    var = br.rolling(window, min_periods=max(10, window // 3)).var()
    beta = cov / (var + 1e-8)  # epsilon avoid blow-ups
    return beta.clip(lower=-10, upper=10)


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


def plot_performance(returns, sharpe_window=63, beta_window=126, pair_name=None,
                     market_ticker='^GSPC', risk_free_ticker='^IRX'):
    """
    Plot strategy performance metrics in a single row of subplots.
    Enhanced to compute rolling beta vs S&P500 excess returns.
    """
    # Create figure with 3 subplots in a row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Performance Metrics - {pair_name}', fontsize=14)
    
    # Clean and prepare returns data
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate cumulative returns without using log
    cumulative = (1 + returns).cumprod()
    
    # Calculate drawdown
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    
    # Calculate rolling metrics with handling for edge cases
    rolling_mean = returns.rolling(sharpe_window, min_periods=1).mean()
    rolling_std = returns.rolling(sharpe_window, min_periods=1).std()
    rolling_sharpe = np.sqrt(252) * np.where(
        rolling_std > 0,
        rolling_mean / rolling_std,
        0
    )
    
    # Calculate rolling beta vs S&P 500 excess returns
    try:
        import yfinance as yf
        start, end = returns.index.min(), returns.index.max()
        
        # Fetch market and risk-free data
        market = yf.download(market_ticker, start=start, end=end, auto_adjust=True, progress=False)['Close']
        rf_rate = yf.download(risk_free_ticker, start=start, end=end, auto_adjust=True, progress=False)['Close']
        
        # Calculate market returns and risk-free rate
        market_ret = market.pct_change().fillna(0)
        rf_daily = ((1 + rf_rate/100) ** (1/252) - 1).fillna(0)  # Convert annual % to daily
        
        # Calculate excess returns
        market_excess = (market_ret - rf_daily).reindex(returns.index).fillna(0)
        strategy_excess = returns - rf_daily.reindex(returns.index).fillna(0)
        
        # Rolling beta calculation on excess returns
        def rolling_beta_excess(strat_ex, mkt_ex, window):
            cov = strat_ex.rolling(window, min_periods=max(10, window//3)).cov(mkt_ex)
            var = mkt_ex.rolling(window, min_periods=max(10, window//3)).var()
            return (cov / var.replace(0, np.nan)).fillna(0).clip(-10, 10)
        
        rolling_beta = rolling_beta_excess(strategy_excess, market_excess, beta_window)
        
    except Exception as e:
        print(f"Warning: Could not fetch market data for beta calculation: {e}")
        # Fallback if benchmarks are unavailable
        rolling_beta = pd.Series(0, index=returns.index)
    
    # Plot 1: Drawdown
    ax1.plot(drawdown, color='crimson', linewidth=1.5)
    ax1.set_title('Strategy Drawdown')
    ax1.set_ylabel('Drawdown %')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rolling Sharpe
    ax2.plot(rolling_sharpe, color='darkblue', linewidth=1.5)
    ax2.set_title(f'Rolling Sharpe ({sharpe_window}d)')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling Beta vs S&P 500
    ax3.plot(rolling_beta, color='darkgreen', linewidth=1.5)
    ax3.set_title(f'Rolling Beta vs S&P 500 ({beta_window}d)')
    ax3.set_ylabel('Beta')
    ax3.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def analyze_pairs_nb(all_data, selected,
                     Z_min=0.5, Z_max=3.0, dZ=0.1, cost=0.0, use_ou=True):
    """
    Analysis of selected 2-asset pairs with optional OU parameter usage.
    """
    from coint_tests import ou_params
    summary = []
    opt_tables = {}

    # Vectorized processing using list comprehension
    def process_pair_analysis(pair):
        df = all_data.get(pair)
        if df is None or df.shape[1] != 2:
            print(f"'{pair}': not found or not a 2-asset series.")
            return None, None

        y_col, x_col = df.columns
        eg = engle_granger(df, y_col, x_col)
        spread, beta = eg['spread'], eg['beta']
        if spread is None:
            print(f"'{pair}' not cointegrated (p={eg['eg_pvalue']:.3f}), skipping.")
            return None, None

        # Get OU parameters for potential use
        ou = ou_params(spread)
        mu_ou, sigma_ou = ou['ou_mu'], ou['ou_sigma']
        
        # Use sample statistics as fallback
        mu_sample, sigma_sample = spread.mean(), spread.std()
        
        opt_df = optimize_thresholds(
            spread, mu_sample, sigma_sample, beta,
            y=df[y_col], x=df[x_col],
            Z_min=Z_min, Z_max=Z_max, dZ=dZ, cost=cost,
            ou_mu=mu_ou, ou_sigma=sigma_ou, use_ou=use_ou
        )

        # Pick best-Z by cum_PnL
        best = opt_df.loc[opt_df['cum_PnL'].idxmax()]
        summary_entry = {
            'pair':     pair,
            'best_Z':   best['Z'],
            'N_trades': best['N_trades'],
            'cum_PnL':  best['cum_PnL'],
            'avg_PnL':  best['avg_PnL']
        }

        # Create and show plot
        fig = plot_threshold_tradeoff(opt_df)
        fig.suptitle(f"Tradeoff: {pair}", y=1.02)
        plt.show()

        return summary_entry, (pair, opt_df)

    # Process all pairs and separate results
    results = [process_pair_analysis(pair) for pair in selected]
    
    # Filter out None results and separate summary and opt_tables
    valid_results = [(summary, opt) for summary, opt in results if summary is not None]
    summary = [s for s, _ in valid_results]
    opt_tables = dict(o for _, o in valid_results)

    summary_df = pd.DataFrame(summary)

    return summary_df, opt_tables


def safe_metrics(returns, name):
    """Calculate safe performance metrics that prevent blow-ups."""
    clean_ret = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    clean_ret = np.clip(clean_ret, -0.5, 0.5)
    
    if clean_ret.std() == 0:
        return {"strategy": name, "max_dd": 0, "sharpe": 0, "vol": 0, "total_ret": 0}
    
    cum = (1 + clean_ret).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    sharpe = np.sqrt(252) * clean_ret.mean() / clean_ret.std()
    vol = clean_ret.std() * np.sqrt(252)
    total_ret = cum.iloc[-1] - 1
    
    return {
        "strategy": name,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "vol": vol,
        "total_ret": total_ret
    }
