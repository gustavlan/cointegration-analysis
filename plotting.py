import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from coint_tests import engle_granger, ou_params
from threshold_optimization import optimize_thresholds, plot_threshold_tradeoff

def normalize_returns_for_beta(returns, scale_window=63, clip_sigma=5.0):
    r = returns.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol = r.rolling(scale_window, min_periods=max(10, scale_window // 3)).std().replace(0, np.nan)
    r_norm = r / (vol + 1e-12)
    mu = r_norm.rolling(scale_window, min_periods=max(10, scale_window // 3)).mean()
    sd = r_norm.rolling(scale_window, min_periods=max(10, scale_window // 3)).std()
    upper, lower = mu + clip_sigma * sd, mu - clip_sigma * sd
    return np.minimum(np.maximum(r_norm, lower), upper).fillna(0.0)

def _safe_rolling_beta(strategy_returns: pd.Series, bench_returns: pd.Series, window: int) -> pd.Series:
    sr = normalize_returns_for_beta(strategy_returns, scale_window=max(20, window // 2))
    br = bench_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cov = sr.rolling(window, min_periods=max(10, window // 3)).cov(br)
    var = br.rolling(window, min_periods=max(10, window // 3)).var()
    return (cov / (var + 1e-8)).clip(lower=-10, upper=10)

def _fetch_benchmarks(index):
    try:
        start, end = index.min(), index.max()
        spx_data = yf.download('^GSPC', start=start, end=end, auto_adjust=True, progress=False)
        irx_data = yf.download('^IRX', start=start, end=end, auto_adjust=True, progress=False)

        if len(spx_data) == 0 or len(irx_data) == 0 or 'Close' not in spx_data.columns or 'Close' not in irx_data.columns:
            raise ValueError('Downloaded data is empty or missing Close column')
        
        spx = spx_data['Close']
        irx = irx_data['Close']
        
        if isinstance(spx, pd.DataFrame):
            spx = spx.iloc[:, 0]
        if isinstance(irx, pd.DataFrame):
            irx = irx.iloc[:, 0]

        spx_ret = spx.pct_change()
        rf_daily = ((1 + irx/100) ** (1/252) - 1)
        df = pd.concat([spx_ret, rf_daily], axis=1).dropna()
        excess = df.iloc[:, 0] - df.iloc[:, 1]
        return excess.reindex(index).dropna()
    except Exception as e:
        print(f"Warning: Could not fetch benchmark data: {e}")
        return pd.Series(0, index=index, name='spx_exc')


def plot_drawdown(strat_ret):
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
    spx_exc = _fetch_benchmarks(strat_ret.index)
    df = pd.concat([strat_ret.rename('strat'), spx_exc], axis=1).dropna()
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
        market_data = yf.download(market_ticker, start=start, end=end, auto_adjust=True, progress=False)
        rf_data = yf.download(risk_free_ticker, start=start, end=end, auto_adjust=True, progress=False)
        
        # Check if downloads were successful and extract Close data
        if (len(market_data) == 0 or len(rf_data) == 0 or 
            'Close' not in market_data.columns or 'Close' not in rf_data.columns):
            raise ValueError('Downloaded market data is empty or missing Close column')
        
        market = market_data['Close']
        rf_rate = rf_data['Close']
        
        # Ensure we got Series, not DataFrames (handle MultiIndex case)
        if isinstance(market, pd.DataFrame):
            market = market.iloc[:, 0]  # Take first column if it's a DataFrame
        if isinstance(rf_rate, pd.DataFrame):
            rf_rate = rf_rate.iloc[:, 0]  # Take first column if it's a DataFrame
        
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


def analyze_pairs_nb(all_data, selected, Z_min=0.5, Z_max=3.0, dZ=0.1, cost=0.0, use_ou=True, normalize: bool = False):
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

        ou = ou_params(spread)
        mu_ou, sigma_ou = ou['ou_mu'], ou['ou_sigma']
        mu_sample, sigma_sample = spread.mean(), spread.std()
        
        opt_df = optimize_thresholds(
            spread, mu_sample, sigma_sample, beta,
            y=df[y_col], x=df[x_col],
            Z_min=Z_min, Z_max=Z_max, dZ=dZ, cost=cost,
            ou_mu=mu_ou, ou_sigma=sigma_ou, use_ou=use_ou,
            normalize=normalize
        )

        best = opt_df.loc[opt_df['cum_PnL'].idxmax()]
        summary.append({
            'pair': pair, 'best_Z': best['Z'], 'N_trades': best['N_trades'],
            'cum_PnL': best['cum_PnL'], 'avg_PnL': best['avg_PnL'],
            'theta': ou['ou_theta'], 'half_life': ou['OU_HalfLife']
        })

        fig = plot_threshold_tradeoff(opt_df)
        fig.suptitle(f"Tradeoff: {pair}", y=1.02)
        plt.show()
        opt_tables[pair] = opt_df

    return pd.DataFrame(summary), opt_tables

def safe_metrics(returns, name):
    clean_ret = returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    clean_ret = np.clip(clean_ret, -0.5, 0.5)
    
    if clean_ret.std() == 0:
        return {"strategy": name, "max_dd": 0, "sharpe": 0, "vol": 0, "total_ret": 0}
    
    cum = (1 + clean_ret).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    sharpe = np.sqrt(252) * clean_ret.mean() / clean_ret.std()
    vol = clean_ret.std() * np.sqrt(252)
    total_ret = cum.iloc[-1] - 1
    
    return {"strategy": name, "max_dd": max_dd, "sharpe": sharpe, "vol": vol, "total_ret": total_ret}


def plot_systematic_performance(stitched_results, selected_pairs, benchmark_returns, 
                               compute_rolling_sharpe, compute_rolling_beta,
                               title="Pairs Trading Strategy Performance Analysis - Stitched CV Results"):
    """
    Plot systematic performance analysis with 4 charts per pair showing fold boundaries.
    """
    # Create comprehensive performance plots using stitched data
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    colors = ['blue', 'red', 'green'] 
    rolling_summary = []

    for i, pair_name in enumerate(selected_pairs):
        stitched = stitched_results[pair_name]
        boundaries = stitched['fold_boundaries']
        
        # 1. Equity Curves
        equity = stitched['cumulative_returns']
        axes[0, i].plot(equity, color=colors[i], linewidth=1)
        
        # Add fold boundaries as dashed vertical lines
        for boundary in boundaries:
            axes[0, i].axvline(x=boundary, color=colors[i], linestyle='--', alpha=0.6, linewidth=1)
        
        axes[0, i].set_title(f'{pair_name.replace("_", " ").title()} - Equity Curve')
        axes[0, i].set_ylabel('Cumulative Return')
        axes[0, i].grid(True, alpha=0.3)
        
        # 2. Drawdown Plots
        drawdown = stitched['drawdowns']['drawdown']
        axes[1, i].fill_between(drawdown.index, drawdown * 100, 0, 
                               color=colors[i], alpha=0.3)
        axes[1, i].plot(drawdown.index, drawdown * 100, color=colors[i], linewidth=1)
        
        # Add fold boundaries
        for boundary in boundaries:
            axes[1, i].axvline(x=boundary, color=colors[i], linestyle='--', alpha=0.6, linewidth=1)
        
        axes[1, i].set_title(f'{pair_name.replace("_", " ").title()} - Drawdown')
        axes[1, i].set_ylabel('Drawdown (%)')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].invert_yaxis()
        
        # 3. Rolling Sharpe Ratio
        strategy_returns = stitched['strategy_returns'].dropna()
        rolling_sharpe = compute_rolling_sharpe(strategy_returns, window=126)
        axes[2, i].plot(rolling_sharpe, color=colors[i], linewidth=2)
        axes[2, i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add fold boundaries
        for boundary in boundaries:
            axes[2, i].axvline(x=boundary, color=colors[i], linestyle='--', alpha=0.6, linewidth=1)
        
        axes[2, i].set_title(f'{pair_name.replace("_", " ").title()} - Rolling Sharpe (6m)')
        axes[2, i].set_ylabel('Sharpe Ratio')
        axes[2, i].grid(True, alpha=0.3)
        
        # 4. Rolling Beta vs S&P 500
        test_period_sp500 = benchmark_returns[strategy_returns.index]
        rolling_beta = compute_rolling_beta(strategy_returns, test_period_sp500, window=126)
        axes[3, i].plot(rolling_beta, color=colors[i], linewidth=2)
        axes[3, i].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        axes[3, i].axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)
        axes[3, i].axhline(y=-0.1, color='gray', linestyle='--', alpha=0.5)
        
        # Add fold boundaries
        for boundary in boundaries:
            axes[3, i].axvline(x=boundary, color=colors[i], linestyle='--', alpha=0.6, linewidth=1)
        
        axes[3, i].set_title(f'{pair_name.replace("_", " ").title()} - Rolling Beta vs S&P500 (6m)')
        axes[3, i].set_ylabel('Beta')
        axes[3, i].set_xlabel('Date')
        axes[3, i].grid(True, alpha=0.3)
        
        # Collect rolling summary statistics
        rolling_summary.append({
            'Pair': pair_name.replace('_', ' ').title(),
            'Avg_Rolling_Sharpe': f"{rolling_sharpe.mean():.2f}",
            'Sharpe_Volatility': f"{rolling_sharpe.std():.2f}",
            'Avg_Rolling_Beta': f"{rolling_beta.mean():.3f}",
            'Beta_Volatility': f"{rolling_beta.std():.3f}",
            'Beta_95_Conf': f"[{rolling_beta.quantile(0.025):.3f}, {rolling_beta.quantile(0.975):.3f}]"
        })

    plt.tight_layout()
    plt.show()
    
    # Return rolling performance summary
    rolling_df = pd.DataFrame(rolling_summary)
    print(rolling_df.to_string(index=False))
    
    return rolling_df


def plot_kalman_beta_evolution(kalman_analysis, selected_pairs):
    """
    Plot the evolution of Kalman filter beta estimates for adaptive cointegration.
    """
    # Plot the evolution of beta estimates for each pair
    fig, axes = plt.subplots(len(selected_pairs), 1, figsize=(12, 4*len(selected_pairs)))
    if len(selected_pairs) == 1:
        axes = [axes]
    
    for i, pair_name in enumerate(selected_pairs):
        detailed = kalman_analysis['detailed_results'][pair_name]
        adaptive_betas = detailed['adaptive_betas']
        initial_beta = detailed['initial_beta']
        final_beta = detailed['final_beta']
        
        # Plot adaptive beta evolution
        axes[i].plot(adaptive_betas.index, adaptive_betas.values, 
                     label='Adaptive β (Kalman)', linewidth=2, color='blue')
        
        axes[i].set_title(f'{pair_name}: Evolution of Cointegrating Weight β\n'
                         f'(Drift: {final_beta-initial_beta:+.4f}, Vol: {detailed["beta_volatility"]:.4f})')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Beta')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    