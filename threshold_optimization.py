import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def backtest_spread(e, mu, sigma, beta, y, x, Z, cost=0.0, normalize=False):
    """Backtest spread trading strategy with specified threshold and cost parameters.
    
    Simulates a mean reversion trading strategy on the spread between two assets.
    Generates long/short signals when spread deviates beyond specified thresholds
    and calculates P&L including transaction costs.
    
    Args:
        e (array-like): Spread time series (residuals from cointegration regression).
        mu (float): Mean of the spread for threshold calculation.
        sigma (float): Standard deviation of spread for threshold calculation.
        beta (float): Hedge ratio between the two assets.
        y (pd.Series): Price series of dependent variable (not used in calculation,
                      kept for interface consistency).
        x (pd.Series): Price series of independent variable (not used in calculation,
                      kept for interface consistency).
        Z (float): Z-score threshold for trade signals (e.g., 2.0 for 2-sigma bands).
        cost (float, optional): Transaction cost per trade as fraction. Defaults to 0.0.
        normalize (bool, optional): Whether to normalize P&L by spread volatility.
                                   Defaults to False.
    
    Returns:
        dict: Backtest results containing:
            - 'N_trades': Number of completed trades
            - 'cum_PnL': Cumulative profit and loss
            - 'avg_PnL': Average P&L per trade (NaN if no trades)
            - 'avg_duration': Average trade duration in periods (NaN if no trades)
    
    Note:
        Strategy goes long when spread < (mu - Z*sigma) and short when 
        spread > (mu + Z*sigma). Exits when spread reverts to mean.
    
    Example:
        >>> result = backtest_spread(spread, 0.0, 1.0, 1.5, y_prices, x_prices, 2.0)
        >>> print(f"Completed {result['N_trades']} trades with P&L {result['cum_PnL']:.4f}")
    """
    e = pd.Series(e).astype(float).dropna()
    mu, sigma = float(mu), float(sigma) if np.isfinite(sigma) and sigma != 0 else np.nan
    
    if not np.isfinite(sigma):
        return {'N_trades': 0, 'cum_PnL': 0.0, 'avg_PnL': np.nan, 'avg_duration': np.nan}
    
    upper, lower = mu + Z * sigma, mu - Z * sigma  # trading thresholds
    signals = pd.Series(0, index=e.index)
    signals[e < lower], signals[e > upper] = 1, -1  # long when below lower, short when above upper
    
    prev_sig = signals.shift(1).fillna(0)
    de = e.diff().fillna(0.0)  # spread changes
    dX = de / sigma if normalize and sigma > 0 else de  # normalize by volatility if requested
    turn = (signals - prev_sig).abs()  # position changes for transaction costs
    ret = prev_sig * dX - cost * turn  # P&L with transaction costs
    
    # Calculate trade-level statistics
    entries = ((prev_sig == 0) & (signals != 0))  # entry points
    exits = ((prev_sig != 0) & (signals == 0))    # exit points
    entry_idx, exit_idx = list(np.where(entries)[0]), list(np.where(exits)[0])
    
    trade_pnls, durations = [], []
    j = 0
    for i in entry_idx:
        while j < len(exit_idx) and exit_idx[j] <= i:  # find corresponding exit
            j += 1
        if j < len(exit_idx):
            k = exit_idx[j]
            trade_pnls.append(float(ret.iloc[i:k+1].sum()))  # P&L for this trade
            durations.append(k - i)  # trade duration in periods
            j += 1
    
    return {
        'N_trades': len(trade_pnls), 
        'cum_PnL': float(ret.sum()),
        'avg_PnL': float(np.mean(trade_pnls)) if trade_pnls else np.nan,
        'avg_duration': float(np.mean(durations)) if durations else np.nan
    }

def optimize_thresholds(e, mu, sigma, beta, y, x, Z_min=0.5, Z_max=3.0, dZ=0.1, cost=0.0, 
                       ou_mu=None, ou_sigma=None, use_ou=False, normalize=False):
    """Optimize trading thresholds across a range of Z values and return performance metrics.
    
    Tests multiple threshold values to find optimal parameters for a mean reversion
    trading strategy. Can use either sample statistics or Ornstein-Uhlenbeck
    process parameters for threshold calculation.
    
    Args:
        e (array-like): Spread time series to backtest on.
        mu (float): Sample mean of the spread.
        sigma (float): Sample standard deviation of the spread.
        beta (float): Hedge ratio between assets.
        y (pd.Series): Price series of first asset.
        x (pd.Series): Price series of second asset.
        Z_min (float, optional): Minimum Z-score threshold to test. Defaults to 0.5.
        Z_max (float, optional): Maximum Z-score threshold to test. Defaults to 3.0.
        dZ (float, optional): Increment between threshold values. Defaults to 0.1.
        cost (float, optional): Transaction cost per trade. Defaults to 0.0.
        ou_mu (float, optional): OU process mean (used if use_ou=True).
        ou_sigma (float, optional): OU process volatility (used if use_ou=True).
        use_ou (bool, optional): Whether to use OU parameters instead of sample
                                statistics. Defaults to False.
        normalize (bool, optional): Normalize returns by spread volatility.
                                  Defaults to False.
    
    Returns:
        pd.DataFrame: Optimization results with columns:
            - 'Z': Z-score threshold values tested
            - 'N_trades': Number of trades for each threshold
            - 'cum_PnL': Cumulative P&L for each threshold
            - 'avg_PnL': Average P&L per trade
            - 'avg_duration': Average trade duration
    
    Example:
        >>> opt_results = optimize_thresholds(spread, spread.mean(), spread.std(),
        ...                                  1.2, prices_y, prices_x, Z_min=1.0, Z_max=2.5)
        >>> best_z = opt_results.loc[opt_results['cum_PnL'].idxmax(), 'Z']
        >>> print(f"Optimal threshold: {best_z}")
    """
    final_mu = ou_mu if use_ou and ou_mu is not None else mu
    final_sigma = ou_sigma if use_ou and ou_sigma is not None else sigma  # use OU params if available
    
    Zs = np.arange(Z_min, Z_max + dZ, dZ)  # threshold range
    records = [{**backtest_spread(e, final_mu, final_sigma, beta, y, x, Z, cost, normalize), 'Z': Z} for Z in Zs]
    return pd.DataFrame(records)

def plot_threshold_tradeoff(df_res):
    """Plot the tradeoff between cumulative P&L and number of trades across threshold values.
    
    Creates a dual-axis plot showing how cumulative P&L and number of trades
    vary with different Z-score thresholds. Helps visualize the tradeoff between
    profitability and trading frequency.
    
    Args:
        df_res (pd.DataFrame): Results from optimize_thresholds() containing
                              'Z', 'cum_PnL', and 'N_trades' columns.
    
    Returns:
        matplotlib.figure.Figure: Figure object containing the plot.
    
    Note:
        Left y-axis shows cumulative P&L (solid line), right y-axis shows
        number of trades (dashed line). Higher thresholds typically mean
        fewer trades but potentially higher P&L per trade.
    
    Example:
        >>> optimization_results = optimize_thresholds(spread, ...)
        >>> fig = plot_threshold_tradeoff(optimization_results)
        >>> fig.suptitle("Threshold Optimization Results")
        >>> plt.show()
    """
    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(df_res['Z'], df_res['cum_PnL'], label='Cumulative P&L')
    ax1.set_xlabel('Z')
    ax1.set_ylabel('Cumulative P&L')
    ax1.grid(True)
    
    ax2 = ax1.twinx()  # secondary y-axis for trade count
    ax2.plot(df_res['Z'], df_res['N_trades'], '--', label='Number of Trades')
    ax2.set_ylabel('Number of Trades')
    
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc='upper right')
    fig.tight_layout()
    return fig