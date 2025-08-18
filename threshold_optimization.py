import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def backtest_spread(e, mu, sigma, beta, y, x, Z, cost=0.0, normalize=False):
    """Backtest spread trading strategy with specified threshold and cost parameters."""
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
    """Optimize trading thresholds across a range of Z values and return performance metrics."""
    final_mu = ou_mu if use_ou and ou_mu is not None else mu
    final_sigma = ou_sigma if use_ou and ou_sigma is not None else sigma  # use OU params if available
    
    Zs = np.arange(Z_min, Z_max + dZ, dZ)  # threshold range
    records = [{**backtest_spread(e, final_mu, final_sigma, beta, y, x, Z, cost, normalize), 'Z': Z} for Z in Zs]
    return pd.DataFrame(records)

def plot_threshold_tradeoff(df_res):
    """Plot the tradeoff between cumulative P&L and number of trades across threshold values."""
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