import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def backtest_spread(e, mu, sigma, beta, y, x, Z, cost=0.0, normalize=False):
    e = pd.Series(e).astype(float).dropna()
    mu, sigma = float(mu), float(sigma) if np.isfinite(sigma) and sigma != 0 else np.nan
    
    if not np.isfinite(sigma):
        return {'N_trades': 0, 'cum_PnL': 0.0, 'avg_PnL': np.nan, 'avg_duration': np.nan}
    
    upper, lower = mu + Z * sigma, mu - Z * sigma
    signals = pd.Series(0, index=e.index)
    signals[e < lower], signals[e > upper] = 1, -1
    
    prev_sig = signals.shift(1).fillna(0)
    de = e.diff().fillna(0.0)
    dX = de / sigma if normalize and sigma > 0 else de
    turn = (signals - prev_sig).abs()
    ret = prev_sig * dX - cost * turn
    
    entries = ((prev_sig == 0) & (signals != 0))
    exits = ((prev_sig != 0) & (signals == 0))
    entry_idx, exit_idx = list(np.where(entries)[0]), list(np.where(exits)[0])
    
    trade_pnls, durations = [], []
    j = 0
    for i in entry_idx:
        while j < len(exit_idx) and exit_idx[j] <= i:
            j += 1
        if j < len(exit_idx):
            k = exit_idx[j]
            trade_pnls.append(float(ret.iloc[i:k+1].sum()))
            durations.append(k - i)
            j += 1
    
    return {
        'N_trades': len(trade_pnls), 
        'cum_PnL': float(ret.sum()),
        'avg_PnL': float(np.mean(trade_pnls)) if trade_pnls else np.nan,
        'avg_duration': float(np.mean(durations)) if durations else np.nan
    }

def optimize_thresholds(e, mu, sigma, beta, y, x, Z_min=0.5, Z_max=3.0, dZ=0.1, cost=0.0, 
                       ou_mu=None, ou_sigma=None, use_ou=False, normalize=False):
    final_mu = ou_mu if use_ou and ou_mu is not None else mu
    final_sigma = ou_sigma if use_ou and ou_sigma is not None else sigma
    
    Zs = np.arange(Z_min, Z_max + dZ, dZ)
    records = [{**backtest_spread(e, final_mu, final_sigma, beta, y, x, Z, cost, normalize), 'Z': Z} for Z in Zs]
    return pd.DataFrame(records)

def plot_threshold_tradeoff(df_res):
    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(df_res['Z'], df_res['cum_PnL'], label='Cumulative P&L')
    ax1.set_xlabel('Z')
    ax1.set_ylabel('Cumulative P&L')
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    ax2.plot(df_res['Z'], df_res['N_trades'], '--', label='Number of Trades')
    ax2.set_ylabel('Number of Trades')
    
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    fig.legend(lines, labels, loc='upper right')
    fig.tight_layout()
    return fig