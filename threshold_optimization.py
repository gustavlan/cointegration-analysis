import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def backtest_spread(e: pd.Series,
                    mu: float,
                    sigma: float,
                    beta: float,
                    y: pd.Series,
                    x: pd.Series,
                    Z: float,
                    cost: float = 0.0
                   ):
    """
    Backtest a simple mean-reversion strategy on the spread e_t
    """
    upper = mu + Z * sigma
    lower = mu - Z * sigma
    
    # Pre-calculate arrays for better performance
    et = e.values
    dates = e.index
    pnls = []
    durations = []
    
    in_trade = False
    direction = 0
    entry_idx = None
    entry_y = entry_x = 0
    
    # Use numpy operations for better performance
    for t in range(1, len(et)):
        if not in_trade:
            if et[t] > upper:
                in_trade, direction = True, -1
            elif et[t] < lower:
                in_trade, direction = True, +1
            if in_trade:
                entry_idx = dates[t]
                entry_y = y.iloc[t]
                entry_x = x.iloc[t]
        else:
            if (direction == 1 and et[t] >= mu) or (direction == -1 and et[t] <= mu):
                exit_y = y.iloc[t]
                exit_x = x.iloc[t]
                pnls.append(direction * ((exit_y - entry_y) - beta * (exit_x - entry_x)) - cost)
                durations.append((dates[t] - entry_idx).days)
                in_trade = False

    N = len(pnls)
    if N == 0:
        return {'N_trades': 0, 'cum_PnL': 0, 'avg_PnL': np.nan, 'avg_duration': np.nan}
        
    pnls = np.array(pnls)  # Convert to numpy array for faster calculations
    return {
        'N_trades': N,
        'cum_PnL': pnls.sum(),
        'avg_PnL': pnls.mean(),
        'avg_duration': np.mean(durations)
    }

def optimize_thresholds(e: pd.Series,
                        mu: float,
                        sigma: float,
                        beta: float,
                        y: pd.Series,
                        x: pd.Series,
                        Z_min: float = 0.5,
                        Z_max: float = 3.0,
                        dZ: float   = 0.1,
                        cost: float = 0.0
                       ):
    """
    Sweep Z from Z_min to Z_max in steps of dZ,
    backtest each, and return a df of results.
    """
    Zs = np.arange(Z_min, Z_max + dZ, dZ)
    records = []

    for Z in Zs:
        stats = backtest_spread(e, mu, sigma, beta, y, x, Z, cost)
        stats['Z'] = Z
        records.append(stats)

    return pd.DataFrame(records)

def plot_threshold_tradeoff(df_res: pd.DataFrame):
    """
    Dual axis plot cum_PnL vs Z (left) and N_trades vs Z (right).
    """
    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(df_res['Z'], df_res['cum_PnL'], label='Cumulative P&L')
    ax1.set_xlabel('Z')
    ax1.set_ylabel('Cumulative P&L')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(df_res['Z'], df_res['N_trades'], '--', label='Number of Trades')
    ax2.set_ylabel('Number of Trades')

    # combine legends
    lines, labels = ax1.get_lines() + ax2.get_lines(), \
                    [l.get_label() for l in ax1.get_lines()+ax2.get_lines()]
    fig.legend(lines, labels, loc='upper right')

    fig.tight_layout()

    return fig