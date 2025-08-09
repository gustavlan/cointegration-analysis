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
    
    # Vectorized signal detection
    long_signals = et < lower
    short_signals = et > upper
    exit_signals = np.abs(et - mu) <= np.abs(et - mu).min()  # exit when closest to mean
    
    # Process trades using state machine with vectorized operations
    position = np.zeros(len(et))
    for t in range(1, len(et)):
        if position[t-1] == 0:  # Not in trade
            if short_signals[t]:
                position[t] = -1
                entry_idx = dates[t]
                entry_y, entry_x = y.iloc[t], x.iloc[t]
            elif long_signals[t]:
                position[t] = 1
                entry_idx = dates[t] 
                entry_y, entry_x = y.iloc[t], x.iloc[t]
        else:  # In trade
            position[t] = position[t-1]  # Maintain position
            # Exit on mean reversion
            if ((position[t-1] == 1 and et[t] >= mu) or 
                (position[t-1] == -1 and et[t] <= mu)):
                exit_y, exit_x = y.iloc[t], x.iloc[t]
                pnl = position[t-1] * ((exit_y - entry_y) - beta * (exit_x - entry_x)) - cost
                pnls.append(pnl)
                durations.append((dates[t] - entry_idx).days)
                position[t] = 0

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
                        cost: float = 0.0,
                        ou_mu: float = None,
                        ou_sigma: float = None,
                        use_ou: bool = False
                       ):
    """
    Sweep Z from Z_min to Z_max in steps of dZ,
    backtest each, and return a df of results.
    If use_ou=True and ou_mu/ou_sigma provided, use OU equilibrium parameters.
    Otherwise use the provided sample mu/sigma.
    """
    # Use OU parameters if specified and available
    if use_ou and ou_mu is not None and ou_sigma is not None:
        final_mu = ou_mu
        final_sigma = ou_sigma
    else:
        final_mu = mu
        final_sigma = sigma
    
    Zs = np.arange(Z_min, Z_max + dZ, dZ)
    
    # Vectorized threshold optimization using list comprehension
    records = [
        {**backtest_spread(e, final_mu, final_sigma, beta, y, x, Z, cost), 'Z': Z}
        for Z in Zs
    ]

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