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
                   ) -> dict:
    """
    Backtest a simple mean-reversion strategy on the spread e_t:
      - Enter SHORT when e_t > mu + Z*sigma
      - Enter LONG  when e_t < mu - Z*sigma
      - Exit when e_t crosses mu
    Returns a dict with N_trades, cum_PnL, avg_PnL, avg_duration.
    """
    upper = mu + Z * sigma
    lower = mu - Z * sigma

    in_trade    = False
    direction   = 0         # +1 for long, -1 for short
    entry_idx   = None
    entry_y     = entry_x = None

    pnls        = []
    durations   = []

    for t in range(1, len(e)):
        et = e.iloc[t]
        idx = e.index[t]

        if not in_trade:
            if et > upper:
                in_trade, direction = True, -1
            elif et < lower:
                in_trade, direction = True, +1
            if in_trade:
                entry_idx   = idx
                entry_y     = y.loc[idx]
                entry_x     = x.loc[idx]
        else:
            # exit signal
            if (direction == 1  and et >= mu) or \
               (direction == -1 and et <= mu):
                exit_y = y.loc[idx]
                exit_x = x.loc[idx]
                raw_pnl = direction * ((exit_y - entry_y)
                                       - beta * (exit_x - entry_x))
                pnl = raw_pnl - cost
                pnls.append(pnl)
                durations.append((idx - entry_idx).days)
                in_trade = False

    N          = len(pnls)
    cum_PnL    = np.nansum(pnls)
    avg_PnL    = np.nan if N == 0 else np.nanmean(pnls)
    avg_dur    = np.nan if not durations else np.nanmean(durations)

    return {
        'N_trades':     N,
        'cum_PnL':      cum_PnL,
        'avg_PnL':      avg_PnL,
        'avg_duration': avg_dur
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
                       ) -> pd.DataFrame:
    """
    Sweep Z from Z_min to Z_max in steps of dZ,
    backtest each, and return a DataFrame of results.
    """
    Zs      = np.arange(Z_min, Z_max + dZ, dZ)
    records = []

    for Z in Zs:
        stats      = backtest_spread(e, mu, sigma, beta, y, x, Z, cost)
        stats['Z'] = Z
        records.append(stats)

    return pd.DataFrame(records)

def plot_threshold_tradeoff(df_res: pd.DataFrame) -> plt.Figure:
    """
    Dual axis plot: cum_PnL vs Z (left) and N_trades vs Z (right).
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
