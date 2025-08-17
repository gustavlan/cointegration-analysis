import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def backtest_spread(e: pd.Series, mu: float, sigma: float, beta: float, y: pd.Series, x: pd.Series, 
                   Z: float, cost: float = 0.0, normalize: bool = False):
    e = pd.Series(e).astype(float).dropna()
    mu, sigma = float(mu), float(sigma) if np.isfinite(sigma) and sigma != 0 else np.nan

    upper = mu + Z * sigma if np.isfinite(sigma) else np.nan
    lower = mu - Z * sigma if np.isfinite(sigma) else np.nan

    signals = pd.Series(0, index=e.index, dtype=int)
    if np.isfinite(upper) and np.isfinite(lower):
        signals[e < lower] = 1
        signals[e > upper] = -1
    else:
        return {'N_trades': 0, 'cum_PnL': 0.0, 'avg_PnL': np.nan, 'avg_duration': np.nan}

    prev_sig = signals.shift(1).fillna(0).astype(int)
    de = e.diff().fillna(0.0)

    dX = de / sigma if normalize and np.isfinite(sigma) and sigma > 0 else de

    turn = (signals - prev_sig).abs().astype(float)
    ret = prev_sig.astype(float) * dX - cost * turn

    # Identify round-trip entries and exits
    entries_mask = (prev_sig == 0) & (signals != 0)
    exits_mask   = (prev_sig != 0) & (signals == 0)
    entry_idx = list(np.where(entries_mask)[0])
    exit_idx  = list(np.where(exits_mask)[0])

    # Align exits to the next exit after each entry
    trade_pnls = []
    durations = []
    j = 0
    for i in entry_idx:
        while j < len(exit_idx) and exit_idx[j] <= i:
            j += 1
        if j < len(exit_idx):
            k = exit_idx[j]
            # Include entry through exit days so costs align with cum_PnL
            trade_pnls.append(float(ret.iloc[i:k+1].sum()))
            durations.append(k - i)
            j += 1
        else:
            # No exit found; close at last bar (conservative include tail)
            k = len(ret) - 1
            if k > i:
                trade_pnls.append(float(ret.iloc[i:k+1].sum()))
                durations.append(k - i)

    N = len(trade_pnls)
    cum = float(ret.sum())
    avg = float(np.mean(trade_pnls)) if N > 0 else np.nan
    dur = float(np.mean(durations)) if N > 0 else np.nan

    return {
        'N_trades': N,
        'cum_PnL': cum,
        'avg_PnL': avg,
        'avg_duration': dur
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
                        use_ou: bool = False,
                        normalize: bool = False
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
        {**backtest_spread(e, final_mu, final_sigma, beta, y, x, Z, cost, normalize=normalize), 'Z': Z}
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