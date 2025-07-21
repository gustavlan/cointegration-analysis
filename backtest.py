import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# Rolling time‐series splits
def rolling_time_series_splits(dates, 
                               train_months=18, 
                               test_months=6, 
                               step_months=3):
    """
    Yields (train_start, train_end, test_start, test_end) tuples.
    """
    start = dates.min()
    end   = dates.max()
    train_start = start
    while True:
        train_end = train_start + relativedelta(months=train_months)
        test_end  = train_end   + relativedelta(months=test_months)
        if test_end > end:
            break
        yield (train_start, train_end, train_end, test_end)
        train_start += relativedelta(months=step_months)

# Purged K‐Fold helper
def purged_split_index(df_index, 
                       train_start, train_end, 
                       test_start, test_end, 
                       purge_days=5):
    """
    Given index and exact train/test boundaries, returns
    boolean masks for train and test with a purge zone around test.
    """
    purge_start = test_start - pd.Timedelta(days=purge_days)
    purge_end   = test_end   + pd.Timedelta(days=purge_days)

    test_mask  = (df_index >= test_start) & (df_index < test_end)
    purge_mask = (df_index >= purge_start) & (df_index < purge_end)
    train_mask = ~test_mask & ~purge_mask
    return train_mask, test_mask

# Signal generation Z‐score entry/exit
def generate_signals(spread, mu_e, sigma_eq, z):
    """
    +1 when spread < μ_e - z·σ_eq (long the spread)
    -1 when spread > μ_e + z·σ_eq (short the spread)
     0 otherwise (flat)
    """
    signals = pd.Series(0, index=spread.index)
    signals[spread < mu_e - z*sigma_eq] =  1
    signals[spread > mu_e + z*sigma_eq] = -1
    return signals

# Performance metrics
def performance_metrics(signals, spread):
    """
    Compute returns = signal_{t-1} * Δspread_t
    Then compute:
      • annualized Sharpe (252 trading days)
      • max drawdown
      • turnover = mean absolute signal change
      • hit ratio = P[returns>0 | position!=0]
    """
    ret = signals.shift(1) * spread.diff()
    cum_ret = ret.cumsum()
    
    # Sharpe
    ann_sharpe = (ret.mean() * 252) / (ret.std() * np.sqrt(252))
    
    # Drawdown
    running_max = cum_ret.cummax()
    drawdown   = cum_ret - running_max
    max_dd     = drawdown.min()
    
    # Turnover
    turnover = signals.diff().abs().mean()
    
    # Hit ratio
    mask = signals.shift(1) != 0
    hit_ratio = (ret[mask] > 0).mean()
    
    return {
        'sharpe':       ann_sharpe,
        'max_drawdown': max_dd,
        'turnover':     turnover,
        'hit_ratio':    hit_ratio
    }

# Nested CV over Z‐score thresholds
def nested_cv(df, 
              spread_col='spread', 
              mu_e=None, 
              sigma_eq=None, 
              z_list=None,
              **split_kwargs):
    """
    df must contain a DateTimeIndex and a column `spread_col`.
    mu_e, sigma_eq: pre‐computed scalars (or series) for the full sample.
    z_list: list of z‐score thresholds to grid‐search.
    split_kwargs passed to rolling_time_series_splits.
    """
    if z_list is None:
        z_list = [0.5, 1.0, 1.5, 2.0]
    results = []
    dates = df.index

    for (t0, t1, v0, v1) in rolling_time_series_splits(dates, **split_kwargs):
        # get masks
        train_mask, test_mask = purged_split_index(dates, t0, t1, v0, v1)
        
        train_spread = df.loc[train_mask, spread_col]
        test_spread  = df.loc[test_mask,  spread_col]
        
        # Inner loop: tune z on TRAIN
        best_z, best_score = None, -np.inf
        for z in z_list:
            sig = generate_signals(train_spread, mu_e, sigma_eq, z)
            perf = performance_metrics(sig, train_spread)
            if perf['sharpe'] > best_score:
                best_score = perf['sharpe']
                best_z     = z
        
        # Evaluate on TEST
        test_sig  = generate_signals(test_spread, mu_e, sigma_eq, best_z)
        test_perf = performance_metrics(test_sig, test_spread)
        
        results.append({
            'train_start': t0.date(),
            'train_end':   t1.date(),
            'test_start':  v0.date(),
            'test_end':    v1.date(),
            'best_z':      best_z,
            'train_sharpe':best_score,
            **{f'test_{k}': v for k,v in test_perf.items()}
        })

    return pd.DataFrame(results)
