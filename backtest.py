import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# Rolling time‐series splits
def rolling_time_series_splits(dates, 
                               train_months=18, 
                               test_months=6, 
                               step_months=3):
    """
    Create train_start, train_end, test_start, test_end tuples.
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
    Given index and train/test boundaries, returns
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

    # Compute period returns and cumulative P/L
    ret = signals.shift(1) * spread.diff()
    cum_ret = ret.cumsum()
    mean_ret = ret.mean() * 252 
    vol = ret.std() * np.sqrt(252)
    if vol == 0 or np.isnan(vol): # Annualized Sharpe, make sure not to divide by zero
        ann_sharpe = np.nan
    else:
        ann_sharpe = mean_ret / vol
    max_dd = (cum_ret - cum_ret.cummax()).min() # max drawdown
    turnover = signals.diff().abs().mean() # Turnover
    hit_ratio = (ret[signals.shift(1) != 0] > 0).mean() # hit ratio

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
    df must have a DateTimeIndex and a column `spread_col`
    """
    if z_list is None:
        z_list = [0.5, 1.0, 1.5, 2.0]
    results = []

    for (t0, t1, v0, v1) in rolling_time_series_splits(df.index, **split_kwargs):
        # get masks
        train_mask, test_mask = purged_split_index(df.index, t0, t1, v0, v1)
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
