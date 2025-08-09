import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Tuple, Optional
import statsmodels.api as sm

_EPS = 1e-12

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


def enhanced_perf_metrics(returns, rf=0.0, freq=252):
    """
    Enhanced performance metrics including tail risk measures (VaR/ES).
    """
    r = pd.Series(returns).dropna().astype(float)
    if len(r) == 0:
        return {}
    
    # Basic metrics
    mu = r.mean() * freq
    vol = r.std(ddof=0) * np.sqrt(freq)
    sharpe = (mu - rf) / vol if vol > 0 else np.nan
    
    # Drawdown metrics
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    maxdd = dd.min()
    
    # Tail risk metrics
    var_95 = -np.percentile(r, 5) if len(r) > 0 else np.nan
    var_99 = -np.percentile(r, 1) if len(r) > 0 else np.nan
    es_95 = -r[r <= np.percentile(r, 5)].mean() if (r <= np.percentile(r, 5)).any() else np.nan
    es_99 = -r[r <= np.percentile(r, 1)].mean() if (r <= np.percentile(r, 1)).any() else np.nan
    
    hit_rate = (r > 0).mean()
    
    return {
        'CAGR': mu, 'Vol_Ann': vol, 'Sharpe': sharpe,
        'MaxDD': float(maxdd), 'HitRate': hit_rate, 'N': int(len(r)),
        'VaR95': var_95, 'ES95': es_95, 'VaR99': var_99, 'ES99': es_99
    }


def _rolling_zscore(spread: pd.Series, window: int = 126, min_periods: int = 60) -> pd.Series:
    """Calculate rolling z-score of spread."""
    mu = spread.rolling(window, min_periods=min_periods).mean()
    sigma = spread.rolling(window, min_periods=min_periods).std()
    z = (spread - mu) / (sigma.replace(0, np.nan))
    return z


def _rolling_sigma(spread: pd.Series, window: int = 126, min_periods: int = 60) -> pd.Series:
    """Calculate rolling standard deviation of spread."""
    return spread.rolling(window, min_periods=min_periods).std()


def backtest_pair_with_stops(
    df_pair: pd.DataFrame,
    y_col: str = None,
    x_col: str = None,
    beta: float = None,
    spread_col: str = 'spread',
    z_entry: float = 1.5,
    z_exit: float = 0.0,
    z_window: int = 126,
    stop_z: float = None,             # hard stop in Z units (adverse move)
    stop_loss_sigma: float = None,    # exit if loss > k * sigma_at_entry
    equity_dd_stop: float = None,     # flatten if equity DD >= threshold
    cool_off_days: int = 20,          # pause trading after DD breach
    max_holding_days: int = None,     # time stop
    cost: float = 0.0,                # per unit turnover cost in spread units
    clip_return: float = 0.50         # hard clip of daily return for numerical safety
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Stateful pairs backtest with Z-entry/exit, hard Z stop, per-trade loss stop,
    time stop, and strategy-level DD circuit breaker.
    
    Returns:
      returns: strategy daily returns (spread units, dimensionless after risk scaling if any)
      signals: position series in {-1, 0, +1}
      zscore:  rolling z-score used for logic
    """
    df = df_pair.dropna().copy()

    # Build spread from provided column or OLS
    if spread_col and spread_col in df.columns:
        spread = df[spread_col].astype(float)
    else:
        if y_col is None or x_col is None:
            y_col, x_col = df.columns[:2]
        if beta is None:
            beta = sm.OLS(df[y_col], sm.add_constant(df[x_col])).fit().params[x_col]
        spread = (df[y_col] - beta * df[x_col]).astype(float)

    idx = spread.index
    z = _rolling_zscore(spread, window=z_window).copy()
    sigma = _rolling_sigma(spread, window=z_window).reindex(idx)

    dS = spread.diff().fillna(0.0)
    signals = pd.Series(0, index=idx, dtype=int)
    returns = pd.Series(0.0, index=idx, dtype=float)

    pos = 0
    hold = 0
    in_trade_entry_idx = None
    entry_spread = None
    entry_sigma = None

    # Equity curve DD circuit breaker
    equity = 1.0
    peak = 1.0
    drawdown_reset_peak = 1.0  # Track peak for resetting drawdown calculation
    cool_off = 0

    for t in range(len(idx)):
        date = idx[t]
        z_t = z.iloc[t]
        s_t = spread.iloc[t]
        dS_t = dS.iloc[t]

        # DD circuit breaker: pause trading
        just_ended_cooloff = False
        if cool_off > 0:
            pos_next = 0
            cool_off -= 1
            # Mark when cool-off period ends (we'll reset drawdown peak after equity calc)
            if cool_off == 0:
                just_ended_cooloff = True
        else:
            pos_next = pos

            if np.isnan(z_t):
                # Keep pos, no entries while z unavailable
                pass
            else:
                if pos == 0:
                    # Entries
                    if z_t > z_entry:
                        pos_next = -1
                        hold = 0
                        in_trade_entry_idx = date
                        entry_spread = s_t
                        entry_sigma = float(sigma.loc[date]) if not np.isnan(sigma.loc[date]) else None
                    elif z_t < -z_entry:
                        pos_next = +1
                        hold = 0
                        in_trade_entry_idx = date
                        entry_spread = s_t
                        entry_sigma = float(sigma.loc[date]) if not np.isnan(sigma.loc[date]) else None
                else:
                    # Exits on mean reversion band
                    if abs(z_t) <= z_exit:
                        pos_next = 0
                        hold = 0
                        in_trade_entry_idx = None
                        entry_spread = None
                        entry_sigma = None
                    # Hard Z stop in adverse direction
                    elif stop_z is not None:
                        if pos == +1 and z_t <= -stop_z:
                            pos_next = 0
                            hold = 0
                            in_trade_entry_idx = None
                            entry_spread = None
                            entry_sigma = None
                        elif pos == -1 and z_t >= +stop_z:
                            pos_next = 0
                            hold = 0
                            in_trade_entry_idx = None
                            entry_spread = None
                            entry_sigma = None
                    # Per-trade loss stop (sigma-based)
                    if pos_next != 0 and stop_loss_sigma is not None and entry_spread is not None and entry_sigma is not None and entry_sigma > 0:
                        open_pnl = (s_t - entry_spread) * pos  # long:+1 multiplies dS; short:-1 multiplies dS
                        if open_pnl < -stop_loss_sigma * entry_sigma:
                            pos_next = 0
                            hold = 0
                            in_trade_entry_idx = None
                            entry_spread = None
                            entry_sigma = None
                    # Time stop
                    if pos_next != 0 and max_holding_days is not None:
                        hold += 1
                        if hold >= max_holding_days:
                            pos_next = 0
                            hold = 0
                            in_trade_entry_idx = None
                            entry_spread = None
                            entry_sigma = None

        # Build returns with turnover costs
        prev_pos = signals.iloc[t-1] if t > 0 else 0
        signals.iloc[t] = pos_next
        pos = pos_next

        gross = (prev_pos * dS_t)
        turn = abs(pos - prev_pos)
        ret_t = gross - cost * turn
        
        # Clip returns to prevent numerical overflow
        ret_t = np.clip(ret_t, -clip_return, clip_return)  # Cap daily returns
        returns.iloc[t] = ret_t

        # Update equity and check DD stop (only if equity_dd_stop is not None)
        if equity_dd_stop is not None:
            equity = equity * (1.0 + ret_t)
            peak = max(peak, equity)
            
            # Update drawdown reset peak: reset on cool-off end, otherwise track new highs
            if just_ended_cooloff:
                drawdown_reset_peak = equity  # Reset to current equity when cool-off ends
            elif cool_off == 0:
                drawdown_reset_peak = max(drawdown_reset_peak, equity)  # Track new highs when trading
            
            # Calculate drawdown from the reset peak (not the all-time peak)
            dd = (equity - drawdown_reset_peak) / max(drawdown_reset_peak, _EPS)
            
            if dd <= -equity_dd_stop:
                # Trigger circuit breaker: flatten and pause
                pos = 0
                signals.iloc[t] = 0
                cool_off = max(cool_off, cool_off_days)

    return returns.astype(float), signals.astype(int), z.astype(float)


def backtest_pair_with_vol_targeting(
    df_pair: pd.DataFrame,
    y_col: str = None,
    x_col: str = None,
    beta: float = None,
    spread_col: str = 'spread',
    z_entry: float = 1.5,
    z_exit: float = 0.25,
    z_window: int = 126,
    stop_z: float = None,
    stop_loss_sigma: float = None,
    equity_dd_stop: float = None,
    cool_off_days: int = 20,
    max_holding_days: int = None,
    cost: float = 0.002,
    # Vol targeting params
    target_vol: float = 0.02,     # target daily vol (e.g., 2%)
    vol_window: int = 63,
    max_leverage: float = 5.0,
    clip_return: float = 0.10     # hard clip of daily return for numerical safety
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Same logic as backtest_pair_with_stops, but scales position size by 1/realized vol of spread.
    Applies transaction costs on sized position changes.
    """
    df = df_pair.dropna().copy()

    if spread_col and spread_col in df.columns:
        spread = df[spread_col].astype(float)
    else:
        if y_col is None or x_col is None:
            y_col, x_col = df.columns[:2]
        if beta is None:
            beta = sm.OLS(df[y_col], sm.add_constant(df[x_col])).fit().params[x_col]
        spread = (df[y_col] - beta * df[x_col]).astype(float)

    idx = spread.index
    z = _rolling_zscore(spread, window=z_window)
    sigma = _rolling_sigma(spread, window=z_window).reindex(idx)
    dS = spread.diff().fillna(0.0)

    # Realized vol of spread increments
    dS_vol = dS.rolling(vol_window, min_periods=max(10, vol_window // 3)).std().replace(0, np.nan)
    size = (target_vol / (dS_vol + _EPS)).clip(upper=max_leverage)  # unitless leverage
    size = size.fillna(0.0)

    signals = pd.Series(0, index=idx, dtype=int)
    sized_pos = pd.Series(0.0, index=idx, dtype=float)
    returns = pd.Series(0.0, index=idx, dtype=float)

    pos = 0
    hold = 0
    entry_spread = None
    entry_sigma = None
    cool_off = 0
    equity = 1.0
    peak = 1.0
    drawdown_reset_peak = 1.0  # Track peak for resetting drawdown calculation

    for t in range(len(idx)):
        date = idx[t]
        z_t = z.iloc[t]
        s_t = spread.iloc[t]
        dS_t = dS.iloc[t]

        # Pause trading on DD
        just_ended_cooloff = False
        if cool_off > 0:
            pos_next = 0
            cool_off -= 1
            # Mark when cool-off period ends (we'll reset drawdown peak after equity calc)
            if cool_off == 0:
                just_ended_cooloff = True
        else:
            pos_next = pos
            if not np.isnan(z_t):
                if pos == 0:
                    if z_t > z_entry:
                        pos_next = -1
                        hold = 0
                        entry_spread = s_t
                        entry_sigma = float(sigma.loc[date]) if not np.isnan(sigma.loc[date]) else None
                    elif z_t < -z_entry:
                        pos_next = +1
                        hold = 0
                        entry_spread = s_t
                        entry_sigma = float(sigma.loc[date]) if not np.isnan(sigma.loc[date]) else None
                else:
                    if abs(z_t) <= z_exit:
                        pos_next = 0
                        hold = 0
                        entry_spread = None
                        entry_sigma = None
                    elif stop_z is not None:
                        if pos == +1 and z_t <= -stop_z:
                            pos_next = 0
                            hold = 0
                            entry_spread = None
                            entry_sigma = None
                        elif pos == -1 and z_t >= +stop_z:
                            pos_next = 0
                            hold = 0
                            entry_spread = None
                            entry_sigma = None
                    if pos_next != 0 and stop_loss_sigma is not None and entry_spread is not None and entry_sigma is not None and entry_sigma > 0:
                        open_pnl = (s_t - entry_spread) * pos
                        if open_pnl < -stop_loss_sigma * entry_sigma:
                            pos_next = 0
                            hold = 0
                            entry_spread = None
                            entry_sigma = None
                    if pos_next != 0 and max_holding_days is not None:
                        hold += 1
                        if hold >= max_holding_days:
                            pos_next = 0
                            hold = 0
                            entry_spread = None
                            entry_sigma = None

        prev_pos = signals.iloc[t-1] if t > 0 else 0
        signals.iloc[t] = pos_next
        pos = pos_next

        # Sized position and returns
        w_t = size.iloc[t]
        sized_pos.iloc[t] = pos * w_t

        gross = (sized_pos.shift(1).fillna(0.0).iloc[t] * dS_t)
        turn = abs(sized_pos.iloc[t] - sized_pos.shift(1).fillna(0.0).iloc[t])
        ret_t = gross - cost * turn

        # Safety clip
        ret_t = float(np.clip(ret_t, -clip_return, clip_return))
        returns.iloc[t] = ret_t

        equity = equity * (1.0 + ret_t)
        peak = max(peak, equity)
        
        # Update drawdown reset peak: reset on cool-off end, otherwise track new highs
        if just_ended_cooloff:
            drawdown_reset_peak = equity  # Reset to current equity when cool-off ends
        elif cool_off == 0:
            drawdown_reset_peak = max(drawdown_reset_peak, equity)  # Track new highs when trading
        
        # Calculate drawdown from the reset peak (not the all-time peak)
        dd = (equity - drawdown_reset_peak) / max(drawdown_reset_peak, _EPS)
        
        if equity_dd_stop is not None and dd <= -equity_dd_stop:
            # flatten and cool off
            signals.iloc[t] = 0
            sized_pos.iloc[t] = 0.0
            pos = 0
            cool_off = max(cool_off, cool_off_days)

    return returns.astype(float), signals.astype(int), z.astype(float)


def normalize_returns_for_beta(returns: pd.Series, scale_window: int = 63, clip_sigma: float = 5.0) -> pd.Series:
    """
    Convert PnL-in-spread-units to dimensionless returns for beta calc:
    - divide by rolling std
    - winsorize to clip_sigma
    """
    r = returns.copy().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol = r.rolling(scale_window, min_periods=max(10, scale_window // 3)).std().replace(0, np.nan)
    r_norm = r / (vol + _EPS)
    # Winsorize
    mu = r_norm.rolling(scale_window, min_periods=max(10, scale_window // 3)).mean()
    sd = r_norm.rolling(scale_window, min_periods=max(10, scale_window // 3)).std()
    upper = mu + clip_sigma * sd
    lower = mu - clip_sigma * sd
    r_norm = np.minimum(np.maximum(r_norm, lower), upper)
    return r_norm.fillna(0.0)

# -----------------------------
# Extended metrics and helpers
# -----------------------------

TRADING_DAYS = 252

def _drawdown_stats(returns: pd.Series) -> Tuple[pd.Series, float, int]:
    """Compute drawdown series, max drawdown and max duration (in bars)."""
    r = returns.fillna(0)
    eq = (1 + r).cumprod()
    peaks = eq.cummax()
    dd = eq / peaks - 1.0
    max_dd = float(dd.min()) if len(dd) else np.nan
    # duration: count bars since last peak
    at_high = (dd == 0).astype(int)
    grp = (at_high.shift(1) != at_high).cumsum()
    dur = at_high.groupby(grp).cumcount()
    max_dur = int(dur.max()) if len(dur) else 0
    return dd, max_dd, max_dur

def _hist_var_es(returns: pd.Series, alpha: float = 0.95) -> Tuple[float, float]:
    """Historical VaR/ES for a return series. Positive numbers = loss."""
    r = returns.dropna().values
    if r.size == 0:
        return np.nan, np.nan
    q = np.quantile(r, 1 - alpha)
    var = -float(q)
    tail = r[r <= q]
    es = -float(tail.mean()) if tail.size else var
    return var, es

def perf_metrics(returns: pd.Series, freq: int = TRADING_DAYS) -> Dict[str, float]:
    """Comprehensive performance metrics computed from returns.

    Returns a dict including CAGR, Ann.Vol, Sharpe, Sortino, Skew, Kurtosis,
    MaxDD, MaxDD_Days, Calmar, VaR/ES at 95% and 99%, HitRate and N.
    """
    r = returns.dropna()
    n = len(r)
    if n == 0:
        return {k: np.nan for k in [
            'CAGR','Vol_Ann','Sharpe','Sortino','Skew','Kurtosis','MaxDD',
            'MaxDD_Days','Calmar','VaR95','ES95','VaR99','ES99','HitRate','N'
        ]}

    mu = r.mean()
    sd = r.std(ddof=0)
    downside = r[r < 0].std(ddof=0)

    cagr = (1 + r).prod()**(freq / n) - 1 if n > 0 else np.nan
    vol = sd * np.sqrt(freq) if sd > 0 else np.nan
    sharpe = (mu / sd) * np.sqrt(freq) if sd > 0 else np.nan
    sortino = (mu / downside) * np.sqrt(freq) if (isinstance(downside, float) and downside > 0) else np.nan
    skew = float(r.skew())
    kurt = float(r.kurtosis())
    _, max_dd, max_dd_dur = _drawdown_stats(r)
    calmar = (cagr / abs(max_dd)) if (max_dd is not None and isinstance(max_dd, float) and max_dd < 0) else np.nan
    var95, es95 = _hist_var_es(r, 0.95)
    var99, es99 = _hist_var_es(r, 0.99)
    hit_rate = float((r > 0).mean()) if n > 0 else np.nan

    return {
        'CAGR': cagr,
        'Vol_Ann': vol,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Skew': skew,
        'Kurtosis': kurt,
        'MaxDD': max_dd,
        'MaxDD_Days': max_dd_dur,
        'Calmar': calmar,
        'VaR95': var95,
        'ES95': es95,
        'VaR99': var99,
        'ES99': es99,
        'HitRate': hit_rate,
        'N': n,
    }

def generate_pair_returns(all_data: Dict[str, pd.DataFrame],
                          summary_df: pd.DataFrame,
                          selected: List[str]) -> pd.DataFrame:
    """Construct per-pair daily return series using best_Z from summary_df.

    Returns a DataFrame of returns (columns = pair, index = dates).
    """
    from coint_tests import engle_granger

    # Vectorized return map generation using dictionary comprehension with validation
    def process_pair(pair):
        if pair not in all_data:
            return None
        df = all_data[pair]
        if df is None or df.shape[1] < 2:
            return None
        y, x = df.columns[:2]
        eg = engle_granger(df, y, x)
        spread = eg.get('spread')
        if spread is None:
            return None
        mu, sigma = spread.mean(), spread.std()
        try:
            best_Z = float(summary_df.loc[summary_df['pair'] == pair, 'best_Z'].iloc[0])
        except Exception:
            return None
        sig = generate_signals(spread, mu, sigma, best_Z)
        ret = sig.shift(1).fillna(0) * spread.diff().fillna(0)
        ret.name = pair
        return ret
    
    # Process all pairs and filter out None results
    ret_map = {pair: ret for pair in selected if (ret := process_pair(pair)) is not None}

    if not ret_map:
        return pd.DataFrame()
    return pd.DataFrame(ret_map).sort_index()

def compute_pair_metrics(returns, pair=None, pair_info=None):
    """
    Compute performance metrics for a single pair. 
    pair_info is an optional dict/Series with per-pair info (e.g., best_Z, OU_HalfLife).
    """
    met = perf_metrics(returns)
    # Robust half-life retrieval with standardized key
    ou_half_life = np.nan
    if isinstance(pair_info, dict):
        if 'OU_HalfLife' in pair_info:
            ou_half_life = pair_info['OU_HalfLife']
        elif 'ou_halflife' in pair_info:
            ou_half_life = pair_info['ou_halflife']
        elif 'halflife' in pair_info:
            ou_half_life = pair_info['halflife']
    elif hasattr(pair_info, 'get'):
        # pandas Series/DataFrame row
        if 'OU_HalfLife' in pair_info:
            ou_half_life = pair_info['OU_HalfLife']
        elif 'ou_halflife' in pair_info:
            ou_half_life = pair_info['ou_halflife']
        elif 'halflife' in pair_info:
            ou_half_life = pair_info['halflife']
    
    out = {'pair': pair, **met}
    # standardize the column name
    out['OU_HalfLife'] = ou_half_life
    # pass through known extras if present
    if hasattr(pair_info, 'get'):
        if 'best_Z' in pair_info:
            out['best_Z'] = pair_info['best_Z']
    elif isinstance(pair_info, dict) and 'best_Z' in pair_info:
        out['best_Z'] = pair_info['best_Z']
    return out


def compute_pair_metrics_bulk(all_data: Dict[str, pd.DataFrame],
                              summary_df: pd.DataFrame,
                              selected: List[str]) -> pd.DataFrame:
    """Compute extended metrics per pair and return a merged summary DataFrame.

    Requires summary_df to contain at least columns ['pair','best_Z'].
    """
    from coint_tests import ou_params, engle_granger

    # Vectorized processing using list comprehension
    def process_individual_pair(pair):
        if pair not in all_data:
            return None
        df = all_data[pair]
        if df is None or df.shape[1] < 2:
            return None
        y, x = df.columns[:2]
        eg = engle_granger(df, y, x)
        spread = eg.get('spread')
        if spread is None:
            return None
        mu, sigma = spread.mean(), spread.std()
        try:
            srow = summary_df.loc[summary_df['pair'] == pair].iloc[0]
            best_Z = float(srow.get('best_Z'))
            N_trades = int(srow.get('N_trades', np.nan))
        except Exception:
            return None
        
        # Generate returns using signal generation
        sig = generate_signals(spread, mu, sigma, best_Z)
        ret = sig.shift(1).fillna(0) * spread.diff().fillna(0)
        if ret is None:
            return None
        
        ou = ou_params(spread)
        pair_info = {'best_Z': best_Z, 'N_trades': N_trades, **ou}
        return compute_pair_metrics(ret, pair=pair, pair_info=pair_info)
    
    # Process all pairs and filter out None results
    rows = [result for pair in selected if (result := process_individual_pair(pair)) is not None]

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return metrics_df

    # Merge back with original summary_df (left on 'pair') to preserve existing fields
    if 'pair' in summary_df.columns:
        # Avoid duplicate columns by dropping keys we recompute
        drop_cols = [c for c in ['best_Z', 'N_trades'] if c in summary_df.columns]
        merged = pd.merge(
            summary_df.drop(columns=drop_cols, errors='ignore'),
            metrics_df,
            on='pair',
            how='inner'
        )
        return merged

    return metrics_df

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
        
        # Compute μ_e and σ_eq on TRAIN data to avoid leakage
        train_mu = train_spread.mean()
        train_sigma = train_spread.std()
        
        # Inner loop: tune z on TRAIN
        best_z, best_score = None, -np.inf
        for z in z_list:
            sig = generate_signals(train_spread, train_mu, train_sigma, z)
            perf = performance_metrics(sig, train_spread)
            if perf['sharpe'] > best_score:
                best_score = perf['sharpe']
                best_z     = z
        
        # Evaluate on TEST using train-derived parameters
        test_sig  = generate_signals(test_spread, train_mu, train_sigma, best_z)
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

def generate_pair_pnl(all_data, summary_df, selected, cost: float = 0.0):
    """Generate cumulative PnL series per pair using best_Z thresholds.

    Applies simple transaction costs proportional to signal turnover.
    """
    pnl = {}
    for pair in selected:
        if pair not in all_data:
            continue
        df = all_data[pair]
        if df is None or df.shape[1] < 2:
            continue
        y, x = df.columns[:2]
        from coint_tests import engle_granger  # Local import to avoid circular dependency
        eg = engle_granger(df, y, x)
        spread = eg.get('spread')
        if spread is None:
            continue
        spread = spread.dropna()
        # best Z for this pair
        try:
            best_Z = float(summary_df.loc[summary_df['pair'] == pair, 'best_Z'].iloc[0])
        except Exception:
            continue
        from backtest import generate_signals  # Local import if needed
        mu, sigma = spread.mean(), spread.std()
        sig = generate_signals(spread, mu, sigma, best_Z).reindex(spread.index).fillna(0)

        dS = spread.diff().fillna(0)
        prev_sig = sig.shift(1).fillna(0)
        gross = prev_sig * dS
        turn = (sig - prev_sig).abs()
        ret = gross - cost * turn

        pnl[pair] = ret.cumsum()

    return pnl


def run_individual_pair_backtests(selected, all_data, summary_df, guarded_returns_dict=None):
    """
    Run individual pair backtests with stops for selected pairs.
    
    Returns:
        Dict of {pair_name: returns_series}
    """
    from coint_tests import engle_granger_bidirectional
    
    if guarded_returns_dict is None:
        guarded_returns_dict = {}
        
    # Vectorized processing using dictionary comprehension
    def process_guarded_pair(pair):
        df = all_data[pair]
        y, x = df.columns[:2]
        eg = engle_granger_bidirectional(df, y, x)
        spread = eg.get('spread')
        if spread is None:
            return None
        
        # Add spread column to dataframe for backtest functions
        df_with_spread = df.copy()
        df_with_spread['spread'] = spread
        
        # Get best_Z from summary
        best_Z = summary_df.loc[summary_df['pair'] == pair, 'best_Z'].iloc[0]
        
        r, sig, z = backtest_pair_with_stops(
            df_pair=df_with_spread,
            z_entry=best_Z,
            z_exit=0.1,
            stop_z=3.0,
            equity_dd_stop=0.15,
            cost=0.002
        )
        return r
    
    # Process all pairs and create results dictionary
    new_results = {pair: result for pair in selected if (result := process_guarded_pair(pair)) is not None}
    
    # Update guarded_returns_dict if provided
    if guarded_returns_dict is not None:
        guarded_returns_dict.update(new_results)
        return guarded_returns_dict
    else:
        return new_results


def run_vol_targeted_backtests(selected, all_data, summary_df, guarded_returns_dict=None):
    """
    Run vol-targeted backtests for selected pairs.
    
    Returns:
        Dict of {pair_name: returns_series}
    """
    from coint_tests import engle_granger_bidirectional
    
    if guarded_returns_dict is None:
        guarded_returns_dict = {}
        
    # Vectorized processing using dictionary comprehension
    def process_vol_targeted_pair(pair):
        df = all_data[pair]
        y, x = df.columns[:2]
        eg = engle_granger_bidirectional(df, y, x)
        spread = eg.get('spread')
        if spread is None:
            return None
        
        # Add spread column to dataframe for backtest functions
        df_with_spread = df.copy()
        df_with_spread['spread'] = spread
        
        # Get best_Z from summary
        best_Z = summary_df.loc[summary_df['pair'] == pair, 'best_Z'].iloc[0]
        
        r, sig, z = backtest_pair_with_vol_targeting(
            df_pair=df_with_spread,
            y_col=y,
            x_col=x,
            beta=eg.get('beta'),
            z_entry=best_Z,
            target_vol=0.15
        )
        return r.fillna(0)
    
    # Process all pairs and create results dictionary
    new_results = {pair: result for pair in selected if (result := process_vol_targeted_pair(pair)) is not None}
    
    # Update guarded_returns_dict if provided
    if guarded_returns_dict is not None:
        guarded_returns_dict.update(new_results)
        return guarded_returns_dict
    else:
        return new_results


def run_guarded_backtests(selected, all_data, backtest_type='stops'):
    """
    Run guarded backtests with improved risk management parameters.
    
    Parameters:
        backtest_type: 'stops' or 'vol_targeted'
    
    Returns:
        Dict of {pair_name: returns_series}
    """
    from coint_tests import engle_granger
    
    guarded_returns = {}

    # Vectorized processing using dictionary comprehension
    def process_guarded_backtest(pair):
        df = all_data[pair].dropna()
        y, x = df.columns[:2]
        eg = engle_granger(df, y, x)
        spread = eg['spread'].dropna()
        df_bt = pd.DataFrame({'spread': spread})

        if backtest_type == 'stops':
            # Stops only (fixed parameters to avoid unit mismatch and sticky trades)
            r, sig, z = backtest_pair_with_stops(
                df_bt,
                spread_col='spread',
                z_entry=1.5, z_exit=0.25, z_window=126,
                stop_z=2.5,
                stop_loss_sigma=2.0,
                equity_dd_stop=0.20,
                cost=0.002
            )
        elif backtest_type == 'vol_targeted':
            r, sig, z = backtest_pair_with_vol_targeting(
                df_bt,
                spread_col='spread',
                z_entry=1.5, z_exit=0.25, z_window=126,
                target_vol=0.12,
                cost=0.002
            )
        else:
            return None
        
        return r.fillna(0)
    
    # Process all pairs and create results dictionary
    guarded_returns = {pair: result for pair in selected if (result := process_guarded_backtest(pair)) is not None}

    return guarded_returns
