import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

TRADING_DAYS_PER_YEAR = 252


def calculate_performance_metrics(strategy_returns, cumulative_returns=None):
    if cumulative_returns is None:
        cumulative_returns = (1 + strategy_returns).cumprod()
    
    if len(strategy_returns) == 0:
        return {'total_return': 0, 'annualized_return': 0, 'annualized_volatility': 0, 'sharpe_ratio': 0}
    
    total_return = cumulative_returns.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / len(strategy_returns)) - 1
    annualized_vol = strategy_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
    
    return {'total_return': total_return, 'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol, 'sharpe_ratio': sharpe_ratio}

def align_price_data(price1, price2):
    data = pd.concat([price1, price2], axis=1).dropna()
    data.columns = ['asset1', 'asset2']
    return data

def split_train_test(data, train_ratio=0.6, train_end_date=None):
    if train_end_date is not None:
        split_date = pd.to_datetime(train_end_date)
        train_data = data[data.index <= split_date]
        test_data = data[data.index > split_date]
    else:
        split_idx = int(len(data) * train_ratio)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        split_date = train_data.index[-1]
    
    return {'train_data': train_data, 'test_data': test_data, 'split_date': split_date,
            'train_size': len(train_data), 'test_size': len(test_data)}

def estimate_cointegration(price1, price2, add_constant=True):
    aligned_data = pd.concat([price1, price2], axis=1).dropna()
    y, x = aligned_data.iloc[:, 0], aligned_data.iloc[:, 1]
    
    if add_constant:
        x_reg = sm.add_constant(x)
        model = sm.OLS(y, x_reg).fit()
        alpha, beta = model.params[0], model.params[1]
    else:
        model = sm.OLS(y, x).fit()
        alpha, beta = 0, model.params[0]
    
    spread = y - alpha - beta * x
    adf_pvalue = adfuller(spread.dropna(), maxlag=1)[1]
    
    return {'alpha': alpha, 'beta': beta, 'spread': spread, 'residuals': spread,
            'adf_pvalue': adf_pvalue, 'r_squared': model.rsquared, 'model': model}

def generate_trading_signals(spread, z_threshold=2.0):
    mean_spread, std_spread = spread.mean(), spread.std()
    z_scores = (spread - mean_spread) / std_spread
    positions = np.where(z_scores > z_threshold, -1, np.where(z_scores < -z_threshold, 1, 0))
    positions = pd.Series(positions, index=spread.index, name='positions')
    entry_signals = (np.abs(z_scores) >= z_threshold).astype(int)
    
    return {'positions': positions, 'z_scores': z_scores,
            'entry_signals': pd.Series(entry_signals, index=spread.index),
            'mean_spread': mean_spread, 'std_spread': std_spread}

def calculate_strategy_returns(price1, price2, positions, beta, alpha=0):
    returns1, returns2 = price1.pct_change(), price2.pct_change()
    aligned_data = pd.concat([returns1, returns2, positions], axis=1).dropna()
    r1, r2, pos = aligned_data.iloc[:, 0], aligned_data.iloc[:, 1], aligned_data.iloc[:, 2]
    pos_lagged = pos.shift(1)
    spread_returns = r1 - beta * r2
    strategy_returns = (pos_lagged * spread_returns).fillna(0)
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    return {'strategy_returns': strategy_returns, 'spread_returns': spread_returns,
            'asset1_returns': r1, 'asset2_returns': r2, 
            'cumulative_returns': cumulative_returns, 'positions_used': pos_lagged}

def compute_drawdowns(cumulative_returns):
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns / peak) - 1
    return {'drawdown': drawdown, 'peak': peak, 
            'max_drawdown': drawdown.min(), 'max_dd_date': drawdown.idxmin()}

def compute_rolling_sharpe(returns, window=TRADING_DAYS_PER_YEAR, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    rolling_mean = excess_returns.rolling(window=window).mean()
    rolling_std = excess_returns.rolling(window=window).std()
    return (rolling_mean / rolling_std) * np.sqrt(TRADING_DAYS_PER_YEAR)

def compute_rolling_beta(strategy_returns, market_returns, window=TRADING_DAYS_PER_YEAR, risk_free_rate=0.0):
    excess_market_returns = market_returns - risk_free_rate/TRADING_DAYS_PER_YEAR
    aligned_data = pd.concat([strategy_returns, excess_market_returns], axis=1).dropna()
    strat_ret, mkt_ret = aligned_data.iloc[:, 0], aligned_data.iloc[:, 1]
    rolling_cov = strat_ret.rolling(window=window).cov(mkt_ret)
    rolling_var = mkt_ret.rolling(window=window).var()
    return rolling_cov / rolling_var

def backtest_pair_strategy(price1, price2, z_threshold=2.0, train_ratio=0.6, transaction_costs=0.0, add_constant=True):
    data = align_price_data(price1, price2)
    split_result = split_train_test(data, train_ratio=train_ratio)
    train_data, test_data = split_result['train_data'], split_result['test_data']
    
    coint_result = estimate_cointegration(train_data['asset1'], train_data['asset2'], add_constant=add_constant)
    test_spread = test_data['asset1'] - coint_result['alpha'] - coint_result['beta'] * test_data['asset2']
    signal_result = generate_trading_signals(test_spread, z_threshold=z_threshold)
    
    returns_result = calculate_strategy_returns(test_data['asset1'], test_data['asset2'], 
                                              signal_result['positions'], coint_result['beta'], coint_result['alpha'])
    
    if transaction_costs > 0:
        position_changes = signal_result['positions'].diff().abs()
        cost_series = position_changes * transaction_costs
        returns_result['strategy_returns'] -= cost_series.shift(1).fillna(0)
        returns_result['cumulative_returns'] = (1 + returns_result['strategy_returns']).cumprod()
    
    drawdown_result = compute_drawdowns(returns_result['cumulative_returns'])
    strategy_ret = returns_result['strategy_returns']
    perf_metrics = calculate_performance_metrics(strategy_ret, returns_result['cumulative_returns'])
    num_trades = signal_result['entry_signals'].sum()
    
    return {
        'split_info': split_result, 'cointegration': coint_result, 'signals': signal_result,
        'returns': returns_result, 'drawdowns': drawdown_result, 'test_spread': test_spread,
        'performance_metrics': {**perf_metrics, 'max_drawdown': drawdown_result['max_drawdown'],
                               'num_trades': int(num_trades), 'avg_return_per_trade': strategy_ret.mean(),
                               'win_rate': (strategy_ret > 0).mean()}
    }

def compute_ts_folds(index, n_splits, min_train_ratio=0.6, min_test_size=63, step=None):
    T = len(index)
    if step is None:
        step = min_test_size // 2
    min_train_size = int(T * min_train_ratio)
    available_for_splits = T - min_train_size - min_test_size
    max_splits = max(1, available_for_splits // step + 1)
    actual_n_splits = min(n_splits, max_splits)
    
    splits = []
    for i in range(actual_n_splits):
        test_end = min_train_size + min_test_size + i * step
        test_start = test_end - min_test_size
        if test_end > T:
            test_end = T
            test_start = test_end - min_test_size
        if test_start < min_train_size:
            break
        splits.append((index[:test_start], index[test_start:test_end]))
    return splits

def run_cross_validation_backtest(price1, price2, z_thresholds=[1.5, 2.0, 2.5], n_splits=3, 
                                  min_train_ratio=0.6, min_test_size=63, return_artifacts=False, transaction_costs=0.002):
    data = align_price_data(price1, price2)
    splits = compute_ts_folds(data.index, n_splits, min_train_ratio, min_test_size)
    
    results, artifacts = [], {} if return_artifacts else None
    
    for split_idx, (train_idx, test_idx) in enumerate(splits):
        train_data, test_data = data.loc[train_idx], data.loc[test_idx]
        coint_result = estimate_cointegration(train_data['asset1'], train_data['asset2'])
        
        for z_thresh in z_thresholds:
            test_spread = (test_data['asset1'] - coint_result['alpha'] - 
                          coint_result['beta'] * test_data['asset2'])
            signal_result = generate_trading_signals(test_spread, z_threshold=z_thresh)
            returns_result = calculate_strategy_returns(test_data['asset1'], test_data['asset2'],
                                                       signal_result['positions'], coint_result['beta'])
            
            if transaction_costs > 0:
                position_changes = signal_result['positions'].diff().abs()
                cost_series = position_changes * transaction_costs
                returns_result['strategy_returns'] -= cost_series.shift(1).fillna(0)
                returns_result['cumulative_returns'] = (1 + returns_result['strategy_returns']).cumprod()
            
            strategy_ret = returns_result['strategy_returns']
            cumulative_ret = returns_result['cumulative_returns']
            perf_metrics = calculate_performance_metrics(strategy_ret, cumulative_ret)
            max_dd = compute_drawdowns(cumulative_ret)['max_drawdown']
            
            results.append({
                'split': split_idx, 'z_threshold': z_thresh, 'hedge_ratio': coint_result['beta'],
                'total_return': perf_metrics['total_return'], 'sharpe_ratio': perf_metrics['sharpe_ratio'],
                'max_drawdown': max_dd, 'num_trades': signal_result['entry_signals'].sum(),
                'train_start': train_idx[0], 'train_end': train_idx[-1],
                'test_start': test_idx[0], 'test_end': test_idx[-1]
            })
            
            if return_artifacts:
                artifacts[(split_idx, z_thresh)] = {
                    'strategy_returns': strategy_ret, 'cumulative_returns': cumulative_ret,
                    'positions': signal_result['positions'], 'spread': test_spread,
                    'alpha': coint_result['alpha'], 'beta': coint_result['beta'],
                    'test_data': test_data, 'drawdowns': compute_drawdowns(cumulative_ret)
                }
    
    return (pd.DataFrame(results), artifacts) if return_artifacts else pd.DataFrame(results)


def run_cv_over_pairs(all_data, selected, z_threshold_by_pair, n_splits=3, min_train_ratio=0.6, 
                     min_test_size=63, return_artifacts=False, transaction_costs=0.002):
    all_results, all_artifacts = [], {} if return_artifacts else None
    
    for pair_name in selected:
        if pair_name not in all_data:
            continue
        df = all_data[pair_name]
        if len(df.columns) != 2:
            continue
            
        asset1, asset2 = df.columns
        z_thresh = z_threshold_by_pair.get(pair_name, 2.0)
        
        try:
            if return_artifacts:
                pair_results, pair_artifacts = run_cross_validation_backtest(
                    df[asset1], df[asset2], z_thresholds=[z_thresh], n_splits=n_splits,
                    min_train_ratio=min_train_ratio, min_test_size=min_test_size,
                    return_artifacts=True, transaction_costs=transaction_costs)
                all_artifacts[pair_name] = pair_artifacts
            else:
                pair_results = run_cross_validation_backtest(
                    df[asset1], df[asset2], z_thresholds=[z_thresh], n_splits=n_splits,
                    min_train_ratio=min_train_ratio, min_test_size=min_test_size,
                    return_artifacts=False, transaction_costs=transaction_costs)
            
            pair_results['pair'] = pair_name
            all_results.append(pair_results)
        except Exception:
            continue
    
    if all_results:
        df_final = pd.concat(all_results, ignore_index=True)
        cols = ['pair'] + [c for c in df_final.columns if c != 'pair']
        df_final = df_final[cols]
        return (df_final, all_artifacts) if return_artifacts else df_final
    else:
        return (pd.DataFrame(), {}) if return_artifacts else pd.DataFrame()

def summarize_cv(cv_df, all_data=None, selected=None):
    group_cols = ['pair', 'z_threshold'] if 'pair' in cv_df.columns else ['z_threshold']
    summary = cv_df.groupby(group_cols).agg({
        'total_return': ['mean', 'std'], 'sharpe_ratio': ['mean', 'std'],
        'max_drawdown': 'mean', 'num_trades': 'mean'
    }).round(4)
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    
    if all_data is not None and selected is not None and 'pair' in cv_df.columns:
        from cointegration_tests import engle_granger, ou_params
        half_lives, thetas = [], []
        for pair in cv_df['pair'].unique():
            if pair in all_data:
                df = all_data[pair]
                y_col, x_col = df.columns[:2]
                eg = engle_granger(df, y_col, x_col)
                if eg['spread'] is not None:
                    ou = ou_params(eg['spread'])
                    half_lives.append((pair, ou['OU_HalfLife']))
                    thetas.append((pair, ou['ou_theta']))
        
        if half_lives:
            hl_df = pd.DataFrame(half_lives, columns=['pair', 'half_life'])
            theta_df = pd.DataFrame(thetas, columns=['pair', 'theta'])
            summary_reset = summary.reset_index()
            summary_reset = summary_reset.merge(hl_df, on='pair', how='left')
            summary_reset = summary_reset.merge(theta_df, on='pair', how='left')
            summary = summary_reset.set_index(group_cols)
    return summary

def stitch_cv_folds(cv_artifacts, pair_name, z_threshold):
    if pair_name not in cv_artifacts:
        raise ValueError(f"Pair {pair_name} not found in artifacts")
    
    pair_artifacts = cv_artifacts[pair_name]
    fold_keys = [(split, z) for (split, z) in pair_artifacts.keys() if z == z_threshold]
    fold_keys.sort()
    
    if not fold_keys:
        raise ValueError(f"No artifacts found for pair {pair_name} and z_threshold {z_threshold}")
    
    all_returns, all_positions, all_spreads, all_cumulative, fold_boundaries = [], [], [], [], []
    
    for split, z in fold_keys:
        artifacts = pair_artifacts[(split, z)]
        
        if all_returns:
            boundary_date = artifacts['strategy_returns'].index[0]
            fold_boundaries.append(boundary_date)
            all_returns.append(pd.Series([np.nan], index=[boundary_date]))
            all_positions.append(pd.Series([0], index=[boundary_date]))
            all_spreads.append(pd.Series([np.nan], index=[boundary_date]))
        
        all_returns.append(artifacts['strategy_returns'])
        all_positions.append(artifacts['positions'])
        all_spreads.append(artifacts['spread'])
        
        if all_cumulative:
            prev_final = all_cumulative[-1].iloc[-1] if len(all_cumulative[-1]) > 0 else 1.0
            fold_cumulative = prev_final * artifacts['cumulative_returns']
        else:
            fold_cumulative = artifacts['cumulative_returns']
        all_cumulative.append(fold_cumulative)
    
    stitched_returns = pd.concat(all_returns).sort_index()
    stitched_positions = pd.concat(all_positions).sort_index()
    stitched_spreads = pd.concat(all_spreads).sort_index()
    stitched_cumulative = pd.concat(all_cumulative).sort_index()
    stitched_drawdowns = compute_drawdowns(stitched_cumulative)
    
    clean_returns = stitched_returns.dropna()
    perf_metrics = calculate_performance_metrics(clean_returns, stitched_cumulative)
    
    return {
        'strategy_returns': stitched_returns, 'cumulative_returns': stitched_cumulative,
        'positions': stitched_positions, 'spread': stitched_spreads,
        'drawdowns': stitched_drawdowns, 'fold_boundaries': fold_boundaries,
        'performance_metrics': {**perf_metrics, 'max_drawdown': stitched_drawdowns['max_drawdown'],
                               'num_trades': (stitched_positions.diff().abs() > 0).sum(), 'num_folds': len(fold_keys)}
    }


def run_systematic_backtest(cv_artifacts, selected_pairs, summary_df):
    stitched_results, systematic_performance = {}, []
    
    for pair in selected_pairs:
        pair_z = summary_df.set_index("pair").loc[pair, "best_Z"]
        try:
            stitched = stitch_cv_folds(cv_artifacts, pair, pair_z)
            stitched_results[pair] = stitched
            perf = stitched['performance_metrics'].copy()
            perf['pair'] = pair
            perf['z_threshold'] = pair_z
            systematic_performance.append(perf)
        except Exception:
            continue
    
    systematic_df = pd.DataFrame(systematic_performance)
    if not systematic_df.empty:
        systematic_df = systematic_df.set_index('pair')
    return stitched_results, systematic_df

def backtest_with_rolling_cointegration(price1, price2, z_threshold=2.0, window_size=126, step_size=10, train_ratio=0.6):
    """Simplified rolling cointegration backtest"""
    data = align_price_data(price1, price2)
    split_result = split_train_test(data, train_ratio=train_ratio)
    test_start_idx = len(split_result['train_data'])
    
    test_data = data.iloc[test_start_idx:]
    all_returns = pd.Series(0.0, index=test_data.index)
    beta_history, dates_history = [], []
    
    current_idx = test_start_idx
    while current_idx < len(data):
        est_start = max(0, current_idx - window_size)
        est_data = data.iloc[est_start:current_idx]
        
        if len(est_data) < 30:
            current_idx += step_size
            continue
            
        coint_result = estimate_cointegration(est_data['asset1'], est_data['asset2'])
        beta_history.append(coint_result['beta'])
        dates_history.append(data.index[current_idx])
        
        next_end = min(current_idx + step_size, len(data))
        period_data = data.iloc[current_idx:next_end]
        
        if len(period_data) == 0:
            break
            
        period_spread = period_data['asset1'] - coint_result['alpha'] - coint_result['beta'] * period_data['asset2']
        est_spread = est_data['asset1'] - coint_result['alpha'] - coint_result['beta'] * est_data['asset2']
        mean_spread, std_spread = est_spread.mean(), est_spread.std()
        
        if std_spread == 0:
            current_idx += step_size
            continue
            
        z_scores = (period_spread - mean_spread) / std_spread
        positions = np.where(z_scores > z_threshold, -1, np.where(z_scores < -z_threshold, 1, 0))
        
        period_returns1 = period_data['asset1'].pct_change()
        period_returns2 = period_data['asset2'].pct_change()
        spread_returns = period_returns1 - coint_result['beta'] * period_returns2
        
        lagged_positions = pd.Series(positions, index=period_data.index).shift(1).fillna(0)
        strategy_returns = (lagged_positions * spread_returns).fillna(0)
        all_returns.loc[period_data.index] = strategy_returns
        
        current_idx += step_size
    
    cumulative_returns = (1 + all_returns).cumprod()
    drawdown_result = compute_drawdowns(cumulative_returns)
    perf_metrics = calculate_performance_metrics(all_returns, cumulative_returns)
    
    return {
        'strategy_returns': all_returns, 'cumulative_returns': cumulative_returns,
        'beta_history': pd.Series(beta_history, index=dates_history), 'drawdowns': drawdown_result,
        'performance_metrics': {**perf_metrics, 'max_drawdown': drawdown_result['max_drawdown'], 'num_rebalances': len(beta_history)},
        'parameters': {'window_size': window_size, 'step_size': step_size, 'z_threshold': z_threshold}
    }

def backtest_with_kalman_filter(price1, price2, z_threshold=2.0, train_ratio=0.6, process_var=1e-4, observation_var=1e-2):
    """Simplified Kalman filter backtest"""
    data = align_price_data(price1, price2)
    split_result = split_train_test(data, train_ratio=train_ratio)
    
    initial_coint = estimate_cointegration(split_result['train_data']['asset1'], split_result['train_data']['asset2'])
    test_data = split_result['test_data']
    
    adaptive_betas = []
    strategy_returns = []
    alpha = 0.95  # exponential smoothing factor
    current_beta = initial_coint['beta']
    
    for i, idx in enumerate(test_data.index):
        y, x = test_data.loc[idx, 'asset1'], test_data.loc[idx, 'asset2']
        
        # beta update via exponential smoothing
        if i > 0:
            implied_beta = y / x if x != 0 else current_beta
            current_beta = alpha * current_beta + (1 - alpha) * implied_beta
        
        adaptive_betas.append(current_beta)
        
        if i > 0:
            ret1 = test_data['asset1'].iloc[i] / test_data['asset1'].iloc[i-1] - 1
            ret2 = test_data['asset2'].iloc[i] / test_data['asset2'].iloc[i-1] - 1
            spread_return = ret1 - adaptive_betas[i-1] * ret2
            
            if abs(spread_return) > z_threshold * 0.01:  # simplified threshold
                position = -1 if spread_return > 0 else 1
            else:
                position = 0
            strategy_returns.append(position * spread_return)
        else:
            strategy_returns.append(0)
    
    strategy_returns = pd.Series(strategy_returns, index=test_data.index)
    cumulative_returns = (1 + strategy_returns).cumprod()
    perf_metrics = calculate_performance_metrics(strategy_returns, cumulative_returns)
    
    return {
        'strategy_returns': strategy_returns, 'cumulative_returns': cumulative_returns,
        'adaptive_betas': pd.Series(adaptive_betas, index=test_data.index),
        'performance_metrics': {**perf_metrics, 'initial_beta': initial_coint['beta'], 
                               'final_beta': adaptive_betas[-1] if adaptive_betas else initial_coint['beta'],
                               'beta_volatility': np.std(adaptive_betas) if len(adaptive_betas) > 1 else 0},
        'parameters': {'z_threshold': z_threshold, 'process_var': process_var, 'observation_var': observation_var}
    }


def rolling_cointegration_analysis(all_data, selected_pairs, best_z_by_pair):
    rolling_configs = {"5m_window_2w_step": (105, 10), "6m_window_2w_step": (126, 10), "8m_window_3w_step": (168, 15)}
    
    rolling_results, static_results = {}, {}
    
    for pair_name in selected_pairs:
        df = all_data[pair_name]
        a1, a2 = df.columns
        z_thresh = float(best_z_by_pair.get(pair_name, 1.0))
        
        static_results[pair_name] = backtest_pair_strategy(df[a1], df[a2], z_threshold=z_thresh, train_ratio=0.6)
        
        pair_roll = {}
        for cfg_name, (win, step) in rolling_configs.items():
            pair_roll[cfg_name] = backtest_with_rolling_cointegration(df[a1], df[a2], z_threshold=z_thresh, window_size=win, step_size=step, train_ratio=0.6)
        rolling_results[pair_name] = pair_roll
    
    rows = []
    for pair_name, cfg_map in rolling_results.items():
        sm = static_results[pair_name]["performance_metrics"]
        rows.append({"Pair": pair_name, "Strategy": "Static", "Total_Return": f"{sm['total_return']:.1%}",
                    "Sharpe": f"{sm['sharpe_ratio']:.2f}", "Max_DD": f"{sm['max_drawdown']:.1%}"})
        for cfg_name, res in cfg_map.items():
            pm = res["performance_metrics"]
            rows.append({"Pair": pair_name, "Strategy": cfg_name.replace("_", " ").title(),
                        "Total_Return": f"{pm['total_return']:.1%}", "Sharpe": f"{pm['sharpe_ratio']:.2f}", 
                        "Max_DD": f"{pm['max_drawdown']:.1%}"})
    
    return pd.DataFrame(rows)

def adaptive_cointegration_analysis(all_data, selected_pairs, best_z_by_pair):
    results = {}
    
    for pair_name in selected_pairs:
        df = all_data[pair_name]
        y_series, x_series = df.iloc[:, 0], df.iloc[:, 1]
        z_thresh = float(best_z_by_pair.get(pair_name, 1.0))
        
        static_result = backtest_pair_strategy(y_series, x_series, z_threshold=z_thresh, train_ratio=0.6)
        kalman_result = backtest_with_kalman_filter(y_series, x_series, z_threshold=z_thresh, train_ratio=0.6)
        
        static_beta = static_result['performance_metrics'].get('hedge_ratio', 'N/A')
        results[pair_name] = {
            'static_beta': static_beta, 'initial_beta': kalman_result['performance_metrics']['initial_beta'], 
            'final_beta': kalman_result['performance_metrics']['final_beta'],
            'beta_drift': kalman_result['performance_metrics']['final_beta'] - kalman_result['performance_metrics']['initial_beta'],
            'beta_volatility': kalman_result['performance_metrics']['beta_volatility'],
            'static_sharpe': static_result['performance_metrics']['sharpe_ratio'],
            'kalman_sharpe': kalman_result['performance_metrics']['sharpe_ratio'],
            'adaptive_betas': kalman_result['adaptive_betas'],
            'adaptive_alphas': kalman_result.get('adaptive_alphas', pd.Series())
        }
    
    summary_rows = []
    for pair_name, res in results.items():
        summary_rows.append({
            'Pair': pair_name,
            'Static_Beta': f"{res['static_beta']:.4f}" if isinstance(res['static_beta'], (int, float)) else str(res['static_beta']),
            'Initial_Beta': f"{res['initial_beta']:.4f}", 'Final_Beta': f"{res['final_beta']:.4f}",
            'Beta_Drift': f"{res['beta_drift']:.4f}", 'Beta_Vol': f"{res['beta_volatility']:.4f}",
            'Static_Sharpe': f"{res['static_sharpe']:.2f}", 'Kalman_Sharpe': f"{res['kalman_sharpe']:.2f}"
        })
    
    return {'summary_df': pd.DataFrame(summary_rows), 'detailed_results': results}