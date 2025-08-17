import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Constants
TRADING_DAYS_PER_YEAR = 252


def calculate_performance_metrics(strategy_returns, cumulative_returns=None):
    if cumulative_returns is None:
        cumulative_returns = (1 + strategy_returns).cumprod()
    
    if len(strategy_returns) == 0:
        return {
            'total_return': 0,
            'annualized_return': 0,
            'annualized_volatility': 0,
            'sharpe_ratio': 0
        }
    
    total_return = cumulative_returns.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / len(strategy_returns)) - 1
    annualized_vol = strategy_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_vol,
        'sharpe_ratio': sharpe_ratio
    }


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
    mean_spread = spread.mean()
    std_spread = spread.std()
    z_scores = (spread - mean_spread) / std_spread
    
    positions = np.where(z_scores > z_threshold, -1,
                np.where(z_scores < -z_threshold, 1, 0))
    positions = pd.Series(positions, index=spread.index, name='positions')
    
    entry_signals = (np.abs(z_scores) >= z_threshold).astype(int)
    
    return {'positions': positions, 'z_scores': z_scores,
            'entry_signals': pd.Series(entry_signals, index=spread.index),
            'mean_spread': mean_spread, 'std_spread': std_spread}


def calculate_strategy_returns(price1, price2, positions, beta, alpha=0):
    """
    Calculate strategy returns based on positions and hedge ratio.
    """
    # Calculate returns
    returns1 = price1.pct_change()
    returns2 = price2.pct_change()
    aligned_data = pd.concat([returns1, returns2, positions], axis=1).dropna()
    r1, r2, pos = aligned_data.iloc[:, 0], aligned_data.iloc[:, 1], aligned_data.iloc[:, 2]
    pos_lagged = pos.shift(1)
    spread_returns = r1 - beta * r2
    strategy_returns = pos_lagged * spread_returns
    strategy_returns = strategy_returns.fillna(0)
    
    # Cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    return {
        'strategy_returns': strategy_returns,
        'spread_returns': spread_returns,
        'asset1_returns': r1,
        'asset2_returns': r2,
        'cumulative_returns': cumulative_returns,
        'positions_used': pos_lagged
    }


def compute_drawdowns(cumulative_returns):
    """
    Compute drawdown series from cumulative returns.
    """
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    return {
        'drawdown': drawdown,
        'peak': peak,
        'max_drawdown': max_drawdown,
        'max_dd_date': max_dd_date
    }


def compute_rolling_sharpe(returns, window=TRADING_DAYS_PER_YEAR, risk_free_rate=0.0):
    """
    Compute rolling Sharpe ratio.
    """
    excess_returns = returns - risk_free_rate / TRADING_DAYS_PER_YEAR  # Convert to daily
    rolling_mean = excess_returns.rolling(window=window).mean()
    rolling_std = excess_returns.rolling(window=window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return rolling_sharpe


def compute_rolling_beta(strategy_returns, market_returns, window=TRADING_DAYS_PER_YEAR, risk_free_rate=0.0):
    """
    Compute rolling beta against market benchmark using excess returns.
    """
    # Convert to excess returns
    excess_market_returns = market_returns - risk_free_rate/TRADING_DAYS_PER_YEAR  # Daily risk-free rate
    
    # Align series
    aligned_data = pd.concat([strategy_returns, excess_market_returns], axis=1).dropna()
    strat_ret, mkt_ret = aligned_data.iloc[:, 0], aligned_data.iloc[:, 1]
    
    # Rolling covariance and variance
    rolling_cov = strat_ret.rolling(window=window).cov(mkt_ret)
    rolling_var = mkt_ret.rolling(window=window).var()
    
    # Beta = Cov(strategy, market) / Var(market)
    rolling_beta = rolling_cov / rolling_var
    
    return rolling_beta


def backtest_pair_strategy(price1, price2, z_threshold=2.0, train_ratio=0.6,
                          transaction_costs=0.0, add_constant=True):
    """
    Complete backtest of pairs trading strategy with train/test split.
    """
    data = align_price_data(price1, price2)
    split_result = split_train_test(data, train_ratio=train_ratio)
    train_data = split_result['train_data']
    test_data = split_result['test_data']
    
    # Estimate cointegration on training data
    coint_result = estimate_cointegration(train_data['asset1'], train_data['asset2'], add_constant=add_constant)
    test_spread = test_data['asset1'] - coint_result['alpha'] - coint_result['beta'] * test_data['asset2']
    signal_result = generate_trading_signals(test_spread, z_threshold=z_threshold)
    
    # Calculate strategy returns on test data
    returns_result = calculate_strategy_returns(
        test_data['asset1'], test_data['asset2'], 
        signal_result['positions'], coint_result['beta'], coint_result['alpha']
    )
    
    # Apply transaction costs
    if transaction_costs > 0:
        position_changes = signal_result['positions'].diff().abs()
        cost_series = position_changes * transaction_costs
        returns_result['strategy_returns'] -= cost_series.shift(1).fillna(0)
        returns_result['cumulative_returns'] = (1 + returns_result['strategy_returns']).cumprod()
    
    # Compute performance metrics
    drawdown_result = compute_drawdowns(returns_result['cumulative_returns'])
    strategy_ret = returns_result['strategy_returns']
    perf_metrics = calculate_performance_metrics(strategy_ret, returns_result['cumulative_returns'])
    num_trades = signal_result['entry_signals'].sum()
    avg_return_per_trade = strategy_ret.mean()
    
    return {
        'split_info': split_result,
        'cointegration': coint_result,
        'signals': signal_result,
        'returns': returns_result,
        'drawdowns': drawdown_result,
        'test_spread': test_spread,
        'performance_metrics': {
            **perf_metrics,
            'max_drawdown': drawdown_result['max_drawdown'],
            'num_trades': int(num_trades),
            'avg_return_per_trade': avg_return_per_trade,
            'win_rate': (strategy_ret > 0).mean()
        }
    }


def backtest_with_rolling_cointegration(price1, price2, z_threshold=2.0, 
                                       window_size=126, step_size=10,
                                       train_ratio=0.6):
    """
    Backtest with rolling re-estimation of cointegration parameters.
    """
    data = align_price_data(price1, price2)
    split_result = split_train_test(data, train_ratio=train_ratio)
    test_start_idx = len(split_result['train_data'])
    
    # Initialize results storage for the entire test period
    test_data = data.iloc[test_start_idx:]
    all_returns = pd.Series(0.0, index=test_data.index)
    all_positions = pd.Series(0, index=test_data.index)
    
    beta_history = []
    alpha_history = []
    dates_history = []
    
    # Rolling estimation loop
    current_idx = test_start_idx
    
    while current_idx < len(data):
        # Define estimation window
        est_start = max(0, current_idx - window_size)
        est_end = current_idx
        
        # Get estimation data
        est_data = data.iloc[est_start:est_end]
        
        if len(est_data) < 30:  # Minimum data requirement
            current_idx += step_size
            continue
            
        # Estimate cointegration
        coint_result = estimate_cointegration(
            est_data['asset1'], est_data['asset2']
        )
        
        # Store parameters
        beta_history.append(coint_result['beta'])
        alpha_history.append(coint_result['alpha'])
        dates_history.append(data.index[current_idx])
        
        # Apply to next period
        next_end = min(current_idx + step_size, len(data))
        period_data = data.iloc[current_idx:next_end]
        
        if len(period_data) == 0:
            break
            
        # Generate spread for this period using current parameters
        period_spread = (period_data['asset1'] - coint_result['alpha'] - 
                        coint_result['beta'] * period_data['asset2'])
        
        # Use estimation period spread statistics for z-score normalization
        est_spread = (est_data['asset1'] - coint_result['alpha'] - 
                     coint_result['beta'] * est_data['asset2'])
        mean_spread = est_spread.mean()
        std_spread = est_spread.std()
        
        if std_spread == 0:
            current_idx += step_size
            continue
            
        # Generate signals based on z-scores
        z_scores = (period_spread - mean_spread) / std_spread
        positions = np.where(z_scores > z_threshold, -1,
                    np.where(z_scores < -z_threshold, 1, 0))
        
        # Store positions for this period
        period_idx = period_data.index
        all_positions.loc[period_idx] = positions
        
        # Calculate returns for this period
        period_returns1 = period_data['asset1'].pct_change()
        period_returns2 = period_data['asset2'].pct_change()
        
        # Portfolio returns: long asset1, short beta*asset2
        spread_returns = period_returns1 - coint_result['beta'] * period_returns2
        
        # Apply positions with lag (use previous position for each day)
        lagged_positions = pd.Series(positions, index=period_idx).shift(1).fillna(0)
        strategy_returns = lagged_positions * spread_returns
        
        # Store returns for this period
        all_returns.loc[period_idx] = strategy_returns.fillna(0)
        
        current_idx += step_size
    
    # Calculate cumulative returns and metrics
    cumulative_returns = (1 + all_returns).cumprod()
    drawdown_result = compute_drawdowns(cumulative_returns)
    
    # Performance metrics
    perf_metrics = calculate_performance_metrics(all_returns, cumulative_returns)
    
    return {
        'strategy_returns': all_returns,
        'cumulative_returns': cumulative_returns,
        'positions': all_positions,
        'beta_history': pd.Series(beta_history, index=dates_history),
        'alpha_history': pd.Series(alpha_history, index=dates_history),
        'drawdowns': drawdown_result,
        'performance_metrics': {
            **perf_metrics,
            'max_drawdown': drawdown_result['max_drawdown'],
            'num_rebalances': len(beta_history)
        },
        'parameters': {
            'window_size': window_size,
            'step_size': step_size,
            'z_threshold': z_threshold
        }
    }


class KalmanPairsFilter:
    """
    Kalman Filter for adaptive hedge ratio estimation in pairs trading.
    """
    
    def __init__(self, initial_beta=1.0, process_var=1e-4, observation_var=1e-2):
        """
        Initialize Kalman Filter for pairs trading.
        """
        # State: [beta, alpha]
        self.state = np.array([initial_beta, 0.0])
        self.P = np.eye(2) * 0.1
        self.Q = np.eye(2) * process_var
        self.R = observation_var
        self.F = np.eye(2)
        
        self.history = {'beta': [], 'alpha': [], 'P_trace': []}
    
    def update(self, y, x):
        """
        Update filter with new observation.
        """
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        H = np.array([x, 1])  # Observation matrix [x, 1] for y = beta*x + alpha
        innovation = y - H @ self.state
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T / S
        self.state = self.state + K * innovation
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P
        
        # Store history
        self.history['beta'].append(self.state[0])
        self.history['alpha'].append(self.state[1])
        self.history['P_trace'].append(np.trace(self.P))
        
        return self.state[0], self.state[1]  # beta, alpha


def backtest_with_kalman_filter(price1, price2, z_threshold=2.0, train_ratio=0.6,
                               process_var=1e-4, observation_var=1e-2):
    """
    Backtest with Kalman Filter for adaptive hedge ratio.

    """
    # Align data
    data = align_price_data(price1, price2)
    
    # Initial split
    split_result = split_train_test(data, train_ratio=train_ratio)
    
    # Get initial beta estimate from training data
    initial_coint = estimate_cointegration(
        split_result['train_data']['asset1'], 
        split_result['train_data']['asset2']
    )
    
    # Initialize Kalman filter
    kf = KalmanPairsFilter(
        initial_beta=initial_coint['beta'],
        process_var=process_var,
        observation_var=observation_var
    )
    
    # Warm up filter on training data
    train_data = split_result['train_data']
    for idx in train_data.index:
        kf.update(train_data.loc[idx, 'asset1'], train_data.loc[idx, 'asset2'])
    
    # Apply to test data
    test_data = split_result['test_data']
    
    adaptive_betas = []
    adaptive_alphas = []
    spreads = []
    z_scores = []
    positions = []
    strategy_returns = []
    
    # Initialize spread statistics
    spread_history = []
    
    for i, idx in enumerate(test_data.index):
        y, x = test_data.loc[idx, 'asset1'], test_data.loc[idx, 'asset2']
        
        # Update Kalman filter
        beta, alpha = kf.update(y, x)
        adaptive_betas.append(beta)
        adaptive_alphas.append(alpha)
        
        # Calculate spread
        spread = y - alpha - beta * x
        spreads.append(spread)
        spread_history.append(spread)
        
        # Calculate z-score using historical spread data
        if len(spread_history) >= 30:  # Minimum history
            mean_spread = np.mean(spread_history)
            std_spread = np.std(spread_history)
            
            if std_spread > 0:
                z_score = (spread - mean_spread) / std_spread
            else:
                z_score = 0
        else:
            z_score = 0
            
        z_scores.append(z_score)
        
        # Generate position
        if z_score > z_threshold:
            position = -1  # Short spread
        elif z_score < -z_threshold:
            position = 1   # Long spread
        else:
            position = 0   # No position
            
        positions.append(position)
        
        # Calculate returns (use previous position to avoid look-ahead)
        if i > 0:
            prev_position = positions[i-1]
            prev_beta = adaptive_betas[i-1]
            
            # Asset returns
            ret1 = test_data['asset1'].iloc[i] / test_data['asset1'].iloc[i-1] - 1
            ret2 = test_data['asset2'].iloc[i] / test_data['asset2'].iloc[i-1] - 1
            
            # Spread return
            spread_return = ret1 - prev_beta * ret2
            
            # Strategy return
            strat_return = prev_position * spread_return
        else:
            strat_return = 0
            
        strategy_returns.append(strat_return)
    
    # Convert to series
    strategy_returns = pd.Series(strategy_returns, index=test_data.index)
    cumulative_returns = (1 + strategy_returns).cumprod()
    positions = pd.Series(positions, index=test_data.index)
    
    # Performance metrics
    drawdown_result = compute_drawdowns(cumulative_returns)
    
    perf_metrics = calculate_performance_metrics(strategy_returns, cumulative_returns)
    
    return {
        'strategy_returns': strategy_returns,
        'cumulative_returns': cumulative_returns,
        'positions': positions,
        'adaptive_betas': pd.Series(adaptive_betas, index=test_data.index),
        'adaptive_alphas': pd.Series(adaptive_alphas, index=test_data.index),
        'spreads': pd.Series(spreads, index=test_data.index),
        'z_scores': pd.Series(z_scores, index=test_data.index),
        'drawdowns': drawdown_result,
        'kalman_filter': kf,
        'performance_metrics': {
            **perf_metrics,
            'max_drawdown': drawdown_result['max_drawdown'],
            'initial_beta': initial_coint['beta'],
            'final_beta': adaptive_betas[-1] if adaptive_betas else initial_coint['beta'],
            'beta_volatility': np.std(adaptive_betas) if len(adaptive_betas) > 1 else 0
        },
        'parameters': {
            'z_threshold': z_threshold,
            'process_var': process_var,
            'observation_var': observation_var
        }
    }


def compute_ts_folds(index, n_splits, min_train_ratio=0.6, min_test_size=63, step=None):
    """
    Fold generator for time-series CV with fixed test size.
    """
    T = len(index)
    if step is None:
        step = min_test_size // 2
        
    # Calculate max feasible splits
    min_train_size = int(T * min_train_ratio)
    available_for_splits = T - min_train_size - min_test_size
    max_splits = max(1, available_for_splits // step + 1)
    
    # Reduce n_splits if needed
    actual_n_splits = min(n_splits, max_splits)
    
    splits = []
    for i in range(actual_n_splits):
        # Calculate test period
        test_end = min_train_size + min_test_size + i * step
        test_start = test_end - min_test_size
        
        # Ensure we don't exceed data bounds
        if test_end > T:
            test_end = T
            test_start = test_end - min_test_size
            
        if test_start < min_train_size:
            break
            
        # Train period: from start to test_start
        train_idx = index[:test_start]
        test_idx = index[test_start:test_end]
        
        splits.append((train_idx, test_idx))
    
    return splits


def run_cross_validation_backtest(price1, price2, z_thresholds=[1.5, 2.0, 2.5], 
                                  n_splits=3, min_train_ratio=0.6, min_test_size=63,
                                  return_artifacts=False, transaction_costs=0.002):
    """
    Run cross-validation backtest across multiple thresholds.
    """
    # Align data
    data = align_price_data(price1, price2)
    
    # Create robust splits
    splits = compute_ts_folds(data.index, n_splits, min_train_ratio, min_test_size)
    
    results = []
    artifacts = {} if return_artifacts else None
    
    for split_idx, (train_idx, test_idx) in enumerate(splits):
        train_data = data.loc[train_idx]
        test_data = data.loc[test_idx]
        
        # Estimate cointegration on training data
        coint_result = estimate_cointegration(
            train_data['asset1'], train_data['asset2']
        )
        
        for z_thresh in z_thresholds:
            # Generate test spread
            test_spread = (test_data['asset1'] - coint_result['alpha'] - 
                          coint_result['beta'] * test_data['asset2'])
            
            # Generate signals and calculate returns
            signal_result = generate_trading_signals(test_spread, z_threshold=z_thresh)
            returns_result = calculate_strategy_returns(
                test_data['asset1'], test_data['asset2'],
                signal_result['positions'], coint_result['beta']
            )
            
            # Apply transaction costs
            if transaction_costs > 0:
                position_changes = signal_result['positions'].diff().abs()
                cost_series = position_changes * transaction_costs
                returns_result['strategy_returns'] -= cost_series.shift(1).fillna(0)
                returns_result['cumulative_returns'] = (1 + returns_result['strategy_returns']).cumprod()
            
            # Calculate metrics
            strategy_ret = returns_result['strategy_returns']
            cumulative_ret = returns_result['cumulative_returns']
            
            perf_metrics = calculate_performance_metrics(strategy_ret, cumulative_ret)
            max_dd = compute_drawdowns(cumulative_ret)['max_drawdown']
            
            results.append({
                'split': split_idx,
                'z_threshold': z_thresh,
                'hedge_ratio': coint_result['beta'],
                'total_return': perf_metrics['total_return'],
                'sharpe_ratio': perf_metrics['sharpe_ratio'],
                'max_drawdown': max_dd,
                'num_trades': signal_result['entry_signals'].sum(),
                'train_start': train_idx[0],
                'train_end': train_idx[-1],
                'test_start': test_idx[0],
                'test_end': test_idx[-1]
            })
            
            # Store artifacts if requested
            if return_artifacts:
                key = (split_idx, z_thresh)
                artifacts[key] = {
                    'strategy_returns': strategy_ret,
                    'cumulative_returns': cumulative_ret,
                    'positions': signal_result['positions'],
                    'spread': test_spread,
                    'alpha': coint_result['alpha'],
                    'beta': coint_result['beta'],
                    'test_data': test_data,
                    'drawdowns': compute_drawdowns(cumulative_ret)
                }
    
    df_results = pd.DataFrame(results)
    
    if return_artifacts:
        return df_results, artifacts
    else:
        return df_results


def run_cv_over_pairs(all_data, selected, z_threshold_by_pair, n_splits=3, 
                     min_train_ratio=0.6, min_test_size=63, return_artifacts=False,
                     transaction_costs=0.002):
    """
    Run cross-validation backtest over multiple pairs.
    """
    all_results = []
    all_artifacts = {} if return_artifacts else None
    
    for pair_name in selected:
        if pair_name not in all_data:
            continue
            
        df = all_data[pair_name]
        if len(df.columns) != 2:
            continue
            
        asset1, asset2 = df.columns
        z_thresh = z_threshold_by_pair.get(pair_name, 2.0)  # Default to 2.0
        
        try:
            if return_artifacts:
                pair_results, pair_artifacts = run_cross_validation_backtest(
                    df[asset1], df[asset2],
                    z_thresholds=[z_thresh],  # Use only the specified threshold
                    n_splits=n_splits,
                    min_train_ratio=min_train_ratio,
                    min_test_size=min_test_size,
                    return_artifacts=True,
                    transaction_costs=transaction_costs
                )
                all_artifacts[pair_name] = pair_artifacts
            else:
                pair_results = run_cross_validation_backtest(
                    df[asset1], df[asset2],
                    z_thresholds=[z_thresh],  # Use only the specified threshold
                    n_splits=n_splits,
                    min_train_ratio=min_train_ratio,
                    min_test_size=min_test_size,
                    return_artifacts=False,
                    transaction_costs=transaction_costs
                )
            
            # Add pair column
            pair_results['pair'] = pair_name
            all_results.append(pair_results)
            
        except Exception as e:
            continue
    
    if all_results:
        df = pd.concat(all_results, ignore_index=True)
        # Reorder columns: pair first, hedge_ratio before total_return
        cols = ['pair'] + [c for c in df.columns if c != 'pair']
        df_final = df[cols]
        
        if return_artifacts:
            return df_final, all_artifacts
        else:
            return df_final
    else:
        if return_artifacts:
            return pd.DataFrame(), {}
        else:
            return pd.DataFrame()


def summarize_cv(cv_df, all_data=None, selected=None):
    """
    Summarize cross-validation results including half-life information.
    """
    if 'pair' in cv_df.columns:
        group_cols = ['pair', 'z_threshold']
    else:
        group_cols = ['z_threshold']
        
    summary = cv_df.groupby(group_cols).agg({
        'total_return': ['mean', 'std'],
        'sharpe_ratio': ['mean', 'std'],
        'max_drawdown': 'mean',
        'num_trades': 'mean'
    }).round(4)
    
    # Flatten column names to match notebook format
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    
    # Add half-life information if data is available
    if all_data is not None and selected is not None and 'pair' in cv_df.columns:
        from coint_tests import engle_granger, ou_params
        half_lives = []
        thetas = []
        
        for pair in cv_df['pair'].unique():
            if pair in all_data:
                df = all_data[pair]
                y_col, x_col = df.columns[:2]
                eg = engle_granger(df, y_col, x_col)
                if eg['spread'] is not None:
                    ou = ou_params(eg['spread'])
                    half_lives.append((pair, ou['OU_HalfLife']))
                    thetas.append((pair, ou['ou_theta']))
        
        # Add to summary if we have half-life data
        if half_lives:
            hl_df = pd.DataFrame(half_lives, columns=['pair', 'half_life'])
            theta_df = pd.DataFrame(thetas, columns=['pair', 'theta'])
            
            # Merge with summary (reset index to access pair column)
            summary_reset = summary.reset_index()
            summary_reset = summary_reset.merge(hl_df, on='pair', how='left')
            summary_reset = summary_reset.merge(theta_df, on='pair', how='left')
            summary = summary_reset.set_index(group_cols)
    
    return summary


def stitch_cv_folds(cv_artifacts, pair_name, z_threshold):
    """
    Stitch together CV fold results chronologically for continuous analysis.
    """
    if pair_name not in cv_artifacts:
        raise ValueError(f"Pair {pair_name} not found in artifacts")
    
    pair_artifacts = cv_artifacts[pair_name]
    
    # Get fold keys for this z_threshold
    fold_keys = [(split, z) for (split, z) in pair_artifacts.keys() if z == z_threshold]
    fold_keys.sort()  # Sort by split number
    
    if not fold_keys:
        raise ValueError(f"No artifacts found for pair {pair_name} and z_threshold {z_threshold}")
    
    # Collect time series from each fold
    all_returns = []
    all_positions = []
    all_spreads = []
    all_cumulative = []
    fold_boundaries = []
    
    for split, z in fold_keys:
        artifacts = pair_artifacts[(split, z)]
        
        # Add fold boundary marker (NaN to break continuity for rolling stats)
        if all_returns:  # Not the first fold
            boundary_date = artifacts['strategy_returns'].index[0]
            fold_boundaries.append(boundary_date)
            
            # Insert NaN at fold boundary
            all_returns.append(pd.Series([np.nan], index=[boundary_date]))
            all_positions.append(pd.Series([0], index=[boundary_date]))
            all_spreads.append(pd.Series([np.nan], index=[boundary_date]))
        
        # Append fold data
        all_returns.append(artifacts['strategy_returns'])
        all_positions.append(artifacts['positions'])
        all_spreads.append(artifacts['spread'])
        
        # For cumulative returns, we need to chain them properly
        if all_cumulative:
            # Start new fold cumulative from the last value of previous fold
            prev_final = all_cumulative[-1].iloc[-1] if len(all_cumulative[-1]) > 0 else 1.0
            fold_cumulative = prev_final * artifacts['cumulative_returns']
        else:
            fold_cumulative = artifacts['cumulative_returns']
            
        all_cumulative.append(fold_cumulative)
    
    # Concatenate all series
    stitched_returns = pd.concat(all_returns).sort_index()
    stitched_positions = pd.concat(all_positions).sort_index()
    stitched_spreads = pd.concat(all_spreads).sort_index()
    stitched_cumulative = pd.concat(all_cumulative).sort_index()
    
    # Compute overall drawdowns on stitched equity curve
    stitched_drawdowns = compute_drawdowns(stitched_cumulative)
    
    # Performance metrics on stitched series (excluding NaNs)
    clean_returns = stitched_returns.dropna()
    perf_metrics = calculate_performance_metrics(clean_returns, stitched_cumulative)
    
    return {
        'strategy_returns': stitched_returns,
        'cumulative_returns': stitched_cumulative,
        'positions': stitched_positions,
        'spread': stitched_spreads,
        'drawdowns': stitched_drawdowns,
        'fold_boundaries': fold_boundaries,
        'performance_metrics': {
            **perf_metrics,
            'max_drawdown': stitched_drawdowns['max_drawdown'],
            'num_trades': (stitched_positions.diff().abs() > 0).sum(),
        'num_folds': len(fold_keys)
    }
}


def run_systematic_backtest(cv_artifacts, selected_pairs, summary_df):
    """
    Run systematic backtesting by stitching CV fold results.
    """
    stitched_results = {}
    systematic_performance = []
    
    for pair in selected_pairs:
        pair_z = summary_df.set_index("pair").loc[pair, "best_Z"]
        
        try:
            # Stitch the CV folds for this pair and z-threshold
            stitched = stitch_cv_folds(cv_artifacts, pair, pair_z)
            stitched_results[pair] = stitched
            
            # Collect performance metrics
            perf = stitched['performance_metrics'].copy()
            perf['pair'] = pair
            perf['z_threshold'] = pair_z
            systematic_performance.append(perf)
            
        except Exception as e:
            continue
    
    # Create systematic backtest performance summary
    systematic_df = pd.DataFrame(systematic_performance)
    if not systematic_df.empty:
        systematic_df = systematic_df.set_index('pair')
    
    return stitched_results, systematic_df


def rolling_cointegration_analysis(all_data, selected_pairs, best_z_by_pair):
    """
    Rolling Cointegration Re-estimation Analysis
    """
    # Define configs strictly matching the brief: 5–8m train, 10–15d step
    rolling_configs = {
        "5m_window_2w_step": (105, 10),
        "6m_window_2w_step": (126, 10), 
        "8m_window_3w_step": (168, 15),
    }
    
    # Build results
    rolling_results = {}
    static_results = {}
    
    for pair_name in selected_pairs:
        df = all_data[pair_name]
        a1, a2 = df.columns
        z_thresh = float(best_z_by_pair.get(pair_name, 1.0))
        
        # Static 60/40 for reference
        static_results[pair_name] = backtest_pair_strategy(
            df[a1], df[a2], z_threshold=z_thresh, train_ratio=0.6
        )
        
        # Rolling variants using existing rolling cointegration function
        pair_roll = {}
        for cfg_name, (win, step) in rolling_configs.items():
            pair_roll[cfg_name] = backtest_with_rolling_cointegration(
                df[a1], df[a2],
                z_threshold=z_thresh,
                window_size=win,
                step_size=step,
                train_ratio=0.6
            )
        rolling_results[pair_name] = pair_roll
    
    # Build report
    rows = []
    
    def fmt_pct(x): return f"{x:.1%}"
    def fmt2(x): return f"{x:.2f}"
    
    for pair_name, cfg_map in rolling_results.items():
        sm = static_results[pair_name]["performance_metrics"]
        rows.append({
            "Pair": pair_name,
            "Strategy": "Static",
            "Total_Return": fmt_pct(sm["total_return"]),
            "Sharpe": fmt2(sm["sharpe_ratio"]),
            "Max_DD": fmt_pct(sm["max_drawdown"])
        })
        for cfg_name, res in cfg_map.items():
            pm = res["performance_metrics"]
            rows.append({
                "Pair": pair_name,
                "Strategy": cfg_name.replace("_", " ").title(),
                "Total_Return": fmt_pct(pm["total_return"]),
                "Sharpe": fmt2(pm["sharpe_ratio"]),
                "Max_DD": fmt_pct(pm["max_drawdown"])
            })
    
    return pd.DataFrame(rows)


def adaptive_cointegration_analysis(all_data, selected_pairs, best_z_by_pair):
    """
    Kalman filter for adaptive estimation of EG Step 1 cointegrating weights.
    """
    results = {}
    
    for pair_name in selected_pairs:
        df = all_data[pair_name]
        y_series, x_series = df.iloc[:, 0], df.iloc[:, 1]
        z_thresh = float(best_z_by_pair.get(pair_name, 1.0))
        
        # Static EG for comparison
        static_result = backtest_pair_strategy(
            y_series, x_series, z_threshold=z_thresh, train_ratio=0.6
        )
        
        # Adaptive Kalman EG
        kalman_result = backtest_with_kalman_filter(
            y_series, x_series, z_threshold=z_thresh, train_ratio=0.6
        )
        
        # Extract key metrics for comparison
        static_beta = static_result['performance_metrics'].get('hedge_ratio', 'N/A')
        initial_beta = kalman_result['performance_metrics']['initial_beta']
        final_beta = kalman_result['performance_metrics']['final_beta']
        beta_volatility = kalman_result['performance_metrics']['beta_volatility']
        
        results[pair_name] = {
            'static_beta': static_beta,
            'initial_beta': initial_beta, 
            'final_beta': final_beta,
            'beta_drift': final_beta - initial_beta,
            'beta_volatility': beta_volatility,
            'static_sharpe': static_result['performance_metrics']['sharpe_ratio'],
            'kalman_sharpe': kalman_result['performance_metrics']['sharpe_ratio'],
            'adaptive_betas': kalman_result['adaptive_betas'],
            'adaptive_alphas': kalman_result['adaptive_alphas']
        }
    
    # Create summary table
    summary_rows = []
    for pair_name, res in results.items():
        summary_rows.append({
            'Pair': pair_name,
            'Static_Beta': f"{res['static_beta']:.4f}" if isinstance(res['static_beta'], (int, float)) else str(res['static_beta']),
            'Initial_Beta': f"{res['initial_beta']:.4f}",
            'Final_Beta': f"{res['final_beta']:.4f}",
            'Beta_Drift': f"{res['beta_drift']:.4f}",
            'Beta_Vol': f"{res['beta_volatility']:.4f}",
            'Static_Sharpe': f"{res['static_sharpe']:.2f}",
            'Kalman_Sharpe': f"{res['kalman_sharpe']:.2f}"
        })
    
    return {
        'summary_df': pd.DataFrame(summary_rows),
        'detailed_results': results
    }