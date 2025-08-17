import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def split_train_test(data, train_ratio=0.6, train_end_date=None):
    """
    Split time series data into training and testing periods.
    """
    if train_end_date is not None:
        split_date = pd.to_datetime(train_end_date)
        train_data = data[data.index <= split_date]
        test_data = data[data.index > split_date]
    else:
        split_idx = int(len(data) * train_ratio)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        split_date = train_data.index[-1]
    
    return {
        'train_data': train_data,
        'test_data': test_data,
        'split_date': split_date,
        'train_size': len(train_data),
        'test_size': len(test_data)
    }


def estimate_cointegration(price1, price2, add_constant=True):
    """
    Estimate cointegration relationship using Engle-Granger approach.
    """
    # Align series
    aligned_data = pd.concat([price1, price2], axis=1).dropna()
    y, x = aligned_data.iloc[:, 0], aligned_data.iloc[:, 1]
    
    # Run cointegration regression
    if add_constant:
        x_reg = sm.add_constant(x)
        model = sm.OLS(y, x_reg).fit()
        alpha, beta = model.params[0], model.params[1]
    else:
        model = sm.OLS(y, x).fit()
        alpha, beta = 0, model.params[0]
    
    # Compute spread and residuals
    spread = y - alpha - beta * x
    
    # Test residuals for stationarity
    adf_result = adfuller(spread.dropna(), maxlag=1)
    adf_pvalue = adf_result[1]
    
    return {
        'alpha': alpha,
        'beta': beta,
        'spread': spread,
        'residuals': spread,
        'adf_pvalue': adf_pvalue,
        'r_squared': model.rsquared,
        'model': model
    }


def generate_trading_signals(spread, z_threshold=2.0, exit_threshold=0.0):
    """
    Generate trading signals based on spread z-score.
    """
    # Compute z-score
    mean_spread = spread.mean()
    std_spread = spread.std()
    z_scores = (spread - mean_spread) / std_spread
    
    # Generate position signals vectorially
    positions = np.where(z_scores > z_threshold, -1,  # Short spread (short asset1, long asset2)
                np.where(z_scores < -z_threshold, 1,   # Long spread (long asset1, short asset2)
                0))  # No position
    
    positions = pd.Series(positions, index=spread.index, name='positions')
    
    # Entry and exit signals
    entry_signals = (np.abs(z_scores) >= z_threshold).astype(int)
    exit_signals = (np.abs(z_scores) <= exit_threshold).astype(int)
    
    return {
        'positions': positions,
        'z_scores': z_scores,
        'entry_signals': pd.Series(entry_signals, index=spread.index),
        'exit_signals': pd.Series(exit_signals, index=spread.index),
        'mean_spread': mean_spread,
        'std_spread': std_spread
    }


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


def compute_rolling_sharpe(returns, window=252, risk_free_rate=0.0):
    """
    Compute rolling Sharpe ratio.
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns series
    window : int
        Rolling window size (default 252 for annual)
    risk_free_rate : float
        Risk-free rate (annualized)
        
    Returns:
    --------
    pd.Series : Rolling Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / 252  # Convert to daily
    
    rolling_mean = excess_returns.rolling(window=window).mean()
    rolling_std = excess_returns.rolling(window=window).std()
    
    # Annualize
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    
    return rolling_sharpe


def compute_rolling_beta(strategy_returns, market_returns, window=252):
    """
    Compute rolling beta against market benchmark.
    """
    # Align series
    aligned_data = pd.concat([strategy_returns, market_returns], axis=1).dropna()
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
    # Align data
    data = pd.concat([price1, price2], axis=1).dropna()
    data.columns = ['asset1', 'asset2']
    
    # Split train/test
    split_result = split_train_test(data, train_ratio=train_ratio)
    train_data = split_result['train_data']
    test_data = split_result['test_data']
    
    # Estimate cointegration on training data
    coint_result = estimate_cointegration(
        train_data['asset1'], train_data['asset2'], add_constant=add_constant
    )
    
    # Generate test spread using training parameters
    test_spread = test_data['asset1'] - coint_result['alpha'] - coint_result['beta'] * test_data['asset2']
    
    # Generate trading signals on test data
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
    
    # Summary statistics
    strategy_ret = returns_result['strategy_returns']
    total_return = returns_result['cumulative_returns'].iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(strategy_ret)) - 1
    annualized_vol = strategy_ret.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return) / annualized_vol if annualized_vol != 0 else 0
    
    num_trades = signal_result['entry_signals'].sum()
    avg_return_per_trade = strategy_ret.mean() if num_trades > 0 else 0
    
    return {
        'split_info': split_result,
        'cointegration': coint_result,
        'signals': signal_result,
        'returns': returns_result,
        'drawdowns': drawdown_result,
        'test_spread': test_spread,
        'performance_metrics': {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': drawdown_result['max_drawdown'],
            'num_trades': int(num_trades),
            'avg_return_per_trade': avg_return_per_trade,
            'win_rate': (strategy_ret > 0).mean() if len(strategy_ret) > 0 else 0
        }
    }


def backtest_with_rolling_cointegration(price1, price2, z_threshold=2.0, 
                                       window_size=126, step_size=10,
                                       train_ratio=0.6):
    """
    Backtest with rolling re-estimation of cointegration parameters.
    """
    # Align data
    data = pd.concat([price1, price2], axis=1).dropna()
    data.columns = ['asset1', 'asset2']
    
    # Initial split for warmup period
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
        # Use lagged positions to avoid look-ahead bias
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
    total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
    annualized_return = (1 + total_return) ** (252 / len(all_returns)) - 1 if len(all_returns) > 0 else 0
    annualized_vol = all_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
    
    return {
        'strategy_returns': all_returns,
        'cumulative_returns': cumulative_returns,
        'positions': all_positions,
        'beta_history': pd.Series(beta_history, index=dates_history),
        'alpha_history': pd.Series(alpha_history, index=dates_history),
        'drawdowns': drawdown_result,
        'performance_metrics': {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
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
        
        # Error covariance
        self.P = np.eye(2) * 0.1
        
        # Process noise covariance
        self.Q = np.eye(2) * process_var
        
        # Observation noise variance
        self.R = observation_var
        
        # State transition (beta and alpha evolve as random walk)
        self.F = np.eye(2)
        
        self.history = {
            'beta': [],
            'alpha': [],
            'P_trace': []
        }
    
    def update(self, y, x):
        """
        Update filter with new observation.
        """
        # Prediction step
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Update step
        H = np.array([x, 1])  # Observation matrix [x, 1] for y = beta*x + alpha
        
        # Innovation
        innovation = y - H @ self.state
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        
        # Kalman gain
        K = self.P @ H.T / S
        
        # State update
        self.state = self.state + K * innovation
        
        # Covariance update
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
    data = pd.concat([price1, price2], axis=1).dropna()
    data.columns = ['asset1', 'asset2']
    
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
    
    total_return = cumulative_returns.iloc[-1] - 1
    annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
    annualized_vol = strategy_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else 0
    
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
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
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
    if actual_n_splits < n_splits:
        print(f"Warning: Reduced n_splits from {n_splits} to {actual_n_splits} due to data constraints")
    
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


def create_timeseries_splits(data, n_splits=5, test_size=0.2):
    """
    Create time series cross-validation splits.
    Kept for backward compatibility.
    """
    # Use the robust fold generator with fixed test size
    min_test_size = max(63, int(len(data) * test_size))  # At least 63 days or test_size ratio
    return compute_ts_folds(data.index, n_splits, min_test_size=min_test_size)


def run_cross_validation_backtest(price1, price2, z_thresholds=[1.5, 2.0, 2.5], 
                                  n_splits=3, min_train_ratio=0.6, min_test_size=63):
    """
    Run cross-validation backtest across multiple thresholds.
    """
    # Align data
    data = pd.concat([price1, price2], axis=1).dropna()
    data.columns = ['asset1', 'asset2']
    
    # Create robust splits
    splits = compute_ts_folds(data.index, n_splits, min_train_ratio, min_test_size)
    
    results = []
    
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
            
            # Calculate metrics
            strategy_ret = returns_result['strategy_returns']
            cumulative_ret = returns_result['cumulative_returns']
            
            total_return = cumulative_ret.iloc[-1] - 1 if len(cumulative_ret) > 0 else 0
            sharpe = (strategy_ret.mean() / strategy_ret.std()) * np.sqrt(252) if strategy_ret.std() != 0 else 0
            max_dd = compute_drawdowns(cumulative_ret)['max_drawdown']
            
            results.append({
                'split': split_idx,
                'z_threshold': z_thresh,
                'total_return': total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'num_trades': signal_result['entry_signals'].sum(),
                'train_start': train_idx[0],
                'train_end': train_idx[-1],
                'test_start': test_idx[0],
                'test_end': test_idx[-1]
            })
    
    return pd.DataFrame(results)


def run_cv_over_pairs(all_data, selected, z_threshold_by_pair, n_splits=3, 
                     min_train_ratio=0.6, min_test_size=63):
    """
    Run cross-validation backtest over multiple pairs.
    """
    all_results = []
    
    for pair_name in selected:
        if pair_name not in all_data:
            print(f"Warning: {pair_name} not found in all_data, skipping...")
            continue
            
        df = all_data[pair_name]
        if len(df.columns) != 2:
            print(f"Warning: {pair_name} has {len(df.columns)} columns, expected 2, skipping...")
            continue
            
        asset1, asset2 = df.columns
        z_thresh = z_threshold_by_pair.get(pair_name, 2.0)  # Default to 2.0
        
        try:
            pair_results = run_cross_validation_backtest(
                df[asset1], df[asset2],
                z_thresholds=[z_thresh],  # Use only the specified threshold
                n_splits=n_splits,
                min_train_ratio=min_train_ratio,
                min_test_size=min_test_size
            )
            
            # Add pair column
            pair_results['pair'] = pair_name
            all_results.append(pair_results)
            
        except Exception as e:
            print(f"Error processing {pair_name}: {e}")
            continue
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


def summarize_cv(cv_df):
    """
    Summarize cross-validation results in the same format as the notebook.
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
    
    return summary
