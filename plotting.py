import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cointegration_tests import engle_granger, ou_params
from threshold_optimization import optimize_thresholds, plot_threshold_tradeoff

def analyze_pairs_nb(all_data, selected, Z_min=0.5, Z_max=3.0, dZ=0.1, cost=0.0, use_ou=True, normalize=False):
    """Analyze pairs trading performance across different threshold values and display results.
    
    Performs comprehensive pairs trading analysis by testing multiple threshold values
    and finding optimal parameters for each pair. Displays threshold optimization plots
    for visual analysis of the tradeoff between number of trades and profitability.
    
    Args:
        all_data (dict): Dictionary mapping pair names to price DataFrames.
        selected (list): List of pair names to analyze from all_data.
        Z_min (float, optional): Minimum threshold Z-score. Defaults to 0.5.
        Z_max (float, optional): Maximum threshold Z-score. Defaults to 3.0.
        dZ (float, optional): Z-score increment step. Defaults to 0.1.
        cost (float, optional): Transaction cost per trade. Defaults to 0.0.
        use_ou (bool, optional): Use Ornstein-Uhlenbeck parameters instead of
                                sample statistics. Defaults to True.
        normalize (bool, optional): Normalize P&L by spread volatility. 
                                   Defaults to False.
    
    Returns:
        tuple: (summary_df, opt_tables) where:
            - summary_df: DataFrame with best parameters and performance for each pair
            - opt_tables: Dictionary mapping pair names to full optimization results
    
    Note:
        Requires cointegrated pairs (Engle-Granger test must pass). Non-cointegrated
        pairs are automatically skipped with a warning.
    
    Example:
        >>> summary, tables = analyze_pairs_nb(data, ['tech_pair', 'energy_pair'])
        >>> best_pair = summary.loc[summary['cum_PnL'].idxmax(), 'pair']
        >>> print(f"Best performing pair: {best_pair}")
    """
    summary, opt_tables = [], {}
    
    for pair in selected:
        df = all_data.get(pair)
        if df is None or df.shape[1] != 2:
            continue
            
        y_col, x_col = df.columns
        eg = engle_granger(df, y_col, x_col)
        if eg['spread'] is None:  # skip if no cointegration found
            continue
            
        spread, beta = eg['spread'], eg['beta']
        ou = ou_params(spread)  # estimate OU process parameters
        
        opt_df = optimize_thresholds(
            spread, spread.mean(), spread.std(), beta, df[y_col], df[x_col],
            Z_min, Z_max, dZ, cost, ou['ou_mu'], ou['ou_sigma'], use_ou, normalize
        )
        
        best = opt_df.loc[opt_df['cum_PnL'].idxmax()]  # find best performing threshold
        summary.append({'pair': pair, 'best_Z': best['Z'], 'N_trades': best['N_trades'],
                       'cum_PnL': best['cum_PnL'], 'avg_PnL': best['avg_PnL'],
                       'theta': ou['ou_theta'], 'half_life': ou['OU_HalfLife']})
        
        fig = plot_threshold_tradeoff(opt_df)
        fig.suptitle(f"Tradeoff: {pair}", y=1.02)
        plt.show()
        opt_tables[pair] = opt_df
    
    return pd.DataFrame(summary), opt_tables

def plot_systematic_performance(stitched_results, selected_pairs, benchmark_returns, 
                               compute_rolling_sharpe, compute_rolling_beta, title="Strategy Performance"):
    """Plot comprehensive performance analysis for multiple pairs trading strategies.
    
    Creates a 4x3 subplot grid showing equity curves, drawdowns, rolling Sharpe ratios,
    and rolling betas for up to 3 pairs trading strategies. Provides visual comparison
    of strategy performance over time.
    
    Args:
        stitched_results (dict): Dictionary mapping pair names to backtest results.
                               Each result must contain 'cumulative_returns', 
                               'drawdowns', and 'strategy_returns'.
        selected_pairs (list): List of up to 3 pair names to plot.
        benchmark_returns (pd.Series): Market benchmark returns (e.g., S&P 500)
                                      for beta calculation.
        compute_rolling_sharpe (callable): Function to compute rolling Sharpe ratio.
                                         Must accept (returns, window) arguments.
        compute_rolling_beta (callable): Function to compute rolling beta.
                                       Must accept (strategy_returns, market_returns, window).
        title (str, optional): Main title for the figure. Defaults to "Strategy Performance".
    
    Returns:
        pd.DataFrame: Summary DataFrame with average rolling Sharpe ratios for each pair.
    
    Note:
        Assumes exactly 3 pairs for layout. Function will need modification for
        different numbers of pairs.
    
    Example:
        >>> results_df = plot_systematic_performance(
        ...     stitched_data, ['pair1', 'pair2', 'pair3'], 
        ...     sp500_returns, compute_rolling_sharpe, compute_rolling_beta
        ... )
    """
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    colors = ['blue', 'red', 'green']
    
    for i, pair_name in enumerate(selected_pairs):
        stitched = stitched_results[pair_name]
        equity = stitched['cumulative_returns']
        drawdown = stitched['drawdowns']['drawdown']
        strategy_returns = stitched['strategy_returns'].dropna()
        
        # Equity curve
        axes[0, i].plot(equity, color=colors[i], linewidth=1)
        axes[0, i].set_title(f'{pair_name.replace("_", " ").title()} - Equity')
        axes[0, i].grid(True, alpha=0.3)
        
        # Drawdown
        axes[1, i].fill_between(drawdown.index, drawdown * 100, 0, color=colors[i], alpha=0.3)
        axes[1, i].set_title(f'{pair_name.replace("_", " ").title()} - Drawdown')
        axes[1, i].grid(True, alpha=0.3)
        
        # Rolling Sharpe
        rolling_sharpe = compute_rolling_sharpe(strategy_returns, window=126)
        axes[2, i].plot(rolling_sharpe, color=colors[i], linewidth=2)
        axes[2, i].set_title(f'{pair_name.replace("_", " ").title()} - Rolling Sharpe')
        axes[2, i].grid(True, alpha=0.3)
        
        # Rolling Beta to market benchmark
        test_sp500 = benchmark_returns[strategy_returns.index]
        rolling_beta = compute_rolling_beta(strategy_returns, test_sp500, window=126)
        axes[3, i].plot(rolling_beta, color=colors[i], linewidth=2)
        axes[3, i].set_title(f'{pair_name.replace("_", " ").title()} - Rolling Beta')
        axes[3, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return pd.DataFrame([{
        'Pair': p.replace('_', ' ').title(),
        'Avg_Sharpe': f"{compute_rolling_sharpe(stitched_results[p]['strategy_returns'].dropna(), 126).mean():.2f}"
    } for p in selected_pairs])

def plot_kalman_beta_evolution(kalman_analysis, selected_pairs):
    """Plot the evolution of adaptive beta coefficients from Kalman filter analysis.
    
    Creates time series plots showing how beta coefficients evolve over time
    when using adaptive estimation methods like Kalman filtering. Useful for
    visualizing the stability of hedge ratios in pairs trading.
    
    Args:
        kalman_analysis (dict): Analysis results containing 'detailed_results' key
                              which maps pair names to dictionaries with 'adaptive_betas'
                              pd.Series.
        selected_pairs (list): List of pair names to plot.
    
    Returns:
        None: Displays matplotlib plots but returns nothing.
    
    Note:
        Creates one subplot per pair in a vertical layout. Each subplot shows
        the time evolution of the adaptive beta coefficient.
    
    Example:
        >>> kalman_results = adaptive_cointegration_analysis(data, pairs, thresholds)
        >>> plot_kalman_beta_evolution(kalman_results, ['tech_pair', 'finance_pair'])
    """
    fig, axes = plt.subplots(len(selected_pairs), 1, figsize=(12, 4*len(selected_pairs)))
    if len(selected_pairs) == 1:
        axes = [axes]
    
    for i, pair_name in enumerate(selected_pairs):
        detailed = kalman_analysis['detailed_results'][pair_name]
        adaptive_betas = detailed['adaptive_betas']
        
        axes[i].plot(adaptive_betas.index, adaptive_betas.values, label='Adaptive β', linewidth=2, color='blue')
        axes[i].set_title(f'{pair_name}: Beta Evolution')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    