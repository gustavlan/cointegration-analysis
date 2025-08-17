import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cointegration_tests import engle_granger, ou_params
from threshold_optimization import optimize_thresholds, plot_threshold_tradeoff

def analyze_pairs_nb(all_data, selected, Z_min=0.5, Z_max=3.0, dZ=0.1, cost=0.0, use_ou=True, normalize=False):
    summary, opt_tables = [], {}
    
    for pair in selected:
        df = all_data.get(pair)
        if df is None or df.shape[1] != 2:
            continue
            
        y_col, x_col = df.columns
        eg = engle_granger(df, y_col, x_col)
        if eg['spread'] is None:
            continue
            
        spread, beta = eg['spread'], eg['beta']
        ou = ou_params(spread)
        
        opt_df = optimize_thresholds(
            spread, spread.mean(), spread.std(), beta, df[y_col], df[x_col],
            Z_min, Z_max, dZ, cost, ou['ou_mu'], ou['ou_sigma'], use_ou, normalize
        )
        
        best = opt_df.loc[opt_df['cum_PnL'].idxmax()]
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
        
        # Rolling Beta
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
    