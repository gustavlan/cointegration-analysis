import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from coint_tests import engle_granger, ou_params
from threshold_optimization import optimize_thresholds, plot_threshold_tradeoff


def analyze_pairs_nb(all_data, selected, Z_min=0.5, Z_max=3.0, dZ=0.1, cost=0.0, use_ou=True, normalize: bool = False):
    summary = []
    opt_tables = {}

    for pair in selected:
        df = all_data.get(pair)
        if df is None or df.shape[1] != 2:
            print(f"'{pair}': not found or not a 2-asset series.")
            continue

        y_col, x_col = df.columns
        eg = engle_granger(df, y_col, x_col)
        spread, beta = eg['spread'], eg['beta']
        if spread is None:
            print(f"'{pair}' not cointegrated (p={eg['eg_pvalue']:.3f}), skipping.")
            continue

        ou = ou_params(spread)
        mu_ou, sigma_ou = ou['ou_mu'], ou['ou_sigma']
        mu_sample, sigma_sample = spread.mean(), spread.std()
        
        opt_df = optimize_thresholds(
            spread, mu_sample, sigma_sample, beta,
            y=df[y_col], x=df[x_col],
            Z_min=Z_min, Z_max=Z_max, dZ=dZ, cost=cost,
            ou_mu=mu_ou, ou_sigma=sigma_ou, use_ou=use_ou,
            normalize=normalize
        )

        best = opt_df.loc[opt_df['cum_PnL'].idxmax()]
        summary.append({
            'pair': pair, 'best_Z': best['Z'], 'N_trades': best['N_trades'],
            'cum_PnL': best['cum_PnL'], 'avg_PnL': best['avg_PnL'],
            'theta': ou['ou_theta'], 'half_life': ou['OU_HalfLife']
        })

        fig = plot_threshold_tradeoff(opt_df)
        fig.suptitle(f"Tradeoff: {pair}", y=1.02)
        plt.show()
        opt_tables[pair] = opt_df

    return pd.DataFrame(summary), opt_tables


def plot_systematic_performance(stitched_results, selected_pairs, benchmark_returns, 
                               compute_rolling_sharpe, compute_rolling_beta,
                               title="Pairs Trading Strategy Performance Analysis - Stitched CV Results"):
    """
    Plot systematic performance analysis with 4 charts per pair showing fold boundaries.
    """
    # Create comprehensive performance plots using stitched data
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    colors = ['blue', 'red', 'green'] 
    rolling_summary = []

    for i, pair_name in enumerate(selected_pairs):
        stitched = stitched_results[pair_name]
        boundaries = stitched['fold_boundaries']
        
        # 1. Equity Curves
        equity = stitched['cumulative_returns']
        axes[0, i].plot(equity, color=colors[i], linewidth=1)
        
        # Add fold boundaries as dashed vertical lines
        for boundary in boundaries:
            axes[0, i].axvline(x=boundary, color=colors[i], linestyle='--', alpha=0.6, linewidth=1)
        
        axes[0, i].set_title(f'{pair_name.replace("_", " ").title()} - Equity Curve')
        axes[0, i].set_ylabel('Cumulative Return')
        axes[0, i].grid(True, alpha=0.3)
        
        # 2. Drawdown Plots
        drawdown = stitched['drawdowns']['drawdown']
        axes[1, i].fill_between(drawdown.index, drawdown * 100, 0, 
                               color=colors[i], alpha=0.3)
        axes[1, i].plot(drawdown.index, drawdown * 100, color=colors[i], linewidth=1)
        
        # Add fold boundaries
        for boundary in boundaries:
            axes[1, i].axvline(x=boundary, color=colors[i], linestyle='--', alpha=0.6, linewidth=1)
        
        axes[1, i].set_title(f'{pair_name.replace("_", " ").title()} - Drawdown')
        axes[1, i].set_ylabel('Drawdown (%)')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].invert_yaxis()
        
        # 3. Rolling Sharpe Ratio
        strategy_returns = stitched['strategy_returns'].dropna()
        rolling_sharpe = compute_rolling_sharpe(strategy_returns, window=126)
        axes[2, i].plot(rolling_sharpe, color=colors[i], linewidth=2)
        axes[2, i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add fold boundaries
        for boundary in boundaries:
            axes[2, i].axvline(x=boundary, color=colors[i], linestyle='--', alpha=0.6, linewidth=1)
        
        axes[2, i].set_title(f'{pair_name.replace("_", " ").title()} - Rolling Sharpe (6m)')
        axes[2, i].set_ylabel('Sharpe Ratio')
        axes[2, i].grid(True, alpha=0.3)
        
        # 4. Rolling Beta vs S&P 500
        test_period_sp500 = benchmark_returns[strategy_returns.index]
        rolling_beta = compute_rolling_beta(strategy_returns, test_period_sp500, window=126)
        axes[3, i].plot(rolling_beta, color=colors[i], linewidth=2)
        axes[3, i].axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        axes[3, i].axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)
        axes[3, i].axhline(y=-0.1, color='gray', linestyle='--', alpha=0.5)
        
        # Add fold boundaries
        for boundary in boundaries:
            axes[3, i].axvline(x=boundary, color=colors[i], linestyle='--', alpha=0.6, linewidth=1)
        
        axes[3, i].set_title(f'{pair_name.replace("_", " ").title()} - Rolling Beta vs S&P500 (6m)')
        axes[3, i].set_ylabel('Beta')
        axes[3, i].set_xlabel('Date')
        axes[3, i].grid(True, alpha=0.3)
        
        # Collect rolling summary statistics
        rolling_summary.append({
            'Pair': pair_name.replace('_', ' ').title(),
            'Avg_Rolling_Sharpe': f"{rolling_sharpe.mean():.2f}",
            'Sharpe_Volatility': f"{rolling_sharpe.std():.2f}",
            'Avg_Rolling_Beta': f"{rolling_beta.mean():.3f}",
            'Beta_Volatility': f"{rolling_beta.std():.3f}",
            'Beta_95_Conf': f"[{rolling_beta.quantile(0.025):.3f}, {rolling_beta.quantile(0.975):.3f}]"
        })

    plt.tight_layout()
    plt.show()
    
    # Return rolling performance summary
    rolling_df = pd.DataFrame(rolling_summary)
    print(rolling_df.to_string(index=False))
    
    return rolling_df


def plot_kalman_beta_evolution(kalman_analysis, selected_pairs):
    """
    Plot the evolution of Kalman filter beta estimates for adaptive cointegration.
    """
    # Plot the evolution of beta estimates for each pair
    fig, axes = plt.subplots(len(selected_pairs), 1, figsize=(12, 4*len(selected_pairs)))
    if len(selected_pairs) == 1:
        axes = [axes]
    
    for i, pair_name in enumerate(selected_pairs):
        detailed = kalman_analysis['detailed_results'][pair_name]
        adaptive_betas = detailed['adaptive_betas']
        initial_beta = detailed['initial_beta']
        final_beta = detailed['final_beta']
        
        # Plot adaptive beta evolution
        axes[i].plot(adaptive_betas.index, adaptive_betas.values, 
                     label='Adaptive β (Kalman)', linewidth=2, color='blue')
        
        axes[i].set_title(f'{pair_name}: Evolution of Cointegrating Weight β\n'
                         f'(Drift: {final_beta-initial_beta:+.4f}, Vol: {detailed["beta_volatility"]:.4f})')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Beta')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    