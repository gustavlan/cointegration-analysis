import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from cointegration_tests import engle_granger, ou_params
from threshold_optimization import optimize_thresholds, plot_threshold_tradeoff

logger = logging.getLogger(__name__)


def analyze_pairs_nb(
    all_data: dict[str, pd.DataFrame],
    selected: list[str],
    Z_min: float = 0.5,
    Z_max: float = 3.0,
    dZ: float = 0.1,
    cost: float = 0.0,
    use_ou: bool = True,
    normalize: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Analyze pairs trading performance across different threshold values and display results.

    Performs comprehensive pairs trading analysis by testing multiple threshold values
    and finding optimal parameters for each pair. Displays threshold optimization plots
    for visual analysis of the tradeoff between number of trades and profitability.

    Args:
        all_data: Dictionary mapping pair names to price DataFrames.
        selected: List of pair names to analyze from all_data.
        Z_min: Minimum threshold Z-score. Defaults to 0.5.
        Z_max: Maximum threshold Z-score. Defaults to 3.0.
        dZ: Z-score increment step. Defaults to 0.1.
        cost: Transaction cost per trade. Defaults to 0.0.
        use_ou: Use Ornstein-Uhlenbeck parameters instead of
                sample statistics. Defaults to True.
        normalize: Normalize P&L by spread volatility.
                   Defaults to False.

    Returns:
        Tuple of (summary_df, opt_tables) where:
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
            logger.warning(f"Skipping {pair}: invalid data")
            continue

        y_col, x_col = df.columns
        eg = engle_granger(df, y_col, x_col)
        if eg["spread"] is None:  # skip if no cointegration found
            logger.warning(f"Skipping {pair}: no cointegration found")
            continue

        spread, beta = eg["spread"], eg["beta"]
        ou = ou_params(spread)  # estimate OU process parameters

        opt_df = optimize_thresholds(
            spread,
            spread.mean(),
            spread.std(),
            beta,
            df[y_col],
            df[x_col],
            Z_min,
            Z_max,
            dZ,
            cost,
            ou["ou_mu"],
            ou["ou_sigma"],
            use_ou,
            normalize,
        )

        best = opt_df.loc[opt_df["cum_PnL"].idxmax()]  # find best performing threshold
        summary.append(
            {
                "pair": pair,
                "best_Z": best["Z"],
                "N_trades": best["N_trades"],
                "cum_PnL": best["cum_PnL"],
                "avg_PnL": best["avg_PnL"],
                "theta": ou["ou_theta"],
                "half_life": ou["OU_HalfLife"],
            }
        )

        fig = plot_threshold_tradeoff(opt_df)
        fig.suptitle(f"Tradeoff: {pair}", y=1.02)
        plt.show()
        opt_tables[pair] = opt_df

    return pd.DataFrame(summary), opt_tables


def plot_systematic_performance(
    systematic_results: dict[str, Any],
    benchmark_returns: pd.Series,
    output_dir: str = "docs/images",
) -> list[str]:
    """Plot comprehensive performance analysis for systematic backtest results.

    Creates performance visualization plots and saves them to the specified directory.
    This function adapts the existing plotting functionality to work with the CLI.

    Args:
        systematic_results: Dictionary containing systematic backtest results.
                          Should have structure compatible with stitched_results format.
        benchmark_returns: Market benchmark returns (e.g., S&P 500) for beta calculation.
        output_dir: Directory where plots will be saved. Defaults to "docs/images".

    Returns:
        List of paths to saved plot files.

    Example:
        >>> plot_paths = plot_systematic_performance(results, benchmark, "docs/images")
        >>> print(f"Generated {len(plot_paths)} plots")
    """
    # Import required functions from backtests module
    from backtests import compute_rolling_beta, compute_rolling_sharpe

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_paths = []

    try:
        # Extract pair names from systematic_results
        selected_pairs = list(systematic_results.keys())[:3]  # Limit to 3 pairs for layout

        if not selected_pairs:
            logger.warning("No systematic results found for plotting")
            return []

        # Create the systematic performance plot
        fig, axes = plt.subplots(4, len(selected_pairs), figsize=(20, 16))
        fig.suptitle("Systematic Strategy Performance", fontsize=16, fontweight="bold")
        colors = ["blue", "red", "green"]

        # If only one pair, make axes 2D for consistency
        if len(selected_pairs) == 1:
            axes = axes.reshape(-1, 1)

        rolling_sharpe_summary = []

        for i, pair_name in enumerate(selected_pairs):
            try:
                stitched = systematic_results[pair_name]
                equity = stitched.get("cumulative_returns", pd.Series())
                drawdown = stitched.get("drawdowns", {}).get("drawdown", pd.Series())
                strategy_returns = stitched.get("strategy_returns", pd.Series()).dropna()

                if len(equity) == 0:
                    logger.warning(f"No equity data for {pair_name}")
                    continue

                color = colors[i % len(colors)]

                # Equity curve
                if len(axes.shape) > 1:
                    ax_equity = axes[0, i] if axes.shape[1] > 1 else axes[0]
                else:
                    ax_equity = axes[0]

                ax_equity.plot(equity, color=color, linewidth=1)
                ax_equity.set_title(f'{pair_name.replace("_", " ").title()} - Equity')
                ax_equity.grid(True, alpha=0.3)

                # Drawdown
                ax_dd = axes[1, i] if len(axes.shape) > 1 and axes.shape[1] > 1 else axes[1]
                if len(drawdown) > 0:
                    ax_dd.fill_between(drawdown.index, drawdown * 100, 0, color=color, alpha=0.3)
                ax_dd.set_title(f'{pair_name.replace("_", " ").title()} - Drawdown')
                ax_dd.grid(True, alpha=0.3)

                # Rolling Sharpe
                ax_sharpe = axes[2, i] if len(axes.shape) > 1 and axes.shape[1] > 1 else axes[2]
                if len(strategy_returns) > 126:
                    rolling_sharpe = compute_rolling_sharpe(strategy_returns, window=126)
                    ax_sharpe.plot(rolling_sharpe, color=color, linewidth=2)
                    rolling_sharpe_summary.append(
                        {"pair": pair_name, "avg_rolling_sharpe": rolling_sharpe.mean()}
                    )
                ax_sharpe.set_title(f'{pair_name.replace("_", " ").title()} - Rolling Sharpe')
                ax_sharpe.grid(True, alpha=0.3)

                # Rolling Beta
                ax_beta = axes[3, i] if len(axes.shape) > 1 and axes.shape[1] > 1 else axes[3]
                if len(strategy_returns) > 126 and len(benchmark_returns) > 0:
                    rolling_beta = compute_rolling_beta(
                        strategy_returns, benchmark_returns, window=126
                    )
                    ax_beta.plot(rolling_beta, color=color, linewidth=2)
                ax_beta.set_title(f'{pair_name.replace("_", " ").title()} - Rolling Beta')
                ax_beta.grid(True, alpha=0.3)

            except Exception as e:
                logger.error(f"Error plotting {pair_name}: {e}")
                continue

        plt.tight_layout()

        # Save the plot
        plot_file = output_path / "systematic_performance.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        plot_paths.append(str(plot_file))

        logger.info(f"Generated systematic performance plot: {plot_file}")

        # Return list of generated plot paths
        return plot_paths

    except Exception as e:
        logger.error(f"Failed to generate systematic performance plots: {e}")
        return []


def plot_systematic_performance_original(
    stitched_results,
    selected_pairs,
    benchmark_returns,
    compute_rolling_sharpe,
    compute_rolling_beta,
    title="Strategy Performance",
):
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
    fig.suptitle(title, fontsize=16, fontweight="bold")
    colors = ["blue", "red", "green"]

    for i, pair_name in enumerate(selected_pairs):
        stitched = stitched_results[pair_name]
        equity = stitched["cumulative_returns"]
        drawdown = stitched["drawdowns"]["drawdown"]
        strategy_returns = stitched["strategy_returns"].dropna()

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

    return pd.DataFrame(
        [
            {
                "Pair": p.replace("_", " ").title(),
                "Avg_Sharpe": f"{compute_rolling_sharpe(stitched_results[p]['strategy_returns'].dropna(), 126).mean():.2f}",
            }
            for p in selected_pairs
        ]
    )


def plot_kalman_beta_evolution(
    kalman_analysis: dict[str, Any], selected_pairs: list[str], output_dir: str | None = None
) -> str | None:
    """Plot the evolution of adaptive beta coefficients from Kalman filter analysis.

    Creates time series plots showing how beta coefficients evolve over time
    when using adaptive estimation methods like Kalman filtering. Useful for
    visualizing the stability of hedge ratios in pairs trading.

    Args:
        kalman_analysis: Analysis results containing 'detailed_results' key
                        which maps pair names to dictionaries with 'adaptive_betas'
                        pd.Series.
        selected_pairs: List of pair names to plot.
        output_dir: Optional directory to save plots. If None, displays interactively.

    Returns:
        Path to saved plot file if output_dir provided, None if displayed interactively.

    Note:
        Creates one subplot per pair in a vertical layout. Each subplot shows
        the time evolution of the adaptive beta coefficient.

    Example:
        >>> kalman_results = adaptive_cointegration_analysis(data, pairs, thresholds)
        >>> plot_kalman_beta_evolution(kalman_results, ['tech_pair', 'finance_pair'])
    """
    if not selected_pairs:
        logger.warning("No pairs selected for Kalman beta evolution plot")
        return None

    try:
        fig, axes = plt.subplots(len(selected_pairs), 1, figsize=(12, 4 * len(selected_pairs)))
        if len(selected_pairs) == 1:
            axes = [axes]

        for i, pair_name in enumerate(selected_pairs):
            try:
                detailed = kalman_analysis["detailed_results"][pair_name]
                adaptive_betas = detailed["adaptive_betas"]

                axes[i].plot(
                    adaptive_betas.index,
                    adaptive_betas.values,
                    label="Adaptive β",
                    linewidth=2,
                    color="blue",
                )
                axes[i].set_title(f"{pair_name}: Beta Evolution")
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            except KeyError as e:
                logger.warning(f"Missing data for pair {pair_name}: {e}")
                continue

        plt.tight_layout()

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            plot_file = output_path / "kalman_beta_evolution.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Generated Kalman beta evolution plot: {plot_file}")
            return str(plot_file)
        else:
            plt.show()
            return None

    except Exception as e:
        logger.error(f"Failed to generate Kalman beta evolution plot: {e}")
        return None
