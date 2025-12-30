import logging

import numpy as np
import pandas as pd

from cointegration_analysis.analytics.backtesting import backtest_pair_strategy

logger = logging.getLogger(__name__)


def check_parameter_sensitivity(
    price1: pd.Series,
    price2: pd.Series,
    base_z_threshold: float,
    perturbation_pct: float = 0.1,
    steps: int = 5,
    transaction_costs: float = 0.0,
) -> pd.DataFrame:
    """Analyze strategy robustness by perturbing the Z-score threshold.

    Runs the backtest multiple times with slightly different Z-score thresholds
    to ensure the strategy's performance is not a result of overfitting to a
    specific parameter value.

    Args:
        price1 (pd.Series): Price series of first asset.
        price2 (pd.Series): Price series of second asset.
        base_z_threshold (float): The optimized Z-score threshold.
        perturbation_pct (float, optional): Percentage to perturb the threshold
            (e.g., 0.1 for 10%). Defaults to 0.1.
        steps (int, optional): Number of steps to test around the base value. Defaults to 5.
        transaction_costs (float, optional): Transaction costs. Defaults to 0.0.

    Returns:
        pd.DataFrame: DataFrame containing performance metrics for each perturbed parameter.
    """
    lower_bound = base_z_threshold * (1 - perturbation_pct)
    upper_bound = base_z_threshold * (1 + perturbation_pct)

    # Generate test values centered around base_z_threshold
    test_values = np.linspace(lower_bound, upper_bound, steps)

    results = []

    for z in test_values:
        try:
            backtest_result = backtest_pair_strategy(
                price1, price2, z_threshold=z, transaction_costs=transaction_costs
            )
            metrics = backtest_result["performance_metrics"]

            results.append(
                {
                    "z_threshold": z,
                    "sharpe_ratio": metrics.get("sharpe_ratio", np.nan),
                    "total_return": metrics.get("total_return", np.nan),
                    "max_drawdown": metrics.get("max_drawdown", np.nan),
                    "num_trades": metrics.get("num_trades", 0),
                }
            )
        except Exception as e:
            logger.warning(f"Backtest failed for Z={z}: {e}")

    return pd.DataFrame(results)
