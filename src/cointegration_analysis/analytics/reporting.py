import numpy as np
import pandas as pd


def calculate_tear_sheet_metrics(
    returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
) -> dict:
    """Calculate comprehensive performance metrics for a strategy.

    Computes institutional-grade performance metrics including Sharpe Ratio,
    Sortino Ratio, Max Drawdown, Calmar Ratio, and Win Rate.

    Args:
        returns (pd.Series): Series of strategy returns (percentage change).
        risk_free_rate (float, optional): Annualized risk-free rate. Defaults to 0.0.
        periods_per_year (int, optional): Number of trading periods per year. Defaults to 252.

    Returns:
        dict: Dictionary containing performance metrics:
            - 'total_return': Total cumulative return
            - 'cagr': Compound Annual Growth Rate
            - 'volatility': Annualized volatility
            - 'sharpe_ratio': Annualized Sharpe Ratio
            - 'sortino_ratio': Annualized Sortino Ratio
            - 'max_drawdown': Maximum Drawdown
            - 'calmar_ratio': Calmar Ratio
            - 'win_rate': Percentage of positive return periods
            - 'skew': Skewness of returns
            - 'kurtosis': Kurtosis of returns
    """
    if returns.empty:
        return {}

    # Cumulative Return
    cum_ret = (1 + returns).cumprod()
    total_return = cum_ret.iloc[-1] - 1

    # CAGR
    n_years = len(returns) / periods_per_year
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    # Volatility
    volatility = returns.std() * np.sqrt(periods_per_year)

    # Sharpe Ratio
    excess_returns = returns - risk_free_rate / periods_per_year
    sharpe_ratio = (
        (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year)
        if returns.std() != 0
        else np.nan
    )

    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    sortino_ratio = (
        (excess_returns.mean() * periods_per_year) / downside_std
        if downside_std != 0
        else np.nan
    )

    # Max Drawdown
    rolling_max = cum_ret.cummax()
    drawdown = cum_ret / rolling_max - 1
    max_drawdown = drawdown.min()

    # Calmar Ratio
    calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # Win Rate
    win_rate = (returns > 0).mean()

    return {
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
        "skew": returns.skew(),
        "kurtosis": returns.kurtosis(),
    }
