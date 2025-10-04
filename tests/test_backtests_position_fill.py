import pandas as pd

from backtests import calculate_strategy_returns


def test_calculate_strategy_returns_fills_missing_positions():
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    price1 = pd.Series([100, 101, 102, 103], index=idx)
    price2 = pd.Series([50, 50.5, 50.3, 50.1], index=idx)
    positions = pd.Series({idx[1]: 1, idx[3]: -1})

    result = calculate_strategy_returns(price1, price2, positions, beta=1.0)

    assert result["positions_used"].isna().sum() == 0
    assert result["strategy_returns"].iloc[0] == 0
