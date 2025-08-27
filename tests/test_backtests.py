"""
Tests for backtests.py module.

Tests all backtesting functions including strategy execution, cross-validation,
performance metrics, and rolling analysis methods.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
import warnings

# Import functions under test
from backtests import (
    calculate_performance_metrics,
    align_price_data,
    split_train_test,
    estimate_cointegration,
    generate_trading_signals,
    calculate_strategy_returns,
    compute_drawdowns,
    compute_rolling_sharpe,
    compute_rolling_beta,
    backtest_pair_strategy,
    compute_ts_folds,
    run_cross_validation_backtest,
    run_cv_over_pairs,
    summarize_cv,
    stitch_cv_folds,
    run_systematic_backtest,
    backtest_with_rolling_cointegration,
    backtest_with_kalman_filter,
    rolling_cointegration_analysis,
    adaptive_cointegration_analysis,
    TRADING_DAYS_PER_YEAR,
)


class TestDataAlignment:
    """Test data alignment and preprocessing functions."""

    def test_align_price_data(self):
        """Test price series alignment with overlapping dates."""
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        price1 = pd.Series(range(100, 110), index=dates, name="asset1")
        price2 = pd.Series(range(200, 210), index=dates, name="asset2")

        aligned = align_price_data(price1, price2)

        assert isinstance(aligned, pd.DataFrame)
        assert list(aligned.columns) == ["asset1", "asset2"]
        assert len(aligned) == 10
        np.testing.assert_array_equal(aligned["asset1"].values, range(100, 110))
        np.testing.assert_array_equal(aligned["asset2"].values, range(200, 210))

    def test_align_price_data_with_nans(self):
        """Test alignment with missing values."""
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        price1 = pd.Series([100, np.nan, 102, 103, 104, 105, 106, np.nan, 108, 109], index=dates)
        price2 = pd.Series([200, 201, np.nan, 203, 204, 205, 206, 207, 208, np.nan], index=dates)

        aligned = align_price_data(price1, price2)

        # Should remove rows with NaN in either series
        assert len(aligned) == 6  # Only complete cases
        assert not aligned.isnull().any().any()

    def test_split_train_test_by_ratio(self):
        """Test train/test split using percentage ratio."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        data = pd.DataFrame({"price": range(100)}, index=dates)

        result = split_train_test(data, train_ratio=0.7)

        assert result["train_size"] == 70
        assert result["test_size"] == 30
        assert len(result["train_data"]) == 70
        assert len(result["test_data"]) == 30

        # Check continuity
        assert result["train_data"].index[-1] < result["test_data"].index[0]

    def test_split_train_test_by_date(self):
        """Test train/test split using specific end date."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        data = pd.DataFrame({"price": range(100)}, index=dates)

        split_date = dates[60]  # 60% point
        result = split_train_test(data, train_end_date=split_date)

        assert result["train_size"] == 61  # Includes split date
        assert result["test_size"] == 39
        assert result["split_date"] == split_date


class TestPerformanceMetrics:
    """Test performance calculation functions."""

    def test_calculate_performance_metrics_basic(self, numerical_tolerances):
        """Test basic performance metrics calculation."""
        # Simple return series: 1% gains for 10 days
        returns = pd.Series([0.01] * 10)

        metrics = calculate_performance_metrics(returns)

        expected_keys = {
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
        }
        assert set(metrics.keys()) == expected_keys

        # Total return should be approximately 10.46% (compound)
        expected_total = (1.01**10) - 1
        np.testing.assert_allclose(metrics["total_return"], expected_total, rtol=1e-6)

        # Annualized volatility should be zero (constant returns)
        assert metrics["annualized_volatility"] == 0

        # Sharpe ratio should be infinite (zero vol), but function returns 0
        assert metrics["sharpe_ratio"] == 0

    def test_calculate_performance_metrics_with_volatility(self):
        """Test metrics with realistic return volatility."""
        np.random.seed(123)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns for 1 year

        metrics = calculate_performance_metrics(returns)

        # All metrics should be finite
        for key, value in metrics.items():
            assert np.isfinite(value), f"Non-finite value for {key}: {value}"

        # Sharpe ratio should be reasonable
        assert -5 < metrics["sharpe_ratio"] < 5

    def test_calculate_performance_metrics_empty_series(self):
        """Test handling of empty return series."""
        empty_returns = pd.Series([])

        metrics = calculate_performance_metrics(empty_returns)

        # Should return zero values for empty series
        expected = {
            "total_return": 0,
            "annualized_return": 0,
            "annualized_volatility": 0,
            "sharpe_ratio": 0,
        }
        assert metrics == expected

    def test_compute_drawdowns(self):
        """Test drawdown calculation."""
        # Equity curve: up, down, up pattern
        equity = pd.Series([1.0, 1.1, 1.2, 1.0, 0.9, 1.1, 1.3])

        dd_result = compute_drawdowns(equity)

        expected_keys = {"drawdown", "peak", "max_drawdown", "max_dd_date"}
        assert set(dd_result.keys()) == expected_keys

        # Peak should be running maximum
        expected_peak = pd.Series([1.0, 1.1, 1.2, 1.2, 1.2, 1.2, 1.3])
        pd.testing.assert_series_equal(dd_result["peak"], expected_peak)

        # Drawdown should be (value/peak) - 1
        expected_dd = equity / expected_peak - 1
        pd.testing.assert_series_equal(dd_result["drawdown"], expected_dd)

        # Max drawdown should be most negative value
        assert dd_result["max_drawdown"] == (0.9 / 1.2 - 1)  # -25%

    def test_compute_rolling_sharpe(self):
        """Test rolling Sharpe ratio calculation."""
        np.random.seed(123)
        returns = pd.Series(np.random.normal(0.001, 0.02, 100))

        rolling_sharpe = compute_rolling_sharpe(returns, window=20)

        # Should have NaN for first 19 values, then valid Sharpe ratios
        assert rolling_sharpe.iloc[:19].isna().all()
        assert rolling_sharpe.iloc[19:].notna().all()

        # Values should be reasonable
        valid_sharpe = rolling_sharpe.dropna()
        assert (valid_sharpe.abs() < 10).all()  # Reasonable range

    def test_compute_rolling_beta(self, make_benchmark):
        """Test rolling beta calculation."""
        benchmark_returns = make_benchmark(T=100, annual_return=0.08)

        # Create strategy with known beta relationship
        np.random.seed(123)
        beta = 0.8
        strategy_returns = beta * benchmark_returns + 0.01 * np.random.randn(100)

        rolling_beta = compute_rolling_beta(strategy_returns, benchmark_returns, window=30)

        # Should have NaN for first 29 values
        assert rolling_beta.iloc[:29].isna().all()

        # Average beta should be close to true beta
        avg_beta = rolling_beta.dropna().mean()
        assert abs(avg_beta - beta) < 0.3  # Allow for estimation error

    def test_rolling_beta_zero_variance_market(self):
        """Test rolling beta with zero market variance periods."""
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        zero_market_returns = pd.Series([0.0] * 50, index=dates)  # Zero variance
        strategy_returns = pd.Series(np.random.randn(50) * 0.01, index=dates)

        rolling_beta = compute_rolling_beta(strategy_returns, zero_market_returns, window=20)

        # Should handle division by zero gracefully
        valid_betas = rolling_beta.dropna()
        assert valid_betas.empty or (valid_betas.abs() == np.inf).any()  # Either empty or infinite


class TestCointegrationEstimation:
    """Test cointegration estimation functions."""

    def test_estimate_cointegration_perfect_relationship(self):
        """Test cointegration estimation on perfect linear relationship."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        x = pd.Series(range(100, 200), index=dates)
        y = 5 + 1.5 * x  # Perfect relationship: y = 5 + 1.5*x

        result = estimate_cointegration(y, x, add_constant=True)

        expected_keys = {"alpha", "beta", "spread", "residuals", "adf_pvalue", "r_squared", "model"}
        assert set(result.keys()) == expected_keys

        # Should recover true parameters
        np.testing.assert_allclose(result["alpha"], 5.0, rtol=1e-6)
        np.testing.assert_allclose(result["beta"], 1.5, rtol=1e-6)

        # R-squared should be perfect
        np.testing.assert_allclose(result["r_squared"], 1.0, rtol=1e-6)

        # Spread should be near zero (perfect fit)
        assert result["spread"].std() < 1e-10

    def test_estimate_cointegration_no_constant(self):
        """Test cointegration through origin (no intercept)."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        x = pd.Series(range(1, 101), index=dates)
        y = 2.0 * x  # y = 2*x (no intercept)

        result = estimate_cointegration(y, x, add_constant=False)

        assert result["alpha"] == 0.0  # No constant term
        np.testing.assert_allclose(result["beta"], 2.0, rtol=1e-6)

    def test_estimate_cointegration_with_noise(self, make_cointegrated_pair):
        """Test estimation with realistic noisy cointegrated pair."""
        data = make_cointegrated_pair(T=300, beta=1.2, sigma=0.02)

        result = estimate_cointegration(data["y"], data["x"])

        # Should detect cointegration (low ADF p-value)
        assert result["adf_pvalue"] < 0.05

        # Beta estimate should be reasonable
        assert abs(result["beta"] - data["true_beta"]) < 0.5

        # R-squared should be high for cointegrated series
        assert result["r_squared"] > 0.5


class TestSignalGeneration:
    """Test trading signal generation."""

    def test_generate_trading_signals_basic(self, make_ou_process):
        """Test signal generation on mean-reverting process."""
        spread = make_ou_process(T=200, theta=0.1, mu=0.0, sigma=0.5)

        signals = generate_trading_signals(spread, z_threshold=2.0)

        expected_keys = {"positions", "z_scores", "entry_signals", "mean_spread", "std_spread"}
        assert set(signals.keys()) == expected_keys

        # Positions should be -1, 0, or 1
        assert set(signals["positions"].unique()).issubset({-1, 0, 1})

        # Z-scores should be standardized
        z_scores = signals["z_scores"]
        np.testing.assert_allclose(z_scores.mean(), 0.0, atol=1e-10)
        np.testing.assert_allclose(z_scores.std(), 1.0, rtol=1e-6)

        # Entry signals should be binary
        assert set(signals["entry_signals"].unique()).issubset({0, 1})

    def test_generate_trading_signals_thresholds(self, make_ou_process):
        """Test that signals respect threshold levels."""
        spread = make_ou_process(T=500, theta=0.05, mu=0.0, sigma=1.0)

        z_threshold = 1.5
        signals = generate_trading_signals(spread, z_threshold=z_threshold)

        z_scores = signals["z_scores"]
        positions = signals["positions"]

        # Long positions should occur when Z < -threshold
        long_mask = positions == 1
        assert (z_scores[long_mask] < -z_threshold).all()

        # Short positions should occur when Z > +threshold
        short_mask = positions == -1
        assert (z_scores[short_mask] > z_threshold).all()

        # Neutral positions when |Z| <= threshold
        neutral_mask = positions == 0
        neutral_z = z_scores[neutral_mask]
        assert (neutral_z.abs() <= z_threshold).all()

    def test_generate_trading_signals_constant_spread(self):
        """Test signal generation on constant spread (edge case)."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        constant_spread = pd.Series([1.0] * 100, index=dates)

        # Should handle zero variance gracefully
        signals = generate_trading_signals(constant_spread, z_threshold=2.0)

        # With zero variance, Z-scores should be NaN or zero
        z_scores = signals["z_scores"]
        assert (z_scores == 0).all() or z_scores.isna().all()


class TestStrategyReturns:
    """Test strategy return calculation."""

    def test_calculate_strategy_returns_basic(self):
        """Test strategy return calculation with known inputs."""
        dates = pd.date_range("2020-01-01", periods=5, freq="B")

        # Simple price series
        price1 = pd.Series([100, 101, 102, 101, 103], index=dates)
        price2 = pd.Series([50, 51, 51.5, 50.5, 52], index=dates)

        # Simple positions: long, neutral, short, neutral
        positions = pd.Series([1, 0, -1, 0, 1], index=dates)
        beta = 2.0

        result = calculate_strategy_returns(price1, price2, positions, beta)

        expected_keys = {
            "strategy_returns",
            "spread_returns",
            "asset1_returns",
            "asset2_returns",
            "cumulative_returns",
            "positions_used",
        }
        assert set(result.keys()) == expected_keys

        # Check that positions are lagged (avoid look-ahead bias)
        expected_lagged_pos = pd.Series([np.nan, 1, 0, -1, 0], index=dates)
        pd.testing.assert_series_equal(
            result["positions_used"], expected_lagged_pos, check_dtype=False
        )

        # First return should be zero (no lagged position)
        assert result["strategy_returns"].iloc[0] == 0

    def test_calculate_strategy_returns_beta_neutral(self):
        """Test that returns are beta-neutral."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")

        # Create correlated price movements
        np.random.seed(123)
        returns = np.random.normal(0.001, 0.02, 100)
        price1 = pd.Series(100 * np.cumprod(1 + returns), index=dates)
        price2 = pd.Series(50 * np.cumprod(1 + 0.8 * returns), index=dates)  # 80% correlation

        # Always long position
        positions = pd.Series([1] * 100, index=dates)
        beta = 0.8

        result = calculate_strategy_returns(price1, price2, positions, beta)

        # Spread returns should be asset1 - beta*asset2
        expected_spread_rets = result["asset1_returns"] - beta * result["asset2_returns"]
        pd.testing.assert_series_equal(
            result["spread_returns"], expected_spread_rets, check_exact=False
        )


class TestCrossValidation:
    """Test cross-validation backtesting framework."""

    def test_compute_ts_folds(self):
        """Test time series cross-validation fold creation."""
        dates = pd.date_range("2020-01-01", periods=200, freq="B")

        folds = compute_ts_folds(dates, n_splits=3, min_train_ratio=0.6, min_test_size=20)

        # Should return list of (train_idx, test_idx) tuples
        assert isinstance(folds, list)
        assert len(folds) <= 3  # Requested splits (may be fewer if constraints)

        for train_idx, test_idx in folds:
            # Train should come before test (temporal order)
            assert train_idx[-1] < test_idx[0]

            # Test size constraint
            assert len(test_idx) >= 20

            # Train size constraint (60% of total)
            assert len(train_idx) >= 0.6 * len(dates)

    def test_run_cross_validation_backtest(self, make_cointegrated_pair):
        """Test cross-validation backtest on synthetic pair."""
        data = make_cointegrated_pair(T=300, beta=1.2, rho=0.8)

        cv_results = run_cross_validation_backtest(
            data["y"], data["x"], z_thresholds=[1.5, 2.0], n_splits=2, transaction_costs=0.001
        )

        # Check return structure
        assert isinstance(cv_results, pd.DataFrame)

        expected_cols = {
            "split",
            "z_threshold",
            "hedge_ratio",
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "num_trades",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
        }
        assert set(cv_results.columns) == expected_cols

        # Should have results for each split x threshold combination
        assert len(cv_results) == 2 * 2  # 2 splits x 2 thresholds

        # Check split chronological order
        for split_id in cv_results["split"].unique():
            split_data = cv_results[cv_results["split"] == split_id]
            assert (split_data["train_end"] < split_data["test_start"]).all()

    def test_run_cv_over_pairs_integration(self, sample_portfolio_data):
        """Test cross-validation over multiple pairs."""
        portfolio = sample_portfolio_data
        z_threshold_by_pair = {pair: 2.0 for pair in portfolio["selected"]}

        cv_results = run_cv_over_pairs(
            portfolio["all_data"],
            portfolio["selected"],
            z_threshold_by_pair,
            n_splits=2,
            return_artifacts=False,
        )

        # Should return results for all pairs
        assert isinstance(cv_results, pd.DataFrame)
        if not cv_results.empty:
            assert "pair" in cv_results.columns
            assert set(cv_results["pair"].unique()).issubset(set(portfolio["selected"]))

    def test_summarize_cv(self, sample_portfolio_data):
        """Test cross-validation result summarization."""
        # Create minimal CV results for testing
        cv_data = pd.DataFrame(
            {
                "pair": ["tech_pair", "tech_pair", "energy_pair", "energy_pair"],
                "z_threshold": [2.0, 2.0, 1.5, 1.5],
                "total_return": [0.1, 0.12, 0.08, 0.09],
                "sharpe_ratio": [1.2, 1.3, 0.9, 1.0],
                "max_drawdown": [-0.05, -0.06, -0.08, -0.07],
                "num_trades": [10, 12, 8, 9],
            }
        )

        summary = summarize_cv(cv_data)

        # Check aggregation structure
        assert isinstance(summary, pd.DataFrame)
        expected_cols_pattern = [
            "total_return_mean",
            "total_return_std",
            "sharpe_ratio_mean",
            "sharpe_ratio_std",
            "max_drawdown_mean",
            "num_trades_mean",
        ]

        for col in expected_cols_pattern:
            assert col in summary.columns

    def test_stitch_cv_folds(self, sample_portfolio_data):
        """Test stitching CV folds into continuous time series."""
        # Create mock artifacts for stitching test
        portfolio = sample_portfolio_data
        pair_name = "tech_pair"
        z_threshold = 2.0

        # Create minimal artifacts structure
        mock_artifacts = {
            pair_name: {
                (0, z_threshold): {
                    "strategy_returns": pd.Series(
                        [0.01, 0.02, -0.01], index=pd.date_range("2020-01-01", periods=3, freq="B")
                    ),
                    "cumulative_returns": pd.Series(
                        [1.01, 1.0302, 1.020398],
                        index=pd.date_range("2020-01-01", periods=3, freq="B"),
                    ),
                    "positions": pd.Series(
                        [1, 1, -1], index=pd.date_range("2020-01-01", periods=3, freq="B")
                    ),
                    "spread": pd.Series(
                        [0.1, 0.2, -0.1], index=pd.date_range("2020-01-01", periods=3, freq="B")
                    ),
                },
                (1, z_threshold): {
                    "strategy_returns": pd.Series(
                        [0.005, -0.01], index=pd.date_range("2020-01-06", periods=2, freq="B")
                    ),
                    "cumulative_returns": pd.Series(
                        [1.005, 0.9950], index=pd.date_range("2020-01-06", periods=2, freq="B")
                    ),
                    "positions": pd.Series(
                        [0, -1], index=pd.date_range("2020-01-06", periods=2, freq="B")
                    ),
                    "spread": pd.Series(
                        [0.05, -0.15], index=pd.date_range("2020-01-06", periods=2, freq="B")
                    ),
                },
            }
        }

        stitched = stitch_cv_folds(mock_artifacts, pair_name, z_threshold)

        expected_keys = {
            "strategy_returns",
            "cumulative_returns",
            "positions",
            "spread",
            "drawdowns",
            "fold_boundaries",
            "performance_metrics",
        }
        assert set(stitched.keys()) == expected_keys

        # Check continuity - cumulative returns should chain properly
        cum_rets = stitched["cumulative_returns"].dropna()
        assert len(cum_rets) > 0

        # Check performance metrics structure
        perf_metrics = stitched["performance_metrics"]
        assert "total_return" in perf_metrics
        assert "sharpe_ratio" in perf_metrics
        assert "max_drawdown" in perf_metrics


class TestFullBacktestIntegration:
    """Test complete backtest pipeline integration."""

    def test_backtest_pair_strategy_integration(self, make_cointegrated_pair):
        """Test full pairs trading backtest pipeline."""
        data = make_cointegrated_pair(T=400, beta=1.3, rho=0.7)

        result = backtest_pair_strategy(
            data["y"], data["x"], z_threshold=2.0, train_ratio=0.6, transaction_costs=0.001
        )

        # Check complete result structure
        expected_keys = {
            "split_info",
            "cointegration",
            "signals",
            "returns",
            "drawdowns",
            "test_spread",
            "performance_metrics",
        }
        assert set(result.keys()) == expected_keys

        # Check performance metrics
        perf = result["performance_metrics"]
        expected_perf_keys = {
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "num_trades",
            "avg_return_per_trade",
            "win_rate",
        }
        assert set(perf.keys()) == expected_perf_keys

        # All metrics should be finite
        for key, value in perf.items():
            assert np.isfinite(value), f"Non-finite value for {key}: {value}"

    def test_backtest_with_no_signals(self, make_cointegrated_pair):
        """Test backtest when no trading signals are generated."""
        data = make_cointegrated_pair(T=200, beta=1.0, rho=0.95)  # Very tight spread

        # Use very wide threshold to prevent signal generation
        result = backtest_pair_strategy(
            data["y"], data["x"], z_threshold=5.0, train_ratio=0.6  # Very wide threshold
        )

        # Should handle no-trade scenario gracefully
        assert result["performance_metrics"]["num_trades"] == 0
        assert result["performance_metrics"]["total_return"] == 0

    def test_transaction_cost_impact(self, make_cointegrated_pair):
        """Test that transaction costs reduce returns."""
        data = make_cointegrated_pair(T=300, beta=1.2, rho=0.6)

        # Run backtest with and without costs
        result_no_cost = backtest_pair_strategy(data["y"], data["x"], transaction_costs=0.0)
        result_with_cost = backtest_pair_strategy(data["y"], data["x"], transaction_costs=0.002)

        # Returns with costs should be lower (assuming trades are made)
        if result_no_cost["performance_metrics"]["num_trades"] > 0:
            no_cost_return = result_no_cost["performance_metrics"]["total_return"]
            with_cost_return = result_with_cost["performance_metrics"]["total_return"]
            assert with_cost_return <= no_cost_return


class TestRollingAnalysis:
    """Test rolling and adaptive analysis methods."""

    def test_backtest_with_rolling_cointegration(self, make_cointegrated_pair):
        """Test backtest with rolling cointegration estimation."""
        # Create pair with parameter change to test rolling adaptation
        data = make_cointegrated_pair(T=500, beta=1.2, break_at=250)

        result = backtest_with_rolling_cointegration(
            data["y"], data["x"], z_threshold=2.0, window_size=100, step_size=20, train_ratio=0.4
        )

        expected_keys = {
            "strategy_returns",
            "cumulative_returns",
            "beta_history",
            "drawdowns",
            "performance_metrics",
            "parameters",
        }
        assert set(result.keys()) == expected_keys

        # Beta should evolve over time
        beta_history = result["beta_history"]
        assert len(beta_history) > 1  # Should have multiple estimates

        # Check parameter storage
        params = result["parameters"]
        assert params["window_size"] == 100
        assert params["step_size"] == 20
        assert params["z_threshold"] == 2.0

    def test_backtest_with_kalman_filter(self, make_cointegrated_pair):
        """Test backtest with adaptive Kalman filter beta."""
        data = make_cointegrated_pair(T=300, beta=1.5)

        result = backtest_with_kalman_filter(data["y"], data["x"], z_threshold=2.0, train_ratio=0.5)

        expected_keys = {
            "strategy_returns",
            "cumulative_returns",
            "adaptive_betas",
            "performance_metrics",
            "parameters",
        }
        assert set(result.keys()) == expected_keys

        # Should have adaptive beta estimates
        adaptive_betas = result["adaptive_betas"]
        assert len(adaptive_betas) > 0

        # Performance metrics should include beta info
        perf = result["performance_metrics"]
        assert "initial_beta" in perf
        assert "final_beta" in perf
        assert "beta_volatility" in perf

    def test_rolling_cointegration_analysis(self, sample_portfolio_data):
        """Test comparison of static vs rolling cointegration approaches."""
        portfolio = sample_portfolio_data
        selected_pairs = portfolio["selected"][:2]  # Use first 2 pairs
        best_z_by_pair = {pair: 1.5 for pair in selected_pairs}

        result = rolling_cointegration_analysis(
            portfolio["all_data"], selected_pairs, best_z_by_pair
        )

        # Should return comparison DataFrame
        assert isinstance(result, pd.DataFrame)

        if not result.empty:
            expected_cols = {"Pair", "Strategy", "Total_Return", "Sharpe", "Max_DD"}
            assert set(result.columns) == expected_cols

            # Should have static and rolling strategies
            strategies = set(result["Strategy"].unique())
            assert "Static" in strategies

    def test_adaptive_cointegration_analysis(self, sample_portfolio_data):
        """Test adaptive vs static cointegration comparison."""
        portfolio = sample_portfolio_data
        selected_pairs = portfolio["selected"][:2]
        best_z_by_pair = {pair: 2.0 for pair in selected_pairs}

        result = adaptive_cointegration_analysis(
            portfolio["all_data"], selected_pairs, best_z_by_pair
        )

        # Should return summary and detailed results
        expected_keys = {"summary_df", "detailed_results"}
        assert set(result.keys()) == expected_keys

        summary_df = result["summary_df"]
        if not summary_df.empty:
            expected_summary_cols = {
                "Pair",
                "Static_Beta",
                "Initial_Beta",
                "Final_Beta",
                "Beta_Drift",
                "Beta_Vol",
                "Static_Sharpe",
                "Kalman_Sharpe",
            }
            assert set(summary_df.columns) == expected_summary_cols

        # Detailed results should contain adaptive beta series
        detailed = result["detailed_results"]
        for pair_name in selected_pairs:
            if pair_name in detailed:
                assert "adaptive_betas" in detailed[pair_name]
                assert isinstance(detailed[pair_name]["adaptive_betas"], pd.Series)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    def test_cross_validation_with_insufficient_data(self):
        """Test CV behavior with insufficient data for splits."""
        # Very short time series
        dates = pd.date_range("2020-01-01", periods=50, freq="B")
        price1 = pd.Series(range(50), index=dates)
        price2 = pd.Series(range(50, 100), index=dates)

        cv_results = run_cross_validation_backtest(
            price1,
            price2,
            z_thresholds=[2.0],
            n_splits=5,  # Too many splits for short series
            min_train_ratio=0.6,
            min_test_size=10,
        )

        # Should return empty or limited results without crashing
        assert isinstance(cv_results, pd.DataFrame)

    def test_constant_price_series(self):
        """Test handling of constant price series."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        constant_price1 = pd.Series([100.0] * 100, index=dates)
        constant_price2 = pd.Series([50.0] * 100, index=dates)

        # Should handle without crashing
        result = backtest_pair_strategy(constant_price1, constant_price2)
        assert isinstance(result, dict)

    def test_perfect_anticorrelation(self):
        """Test handling of perfectly anti-correlated series."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        price1 = pd.Series(range(100, 200), index=dates)
        price2 = pd.Series(range(200, 100, -1), index=dates)  # Perfect negative correlation

        result = backtest_pair_strategy(price1, price2)

        # Should detect strong cointegration
        assert result["cointegration"]["adf_pvalue"] < 0.001  # Very strong cointegration

    def test_single_observation_test_set(self, make_cointegrated_pair):
        """Test CV with single observation in test set."""
        data = make_cointegrated_pair(T=100)

        # Force very large train ratio
        result = run_cross_validation_backtest(
            data["y"],
            data["x"],
            z_thresholds=[2.0],
            n_splits=1,
            min_train_ratio=0.98,  # Force tiny test set
            min_test_size=1,
        )

        # Should handle gracefully
        assert isinstance(result, pd.DataFrame)
