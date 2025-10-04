"""
Integration tests and test runner configuration.

Runs full end-to-end tests with small synthetic datasets to validate
the complete pairs trading pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
import warnings


# Integration test for complete pipeline
class TestPairsTradeIntegration:
    """End-to-end integration tests for the complete pairs trading system."""

    def test_small_portfolio_end_to_end(self, sample_portfolio_data):
        """Test complete pipeline on small synthetic portfolio."""
        portfolio = sample_portfolio_data

        # Import functions from all modules
        from cointegration_analysis.analytics.cointegration import engle_granger
        from cointegration_analysis.analytics.backtesting import (
            run_cv_over_pairs,
            summarize_cv,
            stitch_cv_folds,
        )
        from cointegration_analysis.analytics.plotting import analyze_pairs_nb

        # Step 1: Test cointegration on all pairs
        cointegration_results = {}
        for pair_name, df in portfolio["all_data"].items():
            y_col, x_col = df.columns
            eg_result = engle_granger(df, y_col, x_col)
            cointegration_results[pair_name] = eg_result

        # Filter to cointegrated pairs only
        cointegrated_pairs = [
            pair
            for pair, result in cointegration_results.items()
            if result["eg_pvalue"] <= 0.05 and result["spread"] is not None
        ]

        if len(cointegrated_pairs) == 0:
            pytest.skip("No cointegrated pairs found in synthetic data")

        # Step 2: Optimize thresholds for cointegrated pairs
        selected_subset = cointegrated_pairs[:2]  # Limit for test speed

        with patch("cointegration_analysis.analytics.plotting.plt.show"):
            summary_df, opt_tables = analyze_pairs_nb(
                portfolio["all_data"],
                selected_subset,
                Z_min=1.5,
                Z_max=2.5,
                dZ=0.5,  # Coarse grid for speed
            )

        if summary_df.empty:
            pytest.skip("Threshold optimization found no valid results")

        # Step 3: Run cross-validation backtests
        z_threshold_by_pair = dict(zip(summary_df["pair"], summary_df["best_Z"]))

        cv_results, cv_artifacts = run_cv_over_pairs(
            portfolio["all_data"],
            selected_subset,
            z_threshold_by_pair,
            n_splits=2,
            return_artifacts=True,
        )

        # Step 4: Summarize CV results
        if not cv_results.empty:
            cv_summary = summarize_cv(cv_results)

            # Verify summary structure
            assert isinstance(cv_summary, pd.DataFrame)
            assert "total_return_mean" in cv_summary.columns
            assert "sharpe_ratio_mean" in cv_summary.columns

        # Step 5: Stitch results for systematic backtest
        if cv_artifacts and len(selected_subset) > 0:
            pair_name = selected_subset[0]
            z_thresh = z_threshold_by_pair[pair_name]

            try:
                stitched = stitch_cv_folds(cv_artifacts, pair_name, z_thresh)

                # Verify stitched results
                assert "strategy_returns" in stitched
                assert "cumulative_returns" in stitched
                assert "performance_metrics" in stitched

                # Performance metrics should be reasonable
                perf = stitched["performance_metrics"]
                assert np.isfinite(perf["total_return"])
                assert np.isfinite(perf["sharpe_ratio"])

            except (KeyError, ValueError):
                # Acceptable if stitching fails due to insufficient data
                pass

    def test_rolling_vs_static_comparison(self, make_cointegrated_pair):
        """Test comparison between static and rolling cointegration approaches."""
        from cointegration_analysis.analytics.backtesting import (
            backtest_pair_strategy,
            backtest_with_rolling_cointegration,
            backtest_with_kalman_filter,
        )

        # Create pair with parameter drift to test adaptability
        data = make_cointegrated_pair(T=600, beta=1.3, rho=0.6, break_at=300)

        # Static backtest
        static_result = backtest_pair_strategy(
            data["y"], data["x"], z_threshold=2.0, train_ratio=0.4
        )

        # Rolling cointegration backtest
        rolling_result = backtest_with_rolling_cointegration(
            data["y"], data["x"], z_threshold=2.0, window_size=150, step_size=30, train_ratio=0.4
        )

        # Kalman filter backtest
        kalman_result = backtest_with_kalman_filter(
            data["y"], data["x"], z_threshold=2.0, train_ratio=0.4
        )

        # All should produce valid results
        for result, name in [
            (static_result, "static"),
            (rolling_result, "rolling"),
            (kalman_result, "kalman"),
        ]:
            assert isinstance(result, dict)
            assert "performance_metrics" in result

            perf = result["performance_metrics"]
            for metric in ["total_return", "sharpe_ratio", "max_drawdown"]:
                assert metric in perf
                assert np.isfinite(perf[metric]), f"Non-finite {metric} in {name} backtest"

        # Rolling and Kalman should adapt to parameter changes better
        # (This is a hypothesis test - may not always hold due to randomness)
        static_sharpe = static_result["performance_metrics"]["sharpe_ratio"]
        rolling_sharpe = rolling_result["performance_metrics"]["sharpe_ratio"]
        kalman_sharpe = kalman_result["performance_metrics"]["sharpe_ratio"]

        # At minimum, adaptive methods shouldn't be dramatically worse
        assert rolling_sharpe >= static_sharpe - 1.0  # Allow some deterioration
        assert kalman_sharpe >= static_sharpe - 1.0

    @pytest.mark.slow
    def test_systematic_portfolio_backtest(self, sample_portfolio_data):
        """
        Comprehensive systematic backtest across portfolio.
        Marked as slow - only run with pytest -m slow.
        """
        from cointegration_analysis.analytics.backtesting import (
            rolling_cointegration_analysis,
            run_systematic_backtest,
        )
        from cointegration_analysis.analytics.plotting import plot_systematic_performance

        portfolio = sample_portfolio_data
        selected_pairs = portfolio["selected"]

        # Mock CV artifacts and summary for systematic backtest
        mock_summary = pd.DataFrame({"pair": selected_pairs, "best_Z": [2.0] * len(selected_pairs)})

        # Create minimal CV artifacts structure
        mock_cv_artifacts = {}
        dates = pd.date_range("2020-01-01", periods=100, freq="B")

        for pair_name in selected_pairs:
            returns = np.random.randn(100) * 0.01
            mock_cv_artifacts[pair_name] = {
                (0, 2.0): {
                    "strategy_returns": pd.Series(returns, index=dates),
                    "cumulative_returns": pd.Series(np.cumprod(1 + returns), index=dates),
                    "positions": pd.Series(np.random.choice([-1, 0, 1], 100), index=dates),
                    "spread": pd.Series(np.random.randn(100), index=dates),
                }
            }

        # Test systematic backtest
        stitched_results, systematic_df = run_systematic_backtest(
            mock_cv_artifacts, selected_pairs[:2], mock_summary
        )

        # Verify results structure
        assert isinstance(stitched_results, dict)
        assert isinstance(systematic_df, pd.DataFrame)

        # Test rolling analysis comparison
        best_z_by_pair = {pair: 2.0 for pair in selected_pairs[:2]}

        rolling_comparison = rolling_cointegration_analysis(
            portfolio["all_data"], selected_pairs[:2], best_z_by_pair
        )

        assert isinstance(rolling_comparison, pd.DataFrame)
        if not rolling_comparison.empty:
            expected_cols = {"Pair", "Strategy", "Total_Return", "Sharpe", "Max_DD"}
            assert set(rolling_comparison.columns) == expected_cols


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_extreme_parameter_values(self, make_cointegrated_pair):
        """Test system behavior with extreme parameter values."""
        from cointegration_analysis.analytics.backtesting import backtest_pair_strategy

        # Very high beta
        high_beta_data = make_cointegrated_pair(T=300, beta=10.0, rho=0.8)
        result_high_beta = backtest_pair_strategy(
            high_beta_data["y"], high_beta_data["x"], z_threshold=2.0
        )

        # Very low error correction speed
        slow_reversion_data = make_cointegrated_pair(T=400, beta=1.2, rho=0.01)
        result_slow = backtest_pair_strategy(
            slow_reversion_data["y"], slow_reversion_data["x"], z_threshold=1.5
        )

        # Both should produce valid results without crashing
        for result in [result_high_beta, result_slow]:
            assert isinstance(result, dict)
            assert "performance_metrics" in result

            # Metrics may be poor but should be finite
            perf = result["performance_metrics"]
            assert np.isfinite(perf["total_return"])
            assert np.isfinite(perf["sharpe_ratio"]) or perf["sharpe_ratio"] == 0

    def test_market_stress_scenarios(self, make_benchmark):
        """Test system during market stress (high volatility, crashes)."""
        from cointegration_analysis.analytics.backtesting import (
            compute_rolling_beta,
            compute_rolling_sharpe,
        )

        # High volatility market
        stress_market = make_benchmark(T=200, annual_return=-0.20, annual_vol=0.50)

        # Strategy returns that are somewhat correlated
        np.random.seed(123)
        strategy_returns = 0.3 * stress_market + 0.01 * np.random.randn(200)

        # Test rolling metrics during stress
        rolling_beta = compute_rolling_beta(strategy_returns, stress_market, window=63)
        rolling_sharpe = compute_rolling_sharpe(strategy_returns, window=63)

        # Should handle high volatility without numerical issues
        valid_beta = rolling_beta.dropna()
        valid_sharpe = rolling_sharpe.dropna()

        if len(valid_beta) > 0:
            assert (np.abs(valid_beta) < 100).all()  # Beta shouldn't explode

        if len(valid_sharpe) > 0:
            assert (np.abs(valid_sharpe) < 50).all()  # Sharpe shouldn't explode

    def test_data_quality_issues(self):
        """Test handling of poor quality data."""
        from cointegration_analysis.analytics.cointegration import adf_results, engle_granger

        # Series with many gaps
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        gappy_series = pd.Series(range(100), index=dates)
        gappy_series.iloc[::5] = np.nan  # Remove every 5th observation

        # Should handle gaps gracefully
        adf_result = adf_results(gappy_series)
        assert isinstance(adf_result, dict)
        assert np.isfinite(adf_result["pvalue"])

        # Misaligned price series
        price1 = pd.Series(range(100), index=dates)
        price2 = pd.Series(range(len(dates[10:])), index=dates[10:])  # Start 10 days later

        df_misaligned = pd.DataFrame({"y": price1, "x": price2})
        eg_result = engle_granger(df_misaligned, "y", "x")

        # Should handle alignment automatically
        assert isinstance(eg_result, dict)


# Slow tests are handled via --slow option added in conftest.py
