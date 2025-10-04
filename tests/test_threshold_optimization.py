"""
Tests for ``cointegration_analysis.analytics.optimization``.

Exercises threshold optimization and backtesting functionality for spread trading strategies.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing

# Import functions under test
from cointegration_analysis.analytics.optimization import (
    backtest_spread,
    optimize_thresholds,
    plot_threshold_tradeoff,
)


class TestBacktestSpread:
    """Test spread backtesting functionality."""

    def test_backtest_spread_basic_functionality(self, make_ou_process):
        """Test basic spread backtesting with known OU process."""
        # Create OU process with known parameters
        spread = make_ou_process(T=300, theta=0.2, mu=0.0, sigma=1.0)

        # Create dummy price series (not used in calculation but required by interface)
        dates = spread.index
        y_dummy = pd.Series(range(len(spread)), index=dates)
        x_dummy = pd.Series(range(len(spread)), index=dates)

        result = backtest_spread(
            e=spread, mu=0.0, sigma=1.0, beta=1.0, y=y_dummy, x=x_dummy, Z=2.0, cost=0.0
        )

        # Check return structure
        expected_keys = {"N_trades", "cum_PnL", "avg_PnL", "avg_duration"}
        assert set(result.keys()) == expected_keys

        # Should generate some trades for mean-reverting process
        assert result["N_trades"] >= 0
        assert isinstance(result["N_trades"], int)

        # P&L should be finite
        assert np.isfinite(result["cum_PnL"])

        # If trades were made, avg_PnL should be finite, otherwise NaN
        if result["N_trades"] > 0:
            assert np.isfinite(result["avg_PnL"])
            assert np.isfinite(result["avg_duration"])
        else:
            assert np.isnan(result["avg_PnL"])
            assert np.isnan(result["avg_duration"])

    def test_backtest_spread_threshold_sensitivity(self, make_ou_process):
        """Test that wider thresholds generate fewer trades."""
        spread = make_ou_process(T=500, theta=0.1, mu=0.0, sigma=1.5)
        dates = spread.index
        y_dummy = pd.Series(range(len(spread)), index=dates)
        x_dummy = pd.Series(range(len(spread)), index=dates)

        # Test multiple threshold levels
        thresholds = [0.5, 1.0, 2.0, 3.0]
        results = []

        for Z in thresholds:
            result = backtest_spread(
                e=spread, mu=0.0, sigma=1.5, beta=1.0, y=y_dummy, x=x_dummy, Z=Z, cost=0.0
            )
            results.append((Z, result["N_trades"]))

        # Generally, wider thresholds should produce fewer trades
        # (though not strictly monotonic due to different exit conditions)
        n_trades_by_threshold = [n_trades for _, n_trades in results]

        # At minimum, the widest threshold should have fewer trades than narrowest
        assert n_trades_by_threshold[-1] <= n_trades_by_threshold[0]

    def test_backtest_spread_with_transaction_costs(self, make_ou_process):
        """Test that transaction costs reduce overall P&L."""
        spread = make_ou_process(T=300, theta=0.15, mu=0.0, sigma=1.0)
        dates = spread.index
        y_dummy = pd.Series(range(len(spread)), index=dates)
        x_dummy = pd.Series(range(len(spread)), index=dates)

        # Backtest without costs
        result_no_cost = backtest_spread(
            e=spread, mu=0.0, sigma=1.0, beta=1.0, y=y_dummy, x=x_dummy, Z=1.5, cost=0.0
        )

        # Backtest with costs
        result_with_cost = backtest_spread(
            e=spread,
            mu=0.0,
            sigma=1.0,
            beta=1.0,
            y=y_dummy,
            x=x_dummy,
            Z=1.5,
            cost=0.01,  # 1% cost per trade
        )

        # If trades were made, P&L with costs should be lower
        if result_no_cost["N_trades"] > 0:
            assert result_with_cost["cum_PnL"] <= result_no_cost["cum_PnL"]

            # Number of trades should be the same (costs don't affect signal generation)
            assert result_with_cost["N_trades"] == result_no_cost["N_trades"]

    def test_backtest_spread_normalization(self, make_ou_process):
        """Test P&L normalization by spread volatility."""
        spread = make_ou_process(T=400, theta=0.1, mu=0.0, sigma=2.0)  # High volatility
        dates = spread.index
        y_dummy = pd.Series(range(len(spread)), index=dates)
        x_dummy = pd.Series(range(len(spread)), index=dates)

        # Backtest without normalization
        result_raw = backtest_spread(
            e=spread, mu=0.0, sigma=2.0, beta=1.0, y=y_dummy, x=x_dummy, Z=2.0, normalize=False
        )

        # Backtest with normalization
        result_norm = backtest_spread(
            e=spread, mu=0.0, sigma=2.0, beta=1.0, y=y_dummy, x=x_dummy, Z=2.0, normalize=True
        )

        # Both should generate same number of trades
        assert result_raw["N_trades"] == result_norm["N_trades"]

        # Normalized P&L should be different (scaled by volatility)
        if result_raw["N_trades"] > 0:
            assert result_raw["cum_PnL"] != result_norm["cum_PnL"]

    def test_backtest_spread_edge_cases(self):
        """Test edge cases: zero variance, constant spread, etc."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")

        # Constant spread (zero variance)
        constant_spread = pd.Series([1.0] * 100, index=dates)
        y_dummy = pd.Series(range(100), index=dates)
        x_dummy = pd.Series(range(100), index=dates)

        # Should handle zero variance gracefully
        result = backtest_spread(
            e=constant_spread, mu=1.0, sigma=0.0, beta=1.0, y=y_dummy, x=x_dummy, Z=2.0
        )

        # Should return zero trades and NaN metrics
        assert result["N_trades"] == 0
        assert result["cum_PnL"] == 0.0
        assert np.isnan(result["avg_PnL"])
        assert np.isnan(result["avg_duration"])

        # Test with NaN sigma
        result_nan = backtest_spread(
            e=constant_spread, mu=1.0, sigma=np.nan, beta=1.0, y=y_dummy, x=x_dummy, Z=2.0
        )

        assert result_nan["N_trades"] == 0

    def test_backtest_spread_perfect_reversion(self):
        """Test backtesting on perfect mean reversion pattern."""
        # Create perfect oscillation around mean
        t = np.arange(200)
        spread_values = 2.0 * np.sin(t * np.pi / 10)  # Oscillates between -2 and +2
        dates = pd.date_range("2020-01-01", periods=200, freq="B")
        spread = pd.Series(spread_values, index=dates)

        y_dummy = pd.Series(range(200), index=dates)
        x_dummy = pd.Series(range(200), index=dates)

        result = backtest_spread(
            e=spread,
            mu=0.0,
            sigma=1.0,
            beta=1.0,
            y=y_dummy,
            x=x_dummy,
            Z=1.0,  # Threshold below max deviation
        )

        # Should capture many profitable mean reversion trades
        assert result["N_trades"] > 10  # Should generate many trades
        assert result["cum_PnL"] > 0  # Should be profitable


class TestOptimizeThresholds:
    """Test threshold optimization functionality."""

    def test_optimize_thresholds_basic(self, make_ou_process):
        """Test basic threshold optimization over range."""
        spread = make_ou_process(T=400, theta=0.15, mu=1.0, sigma=1.0)
        dates = spread.index
        y_dummy = pd.Series(range(len(spread)), index=dates)
        x_dummy = pd.Series(range(len(spread)), index=dates)

        opt_results = optimize_thresholds(
            e=spread,
            mu=1.0,
            sigma=1.0,
            beta=1.0,
            y=y_dummy,
            x=x_dummy,
            Z_min=0.5,
            Z_max=2.5,
            dZ=0.5,
        )

        # Check return structure
        assert isinstance(opt_results, pd.DataFrame)
        expected_cols = {"Z", "N_trades", "cum_PnL", "avg_PnL", "avg_duration"}
        assert set(opt_results.columns) == expected_cols

        # Should test all specified thresholds
        expected_z_values = np.arange(0.5, 2.5 + 0.5, 0.5)
        np.testing.assert_array_almost_equal(sorted(opt_results["Z"].values), expected_z_values)

        # All metrics should be finite or NaN (not inf)
        for col in ["cum_PnL", "avg_PnL", "avg_duration"]:
            assert opt_results[col].apply(lambda x: np.isfinite(x) or np.isnan(x)).all()

    def test_optimize_thresholds_with_ou_parameters(self, make_ou_process):
        """Test optimization using OU process parameters vs sample statistics."""
        spread = make_ou_process(T=500, theta=0.2, mu=2.0, sigma=1.5)
        dates = spread.index
        y_dummy = pd.Series(range(len(spread)), index=dates)
        x_dummy = pd.Series(range(len(spread)), index=dates)

        # Sample statistics
        sample_mu = spread.mean()
        sample_sigma = spread.std()

        # OU parameters (closer to true values)
        ou_mu = 2.0
        ou_sigma = 1.5

        # Optimize using sample statistics
        opt_sample = optimize_thresholds(
            e=spread,
            mu=sample_mu,
            sigma=sample_sigma,
            beta=1.0,
            y=y_dummy,
            x=x_dummy,
            Z_min=1.0,
            Z_max=2.0,
            dZ=0.5,
            use_ou=False,
        )

        # Optimize using OU parameters
        opt_ou = optimize_thresholds(
            e=spread,
            mu=sample_mu,
            sigma=sample_sigma,
            beta=1.0,
            y=y_dummy,
            x=x_dummy,
            Z_min=1.0,
            Z_max=2.0,
            dZ=0.5,
            ou_mu=ou_mu,
            ou_sigma=ou_sigma,
            use_ou=True,
        )

        # Both should return valid DataFrames
        assert isinstance(opt_sample, pd.DataFrame)
        assert isinstance(opt_ou, pd.DataFrame)
        assert len(opt_sample) == len(opt_ou)

        # Results may differ due to different parameters
        # At minimum, should have same threshold values tested
        pd.testing.assert_series_equal(opt_sample["Z"], opt_ou["Z"])

    def test_optimize_thresholds_fine_grid(self, make_ou_process):
        """Test optimization with fine threshold grid."""
        spread = make_ou_process(T=300, theta=0.1, mu=0.0, sigma=1.0)
        dates = spread.index
        y_dummy = pd.Series(range(len(spread)), index=dates)
        x_dummy = pd.Series(range(len(spread)), index=dates)

        # Fine grid for detailed optimization
        opt_results = optimize_thresholds(
            e=spread,
            mu=0.0,
            sigma=1.0,
            beta=1.0,
            y=y_dummy,
            x=x_dummy,
            Z_min=1.0,
            Z_max=2.0,
            dZ=0.1,  # Fine grid
        )

        # Should test many thresholds
        expected_n_thresholds = len(np.arange(1.0, 2.0 + 0.1, 0.1))
        assert len(opt_results) == expected_n_thresholds

        # Should provide smooth progression of metrics
        # Number of trades should generally decrease with higher thresholds
        sorted_results = opt_results.sort_values("Z")
        n_trades = sorted_results["N_trades"].values

        # At minimum, highest threshold should have <= trades than lowest
        assert n_trades[-1] <= n_trades[0]

    def test_optimize_thresholds_with_costs(self, make_ou_process):
        """Test that transaction costs are properly incorporated in optimization."""
        spread = make_ou_process(T=400, theta=0.12, mu=0.0, sigma=1.0)
        dates = spread.index
        y_dummy = pd.Series(range(len(spread)), index=dates)
        x_dummy = pd.Series(range(len(spread)), index=dates)

        # Optimize without costs
        opt_no_cost = optimize_thresholds(
            e=spread,
            mu=0.0,
            sigma=1.0,
            beta=1.0,
            y=y_dummy,
            x=x_dummy,
            Z_min=1.0,
            Z_max=2.5,
            dZ=0.5,
            cost=0.0,
        )

        # Optimize with costs
        opt_with_cost = optimize_thresholds(
            e=spread,
            mu=0.0,
            sigma=1.0,
            beta=1.0,
            y=y_dummy,
            x=x_dummy,
            Z_min=1.0,
            Z_max=2.5,
            dZ=0.5,
            cost=0.005,  # 0.5% cost
        )

        # Number of trades should be the same (costs don't affect signals)
        pd.testing.assert_series_equal(opt_no_cost["N_trades"], opt_with_cost["N_trades"])

        # P&L should be lower with costs (for any rows with trades > 0)
        trades_mask = opt_no_cost["N_trades"] > 0
        if trades_mask.any():
            no_cost_pnl = opt_no_cost.loc[trades_mask, "cum_PnL"]
            with_cost_pnl = opt_with_cost.loc[trades_mask, "cum_PnL"]
            assert (with_cost_pnl <= no_cost_pnl).all()


class TestPlotThresholdTradeoff:
    """Test threshold tradeoff plotting."""

    def test_plot_threshold_tradeoff_structure(self, make_ou_process):
        """Test that plot_threshold_tradeoff returns valid matplotlib figure."""
        # Create optimization results
        spread = make_ou_process(T=200, theta=0.1, mu=0.0, sigma=1.0)
        dates = spread.index
        y_dummy = pd.Series(range(len(spread)), index=dates)
        x_dummy = pd.Series(range(len(spread)), index=dates)

        opt_results = optimize_thresholds(
            e=spread,
            mu=0.0,
            sigma=1.0,
            beta=1.0,
            y=y_dummy,
            x=x_dummy,
            Z_min=0.5,
            Z_max=3.0,
            dZ=0.5,
        )

        # Create plot
        fig = plot_threshold_tradeoff(opt_results)

        # Check that it returns a matplotlib figure
        import matplotlib.figure

        assert isinstance(fig, matplotlib.figure.Figure)

        # Check that figure has expected structure
        axes = fig.get_axes()
        assert len(axes) == 2  # Primary and secondary y-axis

        # Check that both axes have data plotted
        for ax in axes:
            lines = ax.get_lines()
            assert len(lines) > 0  # Should have at least one line plotted

    @patch("matplotlib.pyplot.show")
    def test_plot_threshold_tradeoff_with_mock_data(self, mock_show):
        """Test plotting with controlled synthetic data."""
        # Create controlled optimization results
        z_values = np.arange(0.5, 3.0, 0.25)
        n_trades = np.array([20, 18, 15, 12, 10, 8, 6, 5, 4, 3])  # Decreasing
        cum_pnl = np.array([0.8, 1.2, 1.5, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6])  # Peak in middle

        opt_df = pd.DataFrame(
            {
                "Z": z_values,
                "N_trades": n_trades,
                "cum_PnL": cum_pnl,
                "avg_PnL": cum_pnl / n_trades,
                "avg_duration": np.ones(len(z_values)) * 5.0,
            }
        )

        fig = plot_threshold_tradeoff(opt_df)

        # Verify plot structure
        assert len(fig.get_axes()) == 2

        # Verify legend exists
        legend = fig.get_legends()
        assert len(legend) > 0

    def test_plot_threshold_tradeoff_edge_cases(self):
        """Test plotting with edge case data."""
        # Single threshold point
        single_point_df = pd.DataFrame(
            {
                "Z": [2.0],
                "N_trades": [5],
                "cum_PnL": [1.2],
                "avg_PnL": [0.24],
                "avg_duration": [4.0],
            }
        )

        fig = plot_threshold_tradeoff(single_point_df)
        assert isinstance(fig, matplotlib.figure.Figure)

        # Zero trades scenario
        zero_trades_df = pd.DataFrame(
            {
                "Z": [1.0, 2.0, 3.0],
                "N_trades": [0, 0, 0],
                "cum_PnL": [0.0, 0.0, 0.0],
                "avg_PnL": [np.nan, np.nan, np.nan],
                "avg_duration": [np.nan, np.nan, np.nan],
            }
        )

        fig = plot_threshold_tradeoff(zero_trades_df)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestThresholdOptimizationIntegration:
    """Test integration between optimization components."""

    def test_optimization_to_plotting_pipeline(self, make_ou_process):
        """Test full pipeline from spread generation to plotting."""
        # Step 1: Generate mean-reverting spread
        spread = make_ou_process(T=400, theta=0.15, mu=1.0, sigma=1.2)

        # Step 2: Create dummy price series (required by interface)
        dates = spread.index
        y_prices = pd.Series(100 + np.cumsum(np.random.randn(len(spread)) * 0.01), index=dates)
        x_prices = pd.Series(50 + np.cumsum(np.random.randn(len(spread)) * 0.01), index=dates)

        # Step 3: Run optimization
        opt_results = optimize_thresholds(
            e=spread,
            mu=spread.mean(),
            sigma=spread.std(),
            beta=1.0,
            y=y_prices,
            x=x_prices,
            Z_min=0.5,
            Z_max=3.0,
            dZ=0.25,
            cost=0.001,
            normalize=False,
        )

        # Step 4: Find optimal threshold
        best_idx = opt_results["cum_PnL"].idxmax()
        best_z = opt_results.loc[best_idx, "Z"]
        best_pnl = opt_results.loc[best_idx, "cum_PnL"]
        best_trades = opt_results.loc[best_idx, "N_trades"]

        # Verify optimization found reasonable result
        assert 0.5 <= best_z <= 3.0
        assert isinstance(best_trades, (int, np.integer))
        assert best_trades >= 0

        # Step 5: Create tradeoff plot
        fig = plot_threshold_tradeoff(opt_results)
        assert isinstance(fig, matplotlib.figure.Figure)

        # Verify that best result is reasonable
        if best_trades > 0:
            assert np.isfinite(best_pnl)

    def test_parameter_sensitivity_analysis(self, make_ou_process):
        """Test optimization sensitivity to different parameters."""
        spread = make_ou_process(T=500, theta=0.1, mu=0.0, sigma=1.0)
        dates = spread.index
        y_dummy = pd.Series(range(len(spread)), index=dates)
        x_dummy = pd.Series(range(len(spread)), index=dates)

        # Test different cost levels
        cost_levels = [0.0, 0.001, 0.005, 0.01]
        optimal_z_by_cost = {}

        for cost in cost_levels:
            opt_results = optimize_thresholds(
                e=spread,
                mu=0.0,
                sigma=1.0,
                beta=1.0,
                y=y_dummy,
                x=x_dummy,
                Z_min=1.0,
                Z_max=2.5,
                dZ=0.25,
                cost=cost,
            )

            if not opt_results.empty and opt_results["cum_PnL"].max() > 0:
                best_z = opt_results.loc[opt_results["cum_PnL"].idxmax(), "Z"]
                optimal_z_by_cost[cost] = best_z

        # Generally, higher costs should lead to higher optimal thresholds
        # (fewer, more selective trades)
        if len(optimal_z_by_cost) > 1:
            cost_values = sorted(optimal_z_by_cost.keys())
            z_values = [optimal_z_by_cost[c] for c in cost_values]

            # Not strictly monotonic, but highest cost should generally have higher Z
            assert z_values[-1] >= z_values[0]

    def test_optimization_robustness(self):
        """Test optimization robustness with difficult data."""
        dates = pd.date_range("2020-01-01", periods=300, freq="B")

        # Trending spread (not mean-reverting)
        trending_spread = pd.Series(np.cumsum(np.ones(300) * 0.01), index=dates)

        # Noisy spread with little structure
        np.random.seed(42)
        noisy_spread = pd.Series(np.random.randn(300) * 2.0, index=dates)

        y_dummy = pd.Series(range(300), index=dates)
        x_dummy = pd.Series(range(300), index=dates)

        for spread, description in [(trending_spread, "trending"), (noisy_spread, "noisy")]:
            opt_results = optimize_thresholds(
                e=spread,
                mu=spread.mean(),
                sigma=spread.std(),
                beta=1.0,
                y=y_dummy,
                x=x_dummy,
                Z_min=1.0,
                Z_max=3.0,
                dZ=0.5,
            )

            # Should return valid results even for difficult data
            assert isinstance(opt_results, pd.DataFrame)
            assert len(opt_results) > 0

            # All P&L values should be finite (may be negative)
            assert opt_results["cum_PnL"].apply(np.isfinite).all()

            # For non-mean-reverting data, P&L should generally be poor
            if description == "trending":
                # Trending data should not be profitable for mean reversion strategy
                max_pnl = opt_results["cum_PnL"].max()
                assert max_pnl <= 0.1  # Should not be very profitable
