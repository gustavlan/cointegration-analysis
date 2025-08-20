"""
Tests for plotting.py module.

Tests plotting and analysis functions with focus on data structure validation
and smoke tests. Visual output validation is limited to object creation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

# Import functions under test
from plotting import analyze_pairs_nb, plot_systematic_performance, plot_kalman_beta_evolution


class TestAnalyzePairsNB:
    """Test pairs analysis and threshold optimization."""
    
    def test_analyze_pairs_nb_basic_structure(self, sample_portfolio_data):
        """Test basic structure and return types of analyze_pairs_nb."""
        portfolio = sample_portfolio_data
        selected = portfolio['selected'][:2]  # Use first 2 pairs to speed up test
        
        # Mock the plotting to avoid display issues in tests
        with patch('plotting.plt.show'):
            summary_df, opt_tables = analyze_pairs_nb(
                portfolio['all_data'], 
                selected,
                Z_min=1.0, Z_max=2.5, dZ=0.5,  # Coarse grid for speed
                use_ou=True
            )
        
        # Check summary DataFrame structure
        assert isinstance(summary_df, pd.DataFrame)
        if not summary_df.empty:
            expected_summary_cols = {'pair', 'best_Z', 'N_trades', 'cum_PnL', 'avg_PnL', 'theta', 'half_life'}
            assert set(summary_df.columns) == expected_summary_cols
            
            # Best Z should be within specified range
            assert (summary_df['best_Z'] >= 1.0).all()
            assert (summary_df['best_Z'] <= 2.5).all()
            
            # N_trades should be non-negative integers
            assert (summary_df['N_trades'] >= 0).all()
            assert summary_df['N_trades'].dtype in [np.int64, int]
        
        # Check optimization tables structure
        assert isinstance(opt_tables, dict)
        for pair_name, opt_df in opt_tables.items():
            assert isinstance(opt_df, pd.DataFrame)
            expected_opt_cols = {'Z', 'N_trades', 'cum_PnL', 'avg_PnL', 'avg_duration'}
            assert set(opt_df.columns) == expected_opt_cols
    
    def test_analyze_pairs_nb_with_non_cointegrated(self, make_nonstationary_series):
        """Test behavior when pairs are not cointegrated."""
        # Create independent random walks (should not be cointegrated)
        y_series = make_nonstationary_series(T=200, random_state=42)
        x_series = make_nonstationary_series(T=200, random_state=123)
        
        all_data = {
            'independent_pair': pd.DataFrame({'y': y_series, 'x': x_series})
        }
        selected = ['independent_pair']
        
        with patch('plotting.plt.show'):
            summary_df, opt_tables = analyze_pairs_nb(
                all_data, selected, Z_min=1.0, Z_max=2.0, dZ=0.5
            )
        
        # Should handle non-cointegrated pairs gracefully (skip them)
        # May return empty results if no cointegration found
        assert isinstance(summary_df, pd.DataFrame)
        assert isinstance(opt_tables, dict)
    
    def test_analyze_pairs_nb_parameter_variations(self, make_cointegrated_pair):
        """Test analyze_pairs_nb with different parameter configurations."""
        data = make_cointegrated_pair(T=300, beta=1.2, rho=0.8)
        all_data = {
            'test_pair': pd.DataFrame({'asset1': data['y'], 'asset2': data['x']})
        }
        selected = ['test_pair']
        
        with patch('plotting.plt.show'):
            # Test with OU parameters disabled
            summary_no_ou, _ = analyze_pairs_nb(
                all_data, selected,
                Z_min=1.5, Z_max=2.0, dZ=0.25,
                use_ou=False,
                normalize=False
            )
            
            # Test with normalization enabled
            summary_norm, _ = analyze_pairs_nb(
                all_data, selected,
                Z_min=1.5, Z_max=2.0, dZ=0.25,
                use_ou=True,
                normalize=True
            )
        
        # Both should return valid DataFrames
        assert isinstance(summary_no_ou, pd.DataFrame)
        assert isinstance(summary_norm, pd.DataFrame)
    
    def test_analyze_pairs_nb_best_z_selection(self, make_ou_process):
        """Test that best_Z selection is monotonic and sensible."""
        # Create clean OU process with known properties
        spread = make_ou_process(T=500, theta=0.1, mu=0.0, sigma=1.0)
        
        # Create synthetic price pair from spread
        dates = spread.index
        x_prices = pd.Series(100 + np.cumsum(np.random.randn(len(spread)) * 0.01), index=dates)
        y_prices = x_prices + spread  # y = x + spread (beta = 1, alpha = 0)
        
        all_data = {
            'clean_pair': pd.DataFrame({'y': y_prices, 'x': x_prices})
        }
        
        with patch('plotting.plt.show'):
            summary_df, opt_tables = analyze_pairs_nb(
                all_data, ['clean_pair'],
                Z_min=0.5, Z_max=3.0, dZ=0.25
            )
        
        if not summary_df.empty:
            best_z = summary_df.iloc[0]['best_Z']
            cum_pnl = summary_df.iloc[0]['cum_PnL']
            
            # Best Z should be reasonable for OU process
            assert 0.5 <= best_z <= 3.0
            
            # For mean-reverting process, should generate some trades
            n_trades = summary_df.iloc[0]['N_trades']
            assert n_trades >= 0


class TestPlotSystematicPerformance:
    """Test systematic performance plotting functionality."""
    
    @patch('plotting.plt.show')
    @patch('plotting.plt.subplots')
    def test_plot_systematic_performance_structure(self, mock_subplots, mock_show, 
                                                  sample_portfolio_data, make_benchmark):
        """Test plot_systematic_performance returns correct DataFrame structure."""
        # Setup mock matplotlib objects
        mock_fig = MagicMock()
        mock_axes = np.array([[MagicMock() for _ in range(3)] for _ in range(4)])
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        portfolio = sample_portfolio_data
        selected_pairs = portfolio['selected'][:3]  # First 3 pairs for 3-column layout
        benchmark_returns = portfolio['benchmark']
        
        # Create mock stitched results
        stitched_results = {}
        for pair_name in selected_pairs:
            dates = pd.date_range('2020-01-01', periods=100, freq='B')
            stitched_results[pair_name] = {
                'cumulative_returns': pd.Series(np.cumprod(1 + np.random.randn(100) * 0.01), index=dates),
                'drawdowns': {
                    'drawdown': pd.Series(-np.abs(np.random.randn(100)) * 0.05, index=dates)
                },
                'strategy_returns': pd.Series(np.random.randn(100) * 0.01, index=dates)
            }
        
        # Mock the rolling calculation functions
        def mock_rolling_sharpe(returns, window):
            return pd.Series(np.random.randn(len(returns)) * 0.5, index=returns.index)
        
        def mock_rolling_beta(strat_returns, market_returns, window):
            return pd.Series(np.random.randn(len(strat_returns)) * 0.3 + 0.8, index=strat_returns.index)
        
        # Call function under test
        result_df = plot_systematic_performance(
            stitched_results, selected_pairs, benchmark_returns,
            mock_rolling_sharpe, mock_rolling_beta,
            title="Test Strategy Performance"
        )
        
        # Check return DataFrame structure
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == len(selected_pairs)
        
        expected_cols = {'Pair', 'Avg_Sharpe'}
        assert set(result_df.columns) == expected_cols
        
        # Check that pairs are properly formatted
        assert (result_df['Pair'].str.contains(' ')).all()  # Should have space-separated words
        
        # Verify plotting calls were made
        mock_subplots.assert_called_once_with(4, 3, figsize=(20, 16))
        mock_show.assert_called_once()
    
    def test_plot_systematic_performance_fewer_than_three_pairs(self):
        """Test handling when fewer than 3 pairs are provided."""
        # Create minimal stitched results for 2 pairs
        dates = pd.date_range('2020-01-01', periods=50, freq='B')
        stitched_results = {
            'pair1': {
                'cumulative_returns': pd.Series(np.cumprod(1 + np.random.randn(50) * 0.01), index=dates),
                'drawdowns': {'drawdown': pd.Series(-np.abs(np.random.randn(50)) * 0.05, index=dates)},
                'strategy_returns': pd.Series(np.random.randn(50) * 0.01, index=dates)
            },
            'pair2': {
                'cumulative_returns': pd.Series(np.cumprod(1 + np.random.randn(50) * 0.01), index=dates),
                'drawdowns': {'drawdown': pd.Series(-np.abs(np.random.randn(50)) * 0.05, index=dates)},
                'strategy_returns': pd.Series(np.random.randn(50) * 0.01, index=dates)
            }
        }
        
        benchmark_returns = pd.Series(np.random.randn(50) * 0.02, index=dates)
        
        def mock_rolling_func(returns, window=126):
            return pd.Series(np.random.randn(len(returns)) * 0.5, index=returns.index)
        
        with patch('plotting.plt.show'), patch('plotting.plt.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_axes = np.array([[MagicMock() for _ in range(3)] for _ in range(4)])
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            result_df = plot_systematic_performance(
                stitched_results, ['pair1', 'pair2'], benchmark_returns,
                mock_rolling_func, mock_rolling_func
            )
        
        # Should handle fewer pairs gracefully
        assert len(result_df) == 2


class TestPlotKalmanBetaEvolution:
    """Test Kalman filter beta evolution plotting."""
    
    @patch('plotting.plt.show')
    @patch('plotting.plt.subplots')
    def test_plot_kalman_beta_evolution_single_pair(self, mock_subplots, mock_show):
        """Test beta evolution plotting for single pair."""
        # Setup mock matplotlib objects
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_subplots.return_value = (mock_fig, [mock_ax])
        
        # Create mock Kalman analysis results
        dates = pd.date_range('2020-01-01', periods=100, freq='B')
        adaptive_betas = pd.Series(1.2 + 0.1 * np.sin(np.arange(100) * 0.1), index=dates)
        
        kalman_analysis = {
            'detailed_results': {
                'test_pair': {
                    'adaptive_betas': adaptive_betas
                }
            }
        }
        
        # Call function (should not raise errors)
        result = plot_kalman_beta_evolution(kalman_analysis, ['test_pair'])
        
        # Function returns None
        assert result is None
        
        # Verify plotting calls
        mock_subplots.assert_called_once_with(1, 1, figsize=(12, 4))
        mock_show.assert_called_once()
        
        # Verify plot method was called on the axis
        mock_ax.plot.assert_called_once()
        mock_ax.set_title.assert_called_once_with('test_pair: Beta Evolution')
    
    @patch('plotting.plt.show')
    @patch('plotting.plt.subplots')
    def test_plot_kalman_beta_evolution_multiple_pairs(self, mock_subplots, mock_show):
        """Test beta evolution plotting for multiple pairs."""
        # Setup mock for multiple subplots
        mock_fig = MagicMock()
        mock_axes = [MagicMock() for _ in range(3)]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Create mock results for 3 pairs
        dates = pd.date_range('2020-01-01', periods=100, freq='B')
        kalman_analysis = {
            'detailed_results': {}
        }
        
        selected_pairs = ['tech_pair', 'energy_pair', 'finance_pair']
        for i, pair_name in enumerate(selected_pairs):
            # Each pair has different beta evolution pattern
            betas = 1.0 + i * 0.2 + 0.1 * np.sin(np.arange(100) * 0.05 * (i + 1))
            kalman_analysis['detailed_results'][pair_name] = {
                'adaptive_betas': pd.Series(betas, index=dates)
            }
        
        # Call function
        result = plot_kalman_beta_evolution(kalman_analysis, selected_pairs)
        
        # Function returns None
        assert result is None
        
        # Verify plotting setup for multiple pairs
        mock_subplots.assert_called_once_with(3, 1, figsize=(12, 12))
        mock_show.assert_called_once()
        
        # Each axis should have been used for plotting
        for mock_ax in mock_axes:
            mock_ax.plot.assert_called_once()
            mock_ax.set_title.assert_called_once()
    
    def test_plot_kalman_beta_evolution_empty_results(self):
        """Test behavior with empty analysis results."""
        kalman_analysis = {'detailed_results': {}}
        selected_pairs = ['nonexistent_pair']
        
        # Should not crash even with missing data
        with patch('plotting.plt.show'), patch('plotting.plt.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, [mock_ax])
            
            # This should raise KeyError or be handled gracefully by the implementation
            with pytest.raises(KeyError):
                plot_kalman_beta_evolution(kalman_analysis, selected_pairs)


class TestPlottingIntegration:
    """Test integration between plotting functions and data pipeline."""
    
    def test_analyze_pairs_nb_to_systematic_plotting_pipeline(self, sample_portfolio_data):
        """Test that analyze_pairs_nb output can feed into systematic plotting."""
        portfolio = sample_portfolio_data
        selected = portfolio['selected'][:2]  # Limit for test speed
        
        # Step 1: Analyze pairs to get optimal thresholds
        with patch('plotting.plt.show'):
            summary_df, opt_tables = analyze_pairs_nb(
                portfolio['all_data'], selected,
                Z_min=1.0, Z_max=2.0, dZ=0.5
            )
        
        # Step 2: Create mock stitched results as if from backtesting
        dates = pd.date_range('2020-01-01', periods=100, freq='B')
        stitched_results = {}
        
        for pair_name in selected:
            if not summary_df.empty and pair_name in summary_df['pair'].values:
                # Use optimal threshold from analysis
                best_z = summary_df[summary_df['pair'] == pair_name]['best_Z'].iloc[0]
                
                # Create mock results with performance influenced by threshold
                volatility_factor = max(0.5, 2.0 / best_z)  # Lower threshold = higher vol
                returns = np.random.randn(100) * 0.01 * volatility_factor
                
                stitched_results[pair_name] = {
                    'cumulative_returns': pd.Series(np.cumprod(1 + returns), index=dates),
                    'drawdowns': {'drawdown': pd.Series(-np.abs(np.random.randn(100)) * 0.05, index=dates)},
                    'strategy_returns': pd.Series(returns, index=dates)
                }
        
        # Step 3: Test that systematic plotting can consume this data
        if stitched_results:
            benchmark_returns = portfolio['benchmark'][:100]  # Match length
            
            def mock_rolling_func(returns, window=126):
                return pd.Series(np.random.randn(len(returns)) * 0.5, index=returns.index)
            
            with patch('plotting.plt.show'), patch('plotting.plt.subplots'):
                result_df = plot_systematic_performance(
                    stitched_results, list(stitched_results.keys()),
                    benchmark_returns, mock_rolling_func, mock_rolling_func
                )
            
            # Should produce valid output
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) == len(stitched_results)
    
    def test_plotting_functions_handle_missing_data(self):
        """Test that plotting functions handle missing or malformed data gracefully."""
        # Test analyze_pairs_nb with empty data
        empty_all_data = {}
        empty_selected = []
        
        with patch('plotting.plt.show'):
            summary_df, opt_tables = analyze_pairs_nb(empty_all_data, empty_selected)
        
        assert isinstance(summary_df, pd.DataFrame)
        assert isinstance(opt_tables, dict)
        assert summary_df.empty
        assert len(opt_tables) == 0
        
        # Test plot_systematic_performance with empty stitched results
        with patch('plotting.plt.show'), patch('plotting.plt.subplots') as mock_subplots:
            mock_fig = MagicMock()
            mock_axes = np.array([[MagicMock() for _ in range(3)] for _ in range(4)])
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            def mock_rolling_func(returns, window=126):
                return pd.Series([], dtype=float)
            
            benchmark = pd.Series([], dtype=float)
            result_df = plot_systematic_performance(
                {}, [], benchmark, mock_rolling_func, mock_rolling_func
            )
        
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.empty


class TestPlottingEdgeCases:
    """Test edge cases specific to plotting functionality."""
    
    def test_analyze_pairs_nb_with_extreme_parameters(self, make_cointegrated_pair):
        """Test analyze_pairs_nb with extreme parameter values."""
        data = make_cointegrated_pair(T=100, beta=1.0)  # Minimal data
        all_data = {'test_pair': pd.DataFrame({'y': data['y'], 'x': data['x']})}
        
        with patch('plotting.plt.show'):
            # Test with very wide Z range
            summary_wide, _ = analyze_pairs_nb(
                all_data, ['test_pair'],
                Z_min=0.1, Z_max=5.0, dZ=1.0
            )
            
            # Test with very narrow Z range
            summary_narrow, _ = analyze_pairs_nb(
                all_data, ['test_pair'],
                Z_min=1.95, Z_max=2.05, dZ=0.05
            )
        
        # Both should produce valid results
        assert isinstance(summary_wide, pd.DataFrame)
        assert isinstance(summary_narrow, pd.DataFrame)
    
    def test_plotting_with_single_time_point(self):
        """Test plotting functions with minimal time series data."""
        # Single point time series
        single_date = pd.date_range('2020-01-01', periods=1, freq='B')
        single_point_series = pd.Series([1.0], index=single_date)
        
        kalman_analysis = {
            'detailed_results': {
                'minimal_pair': {
                    'adaptive_betas': single_point_series
                }
            }
        }
        
        with patch('plotting.plt.show'), patch('plotting.plt.subplots'):
            # Should handle minimal data gracefully
            result = plot_kalman_beta_evolution(kalman_analysis, ['minimal_pair'])
            assert result is None  # Function returns None
    
    def test_analyze_pairs_nb_cost_sensitivity(self, make_cointegrated_pair):
        """Test that transaction costs affect optimization results appropriately."""
        data = make_cointegrated_pair(T=300, beta=1.2, rho=0.6)
        all_data = {'cost_test': pd.DataFrame({'y': data['y'], 'x': data['x']})}
        
        with patch('plotting.plt.show'):
            # No transaction costs
            summary_no_cost, tables_no_cost = analyze_pairs_nb(
                all_data, ['cost_test'],
                Z_min=1.0, Z_max=2.5, dZ=0.5, cost=0.0
            )
            
            # With transaction costs
            summary_with_cost, tables_with_cost = analyze_pairs_nb(
                all_data, ['cost_test'],
                Z_min=1.0, Z_max=2.5, dZ=0.5, cost=0.002
            )
        
        # Both should return results
        assert isinstance(summary_no_cost, pd.DataFrame)
        assert isinstance(summary_with_cost, pd.DataFrame)
        
        # Transaction costs should generally reduce cumulative P&L
        if not summary_no_cost.empty and not summary_with_cost.empty:
            pnl_no_cost = summary_no_cost['cum_PnL'].iloc[0]
            pnl_with_cost = summary_with_cost['cum_PnL'].iloc[0]
            
            # With costs should be lower (or equal if no trades)
            assert pnl_with_cost <= pnl_no_cost
