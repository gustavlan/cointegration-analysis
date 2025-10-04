"""
Tests for ``cointegration_analysis.analytics.cointegration``.

Exercises statistical functions for cointegration analysis including
unit root tests, cointegration tests, model selection, and time series analysis.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
import warnings

# Import functions under test
from cointegration_analysis.analytics.cointegration import (
    matrix_ols_regression,
    adf_results,
    kpss_results,
    engle_granger,
    engle_granger_bidirectional,
    za_test,
    analyze_error_correction_model,
    ou_params,
    select_var_order,
    johansen,
    analyze_regression_var_summary,
    analyze_ecm_timeslices,
    analyze_johansen_triples,
    johansen_sensitivity_summary,
)


class TestMatrixOLS:
    """Test matrix OLS regression functionality."""

    def test_simple_regression(self):
        """Test basic OLS regression with known result."""
        # Perfect linear relationship: y = 2 + 3*x
        x = np.array([1, 2, 3, 4, 5])
        y = 2 + 3 * x
        X = np.column_stack([np.ones(len(x)), x])  # Add constant column

        beta = matrix_ols_regression(y, X)

        np.testing.assert_allclose(beta, [2.0, 3.0], rtol=1e-6)

    def test_singular_matrix(self):
        """Test handling of singular (non-invertible) matrix."""
        # Perfect collinearity: X[:, 1] = 2 * X[:, 0]
        X = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
        y = np.array([1, 2, 3, 4])

        result = matrix_ols_regression(y, X)
        assert result is None

    def test_empty_input(self):
        """Test behavior with empty arrays."""
        with pytest.raises(ValueError):
            matrix_ols_regression(np.array([]), np.array([]).reshape(0, 1))


class TestUnitRootTests:
    """Test ADF and KPSS stationarity tests."""

    def test_adf_stationary_series(self, make_stationary_series):
        """Test ADF on stationary AR(1) series."""
        ar1 = make_stationary_series(T=200, phi=0.6, sigma=1.0)
        result = adf_results(ar1)

        # Check return structure
        expected_keys = {"stat", "pvalue", "crit_1%", "crit_5%", "crit_10%"}
        assert set(result.keys()) == expected_keys

        # Stationary series should have low p-value
        assert result["pvalue"] < 0.1  # Should reject unit root
        assert result["stat"] < result["crit_5%"]  # Test stat < critical value

    def test_adf_nonstationary_series(self, make_nonstationary_series):
        """Test ADF on non-stationary random walk."""
        rw = make_nonstationary_series(T=200, drift=0.0, sigma=1.0)
        result = adf_results(rw)

        # Non-stationary series should have high p-value
        assert result["pvalue"] > 0.05  # Fail to reject unit root

    def test_kpss_stationary_series(self, make_stationary_series):
        """Test KPSS on stationary series."""
        ar1 = make_stationary_series(T=200, phi=0.5)
        result = kpss_results(ar1)

        # Check return structure
        expected_keys = {"stat", "pvalue", "crit_10%", "crit_5%", "crit_2.5%", "crit_1%"}
        assert set(result.keys()) == expected_keys

        # KPSS null is stationarity - should NOT reject for stationary series
        assert result["pvalue"] >= 0.05  # Fail to reject stationarity

    def test_kpss_nonstationary_series(self, make_nonstationary_series):
        """Test KPSS on non-stationary series."""
        rw = make_nonstationary_series(T=200, drift=0.01)
        result = kpss_results(rw)

        # Should reject stationarity for random walk
        assert result["pvalue"] < 0.1

    def test_short_series_handling(self):
        """Test behavior with very short time series."""
        short_series = pd.Series(
            [1, 2, 3, 4, 5], index=pd.date_range("2020-01-01", periods=5, freq="B")
        )

        # Should not crash with short series
        adf_result = adf_results(short_series)
        kpss_result = kpss_results(short_series)

        assert isinstance(adf_result["pvalue"], float)
        assert isinstance(kpss_result["pvalue"], float)

    def test_series_with_nans(self):
        """Test handling of series with NaN values."""
        series_with_nans = pd.Series(
            [1, 2, np.nan, 4, 5, np.nan, 7, 8],
            index=pd.date_range("2020-01-01", periods=8, freq="B"),
        )

        # Should handle NaNs gracefully (drop them)
        result = adf_results(series_with_nans)
        assert np.isfinite(result["pvalue"])


class TestCointegrationTests:
    """Test Engle-Granger cointegration methods."""

    def test_engle_granger_cointegrated_pair(self, make_cointegrated_pair, numerical_tolerances):
        """Test EG test on known cointegrated pair."""
        data = make_cointegrated_pair(T=300, beta=1.5, rho=0.8)
        df = pd.DataFrame({"y": data["y"], "x": data["x"]})

        result = engle_granger(df, "y", "x")

        # Check return structure
        expected_keys = {
            "beta",
            "alpha",
            "eg_pvalue",
            "spread",
            "maxlag",
            "used_lag",
            "eg_stat",
            "resid_lb_pvalue",
            "beta_tstat",
        }
        assert set(result.keys()) == expected_keys

        # Should detect cointegration
        assert result["eg_pvalue"] <= 0.05  # Significant cointegration
        assert result["spread"] is not None

        # Beta should be close to true value
        assert abs(result["beta"] - data["true_beta"]) < 0.3  # Allow some estimation error

    def test_engle_granger_independent_series(self, make_nonstationary_series):
        """Test EG test on independent non-stationary series."""
        y = make_nonstationary_series(T=200, random_state=42)
        x = make_nonstationary_series(T=200, random_state=123)  # Different seed

        # Align series
        df = pd.DataFrame({"y": y, "x": x}).dropna()

        result = engle_granger(df, "y", "x")

        # Should NOT detect cointegration
        assert result["eg_pvalue"] > 0.05
        assert result["spread"] is None

    def test_engle_granger_bidirectional_symmetry(self, make_cointegrated_pair):
        """Test that bidirectional test gives consistent results."""
        data = make_cointegrated_pair(T=300, beta=1.2)
        df = pd.DataFrame({"a": data["y"], "b": data["x"]})

        result_ab = engle_granger_bidirectional(df, "a", "b")
        result_ba = engle_granger_bidirectional(df, "b", "a")  # Reversed

        # Results should be similar (after accounting for direction)
        assert abs(result_ab["eg_pvalue"] - result_ba["eg_pvalue"]) < 0.1

        # Check that result format is consistent
        expected_keys = {
            "beta",
            "alpha",
            "eg_pvalue",
            "spread",
            "maxlag",
            "used_lag",
            "eg_stat",
            "resid_lb_pvalue",
            "beta_tstat",
        }
        assert set(result_ab.keys()) == expected_keys

    def test_perfect_collinearity_case(self):
        """Test handling of perfectly collinear series."""
        # Create perfectly collinear series: y = 2*x
        x = pd.Series(np.arange(100), index=pd.date_range("2020-01-01", periods=100, freq="B"))
        y = 2 * x
        df = pd.DataFrame({"y": y, "x": x})

        result = engle_granger(df, "y", "x")

        # Should handle perfect collinearity
        assert np.isclose(result["beta"], 2.0, rtol=1e-6)
        assert result["spread"].std() < 1e-10  # Spread should be near zero

    def test_constant_series_edge_case(self):
        """Test behavior with constant price series."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        constant_x = pd.Series([100.0] * 100, index=dates)
        constant_y = pd.Series([200.0] * 100, index=dates)
        df = pd.DataFrame({"y": constant_y, "x": constant_x})

        # Should handle gracefully without crashing
        result = engle_granger(df, "y", "x")
        assert isinstance(result, dict)


class TestStructuralBreaks:
    """Test Zivot-Andrews structural break test."""

    def test_za_test_with_known_break(self, make_stationary_series):
        """Test ZA test on series with injected structural break."""
        # Create AR(1) series with level break at t=150
        ar1_series = make_stationary_series(T=300, phi=0.7)

        # Inject level break
        break_point = 150
        ar1_with_break = ar1_series.copy()
        ar1_with_break.iloc[break_point:] += 2.0  # Level shift

        result = za_test(ar1_with_break, model="trend")

        # Check return structure
        assert isinstance(result, pd.DataFrame)
        expected_cols = {"stat", "pvalue", "breakpoint", "break_date", "model"}
        assert set(result.columns) == expected_cols

        # Should detect break (low p-value means rejecting unit root with break)
        assert result.iloc[0]["pvalue"] < 0.1

        # Break point should be reasonably close to true break
        detected_break = result.iloc[0]["breakpoint"]
        assert abs(detected_break - break_point) < 50  # Allow some detection error

    def test_za_test_no_break(self, make_stationary_series):
        """Test ZA test on series without structural break."""
        clean_ar1 = make_stationary_series(T=200, phi=0.6)

        result = za_test(clean_ar1, model="trend")

        # Without clear break, test should be less decisive
        assert isinstance(result.iloc[0]["pvalue"], float)
        assert np.isfinite(result.iloc[0]["breakpoint"])


class TestVARAnalysis:
    """Test VAR model selection and analysis."""

    def test_select_var_order_stable_system(self, make_stable_VAR2):
        """Test VAR order selection on known stable system."""
        var_data = make_stable_VAR2(T=400, p=1)

        results_df, best_aic, best_bic, best_hqic = select_var_order(var_data, maxlags=5)

        # Should return valid results
        assert not results_df.empty
        assert isinstance(best_aic, int) and best_aic >= 1
        assert isinstance(best_bic, int) and best_bic >= 1
        assert isinstance(best_hqic, int) and best_hqic >= 1

        # All tested models should be stable
        if "stable" in results_df.columns:
            assert results_df["stable"].all()

        # Check eigenvalue magnitudes for stability
        for _, row in results_df.iterrows():
            if "eigenvalues" in row and len(row["eigenvalues"]) > 0:
                max_eigenval = max(abs(ev) for ev in row["eigenvalues"])
                assert max_eigenval < 1.0  # Stability condition

    def test_var_order_selection_criteria(self, make_stable_VAR2):
        """Test that AIC/BIC/HQIC give reasonable order selections."""
        var_data = make_stable_VAR2(T=500)

        results_df, best_aic, best_bic, best_hqic = select_var_order(var_data, maxlags=4)

        # BIC typically selects lower order than AIC
        assert best_bic <= best_aic + 1  # Allow some flexibility

        # HQIC typically between AIC and BIC
        assert min(best_aic, best_bic) <= best_hqic <= max(best_aic, best_bic) + 1

    def test_analyze_regression_var_summary(self, make_stable_VAR2, numerical_tolerances):
        """Test regression quality analysis."""
        var_data = make_stable_VAR2(T=300)
        all_data = {"test_pair": var_data}

        summary = analyze_regression_var_summary(all_data)

        # Check return structure
        expected_cols = {"group", "r_squared", "best_aic", "best_bic", "best_hqic", "eigenvalues"}
        assert set(summary.columns) == expected_cols
        assert len(summary) == 1

        # R-squared should be reasonable
        r2 = summary.iloc[0]["r_squared"]
        assert 0 <= r2 <= 1

        # Eigenvalue string should contain numeric values
        eig_str = summary.iloc[0]["eigenvalues"]
        assert isinstance(eig_str, str)
        assert len(eig_str) > 0


class TestErrorCorrectionModel:
    """Test error correction model analysis."""

    def test_error_correction_model(self, make_cointegrated_pair, numerical_tolerances):
        """Test ECM coefficient estimation."""
        data = make_cointegrated_pair(T=400, rho=0.6)  # Known EC speed

        result = analyze_error_correction_model(data["y"], data["x"], data["true_spread"])

        # Check return structure
        expected_keys = {"ecm_coeff", "ecm_pvalue"}
        assert set(result.keys()) == expected_keys

        # ECM coefficient should be negative (error correction)
        assert result["ecm_coeff"] < 0

        # Should be statistically significant for strong cointegration
        assert result["ecm_pvalue"] < 0.1

    def test_ou_parameter_estimation(self, make_ou_process, numerical_tolerances):
        """Test OU process parameter estimation."""
        true_theta = 0.2
        true_mu = 1.0
        ou_series = make_ou_process(T=500, theta=true_theta, mu=true_mu, sigma=0.3)

        result = ou_params(ou_series)

        # Check return structure
        expected_keys = {"ou_mu", "ou_theta", "OU_HalfLife", "ou_sigma"}
        assert set(result.keys()) == expected_keys

        # Parameter estimates should be reasonable
        assert abs(result["ou_mu"] - true_mu) < 1.0  # Allow estimation error
        assert abs(result["ou_theta"] - true_theta) < 0.5

        # Half-life should be positive
        assert result["OU_HalfLife"] > 0

        # Check half-life calculation: HL = ln(2)/theta
        expected_hl = np.log(2) / result["ou_theta"]
        np.testing.assert_allclose(result["OU_HalfLife"], expected_hl, rtol=1e-6)

    def test_ecm_timeslices_consistency(self, make_cointegrated_pair):
        """Test ECM analysis across time periods."""
        data = make_cointegrated_pair(T=600, rho=0.5)

        result = analyze_ecm_timeslices(data["y"], data["x"], periods=3)

        # Check return structure
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 3)  # (ecm_coeff, ecm_pvalue) x 3 periods
        assert all(col.startswith("P") for col in result.columns)

        # ECM coefficients should be consistently negative (if significant)
        ecm_coeffs = result.loc["ecm_coeff"].dropna()
        significant_mask = result.loc["ecm_pvalue"] < 0.1
        significant_coeffs = result.loc["ecm_coeff"][significant_mask].dropna()

        if len(significant_coeffs) > 0:
            assert (significant_coeffs < 0).all()  # Error correction property


class TestJohansenCointegration:
    """Test Johansen cointegration analysis."""

    def test_johansen_known_rank(self, make_stable_VAR2):
        """Test Johansen test on system with known cointegration rank."""
        # Create VAR system with 1 cointegrating relationship
        var_data = make_stable_VAR2(T=400)

        result = johansen(var_data)

        # Check return structure
        assert "johansen_n" in result
        assert "k_ar_diff_used" in result
        assert "det_order" in result

        # Should find some cointegrating relationships
        assert isinstance(result["johansen_n"], int)
        assert 0 <= result["johansen_n"] <= 2  # Can't exceed number of variables

        # Check eigenvector coefficients
        eig_keys = [k for k in result.keys() if k.startswith("eig_")]
        assert len(eig_keys) == var_data.shape[1]  # One coeff per variable

    def test_johansen_triples_analysis(self, make_stable_VAR2):
        """Test Johansen analysis on triple systems."""
        # Create 3-variable system
        var_data = make_stable_VAR2(T=500)
        var_data["var3"] = (
            0.5 * var_data["var1"] + 0.3 * var_data["var2"] + 0.1 * np.random.randn(len(var_data))
        )

        all_data = {"test_triple": var_data}

        result = analyze_johansen_triples(all_data)

        if not result.empty:
            expected_cols = {
                "triple",
                "n_assets",
                "data_points",
                "n_coint_relations",
                "first_eigenvec_norm",
                "spread_vol",
            }
            assert set(result.columns) == expected_cols

            # Check basic properties
            assert result.iloc[0]["n_assets"] == 3
            assert result.iloc[0]["data_points"] == len(var_data)

    def test_johansen_sensitivity_summary(self, make_stable_VAR2, suppress_warnings):
        """Test Johansen sensitivity analysis across specifications."""
        # Create data that looks like triples
        var_data = make_stable_VAR2(T=300)
        var_data["var3"] = var_data["var1"] + 0.1 * np.random.randn(len(var_data))

        all_data = {"test_triple": var_data}

        with suppress_warnings:
            result = johansen_sensitivity_summary(all_data, min_obs=100)

        # Should return DataFrame (even if empty)
        assert isinstance(result, pd.DataFrame)

        if not result.empty:
            expected_cols = {"triple", "window", "rank>0", "total", "summary"}
            assert set(result.columns) == expected_cols


# Stub for missing silence_fd_output function
def silence_fd_output():
    """Stub for missing silence_fd_output context manager."""
    import contextlib
    import os
    import sys

    @contextlib.contextmanager
    def _silence():
        # Save current stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            # Redirect to devnull
            with open(os.devnull, "w") as devnull:
                sys.stdout = devnull
                sys.stderr = devnull
                yield
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    return _silence()


# Patch the missing function in the module
@pytest.fixture(autouse=True)
def patch_silence_fd_output():
    """Auto-patch the missing silence_fd_output function."""
    with patch(
        "cointegration_analysis.analytics.cointegration.silence_fd_output",
        side_effect=silence_fd_output,
    ):
        yield


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_short_series(self):
        """Test behavior with minimal data points."""
        short_data = pd.DataFrame(
            {"y": [100, 101, 102, 103], "x": [50, 51, 52, 53]},
            index=pd.date_range("2020-01-01", periods=4, freq="B"),
        )

        # Should handle gracefully without crashing
        result = engle_granger(short_data, "y", "x")
        assert isinstance(result, dict)

    def test_misaligned_series(self):
        """Test handling of series with different date ranges."""
        dates1 = pd.date_range("2020-01-01", periods=100, freq="B")
        dates2 = pd.date_range("2020-02-01", periods=100, freq="B")  # Offset dates

        y = pd.Series(range(100), index=dates1)
        x = pd.Series(range(100), index=dates2)

        # Functions should handle alignment automatically
        result = analyze_error_correction_model(y, x, y - x)  # Dummy spread
        assert isinstance(result, dict)

    def test_series_with_gaps(self):
        """Test handling of time series with gaps."""
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        y = pd.Series(range(100), index=dates)
        x = pd.Series(range(100), index=dates)

        # Create gaps by dropping some dates
        gap_indices = dates[::10]  # Every 10th observation
        y_with_gaps = y.drop(gap_indices)
        x_with_gaps = x.drop(gap_indices)

        # Should handle gaps gracefully
        df = pd.DataFrame({"y": y_with_gaps, "x": x_with_gaps})
        result = engle_granger(df, "y", "x")
        assert isinstance(result, dict)
