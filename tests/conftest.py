"""
Test fixtures for cointegration analysis test suite.

Provides synthetic data generators for testing statistical methods,
backtesting algorithms, and plotting functions with known properties.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats


@pytest.fixture(scope="session")
def random_seed():
    """Set global random seed for reproducible tests."""
    np.random.seed(123)
    return 123


@pytest.fixture
def make_cointegrated_pair():
    """Factory fixture to create cointegrated price pairs with known parameters."""

    def _make_pair(T=600, beta=1.3, rho=0.95, sigma=0.01, break_at=None, random_state=123):
        """
        Generate cointegrated price pair using vector error correction model.

        Args:
            T: Number of time periods
            beta: True cointegrating coefficient (hedge ratio)
            rho: Error correction speed (0 < rho < 1)
            sigma: Innovation standard deviation
            break_at: Optional structural break point (None for no break)
            random_state: Random seed for reproducibility

        Returns:
            dict: Contains 'y', 'x' price series and 'true_beta', 'true_spread'
        """
        np.random.seed(random_state)

        # Generate prices using VECM with known cointegrating relationship
        e = np.zeros(T)  # error correction term
        y = np.zeros(T)
        x = np.zeros(T)

        # Initial values
        y[0] = 100.0
        x[0] = 100.0 / beta  # Ensure initial cointegration

        for t in range(1, T):
            # Structural break in beta if specified
            current_beta = beta if break_at is None or t < break_at else beta * 1.2

            # Error correction term
            e[t - 1] = y[t - 1] - current_beta * x[t - 1]

            # Price innovations with error correction
            dy = -rho * e[t - 1] + sigma * np.random.normal()
            dx = sigma * np.random.normal()

            # Accumulate to price levels
            y[t] = y[t - 1] + dy
            x[t] = x[t - 1] + dx

        # Create time index
        dates = pd.date_range("2020-01-01", periods=T, freq="B")
        y_series = pd.Series(y, index=dates, name="asset1")
        x_series = pd.Series(x, index=dates, name="asset2")

        # True spread for validation
        true_spread = y_series - beta * x_series

        return {
            "y": y_series,
            "x": x_series,
            "true_beta": beta,
            "true_spread": true_spread,
            "true_rho": rho,
            "break_at": break_at,
        }

    return _make_pair


@pytest.fixture
def make_nonstationary_series():
    """Factory fixture for non-stationary time series (random walks)."""

    def _make_nonstationary(T=600, drift=0.0, sigma=1.0, random_state=123):
        """Generate random walk series with optional drift."""
        np.random.seed(random_state)
        innovations = np.random.normal(drift, sigma, T)
        series = np.cumsum(innovations)
        dates = pd.date_range("2020-01-01", periods=T, freq="B")
        return pd.Series(series, index=dates, name="random_walk")

    return _make_nonstationary


@pytest.fixture
def make_stationary_series():
    """Factory fixture for stationary AR(1) series."""

    def _make_stationary(T=600, phi=0.7, mu=0.0, sigma=1.0, random_state=123):
        """Generate AR(1) process: x_t = mu + phi*(x_{t-1} - mu) + eps_t"""
        np.random.seed(random_state)
        x = np.zeros(T)
        x[0] = mu

        for t in range(1, T):
            x[t] = mu + phi * (x[t - 1] - mu) + sigma * np.random.normal()

        dates = pd.date_range("2020-01-01", periods=T, freq="B")
        return pd.Series(x, index=dates, name="ar1_process")

    return _make_stationary


@pytest.fixture
def make_stable_VAR2():
    """Factory fixture for stable 2-variable VAR system."""

    def _make_var2(T=800, p=1, random_state=123):
        """
        Generate stable VAR(p) system with eigenvalues inside unit circle.

        Returns:
            pd.DataFrame: Two-column DataFrame with VAR series
        """
        np.random.seed(random_state)

        if p == 1:
            # VAR(1) with stable eigenvalues
            A = np.array([[0.5, 0.1], [0.2, 0.4]])  # Eigenvalues: ~0.6, 0.3
            Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])  # Error covariance
        else:
            # Higher order VAR - simplified to VAR(1) for stability
            A = np.array([[0.6, 0.1], [0.1, 0.6]])
            Sigma = np.eye(2)

        # Check stability
        eigenvals = np.linalg.eigvals(A)
        assert np.all(np.abs(eigenvals) < 1), f"Unstable VAR: eigenvalues {eigenvals}"

        # Generate VAR series
        Y = np.zeros((T, 2))
        for t in range(1, T):
            innovations = np.random.multivariate_normal([0, 0], Sigma)
            Y[t] = A @ Y[t - 1] + innovations

        dates = pd.date_range("2020-01-01", periods=T, freq="B")
        df = pd.DataFrame(Y, index=dates, columns=["var1", "var2"])

        return df

    return _make_var2


@pytest.fixture
def make_benchmark():
    """Factory fixture for market benchmark (e.g., S&P 500 style returns)."""

    def _make_benchmark(T=600, annual_return=0.08, annual_vol=0.16, random_state=123):
        """Generate realistic market benchmark returns."""
        np.random.seed(random_state)

        # Daily parameters
        daily_return = annual_return / 252
        daily_vol = annual_vol / np.sqrt(252)

        # Generate log returns with fat tails
        returns = np.random.normal(daily_return, daily_vol, T)

        # Add occasional large moves (fat tails)
        shock_prob = 0.05
        shocks = np.random.random(T) < shock_prob
        returns[shocks] *= 3  # Amplify shock days

        dates = pd.date_range("2020-01-01", periods=T, freq="B")
        return pd.Series(returns, index=dates, name="market_returns")

    return _make_benchmark


@pytest.fixture
def make_ou_process():
    """Factory fixture for Ornstein-Uhlenbeck mean-reverting process."""

    def _make_ou(T=600, theta=0.1, mu=0.0, sigma=0.2, random_state=123):
        """Generate OU process: dX_t = theta*(mu - X_t)*dt + sigma*dW_t"""
        np.random.seed(random_state)

        dt = 1.0  # Daily time step
        x = np.zeros(T)
        x[0] = mu

        for t in range(1, T):
            dx = theta * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
            x[t] = x[t - 1] + dx

        dates = pd.date_range("2020-01-01", periods=T, freq="B")
        return pd.Series(x, index=dates, name="ou_process")

    return _make_ou


@pytest.fixture
def sample_portfolio_data(make_cointegrated_pair, make_benchmark):
    """Create sample portfolio with multiple pairs and benchmark."""

    def _make_portfolio():
        # Generate 3 cointegrated pairs with different characteristics
        pairs = {}

        # High-frequency reverting pair
        pairs["tech_pair"] = make_cointegrated_pair(T=500, beta=1.2, rho=0.8, sigma=0.01)

        # Slow-reverting pair
        pairs["energy_pair"] = make_cointegrated_pair(T=500, beta=0.9, rho=0.3, sigma=0.02)

        # Pair with structural break
        pairs["finance_pair"] = make_cointegrated_pair(T=500, beta=1.5, rho=0.6, break_at=300)

        # Market benchmark
        benchmark = make_benchmark(T=500)

        # Organize as all_data dictionary expected by functions
        all_data = {}
        for pair_name, pair_data in pairs.items():
            all_data[pair_name] = pd.DataFrame({"asset1": pair_data["y"], "asset2": pair_data["x"]})

        return {
            "all_data": all_data,
            "pairs": pairs,
            "benchmark": benchmark,
            "selected": list(pairs.keys()),
        }

    return _make_portfolio()


@pytest.fixture
def numerical_tolerances():
    """Standard numerical tolerances for testing."""
    return {
        "rtol": 1e-6,  # Relative tolerance
        "atol": 1e-8,  # Absolute tolerance
        "price_rtol": 1e-4,  # More lenient for price/return comparisons
        "stat_rtol": 1e-3,  # Statistical test p-values
    }


# Helper context manager for suppressing output during tests
@pytest.fixture
def suppress_warnings():
    """Context manager to suppress warnings during tests."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
