#!/usr/bin/env python3
"""
Data download and management utilities for pairs trading analysis.

This module provides functions for fetching financial data, ensuring data
availability, and managing the data directory structure.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def ensure_data_availability(data_dir: str) -> bool:
    """Ensure that sample data is available in the data directory.

    Checks for the existence of sample data files and creates placeholder
    files if they don't exist. This is a placeholder implementation for
    the full data download functionality.

    Args:
        data_dir: Directory where data files should be located

    Returns:
        True if data is available or was successfully created

    Raises:
        IOError: If unable to create data directory or files
    """
    try:
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        # Check for sample data files
        expected_files = [
            "oil_pair_data.csv",
            "currency_pair_data.csv",
            "agri_pair_data.csv",
            "sp500_benchmark_data.csv",
        ]

        missing_files = []
        for filename in expected_files:
            file_path = data_path / filename
            if not file_path.exists():
                missing_files.append(filename)

        if missing_files:
            logger.warning(f"Missing data files: {missing_files}")
            logger.info("Consider running actual data download functions or provide sample data")

            # Create minimal sample data for demonstration
            _create_sample_data(data_path, missing_files)

        logger.info(f"Data directory ready: {data_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to ensure data availability: {e}")
        raise OSError(f"Cannot prepare data directory: {e}")


def _create_sample_data(data_path: Path, missing_files: list[str]) -> None:
    """Create minimal sample data files for testing purposes.

    Args:
        data_path: Path to data directory
        missing_files: List of filenames to create
    """
    import numpy as np

    # Generate sample data with some realistic properties
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    np.random.seed(42)  # For reproducible sample data

    for filename in missing_files:
        try:
            if "oil_pair" in filename:
                # Simulate oil prices (WTI, Brent)
                wti = 60 + np.cumsum(np.random.randn(len(dates)) * 0.02)
                brent = wti * 1.05 + np.cumsum(np.random.randn(len(dates)) * 0.01)
                data = pd.DataFrame({"WTI": wti, "Brent": brent}, index=dates)

            elif "currency_pair" in filename:
                # Simulate currency rates (AUD/USD, CAD/USD)
                aud = 0.7 + np.cumsum(np.random.randn(len(dates)) * 0.001)
                cad = 0.75 + np.cumsum(np.random.randn(len(dates)) * 0.001)
                data = pd.DataFrame({"AUD_USD": aud, "CAD_USD": cad}, index=dates)

            elif "agri_pair" in filename:
                # Simulate agricultural commodities (Corn, Soybeans)
                corn = 400 + np.cumsum(np.random.randn(len(dates)) * 2)
                soybeans = 1000 + np.cumsum(np.random.randn(len(dates)) * 5)
                data = pd.DataFrame({"Corn": corn, "Soybeans": soybeans}, index=dates)

            elif "benchmark" in filename:
                # Simulate S&P 500 index
                sp500 = 3000 * (1 + np.cumsum(np.random.randn(len(dates)) * 0.001))
                data = pd.DataFrame({"SP500": sp500}, index=dates)

            else:
                # Generic sample data
                asset1 = 100 + np.cumsum(np.random.randn(len(dates)) * 0.01)
                asset2 = 105 + np.cumsum(np.random.randn(len(dates)) * 0.01)
                data = pd.DataFrame({"Asset1": asset1, "Asset2": asset2}, index=dates)

            # Save to CSV
            filepath = data_path / filename
            data.to_csv(filepath)
            logger.info(f"Created sample data file: {filepath}")

        except Exception as e:
            logger.error(f"Failed to create sample data for {filename}: {e}")


def fetch_asset_data(asset_groups: dict, start_date: datetime, end_date: datetime, freq: str = "B"):
    """Download and process price data for multiple asset groups from Yahoo Finance.

    Downloads historical price data for specified asset groups using yfinance
    and processes it into a standardized format with proper frequency alignment
    and missing data handling.

    Args:
        asset_groups (dict): Dictionary mapping group names to lists of ticker
                           symbols. E.g., {'tech_stocks': ['AAPL', 'MSFT']}
        start_date (datetime): Start date for data download.
        end_date (datetime): End date for data download.
        freq (str, optional): Target frequency for data resampling. Defaults
                             to "B" (business days).

    Returns:
        dict: Dictionary mapping group names to DataFrame objects containing
              adjusted close prices. Each DataFrame has:
              - Index: DatetimeIndex with specified frequency
              - Columns: Ticker symbols
              - Values: Adjusted close prices

    Raises:
        Exception: Individual group download failures are caught and the group
                  is skipped (logged silently).

    Example:
        >>> groups = {'pairs': ['SPY', 'QQQ'], 'commodities': ['GLD', 'SLV']}
        >>> start = datetime(2020, 1, 1)
        >>> end = datetime(2023, 12, 31)
        >>> data = fetch_asset_data(groups, start, end)
        >>> print(data['pairs'].head())
    """
    all_data = {}

    for group_name, tickers in asset_groups.items():
        try:
            raw = yf.download(
                tickers,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
            )

            if isinstance(raw.columns, pd.MultiIndex):  # multiple tickers
                stacked = (
                    raw.stack(level=0, future_stack=True)
                    .rename_axis(["Date", "Ticker"])
                    .reset_index(level="Ticker")
                )
                price = stacked.pivot(columns="Ticker", values="Close")
            else:  # single ticker
                price = raw[["Close"]].copy()
                if len(tickers) == 1:
                    price.columns = tickers

            price = (
                price.ffill().dropna().asfreq(freq, method="ffill")
            )  # forward fill and set frequency
            if not price.empty:
                all_data[group_name] = price
        except Exception:
            continue

    return all_data
