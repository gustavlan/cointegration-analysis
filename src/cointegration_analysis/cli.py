#!/usr/bin/env python3
"""
Main CLI entrypoint for pairs trading cointegration backtester.

This module provides command-line access to the cointegration analysis toolkit,
supporting data download, cross-validation analysis, and systematic backtesting.
"""

import argparse
import logging
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cointegration_analysis.analytics.backtesting import (
    run_cv_over_pairs,
    run_systematic_backtest,
    summarize_cv,
)
from cointegration_analysis.analytics.optimization import get_best_z_threshold
from cointegration_analysis.analytics.plotting import plot_systematic_performance
from cointegration_analysis.data.download import ensure_data_availability

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SEED = 42


def _configure_reproducibility(seed: int = DEFAULT_SEED) -> None:
    """Seed Python and NumPy RNGs for deterministic runs."""

    random.seed(seed)
    np.random.seed(seed)


def _resolve_commit_hash() -> str:
    """Return the current git commit hash, falling back to CI-provided SHAs."""

    env_candidates = ("GIT_COMMIT", "CI_COMMIT_SHA", "GITHUB_SHA")
    for var in env_candidates:
        commit = os.getenv(var)
        if commit:
            return commit

    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        return commit
    except Exception:
        return "unknown"


def _print_reproducibility_metadata(seed: int = DEFAULT_SEED) -> None:
    """Emit a standardized reproducibility hash line."""

    commit = _resolve_commit_hash()
    print(f"Reproducibility hash: seed={seed} | commit={commit}")


def load_pair_data(pairs: list[str], data_dir: str = "data") -> dict[str, pd.DataFrame]:
    """Load pair data from CSV files using naming convention.

    Args:
        pairs: List of pair identifiers (e.g., ['oil_pair', 'currency_pair'])
        data_dir: Directory containing the data files

    Returns:
        Dictionary mapping pair names to DataFrames with DatetimeIndex

    Raises:
        FileNotFoundError: If data files are missing
        ValueError: If data cannot be parsed
    """
    all_data = {}
    data_path = Path(data_dir)

    for pair in pairs:
        csv_file = data_path / f"{pair}_data.csv"
        if not csv_file.exists():
            logger.error(f"Data file not found: {csv_file}")
            raise FileNotFoundError(f"Missing data file for {pair}: {csv_file}")

        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            df = df.sort_index()
            if df.index.duplicated().any():
                logger.warning("Dropping duplicated timestamps in %s", pair)
                df = df[~df.index.duplicated(keep="last")]
            all_data[pair] = df
            logger.info(f"Loaded {pair} data: {len(df)} rows, {df.shape[1]} columns")
        except Exception as e:
            logger.error(f"Failed to load {pair} data from {csv_file}: {e}")
            raise ValueError(f"Cannot parse data file for {pair}: {e}") from e

    return all_data


def cmd_download(args) -> int:
    """Handle data download command.

    Args:
        args: Parsed command-line arguments

        _configure_reproducibility()
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        data_dir = Path(args.out)
        data_dir.mkdir(parents=True, exist_ok=True)

        # Placeholder for actual data download logic
        # In a real implementation, this would call functions from data_download.py
        logger.info(f"Data preparation completed for directory: {data_dir}")
        _print_reproducibility_metadata()

        # Check if we have sample data or need to fetch
        ensure_data_availability(str(data_dir))

        logger.info(f"Data preparation completed for directory: {data_dir}")
        return 0

    except Exception as e:
        logger.error(f"Data download failed: {e}")
        return 1


def cmd_cv(args) -> int:
    """Handle cross-validation command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        _configure_reproducibility()

        # Load data for all pairs
        all_data = load_pair_data(args.pairs)

        # Get optimal Z thresholds (placeholder implementation)
        best_z = {}
        for pair in args.pairs:
            try:
                z_value = get_best_z_threshold(pair, all_data[pair])
                best_z[pair] = z_value
                logger.info(f"Using Z-threshold {z_value:.2f} for {pair}")
            except Exception as e:
                logger.warning(f"Could not compute optimal Z for {pair}, using default 2.0: {e}")
                best_z[pair] = 2.0

        # Run cross-validation
        logger.info(f"Running {args.splits}-fold CV on {len(args.pairs)} pairs")
        cv_results, cv_artifacts = run_cv_over_pairs(
            all_data=all_data,
            selected=args.pairs,
            z_threshold_by_pair=best_z,
            n_splits=args.splits,
            transaction_costs=args.cost,
            return_artifacts=True,
        )

        # Summarize and display results
        summary = summarize_cv(cv_results, all_data, args.pairs)
        print("\\n" + "=" * 80)
        print("CROSS-VALIDATION RESULTS")
        print("=" * 80)
        print(summary.to_string(index=False))

        logger.info("Cross-validation completed successfully")
        _print_reproducibility_metadata()
        return 0

    except Exception as e:
        logger.error(f"Cross-validation failed: {e}")
        return 1


def cmd_systematic(args) -> int:
    """Handle systematic backtesting command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        _configure_reproducibility()

        # Load pair data
        all_data = load_pair_data(args.pairs)

        # Load benchmark data
        benchmark_path = Path(args.benchmark)
        if not benchmark_path.exists():
            logger.error(f"Benchmark file not found: {benchmark_path}")
            return 1

        benchmark = pd.read_csv(benchmark_path, index_col=0, parse_dates=True)
        benchmark_returns = benchmark.iloc[:, 0].pct_change().dropna()
        logger.info(f"Loaded benchmark data: {len(benchmark_returns)} returns")

        # Get Z thresholds
        best_z = {}
        for pair in args.pairs:
            try:
                z_value = get_best_z_threshold(pair, all_data[pair])
                best_z[pair] = z_value
                logger.info(f"Using Z-threshold {z_value} for {pair} (placeholder implementation)")
            except Exception:
                best_z[pair] = 2.0
                logger.info(f"Using default Z-threshold 2.0 for {pair}")

        # First run cross-validation to get artifacts
        logger.info("Running cross-validation for systematic backtest...")
        cv_results, cv_artifacts = run_cv_over_pairs(
            all_data=all_data,
            selected=args.pairs,
            z_threshold_by_pair=best_z,
            n_splits=5,
            transaction_costs=0.002,
            return_artifacts=True,
        )

        # Generate summary for systematic backtest
        summary_df = summarize_cv(cv_results, all_data, args.pairs)

        # Run systematic backtest (stitch CV folds)
        logger.info("Running systematic backtest...")
        systematic_results, systematic_df = run_systematic_backtest(
            cv_artifacts=cv_artifacts, selected_pairs=args.pairs, summary_df=summary_df
        )

        # Generate plots
        docs_dir = Path("docs/images")
        docs_dir.mkdir(parents=True, exist_ok=True)

        plot_paths = plot_systematic_performance(
            systematic_results=systematic_results,
            benchmark_returns=benchmark_returns,
            output_dir=str(docs_dir),
        )

        logger.info("Generated plots:")
        for plot_path in plot_paths:
            logger.info(f"  - {plot_path}")

        print("\\n" + "=" * 80)
        print("SYSTEMATIC BACKTEST COMPLETED")
        print("=" * 80)
        print(f"Plots saved to: {docs_dir}")
        _print_reproducibility_metadata()

        return 0

    except Exception as e:
        logger.error(f"Systematic backtest failed: {e}")
        return 1


def main() -> int:
    """Main CLI function.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="Pairs Trading Cointegration Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s download --out data
  %(prog)s cv --pairs oil_pair currency_pair --cost 0.002 --splits 5
  %(prog)s systematic --pairs oil_pair currency_pair --benchmark data/sp500_data.csv
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download or prepare data")
    download_parser.add_argument(
        "--out", default="data", help="Output data directory (default: data)"
    )

    # CV command
    cv_parser = subparsers.add_parser("cv", help="Run cross-validation analysis")
    cv_parser.add_argument(
        "--pairs", nargs="+", required=True, help="Pair identifiers (e.g., oil_pair currency_pair)"
    )
    cv_parser.add_argument(
        "--cost", type=float, default=0.002, help="Transaction cost per trade (default: 0.002)"
    )
    cv_parser.add_argument("--splits", type=int, default=5, help="Number of CV splits (default: 5)")

    # Systematic command
    systematic_parser = subparsers.add_parser("systematic", help="Run systematic backtest")
    systematic_parser.add_argument("--pairs", nargs="+", required=True, help="Pair identifiers")
    systematic_parser.add_argument(
        "--benchmark",
        default="data/sp500_benchmark_data.csv",
        help="Benchmark CSV file path (default: data/sp500_benchmark_data.csv)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Route to appropriate command handler
    command_handlers = {"download": cmd_download, "cv": cmd_cv, "systematic": cmd_systematic}

    return command_handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
