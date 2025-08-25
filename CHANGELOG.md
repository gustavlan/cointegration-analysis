# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup and refactoring for public release

## [0.1.0] - 2025-08-25

### Added
- Core cointegration testing framework with ADF, KPSS, Engle-Granger, and Zivot-Andrews tests
- Johansen cointegration analysis for multivariate series
- Error Correction Model (ECM) implementation with time-series analysis
- Comprehensive backtesting engine with walk-forward cross-validation
- Systematic backtesting with performance attribution and risk metrics
- Z-score threshold optimization through parameter sweeps
- Rich visualization suite for equity curves, rolling metrics, and statistical plots
- Command-line interface with `download`, `cv`, and `systematic` subcommands
- Production-ready project structure with CI/CD pipeline
- Comprehensive test suite with >80% coverage
- MIT license for open-source distribution
- Complete documentation with methodology overview and usage examples

### Technical Features
- Type hints and docstrings for all public functions
- Logging integration for better debugging and monitoring
- Pre-commit hooks for code quality (black, ruff, nbstripout)
- GitHub Actions CI/CD with automated testing
- Deterministic random seeds for reproducible results
- Memory-efficient vectorized operations using pandas/numpy
- Error handling with informative exception messages

### Documentation
- README with feature overview, installation, and usage examples
- Contributing guidelines with development workflow
- Research notebook demonstrating complete methodology
- Sample plots and performance visualizations
- API documentation for core modules

### Infrastructure
- pyproject.toml configuration for modern Python packaging
- Pre-commit configuration for code quality enforcement  
- GitHub Actions workflow for continuous integration
- Requirements.txt with pinned dependencies for reproducibility
- .gitignore configured for data science projects

### Data Management
- Sample datasets for testing and demonstration
- Data loading utilities with error handling
- Configurable data directory structure
- CSV-based data storage with DatetimeIndex parsing

[Unreleased]: https://github.com/gustavlan/pairs-cointegration-backtester/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gustavlan/pairs-cointegration-backtester/releases/tag/v0.1.0
