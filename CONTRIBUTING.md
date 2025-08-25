# Contributing to Pairs Trading Cointegration Backtester

We welcome contributions to this project! This guide will help you get started.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/pairs-cointegration-backtester.git
   cd pairs-cointegration-backtester
   ```
3. **Create a virtual environment**:
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```
4. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   pip install pre-commit pytest-cov black ruff
   ```
5. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## 🔧 Development Workflow

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards (see below)

3. **Run tests locally**:
   ```bash
   pytest -v --cov=.
   ```

4. **Run code formatting and linting**:
   ```bash
   black .
   ruff check . --fix
   ```

5. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** on GitHub

## 📏 Coding Standards

### Code Style
- **Line length**: 100 characters maximum
- **Formatter**: Black with line-length=100
- **Linter**: Ruff with E, F, I, B, UP rules
- **Import sorting**: isort with black profile

### Type Hints
- Add type hints to all public functions
- Use precise types: `pd.DataFrame`, `np.ndarray`, `Dict[str, Any]`
- Import types from `typing` when needed

### Documentation
- **Docstrings**: Google-style docstrings for all public functions
- **Comments**: Explain complex algorithms and business logic
- **README updates**: Update documentation for new features

### Example Function:
```python
def backtest_pair_strategy(
    data: pd.DataFrame,
    z_threshold: float = 2.0,
    transaction_costs: float = 0.002
) -> Dict[str, Any]:
    """Execute pairs trading strategy on cointegrated time series.
    
    Args:
        data: DataFrame with price series columns
        z_threshold: Entry/exit threshold for standardized residuals
        transaction_costs: Proportional costs per trade
        
    Returns:
        Dictionary containing returns, positions, and performance metrics
        
    Raises:
        ValueError: If data contains insufficient observations
    """
    if len(data) < 50:
        raise ValueError("Insufficient data for backtesting")
    
    # Implementation here...
    return {"returns": returns, "sharpe": sharpe_ratio}
```

## 🧪 Testing

### Test Structure
- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test end-to-end workflows
- **Data tests**: Validate sample data integrity

### Running Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific test file
pytest tests/test_backtests.py -v

# Specific test function
pytest tests/test_backtests.py::test_cv_over_pairs -v
```

### Writing Tests
```python
import pytest
import pandas as pd
from backtests import run_cv_over_pairs

def test_cv_over_pairs_basic():
    """Test CV runs successfully with minimal input."""
    # Arrange
    data = generate_sample_pair_data(n_periods=252)
    pairs = ["test_pair"]
    best_z = {"test_pair": 2.0}
    
    # Act
    results, artifacts = run_cv_over_pairs(
        all_data={"test_pair": data},
        pairs=pairs,
        best_z=best_z,
        n_splits=3
    )
    
    # Assert
    assert len(results) == 3  # 3 CV folds
    assert "mean_return" in results[0]
    assert artifacts is not None
```

## 📊 Data and Notebooks

### Sample Data
- **Keep datasets small**: <10MB per file
- **Document sources**: Add data provenance comments
- **Version control**: Track small sample datasets only

### Jupyter Notebooks
- **Strip outputs**: Use nbstripout pre-commit hook
- **Clear structure**: Use markdown headers for sections
- **Reproducible**: Set random seeds for deterministic results

## 🏷️ Commit Convention

Use conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:
- `feat: add Zivot-Andrews structural break test`
- `fix: handle edge case in ECM parameter estimation`
- `docs: update README with CLI usage examples`

## 🐛 Reporting Issues

When reporting bugs, please include:

1. **Environment**: Python version, OS, dependency versions
2. **Steps to reproduce**: Minimal example that triggers the issue
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happened
5. **Error messages**: Full traceback if applicable

## 💡 Suggesting Features

Feature suggestions should include:

1. **Use case**: Why is this feature needed?
2. **Proposed API**: How should users interact with it?
3. **Implementation ideas**: Any thoughts on approach
4. **Breaking changes**: Will this affect existing code?

## 🔬 Research Contributions

We especially welcome:

- **New cointegration tests**: Additional statistical methods
- **Alternative backtesting strategies**: Different entry/exit rules
- **Performance metrics**: New ways to evaluate strategies  
- **Data sources**: Integration with additional market data APIs
- **Visualization**: Enhanced plots and interactive charts

## ⚡ Performance Considerations

- **Vectorized operations**: Prefer pandas/numpy over loops
- **Memory efficiency**: Avoid creating unnecessary copies
- **Caching**: Cache expensive computations when appropriate
- **Profiling**: Use `cProfile` to identify bottlenecks

## 📋 Review Process

Pull requests are reviewed for:

1. **Code quality**: Style, documentation, test coverage
2. **Functionality**: Does it work as intended?
3. **Performance**: Is it reasonably efficient?
4. **Breaking changes**: Are they necessary and documented?
5. **Security**: No secrets or vulnerabilities

## 🆘 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and brainstorming
- **Code Review**: Tag maintainers for review feedback

Thank you for contributing! 🙏
