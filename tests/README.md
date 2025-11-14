# Testing Guide for Kalshi AI Trading Bot

This document describes the testing infrastructure and how to run tests.

## Test Structure

```
tests/
├── unit/                      # Unit tests (fast, isolated)
│   ├── test_budget_validator.py
│   ├── test_deduplication_validator.py
│   ├── test_volume_validator.py
│   ├── test_category_validator.py
│   └── ...
├── integration/               # Integration tests (slower)
│   └── ...
├── fixtures/                  # Test data and fixtures
│   └── markets.json
└── README.md                  # This file
```

## Running Tests

### Prerequisites

Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only validator tests
pytest -m validator

# Run tests in a specific file
pytest tests/unit/test_budget_validator.py

# Run a specific test
pytest tests/unit/test_budget_validator.py::TestBudgetValidator::test_validation_passes_when_under_budget
```

### Run Tests with Coverage

```bash
# Run with coverage report
pytest --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Run Tests in Parallel

```bash
# Run tests in parallel (faster for large test suites)
pytest -n auto
```

### Run Tests with Specific Verbosity

```bash
# Minimal output
pytest -q

# Normal output
pytest

# Verbose output
pytest -v

# Very verbose output (show all test names and outcomes)
pytest -vv
```

## Test Coverage Goals

| Module | Current Coverage | Target Coverage |
|--------|------------------|-----------------|
| Validators | 100% | 100% |
| Strategies | 90% | 95% |
| Configuration | 85% | 90% |
| Database | 70% | 80% |
| API Clients | 60% | 75% |
| **Overall** | **70%** | **80%+** |

## Writing Tests

### Unit Test Template

```python
"""
Unit tests for MyModule.
"""

import pytest
from unittest.mock import Mock, AsyncMock

@pytest.fixture
def my_fixture():
    """Create test fixture."""
    return SomeObject()

@pytest.mark.asyncio
class TestMyModule:
    """Tests for MyModule."""

    async def test_something(self, my_fixture):
        \"\"\"Test that something works.\"\"\"
        # Arrange
        obj = MyObject()

        # Act
        result = await obj.do_something()

        # Assert
        assert result.success
        assert result.value == expected_value
```

### Test Naming Conventions

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<what_is_being_tested>`

### Test Organization

Follow the **Arrange-Act-Assert** pattern:

```python
async def test_validation_passes(self):
    # Arrange - Set up test data and mocks
    validator = BudgetValidator(mock_db)
    market = create_test_market()

    # Act - Execute the code being tested
    result = await validator.validate(market)

    # Assert - Verify the results
    assert result.passed
    assert result.metadata['cost'] == 10.0
```

## Mocking

### Mock Database Calls

```python
from unittest.mock import Mock, AsyncMock

mock_db = Mock()
mock_db.get_daily_ai_cost = AsyncMock(return_value=10.0)
```

### Mock API Clients

```python
mock_kalshi_client = Mock()
mock_kalshi_client.get_balance = AsyncMock(return_value={"balance": 10000})
```

### Mock Settings

```python
from unittest.mock import patch

with patch('module.settings') as mock_settings:
    mock_settings.trading.daily_ai_budget = 50.0
    # ... test code
```

## Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.unit
@pytest.mark.validator
async def test_budget_validator():
    ...

@pytest.mark.integration
@pytest.mark.slow
async def test_full_trading_workflow():
    ...
```

## Continuous Integration

Tests are automatically run on every commit via GitHub Actions:

- ✅ Unit tests must pass
- ✅ Code coverage must be > 70%
- ✅ No linting errors (black, mypy, bandit)
- ✅ All security checks pass

## Troubleshooting

### Tests Hang

If tests hang, it's likely due to:
- Async tests without `@pytest.mark.asyncio`
- Missing mock for async functions
- Infinite loops in code

### Import Errors

If you get import errors:
```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in editable mode
pip install -e .
```

### Coverage Not Working

```bash
# Install coverage plugin
pip install pytest-cov

# Run with coverage
pytest --cov=src
```

## Performance

### Test Execution Times

- Unit tests: < 0.1s per test
- Integration tests: < 5s per test
- Full suite: < 60s

### Optimizing Slow Tests

If tests are slow:
1. Use mocks instead of real API calls
2. Use in-memory database for tests
3. Run tests in parallel: `pytest -n auto`
4. Mark slow tests: `@pytest.mark.slow`

## Best Practices

1. **Fast Tests**: Unit tests should be < 100ms each
2. **Isolated**: Tests should not depend on each other
3. **Deterministic**: Tests should always produce same results
4. **Clear**: Test names should describe what they test
5. **Comprehensive**: Test edge cases, not just happy path
6. **Maintainable**: Avoid complex test logic

## Examples

### Testing Validators

```python
@pytest.mark.asyncio
async def test_validator_passes():
    # Create validator
    validator = BudgetValidator(mock_db)

    # Create test market
    market = Market(
        market_id="TEST-001",
        title="Test",
        yes_price=0.50,
        ...
    )

    # Validate
    result = await validator.validate(market)

    # Assert
    assert result.passed
```

### Testing Strategies

```python
@pytest.mark.asyncio
async def test_strategy_creates_position():
    # Create strategy
    strategy = StandardStrategy()

    # Create context
    context = DecisionContext(
        market=market,
        db_manager=mock_db,
        kalshi_client=mock_kalshi,
        xai_client=mock_xai,
        available_balance=1000.0
    )

    # Execute
    result = await strategy.decide(context)

    # Assert
    assert result.has_position
    assert result.position.quantity > 0
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Coverage.py](https://coverage.readthedocs.io/)

## Questions?

For questions about testing, please:
1. Check this README
2. Review existing tests for examples
3. Open an issue on GitHub
4. Contact the development team
