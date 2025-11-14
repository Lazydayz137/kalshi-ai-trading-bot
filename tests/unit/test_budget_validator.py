"""
Unit tests for BudgetValidator.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.jobs.decision_validators.budget_validator import BudgetValidator
from src.jobs.decision_validators.validation_result import ValidationStatus
from src.utils.database import Market


@pytest.fixture
def mock_db_manager():
    """Create mock database manager."""
    db = Mock()
    db.get_daily_ai_cost = AsyncMock()
    return db


@pytest.fixture
def sample_market():
    """Create sample market for testing."""
    return Market(
        market_id="TEST-MARKET-001",
        title="Test Market",
        yes_price=0.50,
        no_price=0.50,
        volume=1000,
        expiration_ts=int(datetime.now().timestamp()) + 86400,
        category="test",
        status="active",
        last_updated=datetime.now()
    )


@pytest.mark.asyncio
class TestBudgetValidator:
    """Tests for BudgetValidator."""

    async def test_validation_passes_when_under_budget(self, mock_db_manager, sample_market):
        """Test validation passes when daily cost is under budget."""
        # Arrange
        mock_db_manager.get_daily_ai_cost.return_value = 10.0
        validator = BudgetValidator(mock_db_manager)

        with patch('src.jobs.decision_validators.budget_validator.settings') as mock_settings:
            mock_settings.trading.daily_ai_budget = 50.0

            # Act
            result = await validator.validate(sample_market)

            # Assert
            assert result.passed
            assert result.status == ValidationStatus.PASSED
            assert "Budget OK" in result.reason
            assert result.metadata['daily_cost'] == 10.0
            assert result.metadata['remaining'] == 40.0

    async def test_validation_fails_when_budget_exceeded(self, mock_db_manager, sample_market):
        """Test validation fails when daily budget is exceeded."""
        # Arrange
        mock_db_manager.get_daily_ai_cost.return_value = 55.0
        validator = BudgetValidator(mock_db_manager)

        with patch('src.jobs.decision_validators.budget_validator.settings') as mock_settings:
            mock_settings.trading.daily_ai_budget = 50.0

            # Act
            result = await validator.validate(sample_market)

            # Assert
            assert result.failed
            assert result.status == ValidationStatus.FAILED
            assert "Daily budget exceeded" in result.reason
            assert result.metadata['daily_cost'] == 55.0
            assert result.metadata['budget'] == 50.0

    async def test_validation_fails_when_exactly_at_budget(self, mock_db_manager, sample_market):
        """Test validation fails when daily cost exactly equals budget."""
        # Arrange
        mock_db_manager.get_daily_ai_cost.return_value = 50.0
        validator = BudgetValidator(mock_db_manager)

        with patch('src.jobs.decision_validators.budget_validator.settings') as mock_settings:
            mock_settings.trading.daily_ai_budget = 50.0

            # Act
            result = await validator.validate(sample_market)

            # Assert
            assert result.failed
            assert result.status == ValidationStatus.FAILED

    async def test_validation_passes_with_zero_cost(self, mock_db_manager, sample_market):
        """Test validation passes when no cost incurred yet."""
        # Arrange
        mock_db_manager.get_daily_ai_cost.return_value = 0.0
        validator = BudgetValidator(mock_db_manager)

        with patch('src.jobs.decision_validators.budget_validator.settings') as mock_settings:
            mock_settings.trading.daily_ai_budget = 50.0

            # Act
            result = await validator.validate(sample_market)

            # Assert
            assert result.passed
            assert result.metadata['daily_cost'] == 0.0
            assert result.metadata['remaining'] == 50.0

    async def test_validation_handles_database_error_gracefully(self, mock_db_manager, sample_market):
        """Test validation handles database errors gracefully."""
        # Arrange
        mock_db_manager.get_daily_ai_cost.side_effect = Exception("Database error")
        validator = BudgetValidator(mock_db_manager)

        # Act & Assert
        with pytest.raises(Exception, match="Database error"):
            await validator.validate(sample_market)

    async def test_metadata_contains_correct_information(self, mock_db_manager, sample_market):
        """Test metadata contains all expected information."""
        # Arrange
        mock_db_manager.get_daily_ai_cost.return_value = 25.0
        validator = BudgetValidator(mock_db_manager)

        with patch('src.jobs.decision_validators.budget_validator.settings') as mock_settings:
            mock_settings.trading.daily_ai_budget = 50.0

            # Act
            result = await validator.validate(sample_market)

            # Assert
            assert 'daily_cost' in result.metadata
            assert 'remaining' in result.metadata
            assert result.metadata['daily_cost'] == 25.0
            assert result.metadata['remaining'] == 25.0
