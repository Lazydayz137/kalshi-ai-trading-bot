"""
Unit tests for CategoryValidator.
"""

import pytest
from unittest.mock import patch
from datetime import datetime

from src.jobs.decision_validators.category_validator import CategoryValidator
from src.jobs.decision_validators.validation_result import ValidationStatus
from src.utils.database import Market


@pytest.fixture
def sample_market():
    """Create sample market for testing."""
    def _make_market(category):
        return Market(
            market_id="TEST-MARKET-001",
            title="Test Market",
            yes_price=0.50,
            no_price=0.50,
            volume=1000,
            expiration_ts=int(datetime.now().timestamp()) + 86400,
            category=category,
            status="active",
            last_updated=datetime.now()
        )
    return _make_market


@pytest.mark.asyncio
class TestCategoryValidator:
    """Tests for CategoryValidator."""

    async def test_validation_passes_when_category_allowed(self, sample_market):
        """Test validation passes when category is not excluded."""
        # Arrange
        market = sample_market(category="politics")
        validator = CategoryValidator()

        with patch('src.jobs.decision_validators.category_validator.settings') as mock_settings:
            mock_settings.trading.exclude_low_liquidity_categories = ["weather", "entertainment"]

            # Act
            result = await validator.validate(market)

            # Assert
            assert result.passed
            assert result.status == ValidationStatus.PASSED
            assert "Category OK" in result.reason
            assert result.metadata['category'] == "politics"

    async def test_validation_fails_when_category_excluded(self, sample_market):
        """Test validation fails when category is excluded."""
        # Arrange
        market = sample_market(category="weather")
        validator = CategoryValidator()

        with patch('src.jobs.decision_validators.category_validator.settings') as mock_settings:
            mock_settings.trading.exclude_low_liquidity_categories = ["weather", "entertainment"]

            # Act
            result = await validator.validate(market)

            # Assert
            assert result.failed
            assert result.status == ValidationStatus.FAILED
            assert "Excluded category" in result.reason
            assert result.metadata['category'] == "weather"

    async def test_validation_case_insensitive(self, sample_market):
        """Test validation is case-insensitive."""
        # Arrange
        market = sample_market(category="WEATHER")
        validator = CategoryValidator()

        with patch('src.jobs.decision_validators.category_validator.settings') as mock_settings:
            mock_settings.trading.exclude_low_liquidity_categories = ["weather", "entertainment"]

            # Act
            result = await validator.validate(market)

            # Assert
            assert result.failed
            assert "Excluded category" in result.reason

    async def test_validation_passes_with_empty_exclusion_list(self, sample_market):
        """Test validation passes when exclusion list is empty."""
        # Arrange
        market = sample_market(category="anything")
        validator = CategoryValidator()

        with patch('src.jobs.decision_validators.category_validator.settings') as mock_settings:
            mock_settings.trading.exclude_low_liquidity_categories = []

            # Act
            result = await validator.validate(market)

            # Assert
            assert result.passed

    async def test_validation_passes_for_multiple_allowed_categories(self, sample_market):
        """Test validation passes for various allowed categories."""
        # Arrange
        validator = CategoryValidator()
        allowed_categories = ["politics", "sports", "crypto", "finance"]

        with patch('src.jobs.decision_validators.category_validator.settings') as mock_settings:
            mock_settings.trading.exclude_low_liquidity_categories = ["weather"]

            # Act & Assert
            for category in allowed_categories:
                market = sample_market(category=category)
                result = await validator.validate(market)
                assert result.passed, f"Category {category} should be allowed"

    async def test_validation_fails_for_multiple_excluded_categories(self, sample_market):
        """Test validation fails for various excluded categories."""
        # Arrange
        validator = CategoryValidator()
        excluded_categories = ["weather", "entertainment", "celebrity"]

        with patch('src.jobs.decision_validators.category_validator.settings') as mock_settings:
            mock_settings.trading.exclude_low_liquidity_categories = excluded_categories

            # Act & Assert
            for category in excluded_categories:
                market = sample_market(category=category)
                result = await validator.validate(market)
                assert result.failed, f"Category {category} should be excluded"

    async def test_metadata_contains_category_info(self, sample_market):
        """Test metadata contains correct category information."""
        # Arrange
        market = sample_market(category="politics")
        validator = CategoryValidator()

        with patch('src.jobs.decision_validators.category_validator.settings') as mock_settings:
            mock_settings.trading.exclude_low_liquidity_categories = ["weather"]

            # Act
            result = await validator.validate(market)

            # Assert
            assert 'category' in result.metadata
            assert result.metadata['category'] == "politics"
