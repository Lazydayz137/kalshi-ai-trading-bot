"""
Unit tests for DeduplicationValidator.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.jobs.decision_validators.deduplication_validator import DeduplicationValidator
from src.jobs.decision_validators.validation_result import ValidationStatus
from src.utils.database import Market


@pytest.fixture
def mock_db_manager():
    """Create mock database manager."""
    db = Mock()
    db.was_recently_analyzed = AsyncMock()
    db.get_market_analysis_count_today = AsyncMock()
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
class TestDeduplicationValidator:
    """Tests for DeduplicationValidator."""

    async def test_validation_passes_when_not_recently_analyzed(self, mock_db_manager, sample_market):
        """Test validation passes when market not recently analyzed."""
        # Arrange
        mock_db_manager.was_recently_analyzed.return_value = False
        mock_db_manager.get_market_analysis_count_today.return_value = 1
        validator = DeduplicationValidator(mock_db_manager)

        with patch('src.jobs.decision_validators.deduplication_validator.settings') as mock_settings:
            mock_settings.trading.analysis_cooldown_hours = 6
            mock_settings.trading.max_analyses_per_market_per_day = 4

            # Act
            result = await validator.validate(sample_market)

            # Assert
            assert result.passed
            assert result.status == ValidationStatus.PASSED
            assert "Deduplication OK" in result.reason
            assert result.metadata['analysis_count'] == 1

    async def test_validation_fails_when_recently_analyzed(self, mock_db_manager, sample_market):
        """Test validation fails when market was recently analyzed."""
        # Arrange
        mock_db_manager.was_recently_analyzed.return_value = True
        validator = DeduplicationValidator(mock_db_manager)

        with patch('src.jobs.decision_validators.deduplication_validator.settings') as mock_settings:
            mock_settings.trading.analysis_cooldown_hours = 6

            # Act
            result = await validator.validate(sample_market)

            # Assert
            assert result.failed
            assert result.status == ValidationStatus.FAILED
            assert "Recently analyzed" in result.reason
            assert result.metadata['cooldown_hours'] == 6

    async def test_validation_fails_when_daily_limit_reached(self, mock_db_manager, sample_market):
        """Test validation fails when daily analysis limit reached."""
        # Arrange
        mock_db_manager.was_recently_analyzed.return_value = False
        mock_db_manager.get_market_analysis_count_today.return_value = 4
        validator = DeduplicationValidator(mock_db_manager)

        with patch('src.jobs.decision_validators.deduplication_validator.settings') as mock_settings:
            mock_settings.trading.analysis_cooldown_hours = 6
            mock_settings.trading.max_analyses_per_market_per_day = 4

            # Act
            result = await validator.validate(sample_market)

            # Assert
            assert result.failed
            assert result.status == ValidationStatus.FAILED
            assert "Daily analysis limit reached" in result.reason
            assert result.metadata['analysis_count'] == 4
            assert result.metadata['limit'] == 4

    async def test_validation_passes_when_under_daily_limit(self, mock_db_manager, sample_market):
        """Test validation passes when under daily limit."""
        # Arrange
        mock_db_manager.was_recently_analyzed.return_value = False
        mock_db_manager.get_market_analysis_count_today.return_value = 2
        validator = DeduplicationValidator(mock_db_manager)

        with patch('src.jobs.decision_validators.deduplication_validator.settings') as mock_settings:
            mock_settings.trading.analysis_cooldown_hours = 6
            mock_settings.trading.max_analyses_per_market_per_day = 4

            # Act
            result = await validator.validate(sample_market)

            # Assert
            assert result.passed
            assert "2/4 analyses today" in result.reason

    async def test_validation_fails_when_at_daily_limit(self, mock_db_manager, sample_market):
        """Test validation fails when exactly at daily limit."""
        # Arrange
        mock_db_manager.was_recently_analyzed.return_value = False
        mock_db_manager.get_market_analysis_count_today.return_value = 6
        validator = DeduplicationValidator(mock_db_manager)

        with patch('src.jobs.decision_validators.deduplication_validator.settings') as mock_settings:
            mock_settings.trading.max_analyses_per_market_per_day = 6

            # Act
            result = await validator.validate(sample_market)

            # Assert
            assert result.failed

    async def test_validation_passes_with_zero_analyses_today(self, mock_db_manager, sample_market):
        """Test validation passes with zero analyses today."""
        # Arrange
        mock_db_manager.was_recently_analyzed.return_value = False
        mock_db_manager.get_market_analysis_count_today.return_value = 0
        validator = DeduplicationValidator(mock_db_manager)

        with patch('src.jobs.decision_validators.deduplication_validator.settings') as mock_settings:
            mock_settings.trading.analysis_cooldown_hours = 6
            mock_settings.trading.max_analyses_per_market_per_day = 4

            # Act
            result = await validator.validate(sample_market)

            # Assert
            assert result.passed
            assert result.metadata['analysis_count'] == 0
