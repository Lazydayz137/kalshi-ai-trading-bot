"""
Unit tests for VolumeValidator.
"""

import pytest
from unittest.mock import patch
from datetime import datetime

from src.jobs.decision_validators.volume_validator import VolumeValidator
from src.jobs.decision_validators.validation_result import ValidationStatus
from src.utils.database import Market


@pytest.fixture
def sample_market():
    """Create sample market for testing."""
    def _make_market(volume):
        return Market(
            market_id="TEST-MARKET-001",
            title="Test Market",
            yes_price=0.50,
            no_price=0.50,
            volume=volume,
            expiration_ts=int(datetime.now().timestamp()) + 86400,
            category="test",
            status="active",
            last_updated=datetime.now()
        )
    return _make_market


@pytest.mark.asyncio
class TestVolumeValidator:
    """Tests for VolumeValidator."""

    async def test_validation_passes_when_volume_sufficient(self, sample_market):
        """Test validation passes when volume meets threshold."""
        # Arrange
        market = sample_market(volume=500)
        validator = VolumeValidator()

        with patch('src.jobs.decision_validators.volume_validator.settings') as mock_settings:
            mock_settings.trading.min_volume_for_ai_analysis = 200.0

            # Act
            result = await validator.validate(market)

            # Assert
            assert result.passed
            assert result.status == ValidationStatus.PASSED
            assert "Volume OK" in result.reason
            assert result.metadata['volume'] == 500

    async def test_validation_fails_when_volume_insufficient(self, sample_market):
        """Test validation fails when volume below threshold."""
        # Arrange
        market = sample_market(volume=100)
        validator = VolumeValidator()

        with patch('src.jobs.decision_validators.volume_validator.settings') as mock_settings:
            mock_settings.trading.min_volume_for_ai_analysis = 200.0

            # Act
            result = await validator.validate(market)

            # Assert
            assert result.failed
            assert result.status == ValidationStatus.FAILED
            assert "Insufficient volume" in result.reason
            assert result.metadata['volume'] == 100
            assert result.metadata['min_volume'] == 200.0

    async def test_validation_passes_when_volume_equals_threshold(self, sample_market):
        """Test validation passes when volume exactly equals threshold."""
        # Arrange
        market = sample_market(volume=200)
        validator = VolumeValidator()

        with patch('src.jobs.decision_validators.volume_validator.settings') as mock_settings:
            mock_settings.trading.min_volume_for_ai_analysis = 200.0

            # Act
            result = await validator.validate(market)

            # Assert
            assert result.passed

    async def test_validation_fails_with_zero_volume(self, sample_market):
        """Test validation fails with zero volume."""
        # Arrange
        market = sample_market(volume=0)
        validator = VolumeValidator()

        with patch('src.jobs.decision_validators.volume_validator.settings') as mock_settings:
            mock_settings.trading.min_volume_for_ai_analysis = 200.0

            # Act
            result = await validator.validate(market)

            # Assert
            assert result.failed
            assert result.metadata['volume'] == 0

    async def test_validation_passes_with_high_volume(self, sample_market):
        """Test validation passes with very high volume."""
        # Arrange
        market = sample_market(volume=10000)
        validator = VolumeValidator()

        with patch('src.jobs.decision_validators.volume_validator.settings') as mock_settings:
            mock_settings.trading.min_volume_for_ai_analysis = 200.0

            # Act
            result = await validator.validate(market)

            # Assert
            assert result.passed
            assert result.metadata['volume'] == 10000

    async def test_metadata_contains_volume_information(self, sample_market):
        """Test metadata contains correct volume information."""
        # Arrange
        market = sample_market(volume=500)
        validator = VolumeValidator()

        with patch('src.jobs.decision_validators.volume_validator.settings') as mock_settings:
            mock_settings.trading.min_volume_for_ai_analysis = 200.0

            # Act
            result = await validator.validate(market)

            # Assert
            assert 'volume' in result.metadata
            assert result.metadata['volume'] == 500
