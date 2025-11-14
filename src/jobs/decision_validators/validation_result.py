"""
Validation result dataclass for decision validators.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ValidationStatus(Enum):
    """Status of validation check."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    status: ValidationStatus
    reason: str
    cost: float = 0.0  # Cost incurred during validation (if any)
    metadata: Optional[dict] = None  # Additional context

    @property
    def passed(self) -> bool:
        """Check if validation passed."""
        return self.status == ValidationStatus.PASSED

    @property
    def failed(self) -> bool:
        """Check if validation failed."""
        return self.status == ValidationStatus.FAILED

    @property
    def skipped(self) -> bool:
        """Check if validation was skipped."""
        return self.status == ValidationStatus.SKIPPED

    @classmethod
    def pass_validation(cls, reason: str, **kwargs) -> 'ValidationResult':
        """Create a passed validation result."""
        return cls(status=ValidationStatus.PASSED, reason=reason, **kwargs)

    @classmethod
    def fail_validation(cls, reason: str, **kwargs) -> 'ValidationResult':
        """Create a failed validation result."""
        return cls(status=ValidationStatus.FAILED, reason=reason, **kwargs)

    @classmethod
    def skip_validation(cls, reason: str, **kwargs) -> 'ValidationResult':
        """Create a skipped validation result."""
        return cls(status=ValidationStatus.SKIPPED, reason=reason, **kwargs)
