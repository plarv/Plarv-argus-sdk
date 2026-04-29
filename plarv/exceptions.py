"""
PLARV Argus — Forensic Exception Protocol
=========================================
Standardized exception hierarchy for training interventions and API failures.
"""

class ArgusError(Exception):
    """Base exception for all Argus-related issues."""
    pass

class ArgusConfigurationError(ArgusError):
    """Raised when initialization parameters are invalid or missing."""
    pass

class ArgusConnectionError(ArgusError):
    """Raised when the PLARV API is unreachable or network is down."""
    pass

class ArgusApiError(ArgusError):
    """Raised for non-200 responses from the PLARV API."""
    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response or {}

class ArgusAuthenticationError(ArgusApiError):
    """Raised when the API key is invalid or unauthorized."""
    pass

class ArgusRateLimitError(ArgusApiError):
    """Raised when the API rate limit is exceeded."""
    pass

class ArgusServerError(ArgusApiError):
    """Raised for 5xx backend errors."""
    pass

# =============================================================================
# INTERVENTIONS (Sovereign Signals)
# =============================================================================

class ArgusIntervention(ArgusError):
    """Base for all engine-driven training interventions."""
    def __init__(self, message: str, step: int = 0, response: dict = None):
        super().__init__(message)
        self.step = step
        self.response = response or {}

class ArgusPause(ArgusIntervention):
    """Engine requested a training pause due to instability detection."""
    pass

class ArgusCheckpoint(ArgusIntervention):
    """Engine requested an immediate forensic checkpoint."""
    pass

class ArgusHalt(ArgusIntervention):
    """Sentinel requested a hard stop (emergency halt)."""
    pass