"""
PLARV Argus SDK
================
Forensic ML Training Protection & Real-time Telemetry.
"""

from .argus import Argus
from .local import LocalDetector
from .adqi import get_dqi_score
from .integrations.callback import ArgusCallback
from .integrations.lightning import ArgusLightningCallback
from .utils import extract_signals, probe_gpu
from .exceptions import (
    ArgusError, ArgusPause, ArgusCheckpoint, ArgusHalt,
    ArgusApiError, ArgusConnectionError
)

__version__ = "2.0.0"
__all__ = [
    "Argus",
    "LocalDetector",
    "get_dqi_score",
    "ArgusCallback",
    "ArgusLightningCallback",
    "extract_signals",
    "probe_gpu",
    "ArgusError",
    "ArgusPause",
    "ArgusCheckpoint",
    "ArgusHalt",
    "ArgusApiError",
    "ArgusConnectionError"
]
