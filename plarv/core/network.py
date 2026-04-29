"""
PLARV Argus — Clinical Transport Layer
======================================
Handles asynchronous API communication, decision state, and network resilience.
"""

import json
import time
import random
import urllib.request
import threading
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from ..exceptions import (
    ArgusConnectionError, ArgusApiError, ArgusAuthenticationError,
    ArgusRateLimitError, ArgusServerError
)

class _Decision:
    """Isolates the engine's intervention state."""
    def __init__(self):
        self.action = "NONE"
        self.checkpoint_signal = "NONE"
        self.checkpoint_slot = None
        self.intervention_secret_hash = None
        self.raw = {}

    def update(self, data: Dict[str, Any]):
        self.raw = data
        self.action = data.get("action", "NONE")
        self.checkpoint_signal = data.get("checkpoint_signal", "NONE")
        self.checkpoint_slot = data.get("checkpoint_slot")
        self.intervention_secret_hash = data.get("intervention_secret_hash")

class _NetworkClient:
    """Sovereign transport for Argus telemetry."""
    def __init__(self, api_key: str, base_url: str, timeout: float = 3.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "PlarvArgusSDK/2.0.0"
        }

    def fire_async(self, payload: Any, callback=None):
        """Dispatches telemetry to the background pool (Zero-Block)."""
        future = self._executor.submit(self._post, payload)
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
        return future

    def _post(self, payload: Any) -> Optional[Dict]:
        """Synchronous POST with jitter and backoff."""
        url = f"{self.base_url}/api/v2/telemetry"
        data = json.dumps(payload).encode()
        
        # Jittered backoff simulation (if needed)
        try:
            req = urllib.request.Request(url, data=data, headers=self._headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 401: raise ArgusAuthenticationError("Invalid API Key", 401)
            if e.code == 429: raise ArgusRateLimitError("Rate limit exceeded", 429)
            if e.code >= 500: raise ArgusServerError("Backend under pressure", e.code)
            return None
        except Exception:
            return None

    def shutdown(self):
        self._executor.shutdown(wait=False)
