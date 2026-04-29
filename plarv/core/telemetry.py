"""
PLARV Argus — Run Telemetry Reporter
======================================
Sends non-sensitive run metadata to PLARV analytics endpoint.
Completely separate from training safety telemetry.

Fires on:
  - Every N seconds (default: 10)
  - Key milestones: run start, every 1000 steps, run end

Data collected (non-sensitive, no model weights, no gradients):
  - GPU info (name, count, VRAM)
  - Model parameter count
  - Training throughput (steps/sec, samples/sec)
  - Estimated time remaining
  - Framework (PyTorch version, CUDA version)
  - Run duration

Usage:
    # Automatic — just pass to Argus (not yet wired, see Argus.py integration below)
    reporter = RunTelemetry(api_key="...", run_id="...", model=model)
    reporter.start()
    reporter.on_step(step=100, loss=0.5, batch_size=32)
    reporter.stop()
"""

import json
import os
import platform
import threading
import time
import urllib.request
from typing import Optional, Any, Dict


# =============================================================================
# GPU / SYSTEM PROBE — pure stdlib + optional torch
# =============================================================================

def _probe_gpu() -> Dict:
    """Collect GPU info. Silent if torch/CUDA not available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False, "count": 0, "names": [], "vram_gb": []}

        count = torch.cuda.device_count()
        names = []
        vram  = []
        for i in range(count):
            props = torch.cuda.get_device_properties(i)
            names.append(props.name)
            vram.append(round(props.total_memory / 1024**3, 1))  # GB

        return {
            "available": True,
            "count":     count,
            "names":     names,
            "vram_gb":   vram,
            "cuda_version": torch.version.cuda or "unknown",
        }
    except Exception:
        return {"available": False, "count": 0, "names": [], "vram_gb": []}


def _probe_framework() -> Dict:
    """Collect framework versions."""
    result = {"python": platform.python_version()}
    try:
        import torch
        result["torch"] = torch.__version__
    except ImportError:
        pass
    try:
        import transformers
        result["transformers"] = transformers.__version__
    except ImportError:
        pass
    return result


def _probe_model(model: Any) -> Dict:
    """Collect model size info — parameter count only, no weights."""
    if model is None:
        return {"param_count": None, "param_count_M": None}
    try:
        total  = sum(p.numel() for p in model.parameters())
        active = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "param_count":   total,
            "param_count_M": round(total / 1e6, 1),
            "trainable_M":   round(active / 1e6, 1),
        }
    except Exception:
        return {"param_count": None, "param_count_M": None}


def _probe_model_name(model: Any) -> Optional[str]:
    """Try to get a human-readable model name."""
    if model is None:
        return None
    try:
        cfg = getattr(model, "config", None)
        if cfg:
            return (
                getattr(cfg, "_name_or_path", None) or
                getattr(cfg, "model_type", None) or
                type(model).__name__
            )
        return type(model).__name__
    except Exception:
        return None


# =============================================================================
# RUN TELEMETRY
# =============================================================================

class RunTelemetry:
    """
    Sends non-sensitive run metadata to PLARV analytics.

    Fires:
      - Immediately on start (run_start event)
      - Every `interval_s` seconds in background thread
      - On every `milestone_every` steps
      - On stop (run_end event)
    """

    _ANALYTICS_PATH = "/api/v2/telemetry"

    def __init__(
        self,
        api_key:          str,
        run_id:           str,
        base_url:         Optional[str]  = None,
        model:            Optional[Any]  = None,
        model_name:       Optional[str]  = None,
        interval_s:       float          = 10.0,
        milestone_every:  int            = 1000,
        silent:           bool           = False,
    ):
        self.api_key         = api_key
        self.run_id          = run_id
        self.base_url        = (base_url or os.getenv("PLARV_API_URL", "https://api.plarv.com")).rstrip("/")
        self.interval_s      = interval_s
        self.milestone_every = milestone_every
        self.silent          = silent
        self._pending_milestone = False

        # Probe once at init — these don't change during a run
        self._gpu       = _probe_gpu()
        self._framework = _probe_framework()
        self._model_info = _probe_model(model)
        self._model_name = model_name or _probe_model_name(model)

        # Runtime state
        self._start_time:    Optional[float] = None
        self._last_step:     int   = 0
        self._last_loss:     float = 0.0
        self._batch_size:    int   = 0
        self._step_times:    list  = []   # rolling window for throughput
        self._last_step_ts:  Optional[float] = None

        # DQI Collection
        self._dqi_steps:     list  = []
        self._dqi_losses:    list  = []

        # Background thread
        self._thread:   Optional[threading.Thread] = None
        self._stop_evt: threading.Event = threading.Event()

    # =========================================================================
    # PUBLIC
    # =========================================================================

    def start(self) -> None:
        """Call once before training loop begins."""
        self._start_time = time.time()
        self._fire("run_start", extra={
            "gpu":       self._gpu,
            "framework": self._framework,
            "model":     {**self._model_info, "name": self._model_name},
        })
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def on_step(self, step: int, loss: float, batch_size: int = 0) -> None:
        """
        Call after every Argus.step(). Zero-cost — just updates internal state.
        Background thread reads this state periodically.
        """
        now = time.time()

        # Rolling step-time window for throughput (keep last 50)
        if self._last_step_ts is not None:
            elapsed = now - self._last_step_ts
            if 0 < elapsed < 60:  # sanity bound
                self._step_times.append(elapsed)
                if len(self._step_times) > 50:
                    self._step_times.pop(0)

        self._last_step    = step
        self._last_loss    = loss
        self._batch_size   = batch_size
        self._last_step_ts = now
        
        self._dqi_steps.append(step)
        self._dqi_losses.append(loss)

        # Milestone flag (non-blocking — will be picked up by the background loop)
        if step > 0 and step % self.milestone_every == 0:
            self._pending_milestone = True

    def stop(self, status: str = "COMPLETED", error: Optional[str] = None) -> None:
        """Call once after training loop ends."""
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=3)
            
        extra = self._runtime_snapshot()
        extra["status"] = status
        if error:
            extra["error_msg"] = str(error)
        
        # Compute Data Quality Index
        dqi_score = None
        if len(self._dqi_steps) >= 3:
            try:
                from ..adqi import get_dqi_score
                dqi_score = get_dqi_score(self._dqi_steps, self._dqi_losses)
            except Exception:
                pass # Silently ignore if numpy is not installed or math fails
                
        extra["dqi"] = dqi_score
        
        self._fire("run_end", extra=extra)

    # =========================================================================
    # INTERNAL
    # =========================================================================

    def _loop(self) -> None:
        while not self._stop_evt.wait(timeout=self.interval_s + __import__('random').uniform(-0.5, 0.5)):
            self._fire("heartbeat", extra=self._runtime_snapshot())
            
            if self._pending_milestone:
                self._fire("milestone", extra=self._runtime_snapshot())
                self._pending_milestone = False

    def _runtime_snapshot(self) -> Dict:
        """Current runtime metrics — computed at fire time."""
        now     = time.time()
        elapsed = now - self._start_time if self._start_time else 0.0

        # Throughput
        steps_per_sec  = 0.0
        samples_per_sec = 0.0
        eta_hours      = None

        if self._step_times:
            avg_step_s     = sum(self._step_times) / len(self._step_times)
            steps_per_sec  = round(1.0 / avg_step_s, 2) if avg_step_s > 0 else 0.0
            samples_per_sec = round(steps_per_sec * self._batch_size, 1)

        return {
            "step":           self._last_step,
            "loss":           round(self._last_loss, 4),
            "elapsed_hours":  round(elapsed / 3600, 3),
            "steps_per_sec":  steps_per_sec,
            "samples_per_sec": samples_per_sec,
            "batch_size":     self._batch_size,
        }

    def _fire(self, event: str, extra: Dict = None) -> None:
        """POST telemetry. Silent on failure — never blocks training."""
        payload = {
            "run_id":    self.run_id,
            "event":     event,
            "ts":        int(time.time()),
            **( extra or {}),
        }
        try:
            data = json.dumps(payload).encode()
            req  = urllib.request.Request(
                self.base_url + self._ANALYTICS_PATH,
                data=data,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type":  "application/json",
                },
                method="POST",
            )
            urllib.request.urlopen(req, timeout=3)
        except Exception:
            pass  # never crash training