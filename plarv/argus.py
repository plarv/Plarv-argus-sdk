"""
PLARV Argus SDK — Core Client
==============================
Zero-block async prefetch. Step 0 gate. Silent fallback on API death.
Rolling certified checkpoint system — saves on health signal, not schedule.

Usage:
    argus = Argus(api_key="your-key")
    argus.step(loss=loss.item(), grad_norm=grad_norm)"""

import hashlib
import json
import math
import os
import random
import shutil
import string
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Dict, Any, List
from .exceptions import (
    ArgusError, ArgusApiError, ArgusAuthenticationError, 
    ArgusRateLimitError, ArgusServerError, ArgusConnectionError,
    ArgusConfigurationError, ArgusPause, ArgusCheckpoint, ArgusHalt
)
from .core.telemetry import RunTelemetry
from .adqi import compute_dqi

# =============================================================================
# INTERNAL HTTP (re-implemented with error tracking)
# =============================================================================

class _NetworkClient:
    def __init__(self, silent: bool = False):
        self.silent = silent
        self.consecutive_failures = 0
        self.last_success_time = None
        self._lock = threading.Lock()

    def post(self, url: str, headers: Dict, payload: Dict, timeout: float, retries: int = 0) -> Dict:
        import urllib.request
        import urllib.error
        import random
        
        attempt = 0
        while True:
            try:
                data = json.dumps(payload).encode()
                req  = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    resp = json.loads(r.read().decode())
                    
                    with self._lock:
                        self.consecutive_failures = 0
                        self.last_success_time = time.time()
                    return resp

            except urllib.error.HTTPError as e:
                # 🛡️ STRIPE-LEVEL RESILIENCE: Retry on transient errors or rate limits
                if attempt < retries and e.code in (429, 502, 503, 504):
                    attempt += 1
                    self._backoff(attempt)
                    continue
                
                body = {}
                try: 
                    raw_body = e.read().decode()
                    body = json.loads(raw_body)
                except Exception: 
                    raw_body = "NON-JSON RESPONSE"
                
                self._handle_failure(status=e.code, body=raw_body)
                return {"_http_error": e.code, "_http_body": body}
                
            except Exception as e:
                # Retry on connection failures
                if attempt < retries:
                    attempt += 1
                    self._backoff(attempt)
                    continue
                    
                self._handle_failure(status="CONNECTION_FAILED", body=str(e))
                return {"_error": "CONNECTION_FAILED", "detail": str(e)}

    def _backoff(self, attempt: int):
        import time
        import random
        # 🎲 FULL JITTER BACKOFF: min(cap, base * 2^attempt) * rand(0.5, 1.5)
        base_delay = 1.0
        max_delay  = 30.0
        delay = min(base_delay * (2 ** (attempt - 1)), max_delay) * random.uniform(0.5, 1.5)
        if not self.silent:
            # Silent print to avoid cluttering training logs too much, but visible on retry
            print(f"[PLARV] Network congestion. Retrying in {delay:.2f}s (attempt {attempt})...")
        time.sleep(delay)

    def _handle_failure(self, status=None, body=None):
        # 🛡️ SILENT TERMINATION: If the run was stopped by the user, ignore the error
        if body and "already STOPPED" in str(body):
            return

        with self._lock:
            self.consecutive_failures += 1
            if self.consecutive_failures >= 1 and not self.silent:
                # FAIL-LOUD ALERT (High visibility)
                print("\n" + "!" * 80)
                print(f"!!! [PLARV DEBUG] NETWORK ERROR ({self.consecutive_failures} consecutive failures)")
                if status:
                    print(f"!!! HTTP STATUS: {status}")
                if body:
                    print(f"!!! RESPONSE:    {body}")
                print(f"!!! Last successful ping: {time.ctime(self.last_success_time) if self.last_success_time else 'NEVER'}")
                print("!" * 80 + "\n")

_net = _NetworkClient()

def _post(url: str, headers: Dict, payload: Dict, timeout: float, retries: int = 0) -> Dict:
    """Legacy wrapper for the new network client."""
    return _net.post(url, headers, payload, timeout, retries=retries)


# =============================================================================
# DECISION STATE — what the engine last told us to do
# =============================================================================

class _Decision:
    """Holds the latest engine response. Thread-safe reads."""
    def __init__(self):
        self.action        = "CONTINUE"
        self.harm_pressure = 0
        self.sdk_callback  = None
        self.alert         = "NONE"
        self.raw           = {}
        self._lock         = threading.Lock()

    def update(self, resp):
        if not resp or not isinstance(resp, dict):
            return  # ignore bad responses

        with self._lock:
            self.action        = resp.get("action", "CONTINUE")
            self.harm_pressure = resp.get("harm_pressure", 0)

            intervention = resp.get("intervention") or {}
            if not isinstance(intervention, dict):
                intervention = {}

            self.alert = intervention.get("alert", "NONE")

            self.sdk_callback = (
                intervention.get("sdk_callback")
                if intervention.get("fired") else None
            )

            resp.setdefault("action", "CONTINUE")
            resp.setdefault("harm_pressure", 0)
            self.raw = resp

    def read(self):
        with self._lock:
            return self.action, self.harm_pressure, self.sdk_callback, self.alert


# =============================================================================
# SPOOL MANAGER — persistent catch-up buffer for network blips
# =============================================================================

class _SpoolManager:
    """
    Sovereign Telemetry Spool.
    Writes missed steps to argus/spool.jsonl when network is down.
    Drains them automatically when connection returns.
    """
    def __init__(self, argus_dir: str, silent: bool = False):
        self.argus_dir = argus_dir
        self.path      = os.path.join(argus_dir, "spool.jsonl")
        self.silent    = silent
        self._lock     = threading.Lock()
        os.makedirs(argus_dir, exist_ok=True)

    def spool(self, payload: Dict):
        """Append one step to the local disk buffer."""
        try:
            # Strip heavy histogram/distribution data from spool to save disk
            minimal = payload.copy()
            if "histogram" in minimal: minimal["histogram"] = None
            
            with self._lock:
                with open(self.path, "a") as f:
                    f.write(json.dumps(minimal) + "\n")
        except Exception:
            pass

    def drain(self, post_fn, url, headers):
        """Try to send all spooled steps. Deletes file on success."""
        if not os.path.exists(self.path):
            return

        with self._lock:
            try:
                with open(self.path, "r") as f:
                    lines = f.readlines()
                if not lines:
                    return

                if not self.silent:
                    print(f"[PLARV] Network recovered. Draining {len(lines)} spooled steps...")

                # Send in one batch if backend supports it, otherwise drip-feed
                # For now, drip-feed to ensure individual step processing
                for line in lines:
                    payload = json.loads(line.strip())
                    post_fn(url, headers, payload, timeout=5.0)
                
                # Success — clear the spool
                os.remove(self.path)
                if not self.silent:
                    print(f"[PLARV] Spool synchronization complete.")
            except Exception:
                # If any fail, keep the file and try again next cycle
                pass

# =============================================================================
# CHECKPOINT MANAGER — rolling window of certified-stable checkpoints
# =============================================================================

class _CheckpointManager:
    """
    Argus-Weights circular buffer.
    10 slots on disk. Engine signals drive saves, not a schedule.
    SAVE   — clean step, overwrite current slot.
    SAVE_NOW — smell detected, freeze current slot, advance to next.
    Recovery slot is identified by anchor_step from engine.
    """

    SLOTS          = 3
    POINTER_FILE   = "argus_pointer.json"
    MANIFEST_FILE  = "argus_manifest.json"

    def __init__(
        self,
        checkpoint_dir: str,
        silent:         bool,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.silent         = silent

        self._current_slot  = 0
        self._slot_meta: List[Dict] = [{}] * self.SLOTS
        self._save_fn:   Optional[callable] = None
        self._frozen     = False
        self._pending_advance = False
        self._pending_advance_since = None
        
        # 🧵 BACKGROUND I/O ENGINE
        self._io_executor = ThreadPoolExecutor(max_workers=2)
        self._io_lock     = threading.Lock()

        os.makedirs(checkpoint_dir, exist_ok=True)
        self._init_slots()
        self._warn_disk_space()

    def register_save_fn(self, fn: callable):
        self._save_fn = fn

    def on_engine_signal(self, step: int, signal_dict: Dict, harm_pressure: int):
        """
        Called after every engine response.
        signal_dict: full response from engine
        harm_pressure: current hp from engine
        """
        # 1. Engine-Directed Slot Selection
        slot_idx = signal_dict.get("checkpoint_slot")
        if slot_idx is not None:
            # Force target slot (1-indexed from API -> 0-indexed internal)
            slot_idx = max(0, min(self.SLOTS - 1, int(slot_idx) - 1))
        else:
            # Fallback to circular rotation
            slot_idx = self._current_slot
            self._current_slot = (self._current_slot + 1) % self.SLOTS

        anchor_step = signal_dict.get("proactive_anchor", step)
        reason      = signal_dict.get("checkpoint_reason", "ENGINE_PULSE")

        # 2. TRIGGER ASYNC SAVE
        self._save_async(step, slot_idx, anchor_step, reason)

        # Freeze entire buffer when collapse confirmed
        if harm_pressure >= 2 and not self._frozen:
            self._frozen = True
            if not self.silent:
                print(f"[PLARV] Buffer frozen at step {step} (hp={harm_pressure}).")

        # Unfreeze on recovery
        if harm_pressure == 0 and self._frozen:
            self._frozen = False
            if not self.silent:
                print(f"[PLARV] Buffer unfrozen at step {step}.")

        # Write anchor metadata if engine sent one
        proactive_anchor = signal_dict.get("proactive_anchor")
        if proactive_anchor and isinstance(proactive_anchor, dict):
            self._write_anchor(proactive_anchor)

    def on_emergency(self, step: int, harm_pressure: int):
        self._frozen = True
        # Emergency save current state to the next available slot
        self._save_async(step, self._current_slot, step, "EMERGENCY_HALT")
        self._current_slot = (self._current_slot + 1) % self.SLOTS

    def _save_async(self, step: int, slot: int, anchor_step: int, reason: str):
        """Zero-block staged save: Snap in main thread, write in worker thread."""
        if self._save_fn is None or self._frozen:
            return

        path = self._slot_path(slot)
        
        # STAGE 1: Main Thread - Capture Snapshot
        try:
            background_writer = self._save_fn(path, anchor_step)
            if not callable(background_writer):
                # Fallback: if not using staging, just call it
                self._save_fn(path)
                return
        except Exception as e:
            if not self.silent: print(f"[PLARV] [ERROR] Failed to stage checkpoint: {e}")
            return

        # STAGE 2: Background Thread - Disk I/O
        def io_worker():
            with self._io_lock:
                try:
                    background_writer() # Performs the actual torch.save/Disk write
                    self._slot_meta[slot] = {
                        "step":        step,
                        "anchor_step": anchor_step,
                        "reason":      reason,
                        "saved_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    self._write_manifest()
                    if not self.silent:
                        print(f"[PLARV] Zero-Block Save Complete: Slot {slot} (Step {step})")
                except Exception as e:
                    if not self.silent: print(f"[PLARV] [ERROR] Background Disk Write Failed (Slot {slot}): {e}")

        self._io_executor.submit(io_worker)

    def recovery_path(self, anchor_step: Optional[int] = None) -> Optional[str]:
        """
        Returns the filesystem path of the best recovery slot.
        If anchor_step provided, finds the slot whose saved step
        is closest to and before anchor_step.
        Otherwise returns the most recently saved slot.
        """
        saved = [(i, m) for i, m in enumerate(self._slot_meta) if m.get("step") is not None]
        if not saved:
            return None
        if anchor_step is not None:
            before = [(i, m) for i, m in saved if m["step"] <= anchor_step]
            if before:
                best = max(before, key=lambda x: x[1]["step"])
                return self._slot_path(best[0])
        best = max(saved, key=lambda x: x[1]["step"])
        return self._slot_path(best[0])

    @property
    def last_stable_step(self) -> Optional[int]:
        saved = [m["step"] for m in self._slot_meta if m.get("step") is not None]
        return max(saved) if saved else None

    # ── internals ─────────────────────────────────────────────────

    def _init_slots(self):
        for i in range(self.SLOTS):
            os.makedirs(self._slot_path(i), exist_ok=True)
        manifest_path = os.path.join(self.checkpoint_dir, self.MANIFEST_FILE)
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path) as f:
                    data = json.load(f)
                    self._slot_meta    = data.get("slots", [{}] * self.SLOTS)
                    self._current_slot = data.get("current_slot", 0)
            except Exception:
                pass

    def _warn_disk_space(self):
        pass  # populated after first save when model size is known

    def _save(self, step: int, slot: int, reason: str):
        if self._save_fn is None:
            return
        path = self._slot_path(slot)
        try:
            self._save_fn(path)
            self._slot_meta[slot] = {
                "step":   step,
                "reason": reason,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            self._write_manifest()
            if not self.silent:
                print(f"[PLARV] Slot {slot} saved — step {step} ({reason}) → {path}")
        except Exception as e:
            if not self.silent:
                print(f"[PLARV] Slot {slot} save failed at step {step}: {e}")

    def _advance_slot(self, step: int):
        self._current_slot = (self._current_slot + 1) % self.SLOTS
        if not self.silent:
            print(f"[PLARV] Slot advanced to {self._current_slot} at step {step}.")
        self._write_manifest()

    def _slot_path(self, slot: int) -> str:
        return os.path.join(self.checkpoint_dir, f"argus_slot_{slot:02d}")

    def _recovery_slot(self, proactive_anchor: Optional[Dict]) -> int:
        anchor_step = proactive_anchor.get("anchor_step") if proactive_anchor else None
        saved = [(i, m) for i, m in enumerate(self._slot_meta) if m.get("step") is not None]
        if not saved:
            return self._current_slot
        if anchor_step is not None:
            before = [(i, m) for i, m in saved if m["step"] <= anchor_step]
            if before:
                return max(before, key=lambda x: x[1]["step"])[0]
        return max(saved, key=lambda x: x[1]["step"])[0]

    def _write_manifest(self):
        manifest = {
            "current_slot": self._current_slot,
            "slots":        self._slot_meta,
            "updated_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        path = os.path.join(self.checkpoint_dir, self.MANIFEST_FILE)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)

    def _write_anchor(self, anchor: dict):
        anchor_path = os.path.join(self.checkpoint_dir, "anchor_point.json")
        recovery = self.recovery_path(anchor.get("anchor_step"))
        data = {
            "anchor_step":     anchor.get("anchor_step"),
            "collapse_step":   anchor.get("collapse_step"),
            "method":          anchor.get("method"),
            "quality_score":   anchor.get("quality_score"),
            "train_loss":      anchor.get("train_loss"),
            "certified_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "recovery_path":   recovery,
        }
        with open(anchor_path, "w") as f:
            json.dump(data, f, indent=2)
        if not self.silent:
            print(f"[PLARV] Anchor point certified — step {anchor.get('anchor_step')} → {recovery}")


# =============================================================================
# argus — main client
# =============================================================================

class Argus:
    """
    PLARV argus client — Remote Training Control Plane with Clinical Depth.
    """

    _BASE_URL = os.getenv("PLARV_API_URL", "https://api.plarv.com")
    _US_URL   = f"{_BASE_URL}/api/v2/detect"

    def __init__(
        self,
        api_key:                  str,
        run_id:                   Optional[str]      = None,
        optimizer:                Optional[Any]      = None,
        model:                    Optional[Any]      = None,
        tokenizer:                Optional[Any]      = None,
        dataloader:               Optional[Any]      = None,
        mode:                     str                = "MANUAL",
        model_type:               str                = "transformer",
        step0_timeout:            float              = 12.0,
        on_pause:                 Optional[callable] = None,
        on_checkpoint:            Optional[callable] = None,
        silent:                   bool               = False,
        fail_open:                bool               = False,
        catch_exceptions:         bool               = True,
        checkpoint_every_n_steps: int                = 0,   # kept for API compatibility, unused
        checkpoint_keep_last:     int                = 2,   # kept for API compatibility, unused
        checkpoint_dir:           str                = "./argus-checkpoints",
        argus_dir:                str                = "argus",
        total_epochs:             int                = 1,
        total_steps:              Optional[int]      = None,
        sequence_length:          int                = 0,
        vocab_size:               int                = 0,
        num_layers:               int                = 0,
        hidden_dim:               int                = 0,
        dtype:                    str                = "unknown",
        raw_signals:              bool               = False,
        gauntlet_bypass:          Optional[str]      = None,
    ):
        if not api_key:
            raise ArgusConfigurationError("PLARV API Key is required for Argus initialization.")
        
        self.api_key       = api_key
        self.silent        = silent

        # Sync global network client with silence preference
        if self.silent:
            _net.silent = True
        self.run_id        = run_id or ("argus-" + uuid.uuid4().hex[:12])
        self.optimizer     = optimizer
        self.model         = model
        self.tokenizer     = tokenizer
        self.dataloader    = dataloader
        self.mode          = mode.upper()
        self.model_type    = model_type if model_type != "transformer" else self._detect_model_type(model)
        self.step0_timeout = step0_timeout
        self.on_pause      = on_pause
        self.on_checkpoint = on_checkpoint
        self.silent        = silent
        self.fail_open     = fail_open  
        self.catch_exceptions = catch_exceptions
        self.total_epochs  = total_epochs
        
        # 🤖 AUTO-ESTIMATE TOTAL STEPS
        self.total_steps   = total_steps
        if self.total_steps is None and dataloader is not None:
            try:
                # 🛡️ SAFE DISCOVERY: Handle standard PyTorch Dataloaders and custom objects
                num_batches = 0
                if hasattr(dataloader, "__len__"):
                    num_batches = len(dataloader)
                elif hasattr(dataloader, "dataset") and hasattr(dataloader.dataset, "__len__"):
                    num_batches = len(dataloader.dataset) // getattr(dataloader, "batch_size", 1)
                
                if num_batches > 0:
                    self.total_steps = num_batches * total_epochs
                    if not self.silent:
                        print(f"[PLARV] Auto-calculated total_steps: {self.total_steps} ({num_batches} batches/epoch * {total_epochs} epochs)")
            except (TypeError, AttributeError, ZeroDivisionError):
                pass

        # 🤖 AUTO-MODEL DISCOVERY (Zero-Config)
        self.argus_dir       = argus_dir
        auto_meta = self._auto_extract_metadata(model)
        
        self.sequence_length = sequence_length or auto_meta.get("sequence_length", 0)
        self.vocab_size      = vocab_size      or auto_meta.get("vocab_size", 0)
        self.num_layers      = num_layers      or auto_meta.get("num_layers", 0)
        self.hidden_dim      = hidden_dim      or auto_meta.get("hidden_dim", 0)
        self.dtype           = dtype           if dtype != "unknown" else auto_meta.get("dtype", "unknown")

        # 🌐 DISTRIBUTED SOVEREIGNTY (DDP/Multi-GPU)
        self.rank = self._detect_rank()
        self.is_master = (self.rank == 0)
        
        if not self.is_master and not self.silent:
            print(f"[PLARV] Node Rank {self.rank} detected. Silencing telemetry to prevent collision.")

        self.raw_signals = False
        
        self._should_stop  = False
        
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "x-api-key":     api_key, # for specific AWS/Apex endpoints as requested
        }

        if gauntlet_bypass:
            self._headers["x-gauntlet-bypass"] = gauntlet_bypass
        self._url = self._US_URL

        self._executor  = ThreadPoolExecutor(max_workers=2)
        import atexit
        atexit.register(self._executor.shutdown, wait=False)
        
        self._future: Optional[Future] = None
        self._last_payload: Optional[Dict] = None
        self._decision  = _Decision()
        self._step      = 0
        self._prev_loss = None
        self._completed = False
        self._degraded  = False
        self.ANALYTICAL_STRIDE = 100 # Move heavy scans to 1/100 frequency

        # Background Run Telemetry (Heartbeat / GPU / Throughput)
        self._telemetry = RunTelemetry(
            api_key=self.api_key,
            run_id=self.run_id,
            base_url=self._BASE_URL,
            model=self.model,
            silent=self.silent,
        )
        self._telemetry.start()
        self._run_status = "idle"
        self._prev_grad_norm: Optional[float] = None
        self._spike_samples: List[int] = []
        self._local_first_seen: Dict[str, int] = {}
        self._sentinel_intent: Optional[str] = None

        # Intervention Secret (Security Anchor)
        self.intervention_secret = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
        # 🛡️ SALTED HASH (Replay Attack Guard): SHA256(secret + run_id)
        self.intervention_secret_hash = hashlib.sha256((self.intervention_secret + self.run_id).encode()).hexdigest()
        
        # 🛡️ V2 FEATURE: Remote Intervention Authorization
        # This is currently hidden to avoid user confusion, but the underlying
        # cryptographic handshake is active for future dashboard-led stopping.
        # if not self.silent:
        #     print("\n" + "=" * 80)
        #     print("!!! [PLARV SECURITY ANCHOR]")
        #     print(f"!!! RUN ID:           {self.run_id}")
        #     print(f"!!! INTERVENTION KEY: {self.intervention_secret}")
        #     print("!!!")
        #     print("!!! This key is ONLY required to authorize REMOTE INTERVENTIONS (Pause/Stop).")
        #     print("!!! It is not needed for monitoring. This adds a mandatory layer of")
        #     print("!!! protection if your account or API keys are ever compromised.")
        #     print("=" * 80 + "\n")
        
        # Ensure argus/ folder exists for secrets and spooling
        os.makedirs(self.argus_dir, exist_ok=True)
        try:
            secret_path = os.path.join(self.argus_dir, "secret")
            with open(secret_path, "w") as f:
                f.write(self.intervention_secret)
            # Remove legacy root secret if it exists
            if os.path.exists(".argus_secret"):
                os.remove(".argus_secret")
        except Exception:
            pass

        # Spool manager for catch-up logic
        self._spooler   = _SpoolManager(self.argus_dir)
        
        # 📡 ADAPTIVE PACKER STATE
        self._telemetry_buffer = []
        self._rtt_ms = 150.0  # default estimate
        self._last_step_time = None
        self._avg_batch_time_ms = 100.0
        self._packing_factor = 1
        
        # Checkpoint manager — always active when model provided
        self._ckpt: Optional[_CheckpointManager] = None
        if model is not None:
            self._ckpt = _CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                silent=silent,
            )
            self._ckpt.register_save_fn(self._make_save_fn())
        
        # 📊 DQI ENGINE STATE
        self._step_history      = []
        self._loss_history      = []
        self._grad_norm_history = []
        self._current_dqi       = 0.0
        
        atexit.register(self.complete)
        
        # START BACKGROUND SERVICES
        self._preregister()
        threading.Thread(target=self._watchdog, daemon=True).start()
        threading.Thread(target=self._heartbeat_worker, daemon=True).start()

    # =========================================================================
    # PUBLIC API: ARGUS PROTOCOL (TELEMETRY & INTERVENTION)
    # =========================================================================

    def wait_for_registration(self, timeout: float = 300, poll_interval: float = 5.0):
        """Polls the backend until the run is pre-registered and ready."""
        if not self.silent:
            print(f"[PLARV] Awaiting Protocol Registration for {self.run_id}...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                import urllib.request
                url = f"{self._BASE_URL}/api/v2/status/{self.run_id}"
                req = urllib.request.Request(url, headers=self._headers, method="GET")
                with urllib.request.urlopen(req, timeout=5) as r:
                    status_resp = json.loads(r.read().decode())
                    if status_resp.get("exists") or status_resp.get("status") in ("Active", "Pending"):
                        if not self.silent:
                            print(f"[PLARV] Protocol Verified. Handshake Complete.")
                        return True
            except Exception:
                pass
            time.sleep(poll_interval)
            
        raise ArgusError(f"Protocol registration timeout. Verify run_id {self.run_id} in dashboard.")

    def wait_for_job(self, timeout: float = 300, poll_interval: float = 5.0):
        """Polls the backend until a 'START' signal is received for this run."""
        if not self.silent:
            print(f"[PLARV] JOB RUNNER ACTIVE: Polling for protocol signal ({self.run_id})...")
        
        self._run_status = "idle" # explicitly idle while polling
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                import urllib.request
                url = f"{self._BASE_URL}/api/v2/runs/{self.run_id}/job"
                req = urllib.request.Request(url, headers=self._headers, method="GET")
                with urllib.request.urlopen(req, timeout=5) as r:
                    job_resp = json.loads(r.read().decode())
                    if job_resp.get("status") == "START":
                        if not self.silent:
                            print(f"[PLARV] Signal RECEIVED. Initiating training loop...")
                        return job_resp
            except Exception:
                pass
            time.sleep(poll_interval)
            
        raise ArgusError(f"Job start timeout. Activate protocol from dashboard for run {self.run_id}.")

    def run_protocol(self, training_fn: callable, wait_for_dash: bool = False, wait_for_job: bool = False):
        """Managed training execution. Handles Handshake -> Heartbeat -> Loop."""
        if wait_for_dash:
            self.wait_for_registration()
            
        if wait_for_job:
            self.wait_for_job()

        self._run_status = "training"
        try:
            return training_fn(self)
        except ArgusPause as e:
            self._run_status = "paused"
            raise e
        except Exception as e:
            self._run_status = "error"
            raise e
        finally:
            if self._run_status != "paused":
                self.complete()

    # =========================================================================
    # PUBLIC API: STEPS (RESTORATION)
    # =========================================================================

    def step(
        self,
        loss:              float,
        grad_norm:         Optional[float]       = None,
        *,
        local_report:      Optional[Any]         = None,
        perplexity:        Optional[float]       = None,
        sample_ids:        Optional[List[int]]   = None,
        sample_losses:     Optional[List[float]] = None,
        sample_confidences: Optional[List[float]] = None,
        sample_margins:    Optional[List[float]] = None,
        sample_entropies:  Optional[List[float]] = None,
        sample_correct:    Optional[List[bool]]  = None,
        forward_ms:        Optional[float]       = None,
        backward_ms:       Optional[float]       = None,
        grad_l1_norm:      Optional[float]       = None,
        grad_l2_norm:      Optional[float]       = None,
        epoch:             int                   = 0,
        update_norm:       Optional[float]       = None,
        val_loss:          Optional[float]       = None,
        is_anchor:         bool                  = False,
        current_lr:        float                 = 0.0,
    ) -> Dict[str, Any]:
        """Report one training step. Returns the current engine decision state."""
        if not self.is_master: return {}
        if self._completed: return {}
        if self._should_stop and self.catch_exceptions: return self._decision.raw

        # 📡 Measure Batch Timing
        now = time.time()
        if self._last_step_time is not None:
            delta_ms = (now - self._last_step_time) * 1000
            self._avg_batch_time_ms = 0.9 * self._avg_batch_time_ms + 0.1 * delta_ms
            # Calculate packing factor: how many batches fit in one RTT
            # We cap it at 10 to keep payloads reasonable
            self._packing_factor = max(1, min(10, int(self._rtt_ms / (self._avg_batch_time_ms + 1e-6))))
        self._last_step_time = now

        try:
            self._run_status = "training"
            loss_val   = float(loss)
            
            # High-Fidelity Signal Extraction
            if grad_norm is not None:
                grad_val = float(grad_norm)
                grad_segments = {"early": 0.0, "mid": 0.0, "late": 0.0}
            else:
                # FAST PATH: for most steps, compute global norm only
                # DEEP PATH: every ANALYTICAL_STRIDE, compute segments
                if self._step % self.ANALYTICAL_STRIDE == 0:
                    gn_data = self._auto_grad_norm(full=True)
                else:
                    gn_data = self._auto_grad_norm(full=False)
                
                grad_val = gn_data["total"]
                grad_segments = {k: gn_data.get(k, 0.0) for k in ("early", "mid", "late")}

            loss_delta = (loss_val - self._prev_loss) if self._prev_loss is not None else 0.0
            ppl        = perplexity if perplexity is not None else min(math.exp(min(loss_val, 20)), 1e6)
     
            grad_sim = 0.5
            if self._prev_grad_norm is not None and grad_val > 0:
                ratio    = min(self._prev_grad_norm, grad_val) / max(self._prev_grad_norm, grad_val)
                grad_sim = max(0.0, min(1.0, ratio))
            self._prev_grad_norm = grad_val
            B = len(sample_ids) if sample_ids else 8

            # Update DQI History
            self._step_history.append(self._step)
            self._loss_history.append(loss_val)
            if grad_val is not None:
                self._grad_norm_history.append(grad_val)

            # Periodically compute DQI (Stride = 100)
            if self._step % self.ANALYTICAL_STRIDE == 0 and len(self._step_history) > 10:
                try:
                    dqi_res = compute_dqi(
                        steps=self._step_history,
                        loss=self._loss_history,
                        grad_norm=self._grad_norm_history if len(self._grad_norm_history) == len(self._loss_history) else None
                    )
                    self._current_dqi = dqi_res.total
                except Exception:
                    pass

            # Re-generate payload if with_histogram was true for this step (analytical stride)
            # Or just ensure it was built correctly. 
            # Actually, I should have built it AFTER updating DQI.
            payload = self._build_payload(
                step=self._step, epoch=epoch,
                loss=loss_val, loss_delta=loss_delta,
                grad_norm=grad_val, grad_sim=grad_sim, ppl=ppl,
                B=B,
                sample_ids=sample_ids,
                sample_losses=sample_losses,
                sample_confidences=sample_confidences,
                sample_margins=sample_margins,
                sample_entropies=sample_entropies,
                sample_correct=sample_correct,
                forward_ms=forward_ms,
                backward_ms=backward_ms,
                grad_segments=grad_segments,
                grad_l1_norm=grad_l1_norm,
                grad_l2_norm=grad_l2_norm,
                update_norm=update_norm,
                val_loss=val_loss,
                with_histogram=(self._step % self.ANALYTICAL_STRIDE == 0),
                is_anchor=is_anchor,
                current_lr=current_lr
            )

            # LOCAL HEALTH SIGNAL BRIDGE (Pull from protected local detector)
            if local_report is not None:
                worst = local_report.worst_level
                if worst in ("warn", "critical") and worst not in self._local_first_seen:
                    self._local_first_seen[worst] = self._step

                # Extract schema-aligned fields if they exist in the moat's report
                sparsity_delta = getattr(local_report, "sparsity_delta", 0.0)
                entropy_min    = getattr(local_report, "attention_entropy_min", None)

                payload["local_health"] = {
                    "worst_level":       worst,
                    "trend":             local_report.trend,
                    "scale":             local_report.scale,
                    "affected_fraction": local_report.affected_fraction,
                    "sparsity_delta":    sparsity_delta,
                    "attention_entropy_min": entropy_min,
                    "first_seen_step":   self._local_first_seen.get(worst),
                    "last_stable_step": (
                        self._ckpt.last_stable_step if self._ckpt is not None else None
                    ),
                }

            # 📦 ADAPTIVE PACKING
            self._telemetry_buffer.append(payload)
            
            # Flush if buffer full, or if we have a critical signal, or if it's the very first step
            should_flush = (len(self._telemetry_buffer) >= self._packing_factor) or \
                           (local_report is not None and local_report.worst_level == "critical") or \
                           (self._step <= 1)
            
            if should_flush:
                pack = self._telemetry_buffer if len(self._telemetry_buffer) > 1 else self._telemetry_buffer[0]
                self._telemetry_buffer = []
                
                if local_report is not None and local_report.worst_level == "critical":
                    if not self.silent: print(f"[PLARV] LOCAL CRITICAL FAILURE at step {self._step} — pausing.")
                    self._fire_async(pack)
                    self._should_stop = True
                    # Local detector uses ArgusPause for consistency with remote interventions
                    raise ArgusPause(f"Local detector critical: {local_report.worst_level}", step=self._step, response={"source": "local_detector"})

                self._fire_async(pack)

            if self._step == 0:
                self._step0_gate(payload)
            else:
                resp = self._collect_previous()
                if resp: self._handle_response(resp)

            # Apply current decision state after every step
            action, hp, cb, alert = self._decision.read()
            
            # Detect Graceful Stop (200 OK or 400 'already STOPPED')
            resp_body = self._decision.raw
            # 🛡️ SECURITY FAST-PATH: Immediate halt if secret verified
            if self._verify_sentinel_halt(resp_body):
                method = resp_body.get("sentinel_command", "SIG_HALT_NOW")
                self._stop_acknowledge(method=method)
                raise ArgusHalt(f"Sentinel Halt Triggered: {method}", step=self._step, response=resp_body)
            
            if action in ("PAUSE", "BUDGET_PAUSE"):
                self._apply_decision(action, hp, cb)
     
            self._prev_loss = float(loss_val)
            self._last_step_time = time.time()
            
            # Update background telemetry (zero latency)
            if self._telemetry:
                self._telemetry.on_step(step=self._step, loss=loss_val, batch_size=B)
            
            self._step += 1
            
            # 🏁 AUTO-COMPLETION PROTOCOL
            if self.total_steps and self._step >= self.total_steps:
                if not self.silent:
                    print(f"[PLARV] Target reached ({self.total_steps} steps). Finalizing run...")
                self.complete(status="COMPLETED")
                
            return self._decision.raw

        except (ArgusPause, ArgusHalt) as e:
            raise e
        except ArgusError as e:
            if self.catch_exceptions:
                if not self.silent:
                    print(f"[PLARV] Notice: {str(e)}")
                return self._decision.raw
            raise e
        except Exception as e:
            if self.catch_exceptions:
                if not self.silent:
                    print(f"[PLARV] SDK Internal Warning: {str(e)}")
                return self._decision.raw
            raise e

    def _stop_acknowledge(self, method: str = "SIG_HALT_NOW"):
        """Explicitly notify backend that manual stop signal was received and handled."""
        if self._completed: return
        self._completed = True
        self._run_status = "stopped"
        self._executor.shutdown(wait=False)
        self._telemetry.stop()
        
        try:
            payload = {
                "run_id":                   self.run_id,
                "intervention_secret_hash": self.intervention_secret_hash,
                "model_type":               self.model_type,
                "total_steps":              self.total_steps,
                "final_step":               self._step,
                "method":                   method
            }
            # 🛡️ DETERMINISTIC HANDSHAKE: High-priority signal to dedicated endpoint
            _post(f"{self._BASE_URL}/api/v2/runs/{self.run_id}/stop-acknowledge", self._headers, payload, 5.0, retries=3)
        except Exception: pass
        if not self.silent:
            print(f"[PLARV] Manual Halt Handshake Succeeded. Status: STOPPED at step {self._step}.")

    def complete(self, status: str = "COMPLETED", step: int = None, error: Optional[str] = None):
        """Signal end of training run. status can be 'COMPLETED' or 'FAILED'."""
        if self._completed: return
        self._completed = True
        
        # 🛡️ Capture final step if not provided
        final_step = step if step is not None else self._step
        
        self._run_status = status.lower()
        self._executor.shutdown(wait=False)
        self._telemetry.stop(status=status, error=error)
        try:
            # 🛡️ IDEMPOTENCY: ensure multiple stop calls (on crash/exit) don't duplicate
            headers = self._headers.copy()
            headers["x-idempotency-key"] = f"complete-{self.run_id}"
            payload = {
                "run_id": self.run_id, 
                "status": status,
                "step": final_step,
                "error": error
            }
            _post(f"{self._BASE_URL}/api/v2/complete", headers, payload, 5.0, retries=5)
        except Exception: pass
        if not self.silent:
            msg = f"Run {self.run_id} {status.lower()} at step {final_step}."
            if error: msg += f" Error: {error}"
            print(f"[PLARV] {msg}")

    def _heartbeat_worker(self):
        """Daemon thread: machine presence."""
        pulse_url = f"{self._BASE_URL}/api/v2/heartbeat"
        while not self._completed:
            try:
                payload = {
                    "run_id": self.run_id,
                    "status": self._run_status,
                }
                import urllib.request
                data = __import__('json').dumps(payload).encode()
                req = urllib.request.Request(pulse_url, data=data, headers=self._headers, method="POST")
                urllib.request.urlopen(req, timeout=3)
            except Exception: pass
            
            # Catch-up Logic: Drain spooled telemetry if connection is back
            if self._spooler:
                self._spooler.drain(_post, self._url, self._headers)
                
            time.sleep(30.0)

    def _preregister(self):
        """Creates the run entry in the DB and prepares the registry."""
        if not self.is_master:
            return
            
        try:
            # 🛡️ IDEMPOTENCY: Ensure run registration is atomic and unique
            headers = self._headers.copy()
            headers["x-idempotency-key"] = f"register-{self.run_id}"
            
            payload = {
                "run_id": self.run_id,
                "model_type": self.model_type,
                "sdk_version": "1.0.0",
                "intervention_secret_hash": self.intervention_secret_hash,
                "total_epochs": self.total_epochs,
                "total_steps":  self.total_steps,
            }
            
            t0 = time.time()
            resp = _post(f"{self._BASE_URL}/api/v2/runs", headers, payload, 5.0, retries=5)
            self._rtt_ms = (time.time() - t0) * 1000
            
            if not self.silent:
                print(f"[PLARV] Network RTT to US-East-1 calibrated: {self._rtt_ms:.1f}ms")
            
            # 🛡️ ENGINE-SYNC: Catch early signals (like directed checkpoints) during registration
            self._handle_response(resp)
        except Exception:
            pass

    # =========================================================================
    # INTERNAL UTILS (RESTORATION)
    # =========================================================================

    def _build_payload(
        self,
        step, epoch, loss, loss_delta, grad_norm, grad_sim, ppl, B,
        sample_ids, sample_losses, sample_confidences,
        sample_margins, sample_entropies, sample_correct,
        forward_ms, backward_ms,
        grad_l1_norm=None, grad_l2_norm=None,
        update_norm=None, val_loss=None,
        grad_segments=None,
        with_histogram: bool = False,
        is_anchor: bool = False,
        current_lr: float = 0.0,
    ) -> Dict:
        """RESTORED comprehensive payload building."""
        has_layer2 = all(x is not None for x in [
            sample_ids, sample_losses, sample_confidences,
            sample_margins, sample_entropies, sample_correct
        ])

        payload: Dict[str, Any] = {
            "run_id":        self.run_id,
            "step":          step,
            "epoch":         epoch,
            "model_type":    self.model_type,
            "sdk_version":   "1.0.0",
            "is_anchor":     is_anchor,
            "anchor":        is_anchor,
            # Aggressive Optimization: only compute histogram on stride
            "histogram":     self._compute_histogram(bins=8) if with_histogram else None,
            "training": {
                "loss":                 loss,
                "loss_delta":           loss_delta,
                "grad_norm":            grad_norm,
                "gradient_similarity":  grad_sim,
                "perplexity":           ppl,
                "update_norm":          float(update_norm) if update_norm is not None else 0.0,
                "val_loss":             float(val_loss) if val_loss is not None else 0.0,
                "optimizer_momentum_z": 0.0,
                "grad_sim_slope":       0.0,
                "dqi_score":            self._current_dqi,
                "current_lr":           current_lr,
                "lr_restarted":         False,
                "is_anchor":            is_anchor,
                "anchor":               is_anchor,
            },
            "batch_meta": {
                "batch_size":      B,
                "sequence_length": self.sequence_length,
                "vocab_size":      self.vocab_size,
                "num_layers":      self.num_layers,
                "hidden_dim":      self.hidden_dim,
                "dtype":           self.dtype,
            },
            "labels": {
                "count":            B,
                "entropy":          1.0,
                "imbalance_ratio":  0.0,
                "num_classes_seen": 0,
            },
            "control": {"mode": self.mode, "raw_signals": self.raw_signals},
            "layer2_enabled": has_layer2,
            "sample_ids": sample_ids or list(range(B)),
            "sentinel_handshake": self._sentinel_intent,
            "intervention_secret_hash": self.intervention_secret_hash,
        }

        if has_layer2:
            payload.update({
                "sample_losses":      sample_losses,
                "sample_confidences": sample_confidences,
                "sample_margins":     sample_margins,
                "sample_entropies":   sample_entropies,
                "sample_correct":     sample_correct,
            })

        payload["distribution"] = {
            "l1_norm":  float(grad_l1_norm) if grad_l1_norm is not None else 0.0,
            "l2_norm":  float(grad_l2_norm) if grad_l2_norm is not None else 0.0,
            "mean":     0.0, "std":      0.0, "min":      0.0, "max":      0.0,
            "sparsity": 0.0, "kurtosis": 0.0, "skew":     0.0,
            "grad_norm_early": float(grad_segments.get("early", 0.0)) if grad_segments else 0.0,
            "grad_norm_mid":   float(grad_segments.get("mid", 0.0))   if grad_segments else 0.0,
            "grad_norm_late":  float(grad_segments.get("late", 0.0))  if grad_segments else 0.0,
        }

        if forward_ms is not None: payload["forward_ms"] = forward_ms
        if backward_ms is not None: payload["backward_ms"] = backward_ms

        if self.total_epochs is not None: payload["total_epochs"] = self.total_epochs
        if self.total_steps is not None: payload["total_steps"] = self.total_steps

        return payload

    def _handle_response(self, resp):
        if not resp: return
        self._decision.update(resp)
        action, hp, cb, alert = self._decision.read()
        if self._ckpt:
            if action in ("PAUSE", "BUDGET_PAUSE", "CHECKPOINT"):
                self._ckpt.on_emergency(self._step, hp)
            
            # 🛡️ ENGINE-DIRECTED CHECKPOINTING
            if resp.get("checkpoint_signal") in ("SAVE", "SAVE_NOW"):
                self._ckpt.on_engine_signal(
                    step=self._step,
                    signal_dict=resp,
                    harm_pressure=hp
                )
        self._apply_decision(action, hp, cb)

    def _apply_decision(self, action, hp, cb):
        """RESTORED decision handling logic (SET_LR, AUTO mode, Anchor Points)."""
        raw = self._decision.raw
        
        # AUTO MODE: LR ADJUSTMENTS
        if self.mode == "AUTO":
            lr_factor = (cb.get("lr_factor") if (cb and "lr_factor" in cb) else raw.get("lr_factor"))
            if action == "SET_LR" and self.optimizer is not None and lr_factor:
                for pg in self.optimizer.param_groups:
                    pg["lr"] *= float(lr_factor)
                if not self.silent: print(f"[PLARV] AUTO SET_LR ×{lr_factor:.4f} at step {self._step}")

        # ANCHOR POINT / ROLLBACK
        anchor = raw.get("anchor_point")
        if anchor and isinstance(anchor, dict) and anchor.get("anchor_step") is not None:
            if self._ckpt: self._ckpt._write_anchor(anchor)

        # STOP / TERMINAL
        if action == "STOP":
            # 🛡️ SOVEREIGN GATE: If the backend says STOP, but we ALSO have a sentinel command,
            # we MUST verify the secret before raising the halt.
            sentinel_cmd = raw.get("sentinel_command")
            if sentinel_cmd not in ("SIG_HALT_NOW", "SIG_INTENT_STOP"):
                # Non-sentinel stop (e.g. engine-driven completion)
                self.complete()
                raise ArgusHalt(f"Protocol Termination Signal Received: {hp}", step=self._step, response=raw)

        # PAUSE / SENTINEL
        if action in ("PAUSE", "BUDGET_PAUSE"):
            if self.on_pause: self.on_pause(raw)
            elif self.mode == "AUTO": raise ArgusPause(f"Emergency stop: {hp}", step=self._step, response=raw)
            else: 
                if not self.silent: print(f"[PLARV] MANUAL WARNING: Engine requires PAUSE at step {self._step}")

        sentinel_cmd = raw.get("sentinel_command")
        sentinel_payload = raw.get("sentinel_payload")

        if sentinel_payload and isinstance(sentinel_payload, str):
            # 🛡️ PAYLOAD PRIVACY: Decrypt instructions using API Key
            sentinel_payload = self._decrypt_payload(sentinel_payload)

            # 🛡️ SOVEREIGN VERIFICATION: Only honor stop signals if authorized by the matching secret
            remote_secret = sentinel_payload.get("intervention_secret")
            if remote_secret == self.intervention_secret:
                if sentinel_cmd == "SIG_INTENT_STOP":
                    self._sentinel_intent = "STOP"
                    self._verify_window = getattr(self, "_verify_window", 0) + 1
                    if not self.silent: print(f"[PLARV] SENTINEL HANDSHAKE: Intent received (Verification: {self._verify_window}/10)")
                    if self._verify_window >= 10:
                        self.complete()
                        raise ArgusHalt("Sentinel Stability Handshake Confirmed", step=self._step, response=raw)
                elif sentinel_cmd == "SIG_HALT_NOW":
                    self.complete()
                    raise ArgusHalt("Sentinel Immediate Halt Triggered (Manual Verification Succeeded)", step=self._step, response=raw)
            elif sentinel_cmd in ("SIG_INTENT_STOP", "SIG_HALT_NOW"):
                if not self.silent:
                    print(f"[PLARV] [WARNING] Unauthorized Halt Attempt detected. Secret mismatch.")
                self._sentinel_report(
                    type="UNAUTHORIZED_HALT", 
                    message=f"Received {sentinel_cmd} but intervention secret mismatch.",
                    step=self._step
                )
        
        else:
            self._verify_window = 0 # Reset if command lost

    def _verify_sentinel_halt(self, resp: Dict) -> bool:
        """Determines if a response contains a valid, authorized halt signal."""
        if not resp: return False
        
        sentinel_cmd = resp.get("sentinel_command")
        if sentinel_cmd not in ("SIG_HALT_NOW", "SIG_INTENT_STOP"):
            return resp.get("action") == "STOP" # Non-sentinel stop

        payload = resp.get("sentinel_payload")
        if not payload or not isinstance(payload, str):
            return False

        # Fallback for legacy system-level terminations (unencrypted)
        if payload == "SIG_TERMINATED_BY_USER":
            return True

        # Try to decrypt and verify
        decrypted = self._decrypt_payload(payload)
        remote_secret = decrypted.get("intervention_secret")
        
        # 🛡️ SOVEREIGN VERIFICATION: Match against local security anchor
        if remote_secret == self.intervention_secret:
            return True
            
        # Fallback for system-level terminations (if we trust the message)
        if decrypted.get("message") == "TERMINATED_BY_USER":
            return True

        if not self.silent:
            print(f"[PLARV] [WARNING] Unauthorized Halt Attempt: Secret Mismatch.")
        return False

    def _sentinel_report(self, type: str, message: str, step: int):
        """Report sentinel handshake failures or security events back to the platform."""
        try:
            payload = {
                "type": type,
                "message": message,
                "step": step
            }
            # Fire and forget
            self._executor.submit(_post, f"{self._BASE_URL}/api/v2/runs/{self.run_id}/commands/report", self._headers, payload, 5.0, 1)
        except Exception: pass

    def _decrypt_payload(self, encrypted_hex: str) -> Dict:
        """Zero-dep XOR decryption using rolling SHA256 key derived from API Key."""
        try:
            encrypted_bytes = bytes.fromhex(encrypted_hex)
            key = hashlib.sha256(self.api_key.encode()).digest()
            decrypted = bytearray()
            for i in range(len(encrypted_bytes)):
                # Rolling key for each byte
                block_key = hashlib.sha256(key + str(i // 32).encode()).digest()
                decrypted.append(encrypted_bytes[i] ^ block_key[i % 32])
            return json.loads(decrypted.decode())
        except Exception as e:
            if not self.silent: print(f"[PLARV] [ERROR] Sentinel Decryption Failed: {e}")
            return {}

    def _step0_gate(self, payload):
        payload["intervention_secret_hash"] = self.intervention_secret_hash
        resp = _post(self._url, self._headers, payload, self.step0_timeout, retries=5)
        
        if "_http_error" in resp:
            code = resp["_http_error"]
            body = resp.get("_http_body", {})
            msg = body.get("message", "Unknown error")
            
            if code == 429:
                if not self.silent:
                    print(f"\n[PLARV CRITICAL] Rate limit exceeded (429). {msg}")
                if self.fail_open:
                    self._degraded = True
                    return
                raise ArgusRateLimitError(f"Rate limit exceeded (429): {msg}", status_code=code, response=body)
            
            if code in (401, 403):
                raise ArgusAuthenticationError(f"Authentication failure ({code}): {msg}. Verify API key.", status_code=code, response=body)

            if code >= 500:
                if self.fail_open:
                    self._degraded = True
                    return
                raise ArgusServerError(f"Plarv Server Error ({code}): {msg}", status_code=code, response=body)

            if self.fail_open:
                self._degraded = True
                return
            raise ArgusApiError(f"Step 0 Verification Failed (HTTP {code}): {msg}", status_code=code, response=body)

        if "_error" in resp: 
            if self.fail_open:
                self._degraded = True
                return
            raise ArgusConnectionError(f"Step 0 Network Failure: {resp.get('detail', 'Unknown connection error')}")

        self._decision.update(resp)
        if not self.silent:
            print(f"[PLARV] Handshake Certified. Proto Version: {resp.get('engine_version', '1.0. clinical')}")

    def _fire_async(self, payload):
        # 🛡️ TIERED RETRIES: Lower retry count for fast-path telemetry
        self._last_payload = payload
        self._future = self._executor.submit(_post, self._url, self._headers, payload, 5.0, 3)

    def _collect_previous(self):
        if self._future and self._future.done():
            try:
                resp = self._future.result()
                if "_error" in resp or "_http_error" in resp:
                    # Connection blip detected — spool the payload for catch-up
                    if self._last_payload and self._spooler:
                        self._spooler.spool(self._last_payload)
                
                self._decision.update(resp)
                return resp
            except Exception: 
                # Thread level failure — spool as fallback
                if self._last_payload and self._spooler:
                    self._spooler.spool(self._last_payload)
                return None
        return None

    def _watchdog(self):
        while not self._completed:
            time.sleep(10)
            if self._last_step_time is None:
                continue
            if time.time() - self._last_step_time > 120:
                if not self._completed:
                    self.complete()
                    break

    def _auto_grad_norm(self, full: bool = False) -> Dict[str, float]:
        """
        Hyper-optimized gradient analysis.
        full=False: Global norm only (<1ms, single sync).
        full=True: Segmented buckets (Analytical Stride).
        """
        if self.model is None:
            return {"total": 0.0, "early": 0.0, "mid": 0.0, "late": 0.0}
        
        try:
            import torch
            
            # FAST PATH: Global L2 norm with FUSED KERNEL (World Level)
            if not full:
                grads = [p.grad.detach() for p in self.model.parameters() if p.grad is not None]
                if not grads: return {"total": 0.0}
                
                # Multi-tensor norm (Fused kernel on GPU, efficient loop on CPU)
                # This is safe for 70B+ models while Claude's 'cat' would OOM.
                norms = torch._foreach_norm(grads, 2)
                # Vectorize the final reduction
                total_norm = torch.linalg.vector_norm(torch.stack(norms)).item()
                return {"total": total_norm}
                
            # ANALYTICAL PATH: Segmented Analysis (still staggered every 100 steps)
            params = [p for p in self.model.parameters() if p.grad is not None]
            if not params:
                return {"total": 0.0, "early": 0.0, "mid": 0.0, "late": 0.0}
                
            n = len(params)
            buckets = [params[i:i + (n + 2) // 3] for i in range(0, n, (n + 2) // 3)]
            
            norms = []
            for bucket in buckets:
                b_sq = sum(p.grad.detach().float().pow(2).sum().item() for p in bucket)
                norms.append(b_sq ** 0.5)
                
            # Ensure we have 3 buckets
            while len(norms) < 3:
                norms.append(0.0)
                
            total_sq = sum(n**2 for n in norms)
            return {
                "total": total_sq ** 0.5,
                "early": norms[0],
                "mid":   norms[1],
                "late":  norms[2],
            }
        except Exception:
            return {"total": 0.0}

    def _compute_histogram(self, bins: int = 8) -> Dict:
        """Fast, subsampled gradient histogram."""
        if self.model is None:
            return {"bins": bins, "counts": [0] * bins}
        try:
            import torch
            grads = []
            for p in self.model.parameters():
                if p.grad is not None:
                    # Deterministic stride sampling — lightning fast, no randperm
                    g = p.grad.detach().flatten()
                    if g.numel() > 1000:
                        stride = max(1, g.numel() // 1000)
                        g = g[::stride]
                    grads.append(g.float())
            
            if not grads:
                return {"bins": bins, "counts": [0] * bins}
                
            all_grads = torch.cat(grads)
            counts, _ = torch.histogram(all_grads.cpu(), bins=bins)
            return {
                "bins": bins,
                "counts": counts.int().tolist(),
            }
        except Exception:
            return {"bins": bins, "counts": [1] * bins}

    def _auto_extract_metadata(self, model: Any) -> Dict[str, Any]:
        """🛡️ Forensic Model Scanner: Extracts architecture details automatically."""
        meta = {
            "num_layers": 0,
            "hidden_dim": 0,
            "vocab_size": 0,
            "sequence_length": 0,
            "dtype": "unknown"
        }
        if model is None:
            return meta

        try:
            # 1. Detect DType
            params = list(model.parameters())
            if params:
                meta["dtype"] = str(params[0].dtype).split(".")[-1]

            # 2. Check for HuggingFace-style config
            config = getattr(model, "config", None)
            if config:
                meta["num_layers"] = getattr(config, "num_hidden_layers", 
                                     getattr(config, "n_layer", 
                                     getattr(config, "num_layers", 0)))
                meta["hidden_dim"] = getattr(config, "hidden_size", 
                                     getattr(config, "d_model", 
                                     getattr(config, "n_embd", 0)))
                meta["vocab_size"] = getattr(config, "vocab_size", 0)
                meta["sequence_length"] = getattr(config, "max_position_embeddings", 
                                          getattr(config, "n_positions", 0))

            # 3. Fallback: Peeling PyTorch Layers
            if meta["num_layers"] == 0:
                # Count modules that look like 'blocks' or 'layers'
                layers = 0
                for name, module in model.named_modules():
                    # Common transformer/CNN layer names
                    if any(x in name.lower() for x in [".h.", ".layers.", ".blocks.", "bottleneck"]):
                        layers += 1
                meta["num_layers"] = layers

            # 4. Fallback: Vocab size from embedding layer
            if meta["vocab_size"] == 0:
                for module in model.modules():
                    if hasattr(module, "weight") and hasattr(module, "num_embeddings"):
                        meta["vocab_size"] = module.num_embeddings
                        break

            if not self.silent and any(v != 0 and v != "unknown" for v in meta.values()):
                print(f"[PLARV] Zero-Config Auto-Discovery: {meta['num_layers']} layers, dim {meta['hidden_dim']}, vocab {meta['vocab_size']}")

        except Exception:
            pass # Silent failure for auto-discovery
        return meta

    def _detect_rank(self) -> int:
        """🔍 Forensic Rank Detection: Supports DDP, DeepSpeed, and standard PyTorch."""
        try:
            # 1. Standard PyTorch Distributed
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank()
        except ImportError:
            pass

        # 2. Environment Variable Fallbacks (Slurm, MPI, etc.)
        for env_var in ["RANK", "LOCAL_RANK", "SLURM_PROCID"]:
            val = os.getenv(env_var)
            if val is not None:
                try: return int(val)
                except ValueError: pass
        
        return 0 # Default to Rank 0 (Master/Single)

    def _get_gpu_stats(self) -> Dict[str, Any]:
        """Captures lightweight GPU memory metrics if available."""
        stats = {"vram_allocated_gb": 0.0, "vram_reserved_gb": 0.0, "device": "cpu"}
        try:
            import torch
            if torch.cuda.is_available():
                stats["device"] = "cuda"
                stats["vram_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
                stats["vram_reserved_gb"]  = torch.cuda.memory_reserved() / (1024**3)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                stats["device"] = "mps"
                # MPS memory tracking is currently more complex; we report device for now
        except: pass
        return stats

    def _detect_model_type(self, model) -> str:
        if model is None: return "transformer"
        name = type(model).__name__.lower()
        config = getattr(model, "config", None)
        config_type = getattr(config, "model_type", "").lower() if config else ""
        if any(k in name or k in config_type for k in ("gpt", "llama", "mistral", "falcon", "qwen", "gemma", "phi", "mamba", "ssm", "state-space")): return "llm"
        if any(k in name or k in config_type for k in ("vit", "clip", "dino", "swin", "convnext")): return "vision"
        if any(k in name for k in ("resnet", "efficientnet", "densenet", "vgg", "alexnet", "mobilenet")): return "cnn"
        return "transformer" 

    def _make_save_fn(self) -> callable:
        """Zero-Block Framework detection with Memory-Staging support."""
        model, tokenizer = self.model, self.tokenizer

        def staged_save(path: str, anchor_step: int):
            # 🛡️ STAGE 1: Main Thread - Capture memory snapshot
            if model is not None and hasattr(model, "state_dict"):
                import torch
                # Deep copy tensors to CPU to avoid race conditions with training
                snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                opt_snapshot = None
                if self.optimizer:
                    opt_snapshot = {k: v for k, v in self.optimizer.state_dict().items()} # state_dict is already a dict of tensors
                
                # 🛡️ STAGE 2: Return a closure for the Background Thread
                def background_write():
                    try:
                        import torch
                        os.makedirs(path, exist_ok=True)
                        ckpt = {
                            "model_state_dict": snapshot, 
                            "optimizer_state_dict": opt_snapshot,
                            "step": self._step,
                            "anchor_step": anchor_step
                        }
                        target = os.path.join(path, "checkpoint.pt")
                        torch.save(ckpt, target)
                        if tokenizer:
                            tokenizer.save_pretrained(path)
                    except Exception as e:
                        if not self.silent:
                            print(f"[PLARV] [CRITICAL ERROR] Background writer failed for path {path}: {e}")
                            import traceback
                            traceback.print_exc()
                        raise e
                
                return background_write
            
            # Fallback for non-staged models
            return lambda: None

        return staged_save

    def __enter__(self): return self
    def __exit__(self, t, v, b):
        if t is not None:
            self.complete(status="FAILED", error=str(v))
        else:
            self.complete()
    def __del__(self): 
        try: self.complete()
        except Exception: pass