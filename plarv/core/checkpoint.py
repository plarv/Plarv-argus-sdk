"""
PLARV Argus — Forensic Checkpoint Management
============================================
Handles the 3-slot circular buffer, async weight staging, and anchor recovery.
"""

import os
import json
import time
import shutil
import threading
from typing import Optional, Any, Callable, Dict
from concurrent.futures import ThreadPoolExecutor

class _CheckpointManager:
    """Sovereign manager for engine-directed snapshots."""
    SLOTS = 3

    def __init__(self, checkpoint_dir: str, silent: bool = False):
        self.checkpoint_dir = checkpoint_dir
        self.silent = silent
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.last_stable_step = None
        self._save_fn = None
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    def register_save_fn(self, save_fn: Callable):
        self._save_fn = save_fn

    def dispatch_save(self, step: int, slot: int, is_now: bool = False):
        """Dispatches a Zero-Block save to the background thread."""
        if not self._save_fn:
            return
        
        path = os.path.join(self.checkpoint_dir, f"slot_{slot}.pt")
        self._executor.submit(self._save_async, path, step, is_now)

    def _save_async(self, path: str, step: int, is_now: bool):
        try:
            if not self.silent:
                tag = "SAVE_NOW" if is_now else "SAVE"
                print(f"[PLARV] {tag} initiated for step {step} -> {os.path.basename(path)}")
            
            # STAGE 1: Memory Staging (Handled by the provided save_fn)
            # We assume save_fn handles the blocking clone if necessary
            self._save_fn(path, step)
            
            if not is_now:
                self.last_stable_step = step
                self._update_anchor(step, path)
                
        except Exception as e:
            if not self.silent:
                print(f"[PLARV] ⚠ Checkpoint failure at step {step}: {e}")

    def _update_anchor(self, step: int, path: str):
        anchor_path = os.path.join(self.checkpoint_dir, "anchor_point.json")
        try:
            with open(anchor_path, "w") as f:
                json.dump({
                    "step": step,
                    "path": os.path.abspath(path),
                    "ts": int(time.time()),
                    "version": "2.0.0"
                }, f)
        except: pass

    def shutdown(self):
        self._executor.shutdown(wait=True)

class _SpoolManager:
    """Manages the local forensic telemetry spool for recovery."""
    def __init__(self, spool_dir: str):
        self.spool_dir = spool_dir
        if not os.path.exists(spool_dir):
            os.makedirs(spool_dir)

    def spool(self, step: int, payload: Dict):
        path = os.path.join(self.spool_dir, f"step_{step}.json")
        try:
            with open(path, "w") as f:
                json.dump(payload, f)
        except: pass

    def clear(self):
        if os.path.exists(self.spool_dir):
            shutil.rmtree(self.spool_dir)
