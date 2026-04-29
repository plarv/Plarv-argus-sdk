"""
PLARV Argus — Forensic Utilities
================================
Lightweight helpers for signal extraction, hardware probing, and model discovery.
"""

import math
import time
from typing import Dict, List, Any, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def probe_gpu() -> Dict[str, Any]:
    """Captures lightweight GPU hardware metrics."""
    stats = {"vram_allocated_gb": 0.0, "vram_reserved_gb": 0.0, "device": "cpu"}
    if not HAS_TORCH:
        return stats
    
    try:
        if torch.cuda.is_available():
            stats["device"] = "cuda"
            stats["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / (1024**3), 3)
            stats["vram_reserved_gb"]  = round(torch.cuda.memory_reserved() / (1024**3), 3)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            stats["device"] = "mps"
    except: pass
    return stats

def extract_signals(logits: Any, labels: Any, model: Any = None) -> Dict[str, Any]:
    """
    Extracts forensic signals (confidence, entropy, etc.) from training batches.
    World-Level: Handles both classification (B, C) and LM (B, T, V) logits.
    """
    if not HAS_TORCH or not isinstance(logits, torch.Tensor):
        return {}

    with torch.no_grad():
        # Handle LM logits (B, T, V) -> (B*T, V)
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)

        probs = torch.softmax(logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)
        
        # Per-sample entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        # Correctness
        correct = (pred == labels)
        
        return {
            "sample_confidences": conf.tolist(),
            "sample_entropies":   entropy.tolist(),
            "sample_correct":     correct.tolist(),
        }

def auto_detect_duration(dataloader: Any, epochs: int) -> Optional[int]:
    """Automatically estimates total training steps from the dataloader."""
    try:
        return len(dataloader) * epochs
    except:
        return None