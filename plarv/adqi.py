
"""
Argus Data Quality Index (ADQI) — Model Agnostic
=================================================
Infers training data quality by analyzing training dynamics.
No raw data access required. Works for any model type.

Architecture:
    1. Signal Extractor  — maps any training run to (x, y)
    2. Phase Detector    — splits run into Early / Mid / Late adaptively
    3. ADQI Components   — DS, VS, SC, DU per phase
    4. ADQI Fusion       — weighted scalar per phase + total

Philosophy:
    ADQI is an instrument for indirect inference. It does not measure the dataset itself;
    it measures the manifested properties of the training trajectory and
    infers the quality of the signal source. 

    Volume is not quality. ADQI uses signal manifold properties—specifically loss span, 
    distribution, and curvature—to infer domain coverage and distribution.
    This indirect measurement allows for privacy-preserving, real-time
    quality assessment across any model modality.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Union


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class PhaseResult:
    name: str           # "early" | "mid" | "late"
    step_start: int
    step_end: int
    DS: float           # Domain Spread
    VS: float           # Variation Strength
    SC: float           # Shape Complexity
    DU: float           # Density Uniformity
    dqi: float          # Phase ADQI score


@dataclass
class ADQIResult:
    total: float                    # Final weighted ADQI [0, 1]
    early: PhaseResult
    mid: PhaseResult
    late: PhaseResult
    phase_weights: dict             # Weights used for fusion
    signal_type: str                # What y signal was used


# ============================================================
# SIGNAL EXTRACTOR — Translation Layer
# Converts training run data into (x, y) arrays
# This is what makes ADQI model agnostic
# ============================================================

def extract_signal(
    steps: np.ndarray,
    loss: Optional[np.ndarray] = None,
    grad_norm: Optional[np.ndarray] = None,
    signal_preference: str = "loss"
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Maps any training run to (x, y) for ADQI computation.

    x = training steps (universal domain axis)
    y = chosen signal (universal quality proxy)

    Priority: loss > grad_norm > steps-only fallback

    TODO (Future Task): Add a NaN-strip or NaN-interpolation step here 
    so ADQI degrades gracefully instead of propagating NaNs through the math.

    Args:
        steps:             Training step indices
        loss:              Loss values per step (any model type)
        grad_norm:         Gradient norm per step (any model type)
        signal_preference: "loss" or "grad_norm"

    Returns:
        (x, y, signal_name)
    """
    x_raw = np.array(steps, dtype=float)
    
    if signal_preference == "loss" and loss is not None:
        y_raw = np.array(loss, dtype=float)
        signal_name = "loss"
    elif signal_preference == "grad_norm" and grad_norm is not None:
        y_raw = np.array(grad_norm, dtype=float)
        signal_name = "grad_norm"
    elif loss is not None:
        y_raw = np.array(loss, dtype=float)
        signal_name = "loss"
    elif grad_norm is not None:
        y_raw = np.array(grad_norm, dtype=float)
        signal_name = "grad_norm"
    else:
        raise ValueError("At least one of loss or grad_norm must be provided.")

    # 🛡️ SOVEREIGN ROBUSTNESS: Strip NaNs and Infs to prevent mathematical collapse
    mask = np.isfinite(x_raw) & np.isfinite(y_raw)
    x = x_raw[mask]
    y = y_raw[mask]

    if len(x) < 3:
        # If too many NaNs, fallback to original if possible or raise
        if len(x_raw) >= 3:
            # Try to interpolate if we have enough raw points but some are NaN
            # For simplicity in this clinical engine, we just use what's finite.
            pass
        
        if len(x) < 3:
            # Still not enough data
            return x_raw, np.nan_to_num(y_raw), signal_name

    return x, y, signal_name


# ============================================================
# PHASE DETECTOR — Adaptive Boundary Detection
# Uses first derivative inflection to find natural phase boundaries
# No hardcoded step counts — adapts to each training run
# ============================================================

def detect_phases(x: np.ndarray, y: np.ndarray) -> tuple[int, int, np.ndarray]:
    """
    Finds two split points dividing run into Early / Mid / Late.

    Method: smoothed first derivative of y.
    Early phase ends where loss descent rate first slows significantly.
    Mid phase ends where descent flattens toward convergence.

    Returns:
        (split1, split2, dy) — split indices and the velocity signal
    """
    n = len(x)

    if n < 9:
        # Too short — equal thirds
        return n // 3, 2 * n // 3

    # Smooth y to reduce noise before derivative
    window = max(3, n // 20)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    y_padded = np.pad(y, pad, mode='edge')
    y_smooth = np.convolve(y_padded, np.ones(window) / window, mode='valid')
    y_smooth = y_smooth[:n]

    # First derivative
    dy = np.abs(np.gradient(y_smooth))

    # Cumulative change — find where 33% and 66% of total change occurred
    cumchange = np.cumsum(dy)
    total = cumchange[-1]

    # 🛡️ SOVEREIGN DYNAMIC DETECTION
    # Instead of hardcoded 40/75 splits, we find natural regime changes.
    
    # 1. split1: The "Braking Point" (Max Deceleration)
    # This is where the model finishes "feature discovery" and enters "refinement."
    # We find the peak of the second derivative (curvature) after the initial descent.
    d2y = np.gradient(dy)
    # We only look in the first 60% of the run for the first transition
    search_limit = int(n * 0.6)
    split1 = int(np.argmax(d2y[:search_limit]))
    
    # 2. split2: The "Convergence Gate" (Velocity Floor)
    # This is where the model enters the "plateau" phase.
    # We find where velocity stays below 15% of peak velocity.
    peak_vel = np.max(dy)
    vel_threshold = 0.15 * peak_vel
    
    # Find the last point where velocity was above threshold
    above_threshold = np.where(dy > vel_threshold)[0]
    if len(above_threshold) > 0:
        split2 = int(above_threshold[-1])
    else:
        split2 = int(n * 0.75) # fallback

    # 3. Refinement & Safety Bounds
    # Ensure phases are in order and have minimum sizes
    min_size = max(3, n // 10)
    
    # If split1 is too early (e.g. at step 0), push it forward
    split1 = max(min_size, split1)
    
    # Ensure split2 is at least min_size after split1
    split2 = max(split1 + min_size, split2)
    
    # Ensure split2 is not too close to the end
    split2 = min(split2, n - min_size)

    return split1, split2, dy


# ============================================================
# ADQI COMPONENTS
# ============================================================

def domain_spread_score(y: np.ndarray, y_global_span: float) -> float:
    if len(y) < 3:
        return 0.0
    if y_global_span <= 1e-12:
        return 0.0
    phase_span = float(np.max(y) - np.min(y))
    return float(np.clip(phase_span / y_global_span, 0.0, 1.0))


def variation_score(y: np.ndarray) -> float:
    """
    Signal power: meaningful variation in y.
    Low VS → flat/constant signal (no learning).
    High VS → rich variation (active learning).
    """
    if len(y) < 3:
        return 0.0
    rng = float(np.max(y) - np.min(y))
    if rng <= 1e-12:
        return 0.0
    std = float(np.std(y))
    return float(np.clip(2 * std / rng, 0.0, 1.0))


def shape_complexity_score(x: np.ndarray, y: np.ndarray) -> float:
    """
    Curvature entropy: complexity of the training trajectory shape.
    High SC → rich curve with meaningful structure.
    Low SC → straight line, no curvature information.
    """
    if len(x) < 6:
        return 0.0
    idx = np.argsort(x)
    x_s, y_s = x[idx], y[idx]
    
    # 🛡️ SOVEREIGN SMOOTHING: Add a small Low-Pass Filter (Moving Average)
    # to y_s before calculating d2. This prevents high-frequency jitter
    # from tricking the score into thinking the trajectory is "complex."
    if len(y_s) > 5:
        window = 3
        y_s = np.convolve(y_s, np.ones(window)/window, mode='same')

    dx = np.diff(x_s) + 1e-12
    d1 = np.diff(y_s) / dx
    d2 = np.diff(d1) / (np.diff(x_s[:-1]) + 1e-12)
    var_d2 = np.var(d2)
    mean_d1 = np.mean(np.abs(d1)) + 1e-12
    return float(np.clip(np.tanh(var_d2 / mean_d1), 0.0, 1.0))


def density_uniformity_score(y: np.ndarray, n_bins: int = 10) -> float:
    if len(y) < 4:
        return 0.0
    y_min, y_max = np.min(y), np.max(y)
    if (y_max - y_min) <= 1e-12:
        return 0.0
    counts, _ = np.histogram(y, bins=n_bins)
    mean_count = np.mean(counts) + 1e-12
    cv = np.std(counts) / mean_count
    return float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))


def compute_phase_dqi(
    x: np.ndarray,
    y: np.ndarray,
    name: str,
    step_start: int,
    step_end: int,
    y_global_span: float
) -> PhaseResult:
    """Compute all four ADQI components for one phase slice."""
    DS = domain_spread_score(y, y_global_span)
    VS = variation_score(y)
    SC = shape_complexity_score(x, y)
    DU = density_uniformity_score(y)
    dqi = float(np.clip(DS * VS * SC * DU, 0.0, 1.0))
    return PhaseResult(
        name=name,
        step_start=step_start,
        step_end=step_end,
        DS=DS, VS=VS, SC=SC, DU=DU,
        dqi=dqi
    )





# ============================================================
# MAIN ENTRY POINT
# ============================================================

def compute_dqi(
    steps: Union[list, np.ndarray],
    loss: Optional[Union[list, np.ndarray]] = None,
    grad_norm: Optional[Union[list, np.ndarray]] = None,
    signal_preference: str = "loss",
    phase_weights: Optional[dict] = None,
    scale_100: bool = False
) -> ADQIResult:
    """
    Compute model-agnostic Data Quality Intelligence.

    Works for: LLMs, vision models, tabular models, any architecture.
    Requires: training step indices + loss or grad_norm values.
    Never touches: raw data, weights, embeddings, tokens, pixels.

    Args:
        steps:             Training step indices (list or array)
        loss:              Loss per step — primary signal
        grad_norm:         Gradient norm per step — fallback signal
        signal_preference: "loss" or "grad_norm"
        phase_weights:     Override default early/mid/late weights
        scale_100:         Return scores on 0-100 scale

    Returns:
        ADQIResult with total score + per-phase breakdown
    """
    steps = np.array(steps, dtype=float)
    if loss is not None:
        loss = np.array(loss, dtype=float)
    if grad_norm is not None:
        grad_norm = np.array(grad_norm, dtype=float)


    # Extract universal (x, y) signal
    x, y, signal_name = extract_signal(steps, loss, grad_norm, signal_preference)

    # Detect adaptive phase boundaries and extract velocity energy
    s1, s2, dy = detect_phases(x, y)

    # 🛡️ DYNAMIC ENERGY WEIGHTING
    # Instead of hardcoded heuristics, we calculate weights based on the 
    # integral of the learning signal in each phase.
    if phase_weights:
        weights = phase_weights
    else:
        e_early = float(np.sum(dy[:s1]))
        e_mid   = float(np.sum(dy[s1:s2]))
        e_late  = float(np.sum(dy[s2:]))
        total_e = e_early + e_mid + e_late
        
        if total_e <= 1e-12:
            weights = {"early": 0.333, "mid": 0.333, "late": 0.334}
        else:
            weights = {
                "early": e_early / total_e,
                "mid":   e_mid / total_e,
                "late":  e_late / total_e
            }

    # Slice phases
    slices = {
        "early": (x[:s1],       y[:s1],       0,  s1),
        "mid":   (x[s1:s2],     y[s1:s2],     s1, s2),
        "late":  (x[s2:],       y[s2:],        s2, len(x)),
    }

    y_global_span = float(np.max(y) - np.min(y))

    phases = {}
    for name, (px, py, ps, pe) in slices.items():
        phases[name] = compute_phase_dqi(px, py, name, int(ps), int(pe), y_global_span)

    # Weighted fusion
    total = sum(weights[p] * phases[p].dqi for p in ["early", "mid", "late"])
    total = float(np.clip(total, 0.0, 1.0))

    if scale_100:
        total *= 100
        for p in phases.values():
            p.dqi *= 100
            p.DS  *= 100
            p.VS  *= 100
            p.SC  *= 100
            p.DU  *= 100

    return ADQIResult(
        total=total,
        early=phases["early"],
        mid=phases["mid"],
        late=phases["late"],
        phase_weights=weights,
        signal_type=signal_name
    )


# ============================================================
# CONVENIENCE
# ============================================================

def get_dqi_score(
    steps, loss=None, grad_norm=None,
    signal_preference="loss", scale_100=False
) -> float:
    """Return just the total ADQI scalar."""
    return compute_dqi(steps, loss, grad_norm, signal_preference, scale_100=scale_100).total