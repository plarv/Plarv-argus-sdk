"""
PLARV Argus — Local Detector
==============================
Open source. Runs entirely on user's machine.
No data sent anywhere. No network calls.

Note:
    Runs entirely on the user's machine. No data leaves the process.
    Open source by design — raw model access should never require
    trusting a third party.

Detects 5 failure modes the telemetry engine cannot see:
  1. Dead neurons        — activations stuck at zero
  2. Activation saturation — activations exploding or clamped
  3. Gradient flow blockage — gradients dying in early layers
  4. Weight norm imbalance  — layer weights growing disproportionately
  5. Optimizer corruption   — Adam momentum state poisoned

Usage:
    from plarv.local import LocalDetector

    detector = LocalDetector(model, optimizer)
    detector.attach()   # registers hooks — call once before training

    # inside training loop, after loss.backward():
    report = detector.step()

    # report is always safe to ignore — never raises, never blocks
    if report.any_critical:
        print(report.summary())

    # detach when done
    detector.detach()

    # Or use as context manager:
    with LocalDetector(model, optimizer) as detector:
        for batch in dataloader:
            ...
            loss.backward()
            report = detector.step()
"""

from typing import Optional, Dict, Any, List, Tuple
import math
try:
    from torch import nn
except ImportError:
    nn = None

# =============================================================================
# RESULT
# =============================================================================

class LocalReport:
    """
    Result from one detector step.
    Always safe to ignore — training never pauses from local detection alone.
    """

    LEVELS = ("ok", "warn", "critical")

    def __init__(self):
        self.dead_neurons:     Dict[str, Any] = {"level": "ok", "details": {}}
        self.saturation:       Dict[str, Any] = {"level": "ok", "details": {}}
        self.gradient_flow:    Dict[str, Any] = {"level": "ok", "details": {}}
        self.weight_imbalance: Dict[str, Any] = {"level": "ok", "details": {}}
        self.optimizer_health: Dict[str, Any] = {"level": "ok", "details": {}}
        self.step:  int = 0
        self.trend: str = "stable"  # "worsening" | "stable" | "recovering"
        self.attention_collapse: Dict[str, Any] = {"level": "ok", "details": {}}
        self.hardware_integrity: Dict[str, Any] = {"level": "ok", "details": {}}
        self.precision_erosion: Dict[str, Any] = {"level": "ok", "details": {}}
        self.representation_collapse: Dict[str, Any] = {"level": "ok", "details": {}}
    @property
    def top_affected_layers(self) -> List[str]:
        """
        All affected layer names across every check, sorted by true severity.
        Scores are normalized so higher always = worse, regardless of check type.
        Critical before warn. Within each level, sorted by normalized score.
        """
        LEVEL_WEIGHT = {"critical": 2, "warn": 1, "ok": 0}

        # Each check declares whether lower value = worse (e.g. gradient flow)
        # We store (check_name, ascending) so we can normalize correctly
        CHECK_META = {
            "dead_neurons":     False,  # higher sparsity = worse
            "saturation":       False,  # higher saturation = worse
            "gradient_flow":    True,   # lower norm = worse
            "weight_imbalance": False,  # higher ratio = worse
            "optimizer_health": False,  # higher z = worse
            "attention_collapse": True,   # lower entropy = worse
            "hardware_integrity": False,  # higher anomaly = worse
            "precision_erosion": False,   # higher fraction = worse
            "representation_collapse": True, # lower variance = worse
        }

        checks_named = {
            "dead_neurons":     self.dead_neurons,
            "saturation":       self.saturation,
            "gradient_flow":    self.gradient_flow,
            "weight_imbalance": self.weight_imbalance,
            "optimizer_health": self.optimizer_health,
            "attention_collapse": self.attention_collapse,
            "hardware_integrity": self.hardware_integrity,
            "precision_erosion": self.precision_erosion,
            "representation_collapse": self.representation_collapse,
        }

        entries: Dict[str, tuple] = {}  # name → (level_weight, normalized_score)

        for check_name in CHECK_META.keys():
            check        = checks_named.get(check_name)
            if not check: continue

            level        = check.get("level", "ok")
            level_weight = LEVEL_WEIGHT.get(level, 0)
            if level_weight == 0:
                continue

            details   = check.get("details", {})
            ascending = CHECK_META.get(check_name, False)

            if not isinstance(details, dict):
                continue

            for layer, score in details.items():
                if not isinstance(score, (int, float)):
                    continue
                # Normalize: flip sign for ascending checks so higher = worse
                norm_score = -float(score) if ascending else float(score)
                existing   = entries.get(layer)
                if existing is None or (level_weight, norm_score) > existing:
                    entries[layer] = (level_weight, norm_score)

        sorted_layers = sorted(
            entries.keys(),
            key=lambda n: entries[n],
            reverse=True,
        )
        return sorted_layers[:10]

    @property
    def affected_fraction(self) -> float:
        """
        Fraction of total tracked layers with at least one issue.
        0.0 = nothing wrong. 1.0 = every layer affected.
        Distinguishes isolated problem from systemic failure.
        """
        all_affected = set()
        total        = set()

        checks_named = {
            "dead_neurons":     self.dead_neurons,
            "saturation":       self.saturation,
            "gradient_flow":    self.gradient_flow,
            "weight_imbalance": self.weight_imbalance,
            "optimizer_health": self.optimizer_health,
            "attention_collapse": self.attention_collapse,
            "hardware_integrity": self.hardware_integrity,
            "precision_erosion": self.precision_erosion,
            "representation_collapse": self.representation_collapse,
        }

        for check_name in checks_named.keys():
            check   = checks_named[check_name]
            details = check.get("details", {})
            if isinstance(details, dict):
                total.update(details.keys())
                if check.get("level") in ("warn", "critical"):
                    all_affected.update(details.keys())

        if not total:
            return 0.0
        return round(len(all_affected) / len(total), 3)

    @property
    def scale(self) -> str:
        """Human-readable scale of impact based on affected_fraction."""
        f = self.affected_fraction
        if f == 0.0:   return "none"
        if f < 0.1:    return "isolated"
        if f < 0.33:   return "moderate"
        if f < 0.66:   return "widespread"
        return "systemic"

    @property
    def any_critical(self) -> bool:
        return any(
            check["level"] == "critical"
            for check in self._all_checks()
        )

    @property
    def any_warn(self) -> bool:
        return any(
            check["level"] in ("warn", "critical")
            for check in self._all_checks()
        )

    @property
    def worst_level(self) -> str:
        levels = [c["level"] for c in self._all_checks()]
        if "critical" in levels: return "critical"
        if "warn"     in levels: return "warn"
        return "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step":                self.step,
            "worst_level":         self.worst_level,
            "trend":               self.trend,
            "scale":               self.scale,
            "affected_fraction":   self.affected_fraction,
            "top_affected_layers": self.top_affected_layers,
            "dead_neurons":        self.dead_neurons,
            "saturation":          self.saturation,
            "gradient_flow":       self.gradient_flow,
            "weight_imbalance":    self.weight_imbalance,
            "optimizer_health":    self.optimizer_health,
            "attention_collapse":  self.attention_collapse,
            "hardware_integrity":  self.hardware_integrity,
            "precision_erosion":   self.precision_erosion,
            "representation_collapse": self.representation_collapse,
        }

    def summary(self) -> str:
        lines = [
            f"[PLARV Local] Step {self.step} — {self.worst_level.upper()} "
            f"| {self.trend} | scale={self.scale}"
        ]
        if self.top_affected_layers:
            lines.append(f"  Top affected: {', '.join(self.top_affected_layers[:5])}")
        checks = {
            "activation_sparsity": self.dead_neurons,
            "saturation":          self.saturation,
            "gradient_flow":       self.gradient_flow,
            "weight_imbalance":    self.weight_imbalance,
            "optimizer_health":    self.optimizer_health,
            "hardware_integrity":  self.hardware_integrity,
            "precision_erosion":   self.precision_erosion,
            "representation_collapse": self.representation_collapse,
        }
        for name, check in checks.items():
            if check["level"] != "ok":
                lines.append(f"  {check['level'].upper():8} {name}: {check.get('message', '')}")
        return "\n".join(lines)

    def _all_checks(self):
        return [
            self.dead_neurons,
            self.saturation,
            self.gradient_flow,
            self.weight_imbalance,
            self.optimizer_health,
            self.attention_collapse,
            self.hardware_integrity,
            self.precision_erosion,
            self.representation_collapse,
        ]


# =============================================================================
# MATH HELPERS — pure functions, no state
# =============================================================================

def _mean(arr: List[float]) -> float:
    return sum(arr) / len(arr) if arr else 0.0


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 1e-10 else default


def _top_affected(scores: Dict[str, float], n: int = 5, ascending: bool = False) -> List[str]:
    """
    Return top N layer names sorted by severity.
    ascending=True for gradient flow (lower norm = worse).
    ascending=False for everything else (higher value = worse).
    """
    if not scores:
        return []
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=not ascending)
    return [name for name, _ in sorted_items[:n]]


# =============================================================================
# LOCAL DETECTOR
# =============================================================================

class LocalDetector:
    """
    Attaches hooks to model. Collects activation and gradient stats per step.
    Detects failure modes that require raw model access.

    Designed to be zero-overhead when healthy — hooks are lightweight,
    tensors are never copied, all math runs on CPU after detach.
    """

    # Thresholds — sensitizing the moat to catch real failures (method.py validated)
    DEAD_NEURON_THRESHOLD    = 0.59   # Critical if >59% dead
    DEAD_NEURON_WARN         = 0.54   # Warn if >54% dead
    SATURATION_THRESHOLD     = 0.40   # Critical saturation
    SATURATION_WARN          = 0.20   # Warn saturation
    GRAD_FLOW_DEAD_RATIO     = 0.008  # 0.8%
    GRAD_FLOW_WARN_RATIO     = 0.01   # 1%
    WEIGHT_IMBALANCE_RATIO   = 20.0   # 20x imbalance = critical
    WEIGHT_IMBALANCE_WARN    = 10.0   # 10x imbalance = warn
    OPTIMIZER_CORRUPTION_Z   = 3.0    # 3.0 z-score
    OPTIMIZER_CORRUPTION_WARN = 2.0   # 2.0 z-score
    ATTENTION_ENTROPY_THRESHOLD = 0.05  # entropy below 0.05 = collapsed
    ATTENTION_ENTROPY_WARN      = 0.15  # entropy below 0.15 = warning
    HARDWARE_ANOMALY_THRESHOLD  = 10.0  # 10x unexplained weight norm jump (Bit-flip)
    PRECISION_EROSION_THRESHOLD = 0.20  # 20% of updates are underflow
    REPRESENTATION_VAR_THRESHOLD = 1e-5 # variance below this = collapse

    # Hyper-Performance Constraints — must stay <1% overhead
    MAX_TENSOR_ELEMENTS      = 100       # 500 -> 100 for <1% overhead
    MAX_GRAD_ELEMENTS        = 100       # 250 -> 100 for speed
    CHECK_EVERY_STEPS        = 100       # Global hook stride
    ANALYTICAL_STRIDE        = 100       # Deep scan stride

    # Optimizer warmup — ignore first N steps to let momentum stabilize
    OPTIMIZER_WARMUP_STEPS = 5

    def __init__(
        self,
        model:           Any,
        optimizer:       Optional[Any]       = None,
        silent:          bool                = False,
        sample_layers:   Optional[float]     = None,   # 0.0–1.0, fraction of layers to hook
        include_modules: Optional[List[str]] = None,   # e.g. ["Linear", "Conv2d"]
        check_every:     int                 = 1,       # run checks every N steps
    ):
        self.model           = model
        self.optimizer       = optimizer
        self.silent          = silent
        self.sample_layers   = sample_layers    # None = all layers
        self.include_modules = include_modules  # None = all types
        self.check_every     = max(1, check_every)
        self._prev_loss: Optional[float] = None
        self._hooks:         list = []
        self._activations:   Dict[str, Any]   = {}
        self._grad_norms:    Dict[str, float]  = {}
        self._forward_order: List[str]         = []
        self._step:          int = 0
        self._attached:      bool = False

        # Trend tracking — just previous level, no heavy history
        self._prev_level: str = "ok"
        self._momentum_acc: Dict[str, Dict] = {}
        self._attention_stats: Dict[str, float] = {}  # attention entropy per layer
        # Dynamic threshold calibration — learn normal baseline
        self._calibration_window = 50  # Steps to observe before enforcing
        self._baseline_grad_ratios: List[float] = []
        self._baseline_weight_ratios: List[float] = []
        self._prev_weight_norms: Dict[str, float] = {}
        self._representation_stats: Dict[str, float] = {}  # variance per layer
        self._layer_names: List[str] = []   # All eligible layers (for rolling window)
        self._param_names: List[str] = []   # All parameters (for rolling window)
        self._cached_params: Dict[str, "nn.Parameter"] = {}
        self._ok_report = LocalReport()     # Recycle for zero allocation overhead

    # =========================================================================
    # PUBLIC
    # =========================================================================

    def attach(self) -> "LocalDetector":
        """Register forward and backward hooks. Call once before training."""
        try:
            # Pre-cache model structure for zero overhead in step()
            self._cached_params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
            self._param_names   = sorted(list(self._cached_params.keys()))
            
            # Collect and cache all eligible modules
            eligible = []
            for name, module in self.model.named_modules():
                if self._is_eligible(module):
                    eligible.append((name, module))
            
            self._layer_names = sorted([n for n, m in eligible])

            # Apply layer sampling (optional)
            if self.sample_layers is not None and 0.0 < self.sample_layers < 1.0:
                import random
                k = max(1, int(len(eligible) * self.sample_layers))
                eligible = random.sample(eligible, k)
                self._layer_names = sorted([n for n, m in eligible])

            for name, module in eligible:
                h_fwd = module.register_forward_hook(
                    self._make_forward_hook(name)
                )
                h_bwd = module.register_full_backward_hook(
                    self._make_backward_hook(name)
                )
                self._hooks.extend([h_fwd, h_bwd])

            self._attached = True
        except Exception:
            pass  # silent — never crash training
        return self

    def detach(self) -> None:
        """Remove all hooks."""
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()
        self._activations.clear()
        self._grad_norms.clear()
        self._attached = False

    def step(self, loss: Optional[float] = None) -> LocalReport:
        """
        Run all local checks. Call after loss.backward(), before optimizer.zero_grad().
        Returns LocalReport — safe to ignore, never raises.
        """
        report = LocalReport()
        report.step = self._step

        # Only run checks every N steps — reduces overhead on large models
        if self._step % self.check_every == 0:
            try:
                # Step-wise (Fast)
                report.representation_collapse = self._check_representation_collapse()
                report.dead_neurons = self._check_dead_neurons()
                
                # Analytical Stride — deeper deep scans every 100 steps
                if self._step % self.ANALYTICAL_STRIDE == 0:
                    report.saturation       = self._check_saturation()
                    report.gradient_flow    = self._check_gradient_flow()
                    report.weight_imbalance = self._check_weight_imbalance()
                    report.optimizer_health = self._check_optimizer_health()
                    report.attention_collapse = self._check_attention_collapse()
                    report.hardware_integrity = self._check_hardware_integrity()
                    report.precision_erosion = self._check_precision_erosion()
            except Exception:
                pass  # never crash training

            # Trend — compare current worst level to previous
            LEVEL_INT = {"ok": 0, "warn": 1, "critical": 2}
            curr_int  = LEVEL_INT.get(report.worst_level, 0)
            prev_int  = LEVEL_INT.get(self._prev_level, 0)
            if   curr_int > prev_int: report.trend = "worsening"
            elif curr_int < prev_int: report.trend = "recovering"
            else:                     report.trend = "stable"
            self._prev_level = report.worst_level

            if not self.silent and report.any_warn:
                print(report.summary())

        # Clear per-step state
        self._activations.clear()
        # Refresh layer list if not yet populated (fallback)
        if not self._layer_names and self._activations:
            self._layer_names = sorted(list(self._activations.keys()))

        self._grad_norms.clear()
        self._forward_order.clear()
        self._attention_stats.clear()
        self._representation_stats.clear()
        self._step += 1
        return report

    def __enter__(self):
        self.attach()
        return self

    def __exit__(self, *args):
        self.detach()
        return False

    # =========================================================================
    # HOOKS
    # =========================================================================

    # Max elements to process per tensor — prevents GPU→CPU sync cost on large models
    MAX_TENSOR_ELEMENTS = 10_000

    def _make_forward_hook(self, name: str):
        def hook(module, input, output):
            try:
                if not self._attached: return
                
                # Performance: Rolling Window — only hook 1 layer per step
                if self._layer_names:
                    idx = self._step % len(self._layer_names)
                    if self._layer_names[idx] != name:
                        return
                    
                # Track actual forward execution order
                if name not in self._forward_order:
                    self._forward_order.append(name)

                t = output[0] if isinstance(output, (tuple, list)) else output
                if not hasattr(t, "detach"):
                    return

                data = t.detach()
                original_size = data.numel()

                # Deterministic stride sampling — reproducible, no flickering
                # randperm would cause non-deterministic warnings across steps
                if original_size > self.MAX_TENSOR_ELEMENTS:
                    stride = max(1, original_size // self.MAX_TENSOR_ELEMENTS)
                    data   = data.reshape(-1)[::stride]

                data  = data.float()
                total = data.numel()
                if total == 0:
                    return

                # Detect activation type for context-aware messaging
                module_type    = type(module).__name__
                is_relu_family = any(k in module_type for k in ("ReLU", "GELU", "SiLU", "Mish", "relu", "gelu"))

                # Activation sparsity — fraction near zero
                # ReLU family: high sparsity = activation sparsity anomaly
                # Others: skip (LayerNorm/Linear naturally have many near-zero values)
                near_zero = (data.abs() < 1e-6).sum().item() / total
                
                # Representation Variance — catch delta-function collapse
                if data.size(0) > 1:
                    # Catch cross-batch collapse: all samples receiving same features
                    var = data.view(data.size(0), -1).var(dim=0).mean().item()
                else:
                    # Single sample fallback: cannot detect cross-batch collapse
                    var = data.var().item()
                self._representation_stats[name] = var

                # Saturation — fraction at max absolute value
                data_max = data.abs().max().item()
                at_max   = (
                    (data.abs() > data_max * 0.999).sum().item() / total
                    if data_max > 1e-8 else 0.0
                )

                self._activations[name] = {
                    "near_zero_frac": near_zero,
                    "at_max_frac":    at_max,
                    "mean_abs":       data.abs().mean().item(),
                    "is_relu_family": is_relu_family,
                    "module_type":    module_type,
                    "sampled":        t.numel() > self.MAX_TENSOR_ELEMENTS,
                }
            # Attention entropy — only for attention layers
                # Must be 4D (B, heads, seq, seq) — actual attention weight matrix
                # Softmaxed output values are NOT attention weights — skip those
                if "attn" in name.lower() or "attention" in name.lower():
                    if data.dim() == 4:
                        # Shape: (B, num_heads, seq_len, seq_len) — this IS the weight matrix
                        # Already softmaxed by attention mechanism, values sum to 1 per row
                        attn_weights = data.clamp(min=1e-10)  # (B, H, S, S)
                        # Entropy per row (each query position over all key positions)
                        entropy = -(attn_weights * attn_weights.log()).sum(dim=-1).mean().item()
                        self._attention_stats[name] = entropy
                    # 3D = (B, seq, seq) single-head — also valid
                    elif data.dim() == 3:
                        attn_weights = data.clamp(min=1e-10)
                        entropy = -(attn_weights * attn_weights.log()).sum(dim=-1).mean().item()
                        self._attention_stats[name] = entropy
                    # 2D or other dims = output values, not weights — skip

            except Exception:
                pass
        return hook

    def _make_backward_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            try:
                if not self._attached: return
                
                # Performance: Rolling Window — only compute 1 grad norm per step
                if self._layer_names:
                    idx = self._step % len(self._layer_names)
                    if self._layer_names[idx] != name:
                        return
                    
                g = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
                if g is None or not hasattr(g, "detach"):
                    return

                g = g.detach()
                original_size = g.numel()

                # Deterministic stride sampling — same as forward hook
                if original_size > self.MAX_TENSOR_ELEMENTS:
                    stride = max(1, original_size // self.MAX_TENSOR_ELEMENTS)
                    g      = g.reshape(-1)[::stride]

                sampled_size = g.numel()
                raw_norm     = g.float().norm().item()

                # Scale sampled norm to approximate true norm
                # norm(sampled) * sqrt(original/sampled) ≈ true norm
                if sampled_size < original_size and sampled_size > 0:
                    scale = math.sqrt(original_size / sampled_size)
                    norm  = raw_norm * scale
                else:
                    norm = raw_norm

                self._grad_norms[name] = norm
            except Exception:
                pass
        return hook

    # =========================================================================
    # CHECKS
    # =========================================================================

    def _check_dead_neurons(self) -> Dict[str, Any]:
        """
        Activation sparsity anomaly detection.
        For ReLU/GELU family: high sparsity = activation sparsity anomaly.
        For other types (LayerNorm, Linear): skipped — naturally sparse.
        """
        if not self._activations:
            return {"level": "ok", "details": {}}

        anomaly_layers = {}
        warn_layers    = {}

        for name, stats in self._activations.items():
            frac           = stats["near_zero_frac"]
            is_relu_family = stats.get("is_relu_family", False)

            if not is_relu_family:
                continue

            if frac >= self.DEAD_NEURON_THRESHOLD:
                anomaly_layers[name] = round(frac, 3)
            elif frac >= self.DEAD_NEURON_WARN:
                warn_layers[name] = round(frac, 3)

        if anomaly_layers:
            top = _top_affected(anomaly_layers, n=5)
            return {
                "level":              "critical",
                "message":            f"{len(anomaly_layers)} layers >{int(self.DEAD_NEURON_THRESHOLD*100)}% activation sparsity anomaly",
                "details":            anomaly_layers,
                "top_affected_layers": top,
            }
        if warn_layers:
            top = _top_affected(warn_layers, n=5)
            return {
                "level":              "warn",
                "message":            f"{len(warn_layers)} layers >{int(self.DEAD_NEURON_WARN*100)}% activation sparsity",
                "details":            warn_layers,
                "top_affected_layers": top,
            }
        return {"level": "ok", "details": {}}

    def _check_saturation(self) -> Dict[str, Any]:
        """
        Activation saturation: outputs clumped at maximum values.
        Kills gradient flow — saturated units have near-zero gradients.
        """
        if not self._activations:
            return {"level": "ok", "details": {}}

        saturated = {}
        warned    = {}
        for name, stats in self._activations.items():
            frac = stats["at_max_frac"]
            mean_abs = stats.get("mean_abs", 0.0)

            # Explosion check: activation magnitudes > 2 in Linear layers = critical anomaly
            if mean_abs > 2.0:
                saturated[name] = f"Exploded (mean_abs={mean_abs:.1f})"
                continue

            if frac >= self.SATURATION_THRESHOLD:
                saturated[name] = round(frac, 3)
            elif frac >= self.SATURATION_WARN:
                warned[name] = round(frac, 3)

        if saturated:
            return {
                "level":              "critical",
                "message":            f"{len(saturated)} layers critically saturated",
                "details":            saturated,
                "top_affected_layers": _top_affected(saturated, n=5),
            }
        if warned:
            return {
                "level":              "warn",
                "message":            f"{len(warned)} layers showing saturation",
                "details":            warned,
                "top_affected_layers": _top_affected(warned, n=5),
            }
        return {"level": "ok", "details": {}}

    def _check_gradient_flow(self) -> Dict[str, Any]:
        """
        Gradient flow blockage: gradients dying before reaching early layers.
        Uses actual forward execution order (tracked in forward hook) so
        first/last layer comparison is always correct regardless of dict order.
        """
        if len(self._grad_norms) < 2:
            return {"level": "ok", "details": {}}

        # Use forward order if available — otherwise fall back to dict order
        if len(self._forward_order) >= 2:
            ordered_names = [n for n in self._forward_order if n in self._grad_norms]
        else:
            ordered_names = list(self._grad_norms.keys())

        if len(ordered_names) < 2:
            return {"level": "ok", "details": {}}

        first_norm = self._grad_norms[ordered_names[0]]
        last_norm  = self._grad_norms[ordered_names[-1]]

        if last_norm < 1e-10:
            return {"level": "ok", "details": {}}

        ratio = _safe_div(first_norm, last_norm, default=1.0)

        # Gradient flow distribution — median and min for bottleneck detection
        all_norms   = sorted([self._grad_norms[n] for n in ordered_names])
        median_norm = all_norms[len(all_norms) // 2]
        min_norm    = all_norms[0]

        # Dynamic threshold — use observed baseline for first 50 steps
        if self._step < self._calibration_window:
            self._baseline_grad_ratios.append(ratio)
            # During calibration, only flag extreme cases
            effective_dead_ratio = 0.001  # 0.1% — extremely strict
            effective_warn_ratio = 0.005  # 0.5%
        else:
            # After calibration, use learned baseline + margin
            if self._baseline_grad_ratios:
                baseline_median = sorted(self._baseline_grad_ratios)[len(self._baseline_grad_ratios) // 2]
                # Critical: 50% below observed baseline
                effective_dead_ratio = max(0.005, baseline_median * 0.5)
                # Warn: 70% of observed baseline
                effective_warn_ratio = max(0.01, baseline_median * 0.7)
            else:
                # Fallback to static thresholds
                effective_dead_ratio = self.GRAD_FLOW_DEAD_RATIO
                effective_warn_ratio = self.GRAD_FLOW_WARN_RATIO

        blocked = {
            name: round(self._grad_norms[name], 6)
            for name in ordered_names
            if self._grad_norms[name] < last_norm * effective_dead_ratio
        }
        warned = {
            name: round(self._grad_norms[name], 6)
            for name in ordered_names
            if last_norm * effective_dead_ratio <= self._grad_norms[name] < last_norm * effective_warn_ratio
        }

        flow_stats = {
            "flow_ratio":   round(ratio, 4),
            "median_norm":  round(median_norm, 6),
            "min_norm":     round(min_norm, 6),
            "last_norm":    round(last_norm, 6),
        }

        if blocked:
            return {
                "level":              "critical",
                "message":            f"Gradient flow blocked in {len(blocked)} layers (first/last ratio={ratio:.4f})",
                "details":            blocked,
                "top_affected_layers": _top_affected(blocked, n=5, ascending=True),
                "stats":              flow_stats,
            }
        if warned or ratio < effective_warn_ratio:
            return {
                "level":              "warn",
                "message":            f"Weak gradient flow (first/last ratio={ratio:.4f})",
                "details":            warned,
                "top_affected_layers": _top_affected(warned, n=5, ascending=True),
                "stats":              flow_stats,
            }
        return {"level": "ok", "details": {}, "stats": flow_stats}
   
    def _check_weight_imbalance(self) -> Dict[str, Any]:
        """
        Weight norm imbalance: some layers have grown disproportionately large.
        Causes one layer to dominate the network — others become irrelevant.
        Happens silently over thousands of steps. Loss can look fine throughout.
        """
        try:
            layer_norms = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.data.dim() >= 2:
                    norm = param.data.detach().float().norm().item()
                    if norm > 1e-10:
                        layer_norms[name] = norm

            if len(layer_norms) < 2:
                return {"level": "ok", "details": {}}

            norms     = list(layer_norms.values())
            max_norm  = max(norms)
            min_norm  = min(norms)
            ratio     = _safe_div(max_norm, min_norm, default=1.0)

            # Dynamic threshold — use observed baseline
            if self._step < self._calibration_window:
                self._baseline_weight_ratios.append(ratio)
                # During calibration, only flag extreme cases
                effective_critical_ratio = 200.0  # Very high
                effective_warn_ratio = 100.0
            else:
                # After calibration, use learned baseline + margin
                if self._baseline_weight_ratios:
                    baseline_median = sorted(self._baseline_weight_ratios)[len(self._baseline_weight_ratios) // 2]
                    # Critical: allow sensitivity even if baseline was higher
                    effective_critical_ratio = min(self.WEIGHT_IMBALANCE_RATIO, baseline_median * 1.5)
                    # Warn: allow sensitivity even if baseline was higher
                    effective_warn_ratio = min(self.WEIGHT_IMBALANCE_WARN, baseline_median * 1.2)
                else:
                    # Fallback
                    effective_critical_ratio = self.WEIGHT_IMBALANCE_RATIO
                    effective_warn_ratio = self.WEIGHT_IMBALANCE_WARN

            if ratio >= effective_critical_ratio:
                mean_norm = _mean(norms)
                outliers  = {
                    name: round(norm, 4)
                    for name, norm in layer_norms.items()
                    if norm > mean_norm * 10 or norm < mean_norm * 0.1
                }
                return {
                    "level":              "critical",
                    "message":            f"Weight norm imbalance ratio={ratio:.1f}x (max={max_norm:.3f}, min={min_norm:.3f})",
                    "details":            outliers,
                    "top_affected_layers": _top_affected(outliers, n=5),
                    "ratio":              round(ratio, 2),
                }

            if ratio >= effective_warn_ratio:
                return {
                    "level":              "warn",
                    "message":            f"Weight norm imbalance ratio={ratio:.1f}x",
                    "details":            {}, # Summary stats only, no outliers
                    "top_affected_layers": [],
                    "stats":              {"max_norm": round(max_norm, 4), "min_norm": round(min_norm, 4), "ratio": round(ratio, 2)},
                }

            return {"level": "ok", "details": {}, "stats": {"ratio": round(ratio, 2)}}

        except Exception:
            return {"level": "ok", "details": {}}

    def _check_optimizer_health(self) -> Dict[str, Any]:
        """
        Optimizer state corruption: Adam's momentum vectors (m, v) poisoned
        by a bad batch. The bad gradient is gone but its echo lives in
        momentum for hundreds of steps — silently poisoning updates.

        Detection: track momentum norm per parameter over time via Welford.
        A sudden spike in momentum norm = corruption event.
        """
        if self.optimizer is None:
            return {"level": "ok", "details": {}}

        try:
            corrupted = {}
            warned    = {}

            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    if param not in self.optimizer.state:
                        continue

                    state = self.optimizer.state[param]
                    exp_avg = state.get("exp_avg")        # m — first moment
                    exp_avg_sq = state.get("exp_avg_sq")  # v — second moment

                    if exp_avg is None:
                        continue

                    # Use parameter id as key
                    key = str(id(param))

                    m_norm = exp_avg.detach().float().norm().item()

                    # Initialize or update Welford accumulator
                    if key not in self._momentum_acc:
                        self._momentum_acc[key] = {"n": 0, "mean": 0.0, "M2": 0.0}

                    acc = self._momentum_acc[key]
                    acc["n"] += 1
                    delta      = m_norm - acc["mean"]
                    acc["mean"] += delta / acc["n"]
                    delta2     = m_norm - acc["mean"]
                    acc["M2"] += delta * delta2

                    # Warmup guard — ignore first 50 steps, momentum too noisy
                    if acc["n"] < self.OPTIMIZER_WARMUP_STEPS:
                        continue

                    std = math.sqrt(acc["M2"] / acc["n"]) if acc["n"] >= 2 else 1.0
                    if std < 1e-10:
                        continue

                    z = (m_norm - acc["mean"]) / (std + 1e-10)

                    # Cap sensitivity in early post-warmup phase (steps 50-100)
                    # momentum distribution still settling — require higher z
                    # Dynamic sensitivity: Patient during noisy init, forensic as n increases
                    # Decay multiplier from ~15x at n=0 down to 1x at n>>50
                    scaler = (1.0 + 15.0 / (acc["n"] + 1))
                    z_critical = self.OPTIMIZER_CORRUPTION_Z * scaler
                    z_warn     = self.OPTIMIZER_CORRUPTION_WARN * scaler

                    if z > z_critical:
                        corrupted[key] = round(z, 2)
                    elif z > z_warn:
                        warned[key] = round(z, 2)

            if corrupted:
                return {
                    "level":   "critical",
                    "message": f"{len(corrupted)} parameter momentum vectors show corruption (z>{self.OPTIMIZER_CORRUPTION_Z})",
                    "details": corrupted,
                }
            if warned:
                return {
                    "level":   "warn",
                    "message": f"{len(warned)} parameter momentum vectors elevated",
                    "details": warned,
                }
            return {"level": "ok", "details": {}}

        except Exception:
            return {"level": "ok", "details": {}}
    

    def _check_attention_collapse(self) -> Dict[str, Any]:
        """
        Attention collapse: all attention heads focus on one token.
        Kills information flow — model can't attend to context.
        Common in transformers when training goes wrong.
        """
        if not self._attention_stats:
            return {"level": "ok", "details": {}}

        collapsed = {}
        warned    = {}

        for name, entropy in self._attention_stats.items():
            if entropy < self.ATTENTION_ENTROPY_THRESHOLD:
                collapsed[name] = round(entropy, 4)
            elif entropy < self.ATTENTION_ENTROPY_WARN:
                warned[name] = round(entropy, 4)

        if collapsed:
            return {
                "level": "critical",
                "message": f"{len(collapsed)} attention layers collapsed (entropy < {self.ATTENTION_ENTROPY_THRESHOLD})",
                "details": collapsed,
                "top_affected_layers": _top_affected(collapsed, n=5, ascending=True),
            }
        if warned:
            return {
                "level": "warn",
                "message": f"{len(warned)} attention layers low entropy",
                "details": warned,
                "top_affected_layers": _top_affected(warned, n=5, ascending=True),
            }

        return {"level": "ok", "details": {}}
    

        



    def _check_hardware_integrity(self) -> Dict[str, Any]:
        """
        Detects silent weight corruption (bit-flips).
        Rolling window: Only checks 1 parameter per step to maintain <1% overhead.
        """
        if not self._param_names:
            return {"level": "ok", "details": {}}
            
        try:
            anomalies = {}
            # Rolling index
            idx = self._step % len(self._param_names)
            name = self._param_names[idx]
            param = self._cached_params.get(name)
            
            if self._step > 10 and param is not None and param.data.dim() >= 2:
                curr_norm = param.data.detach().float().norm().item()
                prev_norm = self._prev_weight_norms.get(name)
                
                if prev_norm is not None and prev_norm > 1e-8:
                    ratio = max(curr_norm / prev_norm, prev_norm / curr_norm)
                    if ratio > self.HARDWARE_ANOMALY_THRESHOLD:
                        # 🛡️ SOVEREIGN TOLERANCE: Only flag if the jump is massive 
                        # AND the gradient is tiny (suggesting non-optimizer change).
                        grad_norm = self._grad_norms.get(name, 0.0)
                        if grad_norm < prev_norm * 0.01:
                            anomalies[name] = round(ratio, 2)
                
                self._prev_weight_norms[name] = curr_norm
            
            if anomalies:
                return {
                    "level": "critical",
                    "message": f"Possible silent weight corruption in {len(anomalies)} layers (bit-flip detected)",
                    "details": anomalies,
                }
            return {"level": "ok", "details": {}}
        except Exception:
            return {"level": "ok", "details": {}}

    def _check_precision_erosion(self) -> Dict[str, Any]:
        """
        Detects if updates are too small for hardware precision (BF16/FP16).
        Rolling window: Checks 1 parameter group/param per step.
        """
        if self.optimizer is None or not self._param_names:
            return {"level": "ok", "details": {}}

        try:
            eroded_layers = {}
            import torch
            
            # Select 1 parameter to check this step
            idx = self._step % len(self._param_names)
            target_name = self._param_names[idx]
            target_param = self._cached_params.get(target_name)
            
            if target_param is not None and target_param.grad is not None:
                # SKIP Embedding layers — naturally sparse and noisy at init, causes false positives
                if "embed" in target_name.lower():
                    return {"level": "ok", "details": {}}

                # Find its LR from optimizer
                lr = 1e-3 # fallback
                for group in self.optimizer.param_groups:
                    if any(p is target_param for p in group["params"]):
                        lr = group.get("lr", 1e-3)
                        break
                
                eps = torch.finfo(target_param.dtype).eps
                flat_grad = target_param.grad.detach().view(-1)
                if flat_grad.numel() > self.MAX_GRAD_ELEMENTS:
                    flat_grad = flat_grad[:self.MAX_GRAD_ELEMENTS]
                
                # Only check non-zero gradients — zeros / noise are usually intentional sparsity
                mask = flat_grad.abs() > 1e-12
                if mask.any():
                    underflow_frac = (flat_grad[mask].abs().float() * lr < eps).float().mean().item()
                else:
                    underflow_frac = 0.0
                
                if underflow_frac > self.PRECISION_EROSION_THRESHOLD:
                    eroded_layers[target_name] = round(underflow_frac, 3)

            if eroded_layers:
                return {
                    "level": "warn",
                    "message": f"Precision erosion in {len(eroded_layers)} layers — updates rounding to zero",
                    "details": eroded_layers,
                }
            return {"level": "ok", "details": {}}
        except Exception:
            return {"level": "ok", "details": {}}

    def _check_representation_collapse(self) -> Dict[str, Any]:
        """
        Instant detection of layer-wise delta transformation.
        If variance flips to zero, the layer has "collapsed" and 
        is outputting constant garbage.
        """
        if not self._representation_stats:
            return {"level": "ok", "details": {}}

        collapsed = {}
        for name, var in self._representation_stats.items():
            if var < self.REPRESENTATION_VAR_THRESHOLD:
                collapsed[name] = round(var, 8)

        if collapsed:
            return {
                "level": "critical",
                "message": f"Representation collapse in {len(collapsed)} layers (zero variance)",
                "details": collapsed,
            }
        return {"level": "ok", "details": {}}

   # =========================================================================
    # HELPERS
    # =========================================================================

    def _is_eligible(self, module) -> bool:
        try:
            # Leaf modules only
            if list(module.children()):
                return False

            module_type = type(module).__name__

            # Skip modules that produce no meaningful activation stats
            SKIP_TYPES = (
                "Dropout", "Identity", "Embedding",
                "LayerNorm", "BatchNorm1d", "BatchNorm2d", "GroupNorm", "RMSNorm",
            )
            if any(module_type.startswith(s) for s in SKIP_TYPES):
                return False

            # Must have at least one parameter or be an activation function
            has_params = any(True for _ in module.parameters(recurse=False))
            is_activation = any(k in module_type for k in (
                "ReLU", "GELU", "SiLU", "Mish", "Tanh", "Sigmoid", "Softmax"
            ))

            return has_params or is_activation
        except Exception:
            return False