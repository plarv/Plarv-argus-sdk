# PLARV Argus

Real-time ML training monitor. Detects NaN explosions, gradient collapse, optimizer corruption, and silent divergence — before they destroy your run.

---

```bash
pip install plarv-argus-sdk  # from plarv import Argus
```

---

## Installation

```bash
pip install plarv-argus-sdk
```

*(Note: The package is installed as `plarv-argus-sdk`, but you import it as `plarv`.)*

Zero hard dependencies. PyTorch, Transformers, and Lightning are optional — Argus uses whatever you already have.

---

## How it works

Argus sits alongside your training loop. Every step, it reports telemetry to the Argus engine, which runs a 12-signal analysis and returns an action. If something is wrong, it tells you — and optionally stops the run, saves a checkpoint, or adjusts your learning rate automatically.

All API calls are async. Argus never blocks your training loop.

---

## Quick start

```python

from plarv import Argus

# Initialize the Control Plane
argus = Argus(api_key="your-key")

for batch in dataloader:
    loss = model(inputs, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    argus.step(loss)

argus.complete()

```

`grad_norm` is optional — Argus computes it automatically from your model if not provided. Exceptions are caught by default so Argus never crashes your training run.

---

## Integrations

### HuggingFace Trainer

```python
from plarv import ArgusCallback
from transformers import Trainer

trainer = Trainer(
    model=model,
    callbacks=[ArgusCallback(api_key="your-key")]
)
trainer.train()
```

Argus automatically sets `logging_steps=1` so it sees every step.

### Unsloth

```python
from plarv.integrations.unsloth import patch_unsloth
from transformers import Trainer

argus_cb = patch_unsloth(model, api_key="your-key")

trainer = Trainer(
    model=model,
    callbacks=[argus_cb]
)
trainer.train()
```

`patch_unsloth` auto-extracts model architecture metadata (`num_layers`, `hidden_size`, `vocab_size`) from the Unsloth model config.

### Axolotl

Add to your Axolotl config:

```yaml
plugins:
  - plarv.integrations.axolotl.ArgusAxolotlPlugin

argus_api_key: "your-key"
argus_run_id: "optional-run-id"
```

No code changes required. Argus is injected automatically after the trainer is created.

### PyTorch Lightning

```python
from plarv import ArgusLightningCallback
import lightning as pl

trainer = pl.Trainer(
    callbacks=[ArgusLightningCallback(api_key="your-key")]
)
trainer.fit(model)
```

---

## Local Detector

The Local Detector runs entirely on your machine. No data is sent anywhere. It complements the Argus engine with signals that require direct model access.

```python
from plarv.local import LocalDetector

detector = LocalDetector(model, optimizer)
detector.attach()

for batch in dataloader:
    loss = model(inputs, labels)
    loss.backward()

    report = detector.step(loss=loss.item())

    if report.any_critical:
        print(report.summary())

    optimizer.step()
    optimizer.zero_grad()

detector.detach()
```

Or as a context manager:

```python
with LocalDetector(model, optimizer) as detector:
    for batch in dataloader:
        loss.backward()
        report = detector.step(loss=loss.item())
```

**What it detects:**

| Check | Description |
|---|---|
| Activation sparsity | ReLU/GELU layers with >54% dead neurons |
| Activation saturation | Outputs clumped at maximum values |
| Gradient flow blockage | Gradients dying before reaching early layers |
| Weight norm imbalance | Layers growing disproportionately large |
| Optimizer corruption | Adam momentum vectors poisoned by a bad batch |
| Attention collapse | All heads focusing on a single token |
| Loss spike | Sudden loss increase beyond rolling baseline |
| Representation collapse | Layer outputs with near-zero variance |
| Precision erosion | Updates rounding to zero in BF16/FP16 |
| Hardware integrity | Silent weight corruption (bit-flip detection) |

The report is always safe to ignore — `detector.step()` never raises and never pauses training on its own.

**Report fields:**

```python
report.worst_level       # "ok" | "warn" | "critical"
report.trend             # "stable" | "worsening" | "recovering"
report.scale             # "none" | "isolated" | "moderate" | "widespread" | "systemic"
report.affected_fraction # 0.0–1.0, fraction of layers with issues
report.top_affected_layers  # list of layer names sorted by severity
report.summary()         # human-readable summary string
report.to_dict()         # full structured output
```

Passing `local_report` to `argus.step()` bridges the Local Detector into the Argus engine — the engine uses it as an additional input signal:

```python
report = detector.step(loss=loss.item())
argus.step(loss=loss.item(), local_report=report)
```

---

## Per-sample signals (Layer 2)

For deeper analysis, pass per-sample signals extracted from your logits:

```python
from plarv.utils import extract_signals

loss = criterion(logits, labels)
loss.backward()

signals = extract_signals(logits, labels, model)
argus.step(loss=loss.item(), **signals)
```

`extract_signals` handles both classification `(B, C)` and language model `(B, T, V)` logits. It extracts per-sample loss, confidence, margin, entropy, and correctness — plus gradient L1/L2 norms for canalization detection.

---

## Checkpointing (Zero-Block)

Pass your model to enable **Zero-Block Async Checkpointing**. Unlike standard checkpointing, Argus uses a two-stage protocol:
1. **Stage 1 (Memory Staging)**: Weights are captured in a near-instantaneous memory snapshot.
2. **Stage 2 (Background I/O)**: The actual disk write happens in a non-blocking background thread.

This ensures your training loop never halts for I/O, even with multi-GB models. Argus saves to a **3-slot circular buffer** on disk, driven by engine signals—not a fixed schedule.

```python
argus = Argus(
    api_key="your-key",
    model=model,
    optimizer=optimizer,
    tokenizer=tokenizer,          # optional, saved alongside model
    checkpoint_dir="./argus-checkpoints",
)
```

Argus auto-detects your framework and calls `save_pretrained`, `model.save`, or `torch.save` accordingly.

When a collapse is detected, the buffer freezes. The anchor point file at `argus-checkpoints/anchor_point.json` identifies the last clean checkpoint before contamination began.

---

## World-Level Performance

Argus is designed for massive scale and global network latency:

- **Adaptive Telemetry Packing**: Argus dynamically calculates your network RTT and bundles training steps to ensure 0ms network blocking, even on transatlantic connections.
- **Zero-Touch Discovery**: Automatically extracts model architecture (`num_layers`, `hidden_dim`, `dtype`) and training duration from your `model` and `dataloader` objects.
---

## Performance Philosophy: Speed First

Plarv Argus is built with a **Speed-First** clinical mandate. We believe the telemetry SDK should never be the bottleneck of your training loop.

- **The Core Promise**: The standard Argus heartbeat carries an overhead of **<1.5ms per step**, ensuring your GPU throughput remains at maximum capacity.
- **Modular Forensics**: The `LocalDetector` is a separate, modular engine. While it provides deep protection against rare events (dead neurons, activation collapse), it is **opt-in**. You choose when to deploy the "Moat" based on your specific stability requirements and compute budget.
- **Passive Bridging**: The `Argus` client can receive reports from the `LocalDetector` via the `.step()` method, but it never forces forensic scans on your model by default.

## Privacy & Sovereignty

Plarv Argus is designed with **Sovereign Privacy** at its core. We maintain a strict clinical boundary between run-level telemetry and your private model data:

- **Local-Only Forensics**: The `LocalDetector` engine is strictly local. It never makes network calls and never transmits activations, gradients, or weights outside your hardware.
- **Zero-Weight Telemetry**: The Argus heartbeat only transmits non-sensitive metadata (loss, total grad norm, throughput). Your model weights and proprietary architecture details never leave your machine.
- **Argus Data Quality Score (ADQI)**: Automatically computes a privacy-safe "Quality Score" for your training data. By analyzing the curvature and distribution of your training trajectory (loss/gradients), Argus determines if your dataset is well-distributed or redundant—without ever seeing the raw data.
- **You Own the Data**: All deep forensic reports generated by the `LocalDetector` stay on your local disk. You choose what (if anything) to bridge to the cloud.

---

## Modes

| Mode | Behavior |
|---|---|
| `MANUAL` (default) | Argus reports and warns. You decide what to do. |
| `AUTO` | Argus adjusts learning rate and raises `ArgusPause` on collapse. |

```python
argus = Argus(api_key="your-key", mode="AUTO")
```

In AUTO mode, catch `ArgusPause` to save state before the run stops:

```python
from plarv.exceptions import ArgusPause

try:
    argus.step(loss=loss.item())
except ArgusPause as e:
    save_checkpoint(model, optimizer, step=e.step)
    raise
```

---

## Configuration

```python
argus = Argus(
    api_key="your-key",

    # Run identity
    run_id="experiment-001",          # auto-generated if not provided

    # Model (all optional — enables checkpointing and auto grad_norm)
    model=model,
    optimizer=optimizer,
    tokenizer=tokenizer,

    # Mode
    mode="MANUAL",                    # "MANUAL" | "AUTO"

    # Architecture metadata (improves signal quality)
    sequence_length=2048,
    vocab_size=32000,
    num_layers=32,
    hidden_dim=4096,
    dtype="bfloat16",

    # Checkpointing
    checkpoint_dir="./argus-checkpoints",

    # Behavior
    silent=False,                     # suppress console output
    fail_open=False,                  # if True, continues training on API failure
    catch_exceptions=True,            # if True, never raises inside step()
    step0_timeout=12.0,               # seconds to wait for handshake

    # Callbacks
    on_pause=my_pause_handler,        # called when engine issues PAUSE
)
```

---

## Exceptions

```python
from plarv.exceptions import (
    ArgusError, ArgusConfigurationError, ArgusConnectionError,
    ArgusApiError, ArgusAuthenticationError, ArgusRateLimitError,
    ArgusServerError, ArgusIntervention, ArgusPause,
    ArgusCheckpoint, ArgusHalt
)

# ArgusError                — base exception
#   ArgusConfigurationError — invalid initialization parameters
#   ArgusConnectionError    — network unreachable
#   ArgusApiError           — non-200 response (carries .status_code, .response)
#     ArgusAuthenticationError — 401/403 invalid key
#     ArgusRateLimitError      — 429 rate limit
#     ArgusServerError         — 5xx backend error
#   ArgusIntervention       — base for training interventions (.step, .response)
#     ArgusPause            — engine requested pause
#     ArgusCheckpoint       — engine requested checkpoint
#     ArgusHalt             — sentinel hard stop (SIG_HALT_NOW)
```

---

## Requirements

- Python 3.8+
- No required dependencies
- PyTorch (optional, required for grad_norm auto-computation and Local Detector)
- Transformers (optional, required for HuggingFace callback)
- Lightning (optional, required for Lightning callback)

---

## Repository Structure

- `plarv/`: The primary package.
  - `core/`: Internal machinery (Network, Telemetry, Checkpoints).
  - `integrations/`: Framework connectors (Lightning, HuggingFace, etc.).
  - `client.py`: The main Argus orchestrator.
  - `local.py`: The LocalDetector forensic engine.
- `tests/`: Forensic validation suite.
- `benchmarks/`: Performance auditing tools (`overhead.py`).

---

## Links

- [Dashboard](https://argus.plarv.com)
- [Documentation](https://plarv.com/resources/documentation)
- [GitHub](https://github.com/plarv/plarv-argus-sdk)
- [Homepage](https://plarv.com)