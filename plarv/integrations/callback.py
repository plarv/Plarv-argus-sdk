"""
PLARV Argus — HuggingFace Trainer Callback
==========================================
2-line integration for HuggingFace Trainer.
Includes certified rolling checkpoint support.

Usage:
    from plarv import ArgusCallback

    # Option A: pass api_key directly (simplest)
    trainer = Trainer(callbacks=[ArgusCallback(api_key="your-key")])

    # Option B: pass a pre-built Argus instance (for Unsloth/Axolotl integrations)
    argus = Argus(api_key="your-key", model=model, ...)
    trainer = Trainer(callbacks=[ArgusCallback(argus=argus)])
"""

from typing import Optional, Any
from ..argus import Argus
from ..exceptions import ArgusPause, ArgusCheckpoint

try:
    from transformers import TrainerCallback
    _Base = TrainerCallback
except ImportError:
    _Base = object


class ArgusCallback(_Base):
    """
    HuggingFace TrainerCallback wrapper for PLARV Argus.

    Can be constructed two ways:

    1. With api_key (Argus is created internally on train begin):
        ArgusCallback(api_key="your-key")

    2. With a pre-built Argus instance (used by Unsloth/Axolotl integrations):
        ArgusCallback(argus=argus_instance)

    Args:
        api_key:                  Your PLARV Argus API key. Required if argus not provided.
        argus:                    Pre-built Argus instance. If provided, api_key is ignored.
        run_id:                   Optional run identifier (only used when api_key path is taken).
        mode:                     "MANUAL" or "AUTO" (only used when api_key path is taken).
        silent:                   Suppress console output.
        checkpoint_every_n_steps: Kept for API compatibility — unused internally.
        checkpoint_keep_last:     Kept for API compatibility — unused internally.
        checkpoint_dir:           Where to save checkpoints (default: "./argus-checkpoints").
    """

    def __init__(
        self,
        api_key:                  Optional[str] = None,
        argus:                    Optional[Argus] = None,
        run_id:                   Optional[str] = None,
        mode:                     str           = "MANUAL",
        silent:                   bool          = False,
        checkpoint_every_n_steps: int           = 0,
        checkpoint_keep_last:     int           = 2,
        checkpoint_dir:           str           = "./argus-checkpoints",
    ):
        if argus is None and api_key is None:
            raise ValueError(
                "[PLARV] ArgusCallback requires either api_key= or argus=. "
                "Example: ArgusCallback(api_key='your-key')"
            )

        # Store pre-built instance if provided — skip internal construction
        self._argus: Optional[Argus] = argus

        # Store api_key path params (only used if argus is None)
        self._api_key                  = api_key
        self._run_id                   = run_id
        self._mode                     = mode
        self._silent                   = silent if argus is None else getattr(argus, "silent", silent)
        self._checkpoint_every_n_steps = checkpoint_every_n_steps
        self._checkpoint_keep_last     = checkpoint_keep_last
        self._checkpoint_dir           = checkpoint_dir

    # =========================================================================
    # HuggingFace Trainer hooks
    # =========================================================================

    def on_train_begin(self, args, state, control, **kwargs):
        # If a pre-built Argus instance was injected, use it as-is.
        # Only build internally when api_key path is taken.
        if self._argus is None:
            model     = kwargs.get("model")
            optimizer = kwargs.get("optimizer")
            tokenizer = kwargs.get("tokenizer")

            self._argus = Argus(
                api_key=self._api_key,
                run_id=self._run_id,
                optimizer=optimizer,
                model=model,
                tokenizer=tokenizer,
                mode=self._mode,
                silent=self._silent,
                checkpoint_every_n_steps=self._checkpoint_every_n_steps,
                checkpoint_keep_last=self._checkpoint_keep_last,
                checkpoint_dir=self._checkpoint_dir,
            )

        # Force per-step logging so Argus sees every step.
        # Default HuggingFace logging_steps=500 means Argus would miss 499/500 steps.
        if hasattr(args, "logging_steps") and args.logging_steps > 1:
            args.logging_steps = 1
            if not self._silent:
                print("[PLARV] logging_steps set to 1 for step-level monitoring.")

        # Override checkpoint save to use trainer.save_model() when available
        if self._argus._ckpt is not None:
            trainer_ref = kwargs.get("trainer") or self._get_trainer_ref(kwargs)
            if trainer_ref is not None:
                def hf_save_fn(path: str):
                    trainer_ref.save_model(path)
                self._argus._ckpt.register_save_fn(hf_save_fn)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._argus is None or logs is None:
            return

        loss      = logs.get("loss") or logs.get("train_loss")
        grad_norm = logs.get("grad_norm")

        if loss is None:
            return

        response = self._argus.step(
            loss=float(loss),
            grad_norm=float(grad_norm) if grad_norm is not None else None,
            epoch=int(state.epoch or 0),
        )

        # 🛡️ NaN RECOVERY BRIDGE: clear optimizer state if engine signals reset
        if response.get("reset_optimizer") and self._argus.optimizer:
            self._argus.optimizer.state.clear()
            if not self._silent:
                print("[PLARV] [CRITICAL] RESET_OPTIMIZER RECEIVED: Clearing momentum buffers.")

        if self._argus._should_stop:
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if self._argus:
            self._argus.complete()

    # =========================================================================
    # INTERNAL
    # =========================================================================

    def _get_trainer_ref(self, kwargs: dict):
        """Try to find trainer in kwargs — HuggingFace passes it differently across versions."""
        for key in ("trainer", "model_trainer", "training_args"):
            val = kwargs.get(key)
            if val is not None and hasattr(val, "save_model"):
                return val
        return None