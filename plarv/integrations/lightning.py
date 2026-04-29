"""
PLARV Argus — PyTorch Lightning Callback
========================================
2-line integration for PyTorch Lightning.

Usage:
    from plarv import ArgusLightningCallback
    trainer = pl.Trainer(callbacks=[ArgusLightningCallback(api_key="your-key")])
"""

from typing import Optional
from ..argus import Argus
from ..exceptions import ArgusPause, ArgusCheckpoint

try:
    from pytorch_lightning import Callback as _Base
except ImportError:
    try:
        from lightning import Callback as _Base
    except ImportError:
        _Base = object


class ArgusLightningCallback(_Base):
    """
    PyTorch Lightning Callback wrapper for PLARV Argus.

    Args:
        api_key:  Your PLARV Argus API key.
        run_id:   Optional run identifier.
        mode:     "MANUAL" or "AUTO".
        silent:   Suppress console output.
    """

    def __init__(
        self,
        api_key: str,
        run_id:  Optional[str] = None,
        mode:    str           = "MANUAL",
        silent:  bool          = False,
    ):
        super().__init__()
        self._api_key = api_key
        self._run_id  = run_id
        self._mode    = mode
        self._silent  = silent
        self._argus: Optional[Argus] = None

    def on_train_start(self, trainer, pl_module):
        self._argus = Argus(
            api_key=self._api_key,
            run_id=self._run_id,
            optimizer=trainer.optimizers[0] if trainer.optimizers else None,
            model=pl_module,
            mode=self._mode,
            silent=self._silent,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._argus is None:
            return

        # outputs can be: dict with "loss", a raw tensor, or None
        # depending on Lightning version and what training_step returns
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
        elif hasattr(outputs, "detach"):
            loss = outputs
        else:
            loss = None

        if loss is None:
            return

        grad_norm = trainer.callback_metrics.get("grad_norm")

        # Modernized Argus handles optional grad_norm and silent exceptions internally
        self._argus.step(
            loss=float(loss),
            grad_norm=float(grad_norm) if grad_norm is not None else None,
            epoch=trainer.current_epoch,
        )

        if self._argus._should_stop:
            trainer.should_stop = True

    def on_train_end(self, trainer, pl_module):
        if self._argus:
            self._argus.complete()