import logging
from typing import Any, Dict, Optional
from plarv.argus import Argus
from plarv.callback import ArgusCallback

logger = logging.getLogger("plarv.integrations.unsloth")

def patch_unsloth(model: Any, api_key: str, run_id: Optional[str] = None, silent: bool = False):
    """
    Convenience function for Unsloth users.
    Usage:
        from plarv.integrations.unsloth import patch_unsloth
        argus_cb = patch_unsloth(model, api_key="...")
        trainer = Trainer(..., callbacks=[argus_cb])
    """
    logger.info("[PLARV] Initializing Argus for Unsloth...")
    
    # Unsloth models have specific configs we can extract
    config = getattr(model, "config", None)
    
    argus = Argus(
        api_key=api_key,
        run_id=run_id,
        model=model,
        # Auto-extracting Unsloth/HF metadata
        num_layers=getattr(config, "num_hidden_layers", 0),
        hidden_dim=getattr(config, "hidden_size", 0),
        vocab_size=getattr(config, "vocab_size", 0),
        model_type="llm",  # Unsloth is optimized for LLMs
        silent=silent,
        catch_exceptions=True
    )
    
    return ArgusCallback(argus=argus)

class UnslothArgusCallback(ArgusCallback):
    """
    Optimized callback for Unsloth FastLanguageModel training.
    """
    def __init__(self, api_key: str, run_id: Optional[str] = None, silent: bool = False):
        self.api_key = api_key
        self.run_id = run_id
        self.silent = silent
        self.argus = None
        super().__init__(argus=None)  # Delayed initialization

    def on_init_end(self, args, state, control, model=None, **kwargs):
        """Initialize Argus when the trainer starts."""
        if self.argus is None and model is not None:
            self.argus = Argus(
                api_key=self.api_key,
                run_id=self.run_id,
                model=model,
                silent=self.silent,
                catch_exceptions=True
            )
        super().on_init_end(args, state, control, model=model, **kwargs)
