import os
import logging
from typing import Any, Dict, Optional

# Attempt to import Axolotl's BasePlugin if available in the environment
try:
    from axolotl.integrations.base import BasePlugin
except ImportError:
    # Fallback for environments where axolotl is not installed locally
    class BasePlugin:
        pass

from plarv.argus import Argus
from plarv.callback import ArgusCallback

logger = logging.getLogger("axolotl.integrations.plarv")

class ArgusAxolotlPlugin(BasePlugin):
    """
    PLARV Argus Integration for Axolotl.
    Enables one-line integration for the most popular LLM fine-tuning repo.
    """

    def register(self, cfg: Dict[str, Any]):
        """
        Check if Argus is enabled in the config.
        Expected config:
        
        plugins:
          - plarv.integrations.axolotl.ArgusAxolotlPlugin
        argus_api_key: "your-key"
        argus_run_id: "optional-run-id"
        """
        self.api_key = cfg.get("argus_api_key")
        self.run_id = cfg.get("argus_run_id")
        self.enabled = bool(self.api_key)
        
        if self.enabled:
            logger.info(f"[PLARV] Argus Plugin registered for run: {self.run_id or 'auto'}")
        else:
            logger.debug("[PLARV] Argus Plugin disabled (no api_key found in cfg)")

    def post_trainer_create(self, cfg: Dict[str, Any], trainer: Any):
        """
        Attach the ArgusCallback to the trainer after it's created.
        """
        if not self.enabled:
            return

        logger.info("[PLARV] Injecting Argus Callback into Axolotl trainer...")
        
        # Initialize Argus with the model and tokenizer from the trainer/cfg
        # We use the simplified Argus API for 30-second integration
        argus = Argus(
            api_key=self.api_key,
            run_id=self.run_id,
            model=getattr(trainer, "model", None),
            tokenizer=getattr(trainer, "tokenizer", None),
            catch_exceptions=True,
            silent=False
        )
        
        # Create and attach the HuggingFace-compatible callback
        callback = ArgusCallback(argus=argus)
        trainer.add_callback(callback)
        
        logger.info("[PLARV] Argus clinical monitoring active.")
