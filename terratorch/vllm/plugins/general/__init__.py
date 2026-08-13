from vllm.model_executor.models.registry import ModelRegistry
from vllm.logger import init_logger

def register_terratorch_fix():
    logger = init_logger(__name__)

    try:
        ModelRegistry.register_model(
            "Terratorch",
            "terratorch.vllm.plugins.general.terratorch:TerratorchFix",
        )
        logger.info(
            "Successfully registered TerratorchFix model with vLLM, "
            "this is only required to support vLLM >0.21.0 <=0.24.0 due to a known bug"
        )
    except Exception as e:
        logger.error(f"Failed to register TerratorchFix model: {e}")
        raise
