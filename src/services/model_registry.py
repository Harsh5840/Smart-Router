"""
Model registry - manages available models and their configurations
"""

from typing import Dict, Optional
from src.services.llm_client import LLMClient, ModelConfig
from src.services.model_clients import LlamaClient, OpenAIClient, ClaudeClient
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# PHASE 1: Model Configurations
# ============================================================================

MODEL_CONFIGS = {
    "llama-7b": ModelConfig(
        name="llama-7b",
        provider="local",
        cost_per_1k_tokens=0.0,  # Self-hosted
        max_tokens=2048,
        supports_streaming=True,
        avg_latency_ms=500,
        quality_tier="medium",
    ),
    "gpt-4": ModelConfig(
        name="gpt-4",
        provider="openai",
        cost_per_1k_tokens=0.03,
        max_tokens=8192,
        supports_streaming=True,
        avg_latency_ms=1000,
        quality_tier="high",
    ),
    "claude-sonnet": ModelConfig(
        name="claude-sonnet",
        provider="anthropic",
        cost_per_1k_tokens=0.015,
        max_tokens=4096,
        supports_streaming=True,
        avg_latency_ms=800,
        quality_tier="high",
    ),
}


class ModelRegistry:
    """
    Central registry for all available models
    Provides factory methods for creating model clients
    """

    def __init__(self):
        self._clients: Dict[str, LLMClient] = {}
        self._initialize_clients()

    def _initialize_clients(self) -> None:
        """Initialize all model clients"""
        for model_name, config in MODEL_CONFIGS.items():
            try:
                client = self._create_client(config)
                self._clients[model_name] = client
                logger.info("model_initialized", model=model_name, provider=config.provider)
            except Exception as e:
                logger.error("model_init_failed", model=model_name, error=str(e))

    def _create_client(self, config: ModelConfig) -> LLMClient:
        """Factory method to create appropriate client"""
        if config.provider == "local":
            return LlamaClient(config)
        elif config.provider == "openai":
            return OpenAIClient(config)
        elif config.provider == "anthropic":
            return ClaudeClient(config)
        else:
            raise ValueError(f"Unknown provider: {config.provider}")

    def get_client(self, model_name: str) -> Optional[LLMClient]:
        """Get a model client by name"""
        return self._clients.get(model_name)

    def get_all_models(self) -> Dict[str, ModelConfig]:
        """Get all available model configurations"""
        return MODEL_CONFIGS

    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available"""
        return model_name in self._clients

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all models"""
        health_status = {}
        for model_name, client in self._clients.items():
            try:
                health_status[model_name] = await client.health_check()
            except Exception as e:
                logger.error("health_check_failed", model=model_name, error=str(e))
                health_status[model_name] = False
        return health_status


# Global model registry instance
model_registry = ModelRegistry()
