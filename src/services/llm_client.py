"""
Abstract interface for LLM clients
Each provider implements this interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a specific model"""

    name: str
    provider: str
    cost_per_1k_tokens: float
    max_tokens: int
    supports_streaming: bool
    avg_latency_ms: float
    quality_tier: str  # low, medium, high


class LLMClient(ABC):
    """
    Abstract base class for LLM clients
    All model providers must implement this interface
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM

        Returns:
            Dict containing:
                - response: str
                - tokens_used: int
                - latency_ms: float
                - model: str
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the model is available"""
        pass

    def estimate_cost(self, tokens: int) -> float:
        """Estimate cost for a given number of tokens"""
        return (tokens / 1000) * self.config.cost_per_1k_tokens

    def get_model_name(self) -> str:
        """Get the model name"""
        return self.config.name
