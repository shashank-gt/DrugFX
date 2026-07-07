"""
DrugFX LLM Provider Abstraction
================================
Factory + base interface for swappable LLM backends.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract interface that all LLM providers must implement."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """
        Generate a text response from the LLM.

        Args:
            prompt: The user/input prompt.
            system_prompt: System-level instructions.
            json_mode: If True, request JSON-formatted output.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Returns:
            Raw text response from the model, or empty string on failure.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and ready."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""
        ...


def get_llm_provider(provider_name: Optional[str] = None) -> BaseLLMProvider:
    """
    Factory: returns the configured LLM provider instance.

    Args:
        provider_name: Override provider selection ("groq" or "gemini").
                       If None, reads from config.

    Returns:
        An initialized BaseLLMProvider.
    """
    from config import settings

    provider = (provider_name or settings.LLM_PROVIDER).lower().strip()

    if provider == "groq":
        from .groq_provider import GroqProvider
        return GroqProvider()
    elif provider == "gemini":
        from .gemini_provider import GeminiProvider
        return GeminiProvider()
    else:
        logger.warning(f"Unknown LLM provider '{provider}', falling back to Groq.")
        from .groq_provider import GroqProvider
        return GroqProvider()
