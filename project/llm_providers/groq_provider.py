"""
DrugFX — Groq LLM Provider
============================
Uses the Groq API (OpenAI-compatible) with multi-model fallback and retry.
"""

import time
import logging
from typing import Optional

from . import BaseLLMProvider

logger = logging.getLogger(__name__)

# Lazy-loaded SDK
_groq_client = None
_client_init_attempted = False


def _get_groq_client():
    """Lazy-init the Groq client."""
    global _groq_client, _client_init_attempted
    if _client_init_attempted:
        return _groq_client

    _client_init_attempted = True
    from config import settings

    api_key = settings.GROQ_API_KEY
    if not api_key or api_key in ("your_groq_api_key_here", ""):
        logger.warning("GroqProvider: GROQ_API_KEY not set.")
        return None

    try:
        from groq import Groq
        _groq_client = Groq(api_key=api_key)
        logger.info("GroqProvider: Client initialized successfully.")
    except ImportError:
        logger.error("GroqProvider: 'groq' package not installed. Run: pip install groq")
    except Exception as e:
        logger.error(f"GroqProvider: Failed to initialize client: {e}")

    return _groq_client


class GroqProvider(BaseLLMProvider):
    """Groq inference provider with multi-model fallback and exponential backoff."""

    @property
    def name(self) -> str:
        return "Groq"

    def is_available(self) -> bool:
        return _get_groq_client() is not None

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        client = _get_groq_client()
        if not client:
            logger.warning("GroqProvider: No client available.")
            return ""

        from config import settings

        models_to_try = [
            settings.GROQ_MODEL_PRIMARY,
            settings.GROQ_MODEL_FALLBACK,
        ]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for model in models_to_try:
            kwargs["model"] = model

            for attempt in range(settings.LLM_MAX_RETRIES):
                try:
                    logger.info(
                        f"GroqProvider: model={model} attempt={attempt + 1}/{settings.LLM_MAX_RETRIES}"
                    )
                    response = client.chat.completions.create(**kwargs)

                    text = ""
                    if response.choices and response.choices[0].message:
                        text = (response.choices[0].message.content or "").strip()

                    if text:
                        logger.info(f"GroqProvider: Success with model={model}")
                        return text
                    else:
                        logger.warning(f"GroqProvider: Empty response from {model}")

                except Exception as e:
                    err_str = str(e).lower()
                    is_rate_limit = any(
                        kw in err_str
                        for kw in ["429", "rate_limit", "rate limit", "quota", "resource_exhausted"]
                    )
                    is_server_error = any(kw in err_str for kw in ["503", "502", "500", "unavailable"])

                    if is_rate_limit or is_server_error:
                        delay = settings.LLM_RETRY_BASE_DELAY * (2 ** attempt)
                        if attempt < settings.LLM_MAX_RETRIES - 1:
                            logger.warning(
                                f"GroqProvider: {'Rate limited' if is_rate_limit else 'Server error'} "
                                f"on {model} — retrying in {delay:.1f}s..."
                            )
                            time.sleep(delay)
                        else:
                            logger.warning(
                                f"GroqProvider: Exhausted retries for {model} — trying next model."
                            )
                            break
                    else:
                        logger.error(f"GroqProvider: Non-retryable error on {model}: {e}")
                        break  # Don't retry non-transient errors

        logger.error("GroqProvider: All models and retries exhausted.")
        return ""
