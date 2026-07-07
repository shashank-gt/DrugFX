"""
DrugFX — Gemini LLM Provider
==============================
Wraps the existing Gemini integration into the provider interface.
Also serves as OCR Vision provider (Groq doesn't support vision).
"""

import time
import logging

from . import BaseLLMProvider

logger = logging.getLogger(__name__)

_gemini_client = None
_genai_types = None
_client_init_attempted = False


def _get_gemini_client():
    """Lazy-init the Gemini client."""
    global _gemini_client, _genai_types, _client_init_attempted
    if _client_init_attempted:
        return _gemini_client, _genai_types

    _client_init_attempted = True
    from config import settings

    api_key = settings.GEMINI_API_KEY
    if not api_key or api_key in ("your_gemini_api_key_here", ""):
        logger.warning("GeminiProvider: GEMINI_API_KEY not set.")
        return None, None

    try:
        from google import genai as _genai_mod
        from google.genai import types as _types
        _genai_types = _types
        _gemini_client = _genai_mod.Client(api_key=api_key)
        logger.info("GeminiProvider: Client initialized successfully.")
    except ImportError:
        logger.warning("GeminiProvider: google-genai not installed. Run: pip install google-genai")
    except Exception as e:
        logger.error(f"GeminiProvider: Failed to initialize: {e}")

    return _gemini_client, _genai_types


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider with multi-model fallback."""

    @property
    def name(self) -> str:
        return "Gemini"

    def is_available(self) -> bool:
        client, _ = _get_gemini_client()
        return client is not None

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        client, types = _get_gemini_client()
        if not client:
            logger.warning("GeminiProvider: No client available.")
            return ""

        from config import settings

        models_to_try = [
            settings.GEMINI_MODEL_PRIMARY,
            settings.GEMINI_MODEL_FALLBACK,
        ]

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        config_kwargs = {}
        if json_mode and types:
            config_kwargs["response_mime_type"] = "application/json"
        cfg = types.GenerateContentConfig(**config_kwargs) if config_kwargs and types else None

        for model in models_to_try:
            for attempt in range(settings.LLM_MAX_RETRIES):
                try:
                    logger.info(
                        f"GeminiProvider: model={model} attempt={attempt + 1}/{settings.LLM_MAX_RETRIES}"
                    )
                    response = client.models.generate_content(
                        model=model,
                        contents=full_prompt,
                        config=cfg,
                    )
                    text = (response.text or "").strip() if response.text else ""
                    if text:
                        logger.info(f"GeminiProvider: Success with model={model}")
                        return text
                except Exception as e:
                    err_str = str(e)
                    is_quota = any(
                        kw in err_str
                        for kw in ["429", "RESOURCE_EXHAUSTED", "quota"]
                    )
                    if is_quota:
                        delay = settings.LLM_RETRY_BASE_DELAY * (2 ** attempt)
                        if attempt < settings.LLM_MAX_RETRIES - 1:
                            logger.warning(
                                f"GeminiProvider: Rate limited on {model} — retrying in {delay:.1f}s..."
                            )
                            time.sleep(delay)
                        else:
                            logger.warning(f"GeminiProvider: Exhausted retries for {model}.")
                            break
                    else:
                        logger.error(f"GeminiProvider: Error on {model}: {e}")
                        break

        logger.error("GeminiProvider: All models and retries exhausted.")
        return ""


# ── Vision API (Gemini-specific, not part of base interface) ──

def extract_text_via_vision(image_path: str) -> str:
    """
    Uses Gemini Vision to extract text from a medicine label image.
    This is separate from the base LLM interface since Groq doesn't support vision.
    """
    client, _ = _get_gemini_client()
    if not client:
        logger.warning("GeminiVision: No client available for OCR.")
        return ""

    try:
        from PIL import Image as PILImage
        pil_image = PILImage.open(image_path)

        extraction_prompt = """You are an expert OCR system specialized in medicine and drug label reading.

Extract ALL visible text from this drug/medicine label image EXACTLY as printed.

Include every piece of text you can see:
- Drug/medicine name (brand and generic)
- Composition / active ingredients and strengths
- Dosage and administration instructions
- Manufactured by (MFG / Mfg.)
- Manufacturing date (Mfg. Date / MFG DATE / Date of Mfg)
- Expiry date (Exp / Expiry / Use Before / EXP DATE)
- Batch number / Lot number
- Storage instructions
- Warnings and precautions
- All other text visible on the label

Return ONLY the extracted text, preserving line breaks. Do NOT add commentary."""

        from config import settings
        response = client.models.generate_content(
            model=settings.GEMINI_MODEL_FALLBACK,
            contents=[extraction_prompt, pil_image],
        )
        extracted = (response.text or "").strip()
        logger.info(f"GeminiVision: Extracted {len(extracted)} characters.")
        return extracted

    except Exception as e:
        logger.error(f"GeminiVision: OCR failed: {e}")
        return ""
