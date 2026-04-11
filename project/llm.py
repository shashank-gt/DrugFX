"""
DrugFX LLM Module — kept for backward compatibility.
The main inference is now handled by agent.py.
"""
import os
import json
import logging

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
gemini_client = None
genai_types = None

try:
    from google import genai
    from google.genai import types as _types
    genai_types = _types
    if GEMINI_API_KEY and GEMINI_API_KEY not in ("your_gemini_api_key_here", ""):
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("LLM: Gemini client initialized.")
    else:
        logger.warning("LLM: GEMINI_API_KEY not set.")
except ImportError:
    logger.warning("LLM: google-genai not installed.")


def generate_response(prompt: str, system_prompt: str = "You are a medical AI assistant.",
                      model: str = "gemini-2.0-flash", response_format: str = "text") -> str:
    """
    Sends a prompt to Google Gemini and returns the text response.
    Falls back to mock JSON if the client is not available.
    """
    if not gemini_client:
        logger.warning("LLM: No Gemini client — returning mock.")
        return json.dumps({
            "uses": ["Pain relief", "Fever reduction"],
            "side_effects": ["Nausea", "Dizziness"],
            "dosage": "As prescribed by your doctor",
            "warnings": ["Consult a medical professional before use"],
            "drug_interactions": ["Warfarin", "Alcohol"],
            "alternatives": ["Ibuprofen", "Paracetamol"]
        })

    try:
        full_prompt = f"{system_prompt}\n\n{prompt}"
        config_kwargs = {}
        if response_format == "json_object" and genai_types:
            config_kwargs["response_mime_type"] = "application/json"

        cfg = genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs and genai_types else None
        response = gemini_client.models.generate_content(
            model=model,
            contents=full_prompt,
            config=cfg
        )
        return response.text
    except Exception as e:
        logger.error(f"LLM: Gemini call failed: {e}")
        return json.dumps({"error": str(e)})
