"""
DrugFX AI Agent
===============
Orchestrates the full pipeline:
  1. Metadata extraction (drug name, MFG, expiry)
  2. RAG context retrieval
  3. LLM comprehensive analysis with multi-model fallback
  4. Returns fully structured, validated response
"""

import os
import re
import json
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Load .env as early as possible
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────
# LLM Client — Lazy Initialized
# ─────────────────────────────────────────────────────────────
_gemini_client = None
_genai_types = None
_client_initialized = False


def _get_gemini_client():
    """Lazy-init the Gemini client so env vars are always resolved at call time."""
    global _gemini_client, _genai_types, _client_initialized
    if _client_initialized:
        return _gemini_client, _genai_types

    _client_initialized = True
    api_key = os.environ.get("GEMINI_API_KEY", "")
    try:
        from google import genai as _genai_mod
        from google.genai import types as _types
        _genai_types = _types
        if api_key and api_key not in ("your_gemini_api_key_here", ""):
            _gemini_client = _genai_mod.Client(api_key=api_key)
            logger.info("DrugFX Agent: Gemini client initialized successfully.")
        else:
            logger.warning("DrugFX Agent: GEMINI_API_KEY not set — will use mock responses.")
    except ImportError:
        logger.warning("DrugFX Agent: google-genai not installed. Run: pip install google-genai")

    return _gemini_client, _genai_types


def _get_client():
    return _get_gemini_client()[0]


def _get_types():
    return _get_gemini_client()[1]


# ─────────────────────────────────────────────────────────────
# RAG Retrieval
# ─────────────────────────────────────────────────────────────
def _get_rag_context(query: str) -> str:
    """Retrieve relevant drug context from knowledge base."""
    try:
        from rag.retriever import retrieve_context
        ctx = retrieve_context(query, top_k=3)
        return ctx if ctx else ""
    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────
# Metadata Parser (MFG / Expiry from OCR text)
# ─────────────────────────────────────────────────────────────
def _parse_label_metadata(text: str) -> dict:
    """
    Scans extracted OCR text for manufacturing date, expiry date,
    batch/lot number, and drug name hints.
    Returns a dict with keys: mfg_date, expiry_date, batch_no, detected_drug_name
    """
    metadata = {
        "mfg_date": None,
        "expiry_date": None,
        "batch_no": None,
        "detected_drug_name": None
    }

    if not text:
        return metadata

    text_upper = text.upper()

    # --- Expiry Date patterns ---
    exp_patterns = [
        r"(?:EXP(?:IRY)?|EXPIRY DATE|USE BEFORE|USE BY|BEST BEFORE)[:\s.]+([A-Z]{3}[\s/-]?\d{2,4}|\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
        r"(?:EXP)[:\s.]+(\d{1,2}[/]\d{2,4})",
        r"\bEXP\.?\s*:?\s*([A-Z]{3,9}[\s\-]?\d{2,4}|\d{1,2}[/\-]\d{2,4})",
    ]
    for pat in exp_patterns:
        m = re.search(pat, text_upper)
        if m:
            metadata["expiry_date"] = m.group(1).strip()
            break

    # --- MFG Date patterns ---
    mfg_patterns = [
        r"(?:MFG(?:\.)?|MANUFACTURED(?:\s+ON)?|DATE OF MFG(?:\.)?|MFD)[:\s.]+([A-Z]{3}[\s/-]?\d{2,4}|\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
        r"\bMFG\.?\s*:?\s*([A-Z]{3,9}[\s\-]?\d{2,4}|\d{1,2}[/\-]\d{2,4})",
    ]
    for pat in mfg_patterns:
        m = re.search(pat, text_upper)
        if m:
            metadata["mfg_date"] = m.group(1).strip()
            break

    # --- Batch / Lot Number ---
    batch_patterns = [
        r"(?:BATCH\s*(?:NO|NUMBER)?|LOT\s*(?:NO|NUMBER)?|LOT#)[:\s.]+([A-Z0-9\-]+)",
    ]
    for pat in batch_patterns:
        m = re.search(pat, text_upper)
        if m:
            metadata["batch_no"] = m.group(1).strip()
            break

    # --- Drug name: typically the first prominent word/line ---
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if lines:
        candidate = lines[0]
        if len(candidate) < 60:
            metadata["detected_drug_name"] = candidate

    return metadata


# ─────────────────────────────────────────────────────────────
# LLM Call — Multi-model Fallback + Retry
# ─────────────────────────────────────────────────────────────
def _call_gemini(prompt: str, system: str, json_mode: bool = True) -> str:
    """
    Calls Gemini with automatic model fallback cascade and retry on 429/rate-limit.
    Tries in order: gemini-2.0-flash -> gemini-1.5-flash -> gemini-1.5-flash-8b
    """
    client = _get_client()
    types = _get_types()

    if not client:
        logger.warning("_call_gemini: No Gemini client available.")
        return ""

    # Model cascade — real available models in order of preference
    # 2.5 is confirmed working, 2.0 is quota exhausted
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-flash-latest",
    ]

    full_prompt = f"{system}\n\n{prompt}"
    config_kwargs = {}
    if json_mode and types:
        config_kwargs["response_mime_type"] = "application/json"
    cfg = types.GenerateContentConfig(**config_kwargs) if config_kwargs and types else None

    for model in models_to_try:
        for attempt in range(2):  # Try each model up to 2x (once + one retry after 429)
            try:
                logger.info(f"_call_gemini: model={model} attempt={attempt + 1}")
                response = client.models.generate_content(
                    model=model,
                    contents=full_prompt,
                    config=cfg
                )
                text = response.text.strip() if response.text else ""
                if text:
                    logger.info(f"_call_gemini: Success with model={model}")
                    return text
            except Exception as e:
                err_str = str(e)
                is_quota = (
                    "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "quota" in err_str.lower()
                )
                if is_quota:
                    if attempt == 0:
                        retry_wait = 4
                        logger.warning(f"_call_gemini: 429 on {model} — waiting {retry_wait}s then retrying...")
                        time.sleep(retry_wait)
                    else:
                        logger.warning(f"_call_gemini: Quota exhausted for {model} — trying next model.")
                        break  # Move to next model
                else:
                    logger.error(f"_call_gemini: Non-quota error on {model}: {e}")
                    break  # Non-quota error; don't retry, try next model

    logger.error("_call_gemini: All models failed or quota exhausted.")
    return ""


# ─────────────────────────────────────────────────────────────
# Mock / Fallback Response
# ─────────────────────────────────────────────────────────────
def _get_mock_response(drug_name: str = "Unknown Drug") -> dict:
    return {
        "drug_name": drug_name,
        "synopsis": (
            f"{drug_name} is a pharmaceutical compound. "
            "AI analysis is currently unavailable because the API quota is temporarily exhausted. "
            "Please wait a few minutes and try again, or update the GEMINI_API_KEY in the .env file."
        ),
        "uses": ["Information temporarily unavailable"],
        "side_effects": [
            "Nausea or stomach upset",
            "Dizziness or lightheadedness",
            "Allergic reactions (rash, itching)",
            "Headache",
            "Consult prescribing information for complete side effects"
        ],
        "key_side_effects": [
            "Severe allergic reaction (anaphylaxis)",
            "Overdose risk — do not exceed recommended dose",
            "Consult your pharmacist for drug-specific risks"
        ],
        "dosage": "As prescribed by your doctor. Do not self-medicate. Consult the package insert.",
        "warnings": [
            "Consult a qualified medical professional before use",
            "Keep out of reach of children",
            "Do not exceed recommended dosage",
            "Store as directed on packaging"
        ],
        "drug_interactions": ["Consult pharmacist for interaction information"],
        "alternatives": ["Consult your doctor for suitable alternatives"],
        "mfg_date": None,
        "expiry_date": None,
        "batch_no": None
    }


# ─────────────────────────────────────────────────────────────
# Core Agent Function
# ─────────────────────────────────────────────────────────────
def run_drug_analysis_agent(
    input_text: str,
    label_metadata: Optional[dict] = None
) -> dict:
    """
    Main agent function. Takes extracted drug text and optional label metadata,
    runs RAG + LLM analysis, and returns a fully structured response.

    Args:
        input_text: Drug name or text description (from user or OCR)
        label_metadata: Pre-parsed dict with mfg_date, expiry_date, batch_no, detected_drug_name

    Returns:
        dict with all structured drug information fields
    """
    if not input_text or not input_text.strip():
        return _get_mock_response()

    # --- Step 1: Extract label metadata if not already provided ---
    if label_metadata is None:
        label_metadata = _parse_label_metadata(input_text)

    # --- Step 2: Determine best drug name for RAG query ---
    rag_query = label_metadata.get("detected_drug_name") or input_text[:200]

    # --- Step 3: RAG context retrieval ---
    logger.info(f"Agent: Retrieving RAG context for: {rag_query[:80]}")
    rag_context = _get_rag_context(rag_query)

    # --- Step 4: Compose the LLM prompt ---
    system_prompt = """You are DrugFX, an expert pharmaceutical AI agent. Your job is to analyze drug/medicine information and return a comprehensive, structured JSON response.

IMPORTANT: Always return ONLY valid JSON with EXACTLY these fields:
{
  "drug_name": "Official drug name (brand and/or generic)",
  "synopsis": "A 2-3 sentence professional summary: what this drug is, its pharmacological class, and primary therapeutic purpose",
  "uses": ["Detailed therapeutic use 1", "use2", "use3"],
  "side_effects": ["Complete list of known side effects — be thorough and specific"],
  "key_side_effects": ["Only the 3-5 MOST CRITICAL side effects patients must urgently know — serious/dangerous ones"],
  "dosage": "Detailed dosage: typical adult dose, frequency, max dose, with food or not, special populations",
  "warnings": ["Important warnings, contraindications, and precautions. ALWAYS include: 'Consult a qualified medical professional before use'"],
  "drug_interactions": ["Specific drugs, foods, or substances it interacts with"],
  "alternatives": ["Alternative medications or treatments for the same condition"]
}

Rules:
- Be medically accurate, specific, and professional
- side_effects should have 6-10 entries minimum
- key_side_effects should be the most serious/dangerous ones only (3-5)
- Never leave arrays empty — always provide at least 2-3 items
- Return ONLY the JSON object, no commentary"""

    mfg_expiry_note = ""
    if label_metadata.get("mfg_date") or label_metadata.get("expiry_date"):
        mfg_expiry_note = (
            f"\n\nLabel metadata detected — MFG: {label_metadata.get('mfg_date', 'Not found')}, "
            f"EXP: {label_metadata.get('expiry_date', 'Not found')}."
        )

    prompt = f"""Analyze this drug/medicine and provide comprehensive pharmaceutical information:

DRUG INPUT:
{input_text[:1000]}

RETRIEVED KNOWLEDGE BASE CONTEXT:
{rag_context if rag_context else 'No specific context retrieved — rely on your pharmaceutical knowledge.'}
{mfg_expiry_note}

Return a complete JSON object with all required fields."""

    # --- Step 5: Call LLM ---
    logger.info("Agent: Calling Gemini LLM for drug analysis...")
    llm_response = _call_gemini(prompt, system_prompt, json_mode=True)

    if not llm_response:
        logger.warning("Agent: LLM call failed or quota exceeded — returning informative mock.")
        result = _get_mock_response(rag_query)
    else:
        try:
            # Strip any markdown code fences if model returned them
            clean = llm_response.strip()
            if clean.startswith("```"):
                clean = re.sub(r'^```[a-z]*\n?', '', clean)
                clean = re.sub(r'\n?```$', '', clean)
            result = json.loads(clean)
        except json.JSONDecodeError as e:
            logger.error(f"Agent: Failed to parse LLM JSON: {e} | Response: {llm_response[:200]}")
            result = _get_mock_response(rag_query)

    # --- Step 6: Merge label metadata ---
    result["mfg_date"] = label_metadata.get("mfg_date")
    result["expiry_date"] = label_metadata.get("expiry_date")
    result["batch_no"] = label_metadata.get("batch_no")

    # --- Step 7: Fill any missing required keys ---
    defaults = {
        "drug_name": rag_query,
        "synopsis": "No synopsis available.",
        "uses": [],
        "side_effects": [],
        "key_side_effects": [],
        "dosage": "Consult your prescriber.",
        "warnings": ["Consult a qualified medical professional before use."],
        "drug_interactions": [],
        "alternatives": []
    }
    for key, default_val in defaults.items():
        if key not in result or not result[key]:
            result[key] = default_val

    logger.info(f"Agent: Analysis complete for '{result.get('drug_name', 'Unknown')}'")
    return result
