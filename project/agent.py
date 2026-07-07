"""
DrugFX AI Agent
===============
Orchestrates the full analysis pipeline:
  1. Metadata extraction (drug name, MFG, expiry, batch)
  2. RAG context retrieval with relevance scoring
  3. LLM comprehensive analysis via provider abstraction
  4. Confidence scoring based on RAG quality + model response
  5. Returns fully structured, validated response

All LLM calls go through llm_providers for Groq/Gemini swappability.
"""

import os
import re
import json
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
# RAG Retrieval
# ─────────────────────────────────────────────────────────────
def _get_rag_context(query: str) -> tuple:
    """
    Retrieve relevant drug context from knowledge base.
    Returns (context_string, avg_relevance_score).
    """
    try:
        from rag.retriever import retrieve_context_with_scores
        ctx, score = retrieve_context_with_scores(query, top_k=5)
        return (ctx or "", score)
    except ImportError:
        try:
            from rag.retriever import retrieve_context
            ctx = retrieve_context(query, top_k=5)
            return (ctx or "", 0.5)
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return ("", 0.0)
    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        return ("", 0.0)


# ─────────────────────────────────────────────────────────────
# Metadata Parser (MFG / Expiry / Batch from OCR text)
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
        "detected_drug_name": None,
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
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        candidate = lines[0]
        if len(candidate) < 60:
            metadata["detected_drug_name"] = candidate

    return metadata


# ─────────────────────────────────────────────────────────────
# LLM Call via Provider Abstraction
# ─────────────────────────────────────────────────────────────
_llm_provider = None


def _get_provider():
    """Lazy-init the LLM provider."""
    global _llm_provider
    if _llm_provider is not None:
        return _llm_provider

    try:
        from llm_providers import get_llm_provider
        _llm_provider = get_llm_provider()
        logger.info(f"Agent: Using LLM provider: {_llm_provider.name}")
    except Exception as e:
        logger.error(f"Agent: Failed to initialize LLM provider: {e}")
        _llm_provider = None

    return _llm_provider


def _call_llm(prompt: str, system: str, json_mode: bool = True) -> str:
    """
    Call the configured LLM provider.
    Returns raw text response or empty string on failure.
    """
    provider = _get_provider()
    if not provider or not provider.is_available():
        logger.warning("Agent: LLM provider not available.")
        return ""

    try:
        from config import settings
        return provider.generate(
            prompt=prompt,
            system_prompt=system,
            json_mode=json_mode,
            temperature=settings.GROQ_TEMPERATURE,
            max_tokens=settings.GROQ_MAX_TOKENS,
        )
    except Exception as e:
        logger.error(f"Agent: LLM call failed: {e}")
        return ""


# ─────────────────────────────────────────────────────────────
# Confidence Scoring
# ─────────────────────────────────────────────────────────────
def _calculate_confidence(rag_score: float, llm_response: dict) -> dict:
    """
    Calculate confidence based on:
    - RAG retrieval relevance score
    - Completeness of LLM response
    """
    # RAG component (0–1)
    rag_component = min(1.0, rag_score)

    # Completeness component: how many key fields are populated
    key_fields = [
        "drug_name", "generic_name", "synopsis", "primary_uses",
        "common_side_effects", "dosage", "warnings",
    ]
    filled = sum(1 for f in key_fields if llm_response.get(f))
    completeness = filled / len(key_fields)

    # Weighted score
    score = (rag_component * 0.4) + (completeness * 0.6)
    score = round(min(1.0, max(0.0, score)), 2)

    if score >= 0.75:
        level = "high"
    elif score >= 0.45:
        level = "medium"
    else:
        level = "low"

    source = "both" if rag_component > 0.3 else "llm"
    if rag_component > 0.7 and completeness > 0.7:
        source = "rag+llm"

    return {
        "level": level,
        "score": score,
        "source": source,
    }


# ─────────────────────────────────────────────────────────────
# Mock / Fallback Response
# ─────────────────────────────────────────────────────────────
def _get_mock_response(drug_name: str = "Unknown Drug") -> dict:
    return {
        "drug_name": drug_name,
        "generic_name": "",
        "brand_names": [],
        "drug_class": "",
        "composition": "",
        "synopsis": (
            f"{drug_name} is a pharmaceutical compound. "
            "AI analysis is currently unavailable — the LLM provider may be temporarily "
            "unreachable or the API key may not be configured. Please check your .env file "
            "and try again."
        ),
        "primary_uses": ["Information temporarily unavailable"],
        "dosage": {
            "adult": "Consult your prescriber",
            "pediatric": "Consult your prescriber",
            "elderly": "Consult your prescriber",
            "frequency": "",
            "max_dose": "",
            "with_food": "",
        },
        "administration": "",
        "warnings": [
            "Consult a qualified medical professional before use",
            "Keep out of reach of children",
            "Do not exceed recommended dosage",
        ],
        "contraindications": [],
        "drug_interactions": [],
        "pregnancy_safety": "Consult your doctor",
        "breastfeeding_safety": "Consult your doctor",
        "alcohol_interaction": "Consult your pharmacist",
        "driving_advisory": "Consult your doctor",
        "storage": "Store as directed on packaging",
        "common_side_effects": [
            "Nausea or stomach upset",
            "Dizziness or lightheadedness",
            "Headache",
        ],
        "serious_side_effects": [
            "Severe allergic reaction (anaphylaxis)",
            "Consult your pharmacist for drug-specific risks",
        ],
        "missed_dose": "Take the missed dose as soon as you remember unless it is close to your next dose.",
        "overdose_guidance": "Seek emergency medical attention immediately.",
        "faq": [],
        "alternatives": [],
        "confidence": {"level": "low", "score": 0.15, "source": "fallback"},
        "mfg_date": None,
        "expiry_date": None,
        "batch_no": None,
    }


# ─────────────────────────────────────────────────────────────
# System Prompt — Expanded Schema
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are DrugFX, an expert pharmaceutical AI agent. Your job is to analyze drug/medicine information and return a comprehensive, structured JSON response.

IMPORTANT: Always return ONLY valid JSON with EXACTLY these fields:
{
  "drug_name": "Official drug name (brand and/or generic, e.g. 'Aspirin')",
  "generic_name": "Generic/INN name (e.g. 'Acetylsalicylic acid')",
  "brand_names": ["Known brand names"],
  "drug_class": "Pharmacological class (e.g. 'NSAID', 'Antibiotic', 'Statin')",
  "composition": "Active ingredients and strengths (e.g. 'Aspirin 500mg')",
  "synopsis": "A 2-3 sentence professional summary: what this drug is, its class, and primary purpose",
  "primary_uses": ["Detailed therapeutic uses — be thorough and specific"],
  "dosage": {
    "adult": "Typical adult dosage",
    "pediatric": "Pediatric dosage or 'Not recommended' or 'Consult pediatrician'",
    "elderly": "Elderly dosage adjustments",
    "frequency": "How often (e.g. 'Every 4-6 hours')",
    "max_dose": "Maximum daily dose",
    "with_food": "Take with food? Before/after meals?"
  },
  "administration": "Route and method (oral, topical, injection, etc.)",
  "warnings": ["Important warnings and precautions. ALWAYS include: 'Consult a qualified medical professional before use'"],
  "contraindications": ["Conditions where this drug should NOT be used"],
  "drug_interactions": ["Specific drugs, foods, or substances it interacts with"],
  "pregnancy_safety": "Safety in pregnancy (e.g. 'Category C — use only if benefit outweighs risk')",
  "breastfeeding_safety": "Safety during breastfeeding",
  "alcohol_interaction": "Interaction with alcohol",
  "driving_advisory": "Effect on driving/operating machinery",
  "storage": "Storage conditions (temperature, light, moisture)",
  "common_side_effects": ["Common/mild side effects — at least 5-8 entries"],
  "serious_side_effects": ["Serious/dangerous side effects requiring medical attention — 3-5 entries"],
  "missed_dose": "What to do if a dose is missed",
  "overdose_guidance": "What to do in case of overdose",
  "faq": [{"q": "Common question", "a": "Answer"}],
  "alternatives": ["Alternative medications or treatments"]
}

Rules:
- Be medically accurate, specific, and professional
- common_side_effects should have 5-8 entries minimum
- serious_side_effects should have 3-5 entries
- Never leave arrays empty — always provide at least 2-3 items
- faq should have 3-5 common questions
- Return ONLY the JSON object, no commentary or markdown fences"""


# ─────────────────────────────────────────────────────────────
# Core Agent Function
# ─────────────────────────────────────────────────────────────
def run_drug_analysis_agent(
    input_text: str,
    label_metadata: Optional[dict] = None,
) -> dict:
    """
    Main agent function. Takes extracted drug text and optional label metadata,
    runs RAG + LLM analysis, and returns a fully structured response.

    Args:
        input_text: Drug name or text description (from user or OCR)
        label_metadata: Pre-parsed dict with mfg_date, expiry_date, batch_no, detected_drug_name

    Returns:
        dict with all structured drug information fields + confidence scoring
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
    rag_context, rag_score = _get_rag_context(rag_query)

    # --- Step 4: Compose the LLM prompt ---
    mfg_expiry_note = ""
    if label_metadata.get("mfg_date") or label_metadata.get("expiry_date"):
        mfg_expiry_note = (
            f"\n\nLabel metadata detected — MFG: {label_metadata.get('mfg_date', 'Not found')}, "
            f"EXP: {label_metadata.get('expiry_date', 'Not found')}."
        )

    prompt = f"""Analyze this drug/medicine and provide comprehensive pharmaceutical information:

DRUG INPUT:
{input_text[:1500]}

RETRIEVED KNOWLEDGE BASE CONTEXT:
{rag_context if rag_context else 'No specific context retrieved — rely on your pharmaceutical knowledge.'}
{mfg_expiry_note}

Return a complete JSON object with all required fields."""

    # --- Step 5: Call LLM ---
    logger.info("Agent: Calling LLM for drug analysis...")
    llm_response = _call_llm(prompt, SYSTEM_PROMPT, json_mode=True)

    if not llm_response:
        logger.warning("Agent: LLM call failed — returning fallback response.")
        result = _get_mock_response(rag_query)
    else:
        try:
            # Strip any markdown code fences if model returned them
            clean = llm_response.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```[a-z]*\n?", "", clean)
                clean = re.sub(r"\n?```$", "", clean)
            result = json.loads(clean)
        except json.JSONDecodeError as e:
            logger.error(f"Agent: Failed to parse LLM JSON: {e} | Response: {llm_response[:200]}")
            result = _get_mock_response(rag_query)

    # --- Step 6: Confidence scoring ---
    confidence = _calculate_confidence(rag_score, result)
    result["confidence"] = confidence

    # --- Step 7: Merge label metadata ---
    result["mfg_date"] = label_metadata.get("mfg_date")
    result["expiry_date"] = label_metadata.get("expiry_date")
    result["batch_no"] = label_metadata.get("batch_no")

    # --- Step 8: Fill any missing required keys ---
    defaults = {
        "drug_name": rag_query,
        "generic_name": "",
        "brand_names": [],
        "drug_class": "",
        "composition": "",
        "synopsis": "No synopsis available.",
        "primary_uses": [],
        "dosage": {"adult": "Consult your prescriber", "pediatric": "", "elderly": "",
                   "frequency": "", "max_dose": "", "with_food": ""},
        "administration": "",
        "warnings": ["Consult a qualified medical professional before use."],
        "contraindications": [],
        "drug_interactions": [],
        "pregnancy_safety": "",
        "breastfeeding_safety": "",
        "alcohol_interaction": "",
        "driving_advisory": "",
        "storage": "",
        "common_side_effects": [],
        "serious_side_effects": [],
        "missed_dose": "",
        "overdose_guidance": "",
        "faq": [],
        "alternatives": [],
    }
    for key, default_val in defaults.items():
        if key not in result or not result[key]:
            result[key] = default_val

    # Handle legacy field mapping (backward compat)
    if "uses" in result and "primary_uses" not in result:
        result["primary_uses"] = result.pop("uses")
    if "side_effects" in result and "common_side_effects" not in result:
        result["common_side_effects"] = result.pop("side_effects")
    if "key_side_effects" in result and "serious_side_effects" not in result:
        result["serious_side_effects"] = result.pop("key_side_effects")

    logger.info(f"Agent: Analysis complete for '{result.get('drug_name', 'Unknown')}' "
                f"[confidence={confidence['level']}, score={confidence['score']}]")
    return result
