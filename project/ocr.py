"""
DrugFX OCR Module
=================
Extracts text from drug label images with confidence scoring.
Pipeline:
  1. Tesseract OCR with preprocessing (if installed)
  2. Gemini Vision fallback (if Tesseract unavailable or low confidence)

Returns structured output: { text, confidence, provider }
"""

import os
import logging

logger = logging.getLogger(__name__)

# ─── Try Tesseract ─────────────────────────────────────────────
try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageEnhance, ImageOps
    HAS_TESSERACT = True
    logger.info("OCR: Tesseract available.")
except ImportError:
    HAS_TESSERACT = False
    logger.info("OCR: Tesseract/Pillow not installed — will use Gemini Vision.")

# ─── Try Pillow alone (for image preprocessing before Gemini) ──
HAS_PILLOW = False
try:
    from PIL import Image as PILImage
    HAS_PILLOW = True
except ImportError:
    pass


# ─── Image Preprocessing ───────────────────────────────────────
def _preprocess_image(image: "Image.Image") -> "Image.Image":
    """Converts to grayscale, scales up 2x, sharpens, boosts contrast, auto-levels."""
    image = image.convert("L")  # Grayscale
    w, h = image.size
    image = image.resize((w * 2, h * 2), Image.LANCZOS)  # Scale 2x
    image = image.filter(ImageFilter.SHARPEN)
    image = image.filter(ImageFilter.SHARPEN)
    image = ImageEnhance.Contrast(image).enhance(2.0)
    # Auto-level: stretch histogram
    try:
        image = ImageOps.autocontrast(image, cutoff=1)
    except Exception:
        pass
    return image


# ─── Tesseract OCR with confidence ─────────────────────────────
def _extract_via_tesseract(image_path: str) -> dict:
    """
    Uses Tesseract OCR with preprocessing.
    Returns { text, confidence, provider }.
    """
    if not HAS_TESSERACT:
        return {"text": "", "confidence": 0.0, "provider": "tesseract"}

    try:
        image = Image.open(image_path)
        preprocessed = _preprocess_image(image)

        # PSM 6: Uniform block of text | OEM 3: Default LSTM engine
        config = r"--oem 3 --psm 6"
        extracted = pytesseract.image_to_string(preprocessed, config=config).strip()

        # Get confidence data
        confidence = 0.0
        try:
            data = pytesseract.image_to_data(preprocessed, config=config, output_type=pytesseract.Output.DICT)
            confidences = [int(c) for c in data.get("conf", []) if str(c).isdigit() and int(c) > 0]
            if confidences:
                confidence = round(sum(confidences) / len(confidences) / 100.0, 2)
        except Exception:
            confidence = 0.5 if extracted and len(extracted) > 10 else 0.1

        if extracted and len(extracted) > 10:
            logger.info(f"OCR (Tesseract): Extracted {len(extracted)} chars, confidence={confidence}")
            return {"text": extracted, "confidence": confidence, "provider": "tesseract"}
        else:
            logger.warning("OCR (Tesseract): Returned short/empty result.")
            return {"text": extracted, "confidence": 0.1, "provider": "tesseract"}

    except Exception as e:
        logger.error(f"OCR (Tesseract) error: {e}")
        return {"text": "", "confidence": 0.0, "provider": "tesseract"}


# ─── Gemini Vision OCR ─────────────────────────────────────────
def _extract_via_gemini_vision(image_path: str) -> dict:
    """
    Uses Google Gemini Vision to extract all text from a drug/medicine label.
    Returns { text, confidence, provider }.
    """
    try:
        from llm_providers.gemini_provider import extract_text_via_vision
        text = extract_text_via_vision(image_path)
        if text and len(text) > 10:
            # Gemini Vision generally has high accuracy when it returns results
            confidence = 0.85
            logger.info(f"OCR (Gemini Vision): Extracted {len(text)} chars, confidence={confidence}")
            return {"text": text, "confidence": confidence, "provider": "gemini_vision"}
        else:
            return {"text": text, "confidence": 0.1, "provider": "gemini_vision"}
    except Exception as e:
        logger.error(f"OCR (Gemini Vision) failed: {e}")
        return {"text": "", "confidence": 0.0, "provider": "gemini_vision"}


# ─── Main Entry Point ──────────────────────────────────────────
def extract_text_from_image(image_path: str) -> str:
    """
    Extracts all text from a drug label image (backward-compatible string return).

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        Extracted text string, or empty string if all methods fail.
    """
    result = extract_text_structured(image_path)
    return result.get("text", "")


def extract_text_structured(image_path: str) -> dict:
    """
    Extracts text from a drug label image with full structured output.

    Pipeline (based on OCR_PROVIDER config):
      - 'auto': Tesseract first, Gemini Vision fallback if low confidence
      - 'tesseract': Tesseract only
      - 'gemini_vision': Gemini Vision only

    Args:
        image_path: Path to the image file.

    Returns:
        { text: str, confidence: float, provider: str }
    """
    if not os.path.exists(image_path):
        logger.error(f"OCR: Image file not found: '{image_path}'")
        return {"text": "", "confidence": 0.0, "provider": "none"}

    # Determine OCR strategy
    try:
        from config import settings
        ocr_provider = settings.OCR_PROVIDER.lower()
        confidence_threshold = settings.OCR_CONFIDENCE_THRESHOLD
    except Exception:
        ocr_provider = "auto"
        confidence_threshold = 0.5

    # ── Tesseract-only mode ──────────────────────────────────
    if ocr_provider == "tesseract":
        return _extract_via_tesseract(image_path)

    # ── Gemini Vision-only mode ──────────────────────────────
    if ocr_provider == "gemini_vision":
        return _extract_via_gemini_vision(image_path)

    # ── Auto mode: Tesseract first, then Gemini fallback ─────
    if HAS_TESSERACT:
        result = _extract_via_tesseract(image_path)
        if result["text"] and result["confidence"] >= confidence_threshold:
            return result
        logger.info(f"OCR: Tesseract confidence ({result['confidence']}) below threshold "
                    f"({confidence_threshold}), trying Gemini Vision...")

    # Gemini Vision fallback
    logger.info("OCR: Attempting Gemini Vision extraction...")
    result = _extract_via_gemini_vision(image_path)
    if result["text"]:
        return result

    logger.warning("OCR: All extraction methods failed.")
    return {"text": "", "confidence": 0.0, "provider": "none"}
