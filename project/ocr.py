"""
DrugFX OCR Module
=================
Extracts ALL text from drug label images.
Pipeline:
  1. Tesseract OCR with preprocessing (if installed)
  2. Gemini Vision fallback (primary if Tesseract unavailable)
"""

import os
import logging

logger = logging.getLogger(__name__)

# ─── Try Tesseract ─────────────────────────────────────────────
try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageEnhance
    HAS_TESSERACT = True
    logger.info("OCR: Tesseract available.")
except ImportError:
    HAS_TESSERACT = False
    logger.info("OCR: Tesseract/Pillow not installed — will use Gemini Vision.")


# ─── Image Preprocessing ───────────────────────────────────────
def _preprocess_image(image: "Image.Image") -> "Image.Image":
    """Converts to grayscale, scales up 2x, sharpens, boosts contrast."""
    image = image.convert("L")              # Grayscale
    w, h = image.size
    image = image.resize((w * 2, h * 2), Image.LANCZOS)  # Scale 2x
    image = image.filter(ImageFilter.SHARPEN)
    image = image.filter(ImageFilter.SHARPEN)
    image = ImageEnhance.Contrast(image).enhance(2.0)
    return image


# ─── Gemini Vision OCR ─────────────────────────────────────────
def _extract_via_gemini_vision(image_path: str) -> str:
    """
    Uses Google Gemini Vision to extract all text from a drug/medicine label.
    Returns the raw extracted string, or empty string on failure.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key in ("your_gemini_api_key_here", ""):
        logger.warning("OCR (Gemini Vision): No GEMINI_API_KEY set.")
        return ""

    try:
        from google import genai
        from PIL import Image as PILImage

        client = genai.Client(api_key=api_key)
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

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[extraction_prompt, pil_image]
        )

        extracted = response.text.strip() if response.text else ""
        logger.info(f"OCR (Gemini Vision): Extracted {len(extracted)} characters.")
        return extracted

    except Exception as e:
        logger.error(f"OCR (Gemini Vision) failed: {e}")
        return ""


# ─── Main Entry Point ──────────────────────────────────────────
def extract_text_from_image(image_path: str) -> str:
    """
    Extracts all text from a drug label image.

    Pipeline:
      1. Tesseract OCR with preprocessing (if available)
      2. Gemini Vision fallback

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        Extracted text string, or empty string if all methods fail.
    """
    if not os.path.exists(image_path):
        logger.error(f"OCR: Image file not found: '{image_path}'")
        return ""

    extracted = ""

    # ── Method 1: Tesseract ──────────────────────────────────
    if HAS_TESSERACT:
        try:
            image = Image.open(image_path)
            preprocessed = _preprocess_image(image)
            # PSM 6: Uniform block of text | OEM 3: Default LSTM engine
            config = r"--oem 3 --psm 6"
            extracted = pytesseract.image_to_string(preprocessed, config=config).strip()

            if extracted and len(extracted) > 10:
                logger.info(f"OCR (Tesseract): Extracted {len(extracted)} characters.")
                return extracted
            else:
                logger.warning("OCR (Tesseract): Returned short/empty result, trying Gemini Vision.")
        except Exception as e:
            logger.error(f"OCR (Tesseract) error: {e}")

    # ── Method 2: Gemini Vision ──────────────────────────────
    logger.info("OCR: Attempting Gemini Vision extraction...")
    extracted = _extract_via_gemini_vision(image_path)

    if extracted:
        return extracted

    logger.warning("OCR: All extraction methods failed.")
    return ""
