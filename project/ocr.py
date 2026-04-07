import os

try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

def extract_text_from_image(image_path: str) -> str:
    """
    Extracts text from an image using Tesseract OCR.
    If Tesseract or Pillow is not installed, returns a mock text based on filename.
    """
    if not os.path.exists(image_path):
        return f"Error: Image '{image_path}' not found."

    if HAS_TESSERACT:
        try:
            image = Image.open(image_path)
            extracted_text = pytesseract.image_to_string(image)
            return extracted_text.strip()
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    else:
        print("WARNING: pytesseract/Pillow not installed or misconfigured. Returning mock extracted text.")
        # Fallback mechanism for demonstration
        if "drug" in image_path.lower() or "aspirin" in image_path.lower():
            return "Aspirin 500mg tablet"
        elif "job" in image_path.lower() or "resume" in image_path.lower():
            return "Data Scientist Required Skills: Python, Machine Learning, SQL"
        return "Mock extracted text from image: Unknown content"
