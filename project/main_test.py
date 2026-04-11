import json
import sys
import os
import logging

# Configure basic logging to see output during tests
logging.basicConfig(level=logging.INFO)

# Ensure the project root is in the path to allow absolute imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from utils.router import route_request
from ocr import extract_text_from_image

def test_drug_text():
    print("\n" + "="*40)
    print("--- Testing Drug Analysis (Text) ---")
    print("="*40)
    try:
        result = route_request("drug", "Cipladine")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"FAILED: {e}")

def test_ocr_flow():
    print("\n" + "="*40)
    print("--- Testing OCR Flow (Real image path) ---")
    print("="*40)
    
    # Use absolute path to ensure it's found regardless of where the script is run from
    image_path = os.path.join(PROJECT_ROOT, "mock_drug_label.png")
    
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found at expected location.")
        return

    print(f"Reading from: {image_path}")
    extracted_text = extract_text_from_image(image_path)
    print(f"Extracted Text: {extracted_text if extracted_text else '[EMPTY - OCR found no text]'}")

    if not extracted_text.strip():
        print("Skipping drug analysis: No text was extracted from the image.")
        return

    print("\nRouting extracted text to Drug module:")
    
    try:
        result = route_request("drug", extracted_text)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    print("Starting AI Pipeline Test (DrugFX Only)...")
    test_drug_text()
    test_ocr_flow()
    print("\nTest completed.")
