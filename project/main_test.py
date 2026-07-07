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

from agent import run_drug_analysis_agent
from ocr import extract_text_structured

def test_drug_text():
    print("\n" + "="*45)
    print("--- Testing New Drug Analysis Agent (Text) ---")
    print("="*45)
    try:
        # Test using the new agent entry point
        result = run_drug_analysis_agent("Aspirin")
        print(json.dumps(result, indent=2))
        print("SUCCESS: Text analysis completed successfully.")
    except Exception as e:
        print(f"FAILED: {e}")

def test_ocr_flow():
    print("\n" + "="*45)
    print("--- Testing New OCR Flow (Real image path) ---")
    print("="*45)
    
    # Use absolute path to ensure it's found regardless of where the script is run from
    image_path = os.path.join(PROJECT_ROOT, "mock_drug_label.png")
    
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found at expected location.")
        return

    print(f"Reading from: {image_path}")
    ocr_result = extract_text_structured(image_path)
    print(f"Extracted OCR Result:")
    print(f"  - Provider: {ocr_result.get('provider')}")
    print(f"  - Confidence: {ocr_result.get('confidence')}")
    print(f"  - Text Length: {len(ocr_result.get('text', ''))}")
    
    extracted_text = ocr_result.get('text', '')
    if not extracted_text.strip():
        print("Skipping drug analysis: No text was extracted from the image.")
        return

    print("\nRouting extracted text to New Agent:")
    try:
        result = run_drug_analysis_agent(extracted_text)
        print(json.dumps(result, indent=2))
        print("SUCCESS: Image analysis OCR flow completed successfully.")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    print("Starting AI Pipeline Test (DrugFX v3.0)...")
    test_drug_text()
    test_ocr_flow()
    print("\nTest completed.")
