import json
import sys
import os

# Ensure the project root is in the path to allow absolute imports within standard running
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.router import route_request
from ocr import extract_text_from_image

def test_drug_text():
    print("\n" + "="*40)
    print("--- Testing Drug Analysis (Text) ---")
    print("="*40)
    result = route_request("drug", "Aspirin")
    print(json.dumps(result, indent=2))

def test_job_text():
    print("\n" + "="*40)
    print("--- Testing Job Analysis (Text) ---")
    print("="*40)
    result = route_request("job", "Looking for a Senior Data Scientist skilled in Python, ML, and SQL.")
    print(json.dumps(result, indent=2))

def test_ocr_flow():
    print("\n" + "="*40)
    print("--- Testing OCR Flow (Simulated image) ---")
    print("="*40)
    # mock_drug_label.png doesn't have to exist; ocr.py handles fallback gracefully
    image_path = "mock_drug_label.png"
    print(f"Reading from: {image_path}")
    extracted_text = extract_text_from_image(image_path)
    print(f"Extracted Text: {extracted_text}")
    print("\nRouting extracted text to Drug module:")
    result = route_request("drug", extracted_text)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    print("Starting AI Pipeline Test...")
    test_drug_text()
    test_job_text()
    test_ocr_flow()
    print("\nDone.")
