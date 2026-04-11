"""
DrugFX FastAPI Backend
======================
Serves the frontend and exposes:
  POST /analyze/text   — text drug query
  POST /analyze/image  — image upload with OCR → agent pipeline
"""

import os
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Load env ──────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(
    title="DrugFX AI Assistant API",
    description="AI-powered drug information extraction and analysis",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static dir
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found. Place index.html in the static/ folder."}


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/analyze/text")
def analyze_text(text: str = Form(...)):
    """
    Accepts a drug name / description as form text.
    Runs the full DrugFX agent pipeline and returns structured data.
    """
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    logger.info(f"/analyze/text — input: '{text[:80]}'")

    try:
        from agent import run_drug_analysis_agent, _parse_label_metadata
        metadata = _parse_label_metadata(text)
        result = run_drug_analysis_agent(text, label_metadata=metadata)
        return {
            "success": True,
            "input_type": "text",
            "extracted_text": None,
            "data": result
        }
    except Exception as e:
        logger.error(f"/analyze/text error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded drug label image.
    Pipeline: OCR extraction → metadata parse → DrugFX agent → structured response.
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    # Save temporarily
    safe_filename = os.path.basename(file.filename).replace(" ", "_")
    temp_path = os.path.join(os.path.dirname(__file__), f"_tmp_{safe_filename}")

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        with open(temp_path, "wb") as f:
            f.write(content)

        logger.info(f"/analyze/image — file: '{safe_filename}', size: {len(content)} bytes")

        # Step 1: OCR extraction
        from ocr import extract_text_from_image
        extracted_text = extract_text_from_image(temp_path)

        if not extracted_text or not extracted_text.strip():
            return {
                "success": False,
                "input_type": "image",
                "extracted_text": "",
                "error": "No text could be extracted from this image. Please ensure the image is clear and contains readable text.",
                "data": None
            }

        logger.info(f"OCR extracted {len(extracted_text)} characters.")

        # Step 2: Parse label metadata
        from agent import run_drug_analysis_agent, _parse_label_metadata
        label_metadata = _parse_label_metadata(extracted_text)
        logger.info(f"Parsed metadata: MFG={label_metadata.get('mfg_date')}, EXP={label_metadata.get('expiry_date')}")

        # Step 3: Run agent
        result = run_drug_analysis_agent(extracted_text, label_metadata=label_metadata)

        return {
            "success": True,
            "input_type": "image",
            "extracted_text": extracted_text,
            "data": result
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/analyze/image error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ─── Dev Entry Point ───────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
