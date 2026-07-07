"""
DrugFX FastAPI Backend — v3.0
==============================
Production-grade API serving the frontend and exposing:
  GET  /                     — Frontend SPA
  GET  /api/health           — Health check with dependency status
  GET  /api/search/suggest   — Auto-complete suggestions
  POST /api/analyze/text     — Text drug query → structured analysis
  POST /api/analyze/image    — Image upload → OCR → analysis
  POST /analyze/text         — Backward-compatible alias
  POST /analyze/image        — Backward-compatible alias
"""

import os
import time
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ─── Load env ──────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except ImportError:
    pass

from config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=f"{settings.APP_NAME} API",
    description="AI-powered medicine intelligence — OCR, RAG, and LLM analysis",
    version=settings.APP_VERSION,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static dir
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─────────────────────────────────────────────────────────────
# Helper: Structured Error Response
# ─────────────────────────────────────────────────────────────
def _error_response(status_code: int, error_code: str, message: str, detail: str = ""):
    """Return a structured error JSON."""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "error": {
                "code": error_code,
                "message": message,
                "detail": detail,
            },
        },
    )


def _validate_upload(file: UploadFile, content: bytes) -> None:
    """Validate uploaded file: type, size, content."""
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # Check file size
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    # Check MIME type
    if file.content_type and file.content_type not in settings.allowed_image_types_set:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {settings.ALLOWED_IMAGE_TYPES}",
        )

    # Check extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext and ext not in settings.allowed_extensions_set:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {ext}",
        )


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    """Serve the single-page frontend."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found. Place index.html in the static/ folder."}


@app.get("/api/health")
def health_check():
    """Detailed health check with dependency status."""
    deps = {}

    # Check LLM provider
    try:
        from llm_providers import get_llm_provider
        provider = get_llm_provider()
        deps["llm"] = {
            "provider": provider.name,
            "available": provider.is_available(),
        }
    except Exception as e:
        deps["llm"] = {"provider": "unknown", "available": False, "error": str(e)}

    # Check RAG
    try:
        from rag.retriever import drug_store
        deps["rag"] = {
            "available": drug_store is not None,
            "documents": len(drug_store.texts) if drug_store else 0,
        }
    except Exception:
        deps["rag"] = {"available": False, "documents": 0}

    # Check OCR
    try:
        from ocr import HAS_TESSERACT
        deps["ocr"] = {
            "tesseract": HAS_TESSERACT,
            "gemini_vision": deps.get("llm", {}).get("available", False),
        }
    except Exception:
        deps["ocr"] = {"tesseract": False, "gemini_vision": False}

    all_ok = deps.get("llm", {}).get("available", False)
    return {
        "status": "ok" if all_ok else "degraded",
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "dependencies": deps,
    }


# Keep old endpoint for backward compat
@app.get("/health")
def health_check_legacy():
    return {"status": "ok", "version": settings.APP_VERSION}


# ─────────────────────────────────────────────────────────────
# Search Suggestions
# ─────────────────────────────────────────────────────────────

@app.get("/api/search/suggest")
def search_suggest(q: str = Query("", min_length=1, max_length=100)):
    """Auto-complete suggestions from the knowledge base."""
    if not q or len(q.strip()) < 1:
        return {"suggestions": []}

    try:
        from rag.retriever import get_suggestions
        suggestions = get_suggestions(q.strip(), limit=8)
        return {"suggestions": suggestions}
    except ImportError:
        # Fallback: basic search from knowledge base titles
        try:
            from rag.retriever import drug_store
            if drug_store and drug_store.metadata:
                query_lower = q.strip().lower()
                matches = []
                for item in drug_store.metadata:
                    title = item.get("title", "")
                    if query_lower in title.lower():
                        matches.append(title)
                return {"suggestions": matches[:8]}
        except Exception:
            pass
        return {"suggestions": []}
    except Exception as e:
        logger.error(f"Search suggest error: {e}")
        return {"suggestions": []}


# ─────────────────────────────────────────────────────────────
# Text Analysis
# ─────────────────────────────────────────────────────────────

def _do_analyze_text(text: str) -> dict:
    """Shared logic for text analysis."""
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    text = text.strip()
    logger.info(f"analyze/text — input: '{text[:80]}'")
    start_time = time.time()

    try:
        from agent import run_drug_analysis_agent, _parse_label_metadata
        metadata = _parse_label_metadata(text)
        result = run_drug_analysis_agent(text, label_metadata=metadata)
        elapsed = round(time.time() - start_time, 2)
        logger.info(f"analyze/text — completed in {elapsed}s")
        return {
            "success": True,
            "input_type": "text",
            "extracted_text": None,
            "processing_time": elapsed,
            "data": result,
        }
    except Exception as e:
        logger.error(f"analyze/text error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/analyze/text")
def analyze_text_v3(text: str = Form(...)):
    return _do_analyze_text(text)


@app.post("/analyze/text")
def analyze_text_legacy(text: str = Form(...)):
    return _do_analyze_text(text)


# ─────────────────────────────────────────────────────────────
# Image Analysis
# ─────────────────────────────────────────────────────────────

async def _do_analyze_image(file: UploadFile) -> dict:
    """Shared logic for image analysis."""
    content = await file.read()
    _validate_upload(file, content)

    # Sanitize filename
    safe_filename = "".join(
        c if c.isalnum() or c in ".-_" else "_"
        for c in os.path.basename(file.filename or "upload")
    )
    temp_path = os.path.join(os.path.dirname(__file__), f"_tmp_{safe_filename}")

    try:
        with open(temp_path, "wb") as f:
            f.write(content)

        logger.info(f"analyze/image — file: '{safe_filename}', size: {len(content)} bytes")
        start_time = time.time()

        # Step 1: OCR extraction (structured)
        from ocr import extract_text_structured
        ocr_result = extract_text_structured(temp_path)
        extracted_text = ocr_result.get("text", "")
        ocr_confidence = ocr_result.get("confidence", 0.0)
        ocr_provider = ocr_result.get("provider", "unknown")

        if not extracted_text or not extracted_text.strip():
            return {
                "success": False,
                "input_type": "image",
                "extracted_text": "",
                "ocr": {"confidence": 0.0, "provider": ocr_provider},
                "error": "No text could be extracted from this image. Please ensure the image is clear and contains readable text.",
                "data": None,
            }

        logger.info(f"OCR: {len(extracted_text)} chars, confidence={ocr_confidence}, provider={ocr_provider}")

        # Step 2: Parse label metadata
        from agent import run_drug_analysis_agent, _parse_label_metadata
        label_metadata = _parse_label_metadata(extracted_text)
        logger.info(f"Parsed metadata: MFG={label_metadata.get('mfg_date')}, EXP={label_metadata.get('expiry_date')}")

        # Step 3: Run agent
        result = run_drug_analysis_agent(extracted_text, label_metadata=label_metadata)

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"analyze/image — completed in {elapsed}s")

        return {
            "success": True,
            "input_type": "image",
            "extracted_text": extracted_text,
            "ocr": {
                "confidence": ocr_confidence,
                "provider": ocr_provider,
            },
            "processing_time": elapsed,
            "data": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"analyze/image error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.post("/api/analyze/image")
async def analyze_image_v3(file: UploadFile = File(...)):
    return await _do_analyze_image(file)


@app.post("/analyze/image")
async def analyze_image_legacy(file: UploadFile = File(...)):
    return await _do_analyze_image(file)


# ─── Dev Entry Point ───────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # When reload is True, uvicorn requires the app to be passed as an import string
    uvicorn.run("api:app", host=settings.HOST, port=settings.PORT, reload=not settings.is_production)
