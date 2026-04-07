import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from utils.router import route_request
from ocr import extract_text_from_image

app = FastAPI(title="Modular AI Analyzer API")

# Setup CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the static directory exists
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Mount the static directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_frontend():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not found! Place index.html in the static/ folder."}

@app.post("/analyze/text")
def analyze_text(module: str = Form(...), text: str = Form(...)):
    """
    Analyzes pure text based on the selected module.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
    try:
        result = route_request(module, text)
        return {"success": True, "module": module, "input_type": "text", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/image")
async def analyze_image(module: str = Form(...), file: UploadFile = File(...)):
    """
    Saves an uploaded image, extracts text using OCR, and routes to the selected module.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    
    # Save the file temporarily
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())
            
        # Extract text
        extracted_text = extract_text_from_image(temp_path)
        
        # Analyze using the generated text
        result = route_request(module, extracted_text)
        
        return {
            "success": True, 
            "module": module, 
            "input_type": "image", 
            "extracted_text": extracted_text, 
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
