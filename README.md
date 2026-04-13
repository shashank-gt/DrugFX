# DrugFX

DrugFX is an AI-powered pharmaceutical platform that provides medicine details like uses, side effects, and dosage from text or image input. It is designed to extract, analyze, and structure drug information from medication labels and text documents.

## Features

- **Document Analysis:** Upload drug labels or text to extract text using OCR (Optical Character Recognition).
- **Intelligent LLM Processing:** Uses an Advanced Large Language Model (LLM) to structure and understand pharmaceutical text automatically.
- **RAG System:** Employs a Retrieval-Augmented Generation approach and FAISS vector database to retrieve verified drug knowledge on demand.
- **RESTful API Endpoint:** Interact with the application using an intuitive API powered by FastAPI.
- **Clean Interface:** Provides a user-friendly frontend allowing you to quickly get insights about specific medications.

## Tech Stack

- **Backend:** FastAPI, Python
- **AI/ML:** PyTorch, Hugging Face Transformers, LangChain, FAISS
- **OCR:** EasyOCR / Tesseract
- **Frontend:** HTML, CSS, JavaScript

## Setup & Running

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shashank-gt/DrugFX.git
   cd DrugFX/FP
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install libraries:**
   ```bash
   pip install -r project/requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the `project/` directory with the necessary API keys and configurations (such as your GEMINI_API_KEY).

5. **Start the API Server:**
   ```bash
   cd project
   uvicorn api:app --reload
   ```

## Folder Structure

- `project/`: Contains the main application, including API routes (`api.py`), LLM agents (`agent.py`, `llm.py`), OCR handlers (`ocr.py`), and RAG utilities.
- `project/static/`: Contains the user interface assets.
- `project/modules/`: Other backend processing modules.

## License

This project is licensed under the MIT License.
