import os
import logging
from typing import Optional

from .vector_store import DocumentStore

logger = logging.getLogger(__name__)

# Setup paths to data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRUG_DATA_FILE = os.path.join(BASE_DIR, 'data', 'drug_knowledge.json')

# Initialize document store for DrugFX
try:
    drug_store = DocumentStore(DRUG_DATA_FILE)
except Exception as e:
    logger.error(f"Failed to load drug knowledge base: {e}")
    drug_store = None

def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Retrieves health/drug knowledge base context for a given query using the RAG vector store.
    
    Args:
        query (str): The search query for the drug/medicine.
        top_k (int): Number of top documents to retrieve.
        
    Returns:
        str: Formatted context string to be fed into the LLM.
    """
    if drug_store is None:
        logger.warning("Drug store uninitialized.")
        return ""

    try:
        results = drug_store.search(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error during vector search: {e}")
        return ""

    if not results:
        return "No relevant context found in the medical database."

    context_lines = ["Relevant medical knowledge base context:"]
    for idx, res in enumerate(results, 1):
        title = res.get('title', 'Knowledge Item')
        content = res.get('content', '').strip()
        if content:
            context_lines.append(f"[{idx}] {title}: {content}")

    return "\n".join(context_lines)