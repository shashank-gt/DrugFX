import os
import logging
from typing import Optional, Tuple

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
    Legacy method. Retrieves health/drug knowledge base context for a given query.
    """
    ctx, _ = retrieve_context_with_scores(query, top_k)
    return ctx


def retrieve_context_with_scores(query: str, top_k: int = 3) -> Tuple[str, float]:
    """
    Retrieves health/drug knowledge base context for a given query.
    Also returns the average relevance score of the retrieved items.
    
    Args:
        query (str): The search query for the drug/medicine.
        top_k (int): Number of top documents to retrieve.
        
    Returns:
        Tuple[str, float]: (Formatted context string, average relevance score)
    """
    if drug_store is None:
        logger.warning("Drug store uninitialized.")
        return "", 0.0

    try:
        results = drug_store.search_with_scores(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Error during vector search: {e}")
        return "", 0.0

    if not results:
        return "No relevant context found in the medical database.", 0.0

    context_lines = ["Relevant medical knowledge base context:"]
    total_score = 0.0
    valid_count = 0
    
    for idx, (res, score) in enumerate(results, 1):
        title = res.get('title', 'Knowledge Item')
        content = res.get('content', '').strip()
        if content:
            context_lines.append(f"[{idx}] {title}: {content}")
            total_score += score
            valid_count += 1

    avg_score = total_score / valid_count if valid_count > 0 else 0.0
    return "\n".join(context_lines), avg_score


def get_suggestions(query: str, limit: int = 8) -> list:
    """
    Retrieves a list of auto-complete suggestions matching the query.
    """
    if drug_store is None:
        return []
    try:
        return drug_store.suggest(query, limit=limit)
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        return []