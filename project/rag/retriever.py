import os
from .vector_store import DocumentStore

# Setup paths to mock data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRUG_DATA_FILE = os.path.join(BASE_DIR, 'data', 'drug_knowledge.json')
JOB_DATA_FILE = os.path.join(BASE_DIR, 'data', 'job_knowledge.json')

# Initialize document stores for each domain
drug_store = DocumentStore(DRUG_DATA_FILE)
job_store = DocumentStore(JOB_DATA_FILE)

def retrieve_context(domain: str, query: str, top_k: int = 2) -> str:
    """
    Retrieves context for a given domain and query using RAG setup.
    """
    domain = domain.lower()
    store = None
    if domain == "drug":
        store = drug_store
    elif domain == "job":
        store = job_store
    else:
        return ""

    results = store.search(query, top_k=top_k)
    if not results:
        return ""

    context_str = "Relevant knowledge base context:\n"
    for res in results:
        context_str += f"- {res.get('content', '')}\n"
    return context_str
