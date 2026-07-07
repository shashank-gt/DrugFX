"""
DrugFX Vector Store
====================
Provides semantic search using:
  - FAISS + sentence-transformers (if installed) — full semantic search
  - Keyword/TF-IDF fallback (always available) — no extra dependencies

Returns results with similarity/relevance scores for confidence scoring.
"""

import json
import os
import logging

logger = logging.getLogger(__name__)

# ── FAISS / sentence-transformers (optional) ──────────────────
_st_model = None      # Cached globally — loaded only once
_HAS_FAISS = False

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    _HAS_FAISS = True
    logger.info("VectorStore: FAISS + sentence-transformers available.")
except ImportError:
    logger.info("VectorStore: FAISS/sentence-transformers not installed — using keyword fallback.")


def _load_st_model():
    """Load sentence-transformers model once, cache globally."""
    global _st_model
    if _st_model is not None:
        return _st_model
    if not _HAS_FAISS:
        return None
    try:
        logger.info("VectorStore: Loading sentence-transformers model...")
        _st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("VectorStore: Model loaded and cached.")
    except Exception as e:
        logger.error(f"VectorStore: Could not load ST model: {e}")
        _st_model = None
    return _st_model


def _keyword_score(query: str, text: str) -> float:
    """Simple keyword overlap score for fallback search (0.0 to 1.0)."""
    q_words = set(query.lower().split())
    t_words = set(text.lower().split())
    if not q_words or not t_words:
        return 0.0
    overlap = q_words & t_words
    return len(overlap) / (len(q_words) + 1.0)


class DocumentStore:
    """Holds a drug knowledge base and supports semantic or keyword search."""

    def __init__(self, data_file: str):
        self.texts: list = []
        self.metadata: list = []
        self.index = None          # FAISS index
        self._np = None            # numpy module reference

        if not os.path.exists(data_file):
            logger.warning(f"VectorStore: Data file not found: {data_file}")
            return

        # Load knowledge base
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data:
                content = item.get('content', '')
                title   = item.get('title', '')
                self.texts.append(f"{title}: {content}")
                self.metadata.append(item)
            logger.info(f"VectorStore: Loaded {len(self.texts)} documents from {os.path.basename(data_file)}")
        except Exception as e:
            logger.error(f"VectorStore: Failed to load data file: {e}")
            return

    def _ensure_index(self):
        """Build FAISS index lazily on first search."""
        if self.index is not None:
            return  # Already built

        model = _load_st_model()
        if model is None or not self.texts:
            return

        try:
            import numpy as np
            self._np = np
            embeddings = model.encode(self.texts, show_progress_bar=False)
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(np.array(embeddings, dtype='float32'))
            logger.info(f"VectorStore: FAISS index built with {len(self.texts)} vectors.")
        except Exception as e:
            logger.error(f"VectorStore: Failed to build FAISS index: {e}")
            self.index = None

    def search(self, query: str, top_k: int = 3) -> list:
        """
        Legacy search interface. Returns a list of documents.
        """
        results_with_scores = self.search_with_scores(query, top_k)
        return [doc for doc, _ in results_with_scores]

    def search_with_scores(self, query: str, top_k: int = 3) -> list:
        """
        Search for top_k most relevant documents.
        Returns a list of tuples (document_metadata_dict, confidence_score_0_to_1).
        """
        top_k = int(top_k)
        if not self.metadata:
            return []

        # ── FAISS semantic search ─────────────────────────────
        if _HAS_FAISS:
            self._ensure_index()

        if self.index is not None:
            try:
                model = _load_st_model()
                np = self._np
                q_emb = model.encode([query], show_progress_bar=False)
                distances, indices = self.index.search(
                    np.array(q_emb, dtype='float32'),
                    min(top_k, len(self.metadata))
                )
                
                results = []
                for dist, i in zip(distances[0], indices[0]):
                    if 0 <= i < len(self.metadata):
                        # Convert L2 distance to a 0-1 similarity score.
                        # For L2 normalized embeddings, L2 distance ranges from 0 to 4.
                        # Similarity = 1 - (distance / 4)
                        score = max(0.0, min(1.0, 1.0 - (float(dist) / 4.0)))
                        results.append((self.metadata[i], score))
                return results
            except Exception as e:
                logger.error(f"VectorStore: FAISS search failed: {e}")

        # ── Keyword fallback ──────────────────────────────────
        scored = []
        for i in range(len(self.texts)):
            score = _keyword_score(query, self.texts[i])
            if score > 0.0:
                scored.append((self.metadata[i], score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # If no keyword match, just return top_k with very low score
        if not scored:
            return [(item, 0.1) for item in self.metadata[:top_k]]
            
        return scored[:top_k]

    def suggest(self, query: str, limit: int = 8) -> list:
        """
        Returns simple suggestions matching title of the drug/medicine.
        """
        if not query:
            return []
        query_lower = query.lower().strip()
        matches = []
        for meta in self.metadata:
            title = meta.get('title', '')
            if query_lower in title.lower():
                matches.append(title)
        return matches[:limit]
