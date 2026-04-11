"""
DrugFX Vector Store
====================
Provides simple semantic search using:
  - FAISS + sentence-transformers (if installed) — full semantic search
  - Keyword/TF-IDF fallback (always available) — no extra dependencies
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
        logger.info("VectorStore: Loading sentence-transformers model (first run only)...")
        _st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("VectorStore: Model loaded and cached.")
    except Exception as e:
        logger.error(f"VectorStore: Could not load ST model: {e}")
        _st_model = None
    return _st_model


def _keyword_score(query: str, text: str) -> float:
    """Simple keyword overlap score for fallback search."""
    q_words = set(query.lower().split())
    t_words = set(text.lower().split())
    if not q_words or not t_words:
        return 0.0
    overlap = q_words & t_words
    return len(overlap) / (len(q_words) + 1)


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

        # Build FAISS index (deferred — only if model is already loaded or we force it)
        # We do NOT load the ST model here to avoid slow startup.
        # It will be loaded lazily on first search.

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
        Search for top_k most relevant documents.
        Uses FAISS semantic search if available, else keyword overlap.
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
                for i in indices[0]:
                    if 0 <= i < len(self.metadata):
                        results.append(self.metadata[i])
                return results
            except Exception as e:
                logger.error(f"VectorStore: FAISS search failed: {e}")

        # ── Keyword fallback ──────────────────────────────────
        scored = [
            (self.metadata[i], _keyword_score(query, self.texts[i]))
            for i in range(len(self.texts))
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, score in scored[:top_k]]


def build_vector_store(data_texts: list):
    """Legacy helper — kept for backward compatibility."""
    model = _load_st_model()
    if model is None:
        return None
    try:
        import numpy as np
        embeddings = model.encode(data_texts, show_progress_bar=False)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings, dtype='float32'))
        return index
    except Exception as e:
        logger.error(f"build_vector_store failed: {e}")
        return None
