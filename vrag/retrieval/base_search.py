"""
VRAG Base Text Search Module

Provides a reusable base class for BM25/dense text-based search,
shared by TextSearch (OCR) and AudioSearch (transcripts).
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaseTextSearch:
    """
    Base class for BM25 and dense text search over document corpora.

    Subclasses (TextSearch, AudioSearch) override `build_index()` to
    populate `_corpus` and `_doc_map` from their respective data sources.
    """

    def __init__(
        self,
        method: str = "bm25",
        top_n: int = 50,
    ):
        self.method = method
        self.top_n = top_n
        self._bm25 = None
        self._corpus: List[str] = []
        self._doc_map: List[Dict] = []
        self._dense_index = None
        self._dense_model = None

    def _build_bm25_index(self):
        """Build BM25 index over corpus."""
        if not self._corpus:
            logger.warning("No text found, skipping BM25 index")
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Install rank_bm25: pip install rank-bm25")

        tokenized = [doc.split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    def _build_dense_index(self):
        """Build dense vector index using sentence transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
        except ImportError:
            raise ImportError(
                "Install sentence-transformers and faiss: "
                "pip install sentence-transformers faiss-cpu"
            )

        self._dense_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = self._dense_model.encode(
            self._corpus, show_progress_bar=True
        )
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self._dense_index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self._dense_index.add(embeddings)

    def _finalize_index(self):
        """Call after populating _corpus and _doc_map."""
        if self.method == "bm25":
            self._build_bm25_index()
        else:
            self._build_dense_index()

        logger.info(
            f"Built {self.__class__.__name__} index "
            f"with {len(self._corpus)} documents"
        )

    def search(
        self,
        query: str,
        top_n: Optional[int] = None,
    ) -> List[Dict]:
        """Search the index."""
        top_n = top_n or self.top_n

        if not self._corpus:
            logger.warning(f"{self.__class__.__name__} index is empty")
            return []

        if self.method == "bm25":
            return self._search_bm25(query, top_n)
        else:
            return self._search_dense(query, top_n)

    def _search_bm25(self, query: str, top_n: int) -> List[Dict]:
        """Search using BM25."""
        if self._bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[::-1][:top_n]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                entry = self._doc_map[idx].copy()
                entry["score"] = float(scores[idx])
                results.append(entry)

        return results

    def _search_dense(self, query: str, top_n: int) -> List[Dict]:
        """Search using dense embeddings."""
        import faiss

        query_embedding = self._dense_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self._dense_index.search(query_embedding, top_n)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                entry = self._doc_map[idx].copy()
                entry["score"] = float(score)
                results.append(entry)

        return results
