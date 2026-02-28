"""
VRAG On-Screen Text Search Module

Retrieves video segments based on on-screen text (OCR) content.
From the paper (Section 3.1): "To enhance text-based video retrieval, we employ 
optical character recognition (OCR) to extract textual content from video frames."
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TextSearch:
    """
    Search for video shots based on on-screen text content.
    Uses BM25 or dense vector search over extracted OCR text.
    """

    def __init__(
        self,
        method: str = "bm25",
        top_n: int = 50,
    ):
        self.method = method
        self.top_n = top_n
        self._bm25 = None
        self._corpus = []
        self._shot_map = []  # Maps corpus index to (video_id, shot_id)
        self._dense_index = None

    def build_index(
        self,
        ocr_data: Dict[str, Dict[int, str]],
    ):
        """
        Build search index from OCR data.

        Args:
            ocr_data: Dict mapping video_id -> {shot_id: ocr_text}
        """
        self._corpus = []
        self._shot_map = []

        for video_id, shots in ocr_data.items():
            for shot_id, text in shots.items():
                if text.strip():
                    self._corpus.append(text.lower())
                    self._shot_map.append({
                        "video_id": video_id,
                        "shot_id": int(shot_id),
                        "text": text,
                    })

        if self.method == "bm25":
            self._build_bm25_index()
        else:
            self._build_dense_index()

        logger.info(
            f"Built text search index with {len(self._corpus)} documents"
        )

    def _build_bm25_index(self):
        """Build BM25 index over OCR text."""
        if not self._corpus:
            logger.warning("No OCR text found, skipping BM25 index")
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Install rank_bm25: pip install rank-bm25")

        tokenized = [doc.split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    def _build_dense_index(self):
        """Build dense vector index over OCR text using sentence transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError(
                "Install sentence-transformers and faiss: "
                "pip install sentence-transformers faiss-cpu"
            )

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(self._corpus, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        # Build FAISS index
        dim = embeddings.shape[1]
        self._dense_index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self._dense_index.add(embeddings)
        self._dense_model = model

    def search(
        self,
        query: str,
        top_n: Optional[int] = None,
    ) -> List[Dict]:
        """
        Search for shots containing specific on-screen text.

        Args:
            query: Text query to search for.
            top_n: Number of results to return.

        Returns:
            List of dicts with 'video_id', 'shot_id', 'score', 'matched_text'.
        """
        top_n = top_n or self.top_n

        if not self._corpus:
            logger.warning("Text search index is empty")
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

        # Get top-N indices
        top_indices = scores.argsort()[::-1][:top_n]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                entry = self._shot_map[idx]
                results.append({
                    "video_id": entry["video_id"],
                    "shot_id": entry["shot_id"],
                    "score": float(scores[idx]),
                    "matched_text": entry["text"],
                    "source": "text_search",
                })

        return results

    def _search_dense(self, query: str, top_n: int) -> List[Dict]:
        """Search using dense embeddings."""
        import numpy as np
        import faiss

        query_embedding = self._dense_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self._dense_index.search(query_embedding, top_n)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                entry = self._shot_map[idx]
                results.append({
                    "video_id": entry["video_id"],
                    "shot_id": entry["shot_id"],
                    "score": float(score),
                    "matched_text": entry["text"],
                    "source": "text_search",
                })

        return results

    def exact_match_search(
        self,
        query: str,
        case_sensitive: bool = False,
    ) -> List[Dict]:
        """
        Search for exact text matches (useful for specific text queries).

        Args:
            query: Exact text to search for.
            case_sensitive: Whether to use case-sensitive matching.

        Returns:
            List of matching results.
        """
        results = []
        pattern = re.compile(
            re.escape(query),
            0 if case_sensitive else re.IGNORECASE
        )

        for i, entry in enumerate(self._shot_map):
            if pattern.search(entry["text"]):
                results.append({
                    "video_id": entry["video_id"],
                    "shot_id": entry["shot_id"],
                    "score": 1.0,
                    "matched_text": entry["text"],
                    "source": "text_search_exact",
                })

        return results
