"""
VRAG Audio-based Search Module

Retrieves video segments based on spoken content/audio transcription.
From the paper (Section 3.1): "Our system enables video retrieval based on spoken 
content by utilizing Whisper for automatic speech transcription."
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class AudioSearch:
    """
    Search for video shots based on spoken content (audio transcription).
    Uses BM25 or dense search over Whisper transcriptions.
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
        self._segment_map = []

    def build_index(
        self,
        transcript_data: Dict[str, Dict],
    ):
        """
        Build search index from audio transcription data.

        Args:
            transcript_data: Dict mapping video_id -> transcription result.
                Each transcription has 'segments' with start/end times and text.
        """
        self._corpus = []
        self._segment_map = []

        for video_id, transcription in transcript_data.items():
            for segment in transcription.get("segments", []):
                text = segment.get("text", "").strip()
                if text:
                    self._corpus.append(text.lower())
                    self._segment_map.append({
                        "video_id": video_id,
                        "start_time": segment.get("start", 0),
                        "end_time": segment.get("end", 0),
                        "text": text,
                    })

        if self.method == "bm25":
            self._build_bm25_index()
        else:
            self._build_dense_index()

        logger.info(
            f"Built audio search index with {len(self._corpus)} segments"
        )

    def _build_bm25_index(self):
        """Build BM25 index."""
        if not self._corpus:
            logger.warning("No audio transcripts found, skipping BM25 index")
            return

        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Install rank_bm25: pip install rank-bm25")

        tokenized = [doc.split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    def _build_dense_index(self):
        """Build dense vector index."""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError(
                "Install sentence-transformers and faiss."
            )

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(self._corpus, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

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
        Search for video segments with matching spoken content.

        Args:
            query: Text query to match against audio transcriptions.
            top_n: Number of results to return.

        Returns:
            List of dicts with 'video_id', 'start_time', 'end_time', 
            'score', 'transcript'.
        """
        top_n = top_n or self.top_n

        if not self._corpus:
            logger.warning("Audio search index is empty")
            return []

        if self.method == "bm25":
            return self._search_bm25(query, top_n)
        else:
            return self._search_dense(query, top_n)

    def _search_bm25(self, query: str, top_n: int) -> List[Dict]:
        """BM25 search over audio transcriptions."""
        if self._bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[::-1][:top_n]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                entry = self._segment_map[idx]
                results.append({
                    "video_id": entry["video_id"],
                    "shot_id": entry.get("shot_id"),
                    "start_time": entry["start_time"],
                    "end_time": entry["end_time"],
                    "score": float(scores[idx]),
                    "transcript": entry["text"],
                    "source": "audio_search",
                })

        return results

    def _search_dense(self, query: str, top_n: int) -> List[Dict]:
        """Dense embedding search over audio transcriptions."""
        import numpy as np
        import faiss

        query_embedding = self._dense_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self._dense_index.search(query_embedding, top_n)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                entry = self._segment_map[idx]
                results.append({
                    "video_id": entry["video_id"],
                    "shot_id": entry.get("shot_id"),
                    "start_time": entry["start_time"],
                    "end_time": entry["end_time"],
                    "score": float(score),
                    "transcript": entry["text"],
                    "source": "audio_search",
                })

        return results

    def search_by_shot(
        self,
        query: str,
        shots_data: Dict[str, list],
        top_n: Optional[int] = None,
    ) -> List[Dict]:
        """
        Search and map results to shots.

        Args:
            query: Query text.
            shots_data: Dict mapping video_id -> list of Shot objects.
            top_n: Number of results.

        Returns:
            List of results mapped to shot IDs.
        """
        raw_results = self.search(query, top_n)

        # Map time-based results to shots
        mapped_results = []
        for result in raw_results:
            video_id = result["video_id"]
            mid_time = (result["start_time"] + result["end_time"]) / 2

            if video_id in shots_data:
                for shot in shots_data[video_id]:
                    if shot.start_time <= mid_time <= shot.end_time:
                        result["shot_id"] = shot.shot_id
                        mapped_results.append(result)
                        break

        return mapped_results
