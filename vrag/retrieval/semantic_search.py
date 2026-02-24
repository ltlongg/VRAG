"""
VRAG Semantic Search Module

Semantic-based video retrieval using vision-language model embeddings.
From the paper (Section 3.1): "Our multimodal retrieval system employs a late-fusion 
approach at the shot level, integrating results from InternVL-G, BLIP-2, BEiT-3, 
and CLIP."
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SemanticSearch:
    """
    Semantic-based video retrieval using multiple vision-language models
    with late fusion at the shot level.
    
    This is the primary retrieval method used in the fully-automated track.
    """

    def __init__(
        self,
        index_manager=None,
        models: List[str] = None,
        fusion_method: str = "late_fusion",
        fusion_weights: Dict[str, float] = None,
        top_n: int = 100,
    ):
        """
        Args:
            index_manager: IndexBuilder instance with loaded indices.
            models: List of model names to use for retrieval.
            fusion_method: "late_fusion" or "weighted_sum".
            fusion_weights: Weight for each model in fusion.
            top_n: Number of top results to return.
        """
        self.index_manager = index_manager
        self.models = models or ["clip", "blip2", "beit3", "internvl"]
        self.fusion_method = fusion_method
        self.fusion_weights = fusion_weights or {
            "clip": 0.3, "blip2": 0.2, "beit3": 0.2, "internvl": 0.3
        }
        self.top_n = top_n

    def search(
        self,
        query_text: str,
        top_n: Optional[int] = None,
    ) -> List[Dict]:
        """
        Search for relevant video shots using semantic similarity.

        Performs late fusion: each model independently retrieves candidates,
        scores are combined using reciprocal rank fusion or weighted averaging.

        Args:
            query_text: Text query for retrieval.
            top_n: Override for number of results.

        Returns:
            List of dicts with keys: 'video_id', 'shot_id', 'score', 'metadata'
        """
        top_n = top_n or self.top_n

        if self.fusion_method == "late_fusion":
            return self._late_fusion_search(query_text, top_n)
        elif self.fusion_method == "weighted_sum":
            return self._weighted_sum_search(query_text, top_n)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

    def _late_fusion_search(
        self, query_text: str, top_n: int
    ) -> List[Dict]:
        """
        Late fusion: Each model retrieves independently, then combine using
        Reciprocal Rank Fusion (RRF).
        """
        all_rankings = {}

        for model_name in self.models:
            if not self.index_manager or not self.index_manager.has_index(model_name):
                logger.warning(f"No index for model: {model_name}")
                continue

            # Get per-model results
            results = self.index_manager.search(
                model_name, query_text, top_k=top_n * 2
            )

            # Store rankings
            for rank, result in enumerate(results):
                key = (result["video_id"], result["shot_id"])
                if key not in all_rankings:
                    all_rankings[key] = {
                        "video_id": result["video_id"],
                        "shot_id": result["shot_id"],
                        "rrf_score": 0.0,
                        "model_scores": {},
                        "metadata": result.get("metadata", {}),
                    }
                # RRF score with weight
                k = 60  # RRF constant
                weight = self.fusion_weights.get(model_name, 1.0)
                all_rankings[key]["rrf_score"] += weight / (k + rank + 1)
                all_rankings[key]["model_scores"][model_name] = result.get("score", 0)

        # Sort by fused score
        results = sorted(
            all_rankings.values(), key=lambda x: x["rrf_score"], reverse=True
        )

        # Normalize scores
        if results:
            max_score = results[0]["rrf_score"]
            for r in results:
                r["score"] = r["rrf_score"] / max_score if max_score > 0 else 0

        return results[:top_n]

    def _weighted_sum_search(
        self, query_text: str, top_n: int
    ) -> List[Dict]:
        """
        Weighted sum fusion: Combine cosine similarity scores directly.
        """
        all_scores = {}

        for model_name in self.models:
            if not self.index_manager or not self.index_manager.has_index(model_name):
                continue

            results = self.index_manager.search(
                model_name, query_text, top_k=top_n * 3
            )

            weight = self.fusion_weights.get(model_name, 1.0)
            for result in results:
                key = (result["video_id"], result["shot_id"])
                if key not in all_scores:
                    all_scores[key] = {
                        "video_id": result["video_id"],
                        "shot_id": result["shot_id"],
                        "score": 0.0,
                        "metadata": result.get("metadata", {}),
                    }
                all_scores[key]["score"] += weight * result.get("score", 0)

        results = sorted(
            all_scores.values(), key=lambda x: x["score"], reverse=True
        )

        return results[:top_n]

    def search_with_image(
        self,
        query_image: np.ndarray,
        top_n: Optional[int] = None,
    ) -> List[Dict]:
        """
        Search using an image query (for visual similarity retrieval).

        Args:
            query_image: Query image as numpy array.
            top_n: Number of results to return.

        Returns:
            List of result dicts.
        """
        top_n = top_n or self.top_n

        if not self.index_manager:
            return []

        # Use CLIP for image-to-image search
        from PIL import Image
        import cv2
        pil_image = Image.fromarray(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB))

        results = self.index_manager.search_by_image(
            "clip", pil_image, top_k=top_n
        )

        return results
