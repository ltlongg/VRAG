"""
VRAG Multi-modal Retrieval System

Combines all retrieval modalities (semantic, text, audio, object, temporal)
into a unified multi-modal retrieval system.
From the paper (Section 3.1): The system employs a late-fusion approach integrating
multiple vision-language models and retrieval modalities.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from vrag.retrieval.semantic_search import SemanticSearch
from vrag.retrieval.text_search import TextSearch
from vrag.retrieval.audio_search import AudioSearch
from vrag.retrieval.object_filter import ObjectFilter
from vrag.retrieval.temporal_search import TemporalSearch
from vrag.utils.video_utils import Shot

logger = logging.getLogger(__name__)


class MultimodalRetrievalSystem:
    """
    Unified Multi-modal Retrieval System for VRAG.
    
    Combines results from:
      1. Semantic-based search (CLIP, BLIP-2, BEiT-3, InternVL)
      2. On-screen text search (OCR-based)
      3. Audio-based search (Whisper transcriptions)
      4. Object filtering (Co-DETR post-processing)
      5. Temporal search (temporal event relations)
    
    Results are fused using reciprocal rank fusion.
    """

    def __init__(
        self,
        semantic_search: SemanticSearch = None,
        text_search: TextSearch = None,
        audio_search: AudioSearch = None,
        object_filter: ObjectFilter = None,
        temporal_search: TemporalSearch = None,
        top_n: int = 100,
    ):
        self.semantic_search = semantic_search
        self.text_search = text_search
        self.audio_search = audio_search
        self.object_filter = object_filter
        self.temporal_search = temporal_search
        self.top_n = top_n

    def search(
        self,
        query: str,
        modalities: Optional[List[str]] = None,
        object_constraints: Optional[List[str]] = None,
        temporal_constraint: Optional[Dict] = None,
        top_n: Optional[int] = None,
    ) -> List[Dict]:
        """
        Perform multi-modal retrieval.

        Args:
            query: Text query for retrieval.
            modalities: Which modalities to use. Default: ["semantic"].
                Possible values: "semantic", "text", "audio", "temporal".
            object_constraints: List of required objects for post-filtering.
            temporal_constraint: Dict with 'query_a', 'query_b', 'relation'.
            top_n: Number of final results.

        Returns:
            Sorted list of retrieval results.
        """
        top_n = top_n or self.top_n
        if modalities is None:
            modalities = ["semantic"]

        all_results = []

        # --- Semantic Search ---
        if "semantic" in modalities and self.semantic_search:
            logger.info("Running semantic search...")
            semantic_results = self.semantic_search.search(query, top_n=top_n * 2)
            for r in semantic_results:
                r["source"] = "semantic"
            all_results.extend(semantic_results)

        # --- On-Screen Text Search ---
        if "text" in modalities and self.text_search:
            logger.info("Running on-screen text search...")
            text_results = self.text_search.search(query, top_n=top_n)
            for r in text_results:
                r["source"] = "text"
            all_results.extend(text_results)

        # --- Audio Search ---
        if "audio" in modalities and self.audio_search:
            logger.info("Running audio-based search...")
            audio_results = self.audio_search.search(query, top_n=top_n)
            for r in audio_results:
                r["source"] = "audio"
            all_results.extend(audio_results)

        # --- Temporal Search ---
        if "temporal" in modalities and self.temporal_search and temporal_constraint:
            logger.info("Running temporal search...")
            query_a = temporal_constraint.get("query_a", query)
            query_b = temporal_constraint.get("query_b", "")
            relation = temporal_constraint.get("relation", "before")

            if query_b and self.semantic_search:
                results_a = self.semantic_search.search(query_a, top_n=top_n)
                results_b = self.semantic_search.search(query_b, top_n=top_n)
                temporal_results = self.temporal_search.search_temporal(
                    results_a, results_b, relation=relation
                )
                all_results.extend(temporal_results)

        # --- Fuse all results ---
        fused = self._reciprocal_rank_fusion(all_results, top_n=top_n * 2)

        # --- Object Filtering (post-processing) ---
        if object_constraints and self.object_filter:
            logger.info(f"Applying object filter: {object_constraints}")
            fused = self.object_filter.filter(
                fused, required_objects=object_constraints
            )

        # Final top-N
        final_results = fused[:top_n]
        logger.info(
            f"Multi-modal retrieval: {len(all_results)} candidates -> "
            f"{len(final_results)} final results"
        )
        return final_results

    def search_for_kis(
        self,
        query: str,
        top_n: Optional[int] = None,
    ) -> List[Dict]:
        """
        Simplified search for KIS (Known-Item Search) task.
        
        From the paper (Section 4.3.1): "We adopted a straightforward strategy 
        for the fully-automated track where the query was directly processed 
        using the semantic-based retrieval module for the KIS task."

        Args:
            query: KIS query description.
            top_n: Number of results.

        Returns:
            Top-N retrieval results.
        """
        return self.search(
            query=query,
            modalities=["semantic"],
            top_n=top_n,
        )

    def _reciprocal_rank_fusion(
        self,
        all_results: List[Dict],
        k: int = 60,
        top_n: int = 100,
    ) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) to combine results from multiple sources.

        Args:
            all_results: All results from different modalities.
            k: RRF constant (default 60).
            top_n: Number of results to return.

        Returns:
            Fused and sorted results.
        """
        # Group by source and sort each source by score
        by_source = defaultdict(list)
        for r in all_results:
            by_source[r.get("source", "unknown")].append(r)

        for source in by_source:
            by_source[source].sort(key=lambda x: x.get("score", 0), reverse=True)

        # Compute RRF scores
        fused_scores = {}
        for source, results in by_source.items():
            for rank, result in enumerate(results):
                key = (result.get("video_id"), result.get("shot_id"))
                if key not in fused_scores:
                    fused_scores[key] = {
                        "video_id": result.get("video_id"),
                        "shot_id": result.get("shot_id"),
                        "rrf_score": 0.0,
                        "sources": [],
                        "metadata": result.get("metadata", {}),
                    }
                fused_scores[key]["rrf_score"] += 1.0 / (k + rank + 1)
                fused_scores[key]["sources"].append(source)

        # Sort by RRF score
        fused_list = sorted(
            fused_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        # Normalize to 0-1 range
        if fused_list:
            max_score = fused_list[0]["rrf_score"]
            for item in fused_list:
                item["score"] = item["rrf_score"] / max_score if max_score > 0 else 0

        return fused_list[:top_n]
