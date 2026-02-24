"""
VRAG Temporal Search Module

Enhances retrieval by incorporating temporal constraints.
From the paper (Section 3.1): "This module enhances retrieval by incorporating 
temporal constraints, allowing users to locate relevant video segments based on 
temporal relationships between events."
"""

import logging
from typing import Dict, List, Optional, Tuple

from vrag.utils.video_utils import Shot

logger = logging.getLogger(__name__)


class TemporalSearch:
    """
    Temporal-aware search that finds video segments based on temporal 
    relationships between events.
    
    Supports:
      - "A before B": Event A happens before event B
      - "A after B": Event A happens after event B
      - "A during B": Event A happens during event B (overlapping)
      - "A then B": A immediately followed by B (sequential)
    """

    def __init__(self, shots_data: Dict[str, List[Shot]] = None):
        """
        Args:
            shots_data: Dict mapping video_id -> list of Shot objects.
        """
        self.shots_data = shots_data or {}

    def set_shots_data(self, shots_data: Dict[str, List[Shot]]):
        """Set the shots data for temporal reasoning."""
        self.shots_data = shots_data

    def search_temporal(
        self,
        results_a: List[Dict],
        results_b: List[Dict],
        relation: str = "before",
        max_gap_seconds: float = 60.0,
    ) -> List[Dict]:
        """
        Find video segments where event A has a temporal relation to event B.

        Args:
            results_a: Retrieval results for event A.
            results_b: Retrieval results for event B.
            relation: Temporal relation ("before", "after", "during", "then").
            max_gap_seconds: Maximum time gap for "before"/"after"/"then".

        Returns:
            List of combined results with temporal scores.
        """
        if relation == "before":
            return self._find_before(results_a, results_b, max_gap_seconds)
        elif relation == "after":
            return self._find_before(results_b, results_a, max_gap_seconds)
        elif relation == "during":
            return self._find_during(results_a, results_b)
        elif relation == "then":
            return self._find_sequential(results_a, results_b, max_gap_seconds)
        else:
            raise ValueError(f"Unknown temporal relation: {relation}")

    def _find_before(
        self,
        results_first: List[Dict],
        results_second: List[Dict],
        max_gap: float,
    ) -> List[Dict]:
        """Find cases where first event occurs before second event."""
        # Group results by video_id
        first_by_video = self._group_by_video(results_first)
        second_by_video = self._group_by_video(results_second)

        combined = []
        for video_id in set(first_by_video) & set(second_by_video):
            for r1 in first_by_video[video_id]:
                shot1 = self._get_shot(video_id, r1.get("shot_id", -1))
                if not shot1:
                    continue

                for r2 in second_by_video[video_id]:
                    shot2 = self._get_shot(video_id, r2.get("shot_id", -1))
                    if not shot2:
                        continue

                    # Check temporal order: shot1 ends before shot2 starts
                    time_gap = shot2.start_time - shot1.end_time
                    if 0 <= time_gap <= max_gap:
                        combined_score = (r1["score"] + r2["score"]) / 2
                        # Closer temporal proximity = higher score bonus
                        temporal_bonus = 1.0 - (time_gap / max_gap) if max_gap > 0 else 1.0
                        final_score = combined_score * (1.0 + 0.2 * temporal_bonus)

                        combined.append({
                            "video_id": video_id,
                            "shot_id": shot1.shot_id,
                            "score": final_score,
                            "temporal_info": {
                                "first_shot": shot1.shot_id,
                                "second_shot": shot2.shot_id,
                                "gap_seconds": time_gap,
                                "relation": "before",
                            },
                            "source": "temporal_search",
                        })

        return sorted(combined, key=lambda x: x["score"], reverse=True)

    def _find_during(
        self,
        results_a: List[Dict],
        results_b: List[Dict],
    ) -> List[Dict]:
        """Find cases where events A and B overlap temporally."""
        a_by_video = self._group_by_video(results_a)
        b_by_video = self._group_by_video(results_b)

        combined = []
        for video_id in set(a_by_video) & set(b_by_video):
            for ra in a_by_video[video_id]:
                shot_a = self._get_shot(video_id, ra.get("shot_id", -1))
                if not shot_a:
                    continue

                for rb in b_by_video[video_id]:
                    shot_b = self._get_shot(video_id, rb.get("shot_id", -1))
                    if not shot_b:
                        continue

                    # Check overlap
                    overlap_start = max(shot_a.start_time, shot_b.start_time)
                    overlap_end = min(shot_a.end_time, shot_b.end_time)

                    if overlap_start < overlap_end:
                        combined_score = (ra["score"] + rb["score"]) / 2
                        combined.append({
                            "video_id": video_id,
                            "shot_id": shot_a.shot_id,
                            "score": combined_score * 1.2,  # Bonus for overlap
                            "temporal_info": {
                                "shot_a": shot_a.shot_id,
                                "shot_b": shot_b.shot_id,
                                "overlap_duration": overlap_end - overlap_start,
                                "relation": "during",
                            },
                            "source": "temporal_search",
                        })

        return sorted(combined, key=lambda x: x["score"], reverse=True)

    def _find_sequential(
        self,
        results_first: List[Dict],
        results_second: List[Dict],
        max_gap: float,
    ) -> List[Dict]:
        """Find immediately sequential events (A then B)."""
        return self._find_before(results_first, results_second, max_gap=max_gap * 0.5)

    def _group_by_video(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """Group results by video_id."""
        grouped = {}
        for r in results:
            vid = r.get("video_id", "")
            if vid not in grouped:
                grouped[vid] = []
            grouped[vid].append(r)
        return grouped

    def _get_shot(self, video_id: str, shot_id: int) -> Optional[Shot]:
        """Look up a shot object."""
        if video_id in self.shots_data:
            for shot in self.shots_data[video_id]:
                if shot.shot_id == shot_id:
                    return shot
        return None
