"""
VRAG Object Filter Module

Post-processing filter that removes retrieved shots not matching object constraints.
From the paper (Section 3.1): "This module refines retrieval results by filtering 
out shots that do not satisfy predefined object conditions."
"""

import logging
from typing import Dict, List, Optional, Set

from vrag.utils.video_utils import Shot

logger = logging.getLogger(__name__)


class ObjectFilter:
    """
    Post-processing filter for retrieval results based on object presence.
    
    Uses pre-computed object detection results (from Co-DETR) to filter
    retrieved shots that don't contain required objects.
    """

    def __init__(self, object_detector=None):
        """
        Args:
            object_detector: ObjectDetector instance for on-the-fly detection.
        """
        self.object_detector = object_detector
        self._object_cache: Dict[str, Set[str]] = {}  # (video_id, shot_id) -> objects

    def load_object_data(
        self,
        object_data: Dict[str, Dict[int, List[str]]],
    ):
        """
        Load pre-computed object detection results.

        Args:
            object_data: Dict mapping video_id -> {shot_id: [object_labels]}
        """
        for video_id, shots in object_data.items():
            for shot_id, labels in shots.items():
                key = f"{video_id}_{shot_id}"
                self._object_cache[key] = set(l.lower() for l in labels)

        logger.info(
            f"Loaded object data for {len(self._object_cache)} shots"
        )

    def get_objects(self, video_id: str, shot_id: int) -> Set[str]:
        """Get detected objects for a specific shot."""
        key = f"{video_id}_{shot_id}"
        return self._object_cache.get(key, set())

    def filter(
        self,
        results: List[Dict],
        required_objects: List[str],
        mode: str = "any",
    ) -> List[Dict]:
        """
        Filter retrieval results based on object conditions.

        Args:
            results: List of retrieval result dicts.
            required_objects: Object labels that must be present.
            mode: "any" = at least one required object present,
                  "all" = all required objects must be present.

        Returns:
            Filtered list of results.
        """
        if not required_objects:
            return results

        required = set(obj.lower() for obj in required_objects)
        filtered = []

        for result in results:
            video_id = result["video_id"]
            shot_id = result.get("shot_id", -1)
            detected = self.get_objects(video_id, shot_id)

            if mode == "any":
                if required & detected:
                    filtered.append(result)
            elif mode == "all":
                if required.issubset(detected):
                    filtered.append(result)

        logger.info(
            f"Object filter: {len(results)} -> {len(filtered)} results "
            f"(required: {required_objects}, mode: {mode})"
        )
        return filtered

    def filter_with_detection(
        self,
        results: List[Dict],
        required_objects: List[str],
        shots_data: Dict[str, List[Shot]],
        mode: str = "any",
    ) -> List[Dict]:
        """
        Filter with on-the-fly object detection for shots not in cache.

        Args:
            results: Retrieval results.
            required_objects: Required objects.
            shots_data: Shot data for accessing keyframes.
            mode: Filtering mode.

        Returns:
            Filtered results.
        """
        if not required_objects or not self.object_detector:
            return self.filter(results, required_objects, mode)

        required = set(obj.lower() for obj in required_objects)
        filtered = []

        for result in results:
            video_id = result["video_id"]
            shot_id = result.get("shot_id", -1)
            key = f"{video_id}_{shot_id}"

            if key in self._object_cache:
                detected = self._object_cache[key]
            elif video_id in shots_data:
                # On-the-fly detection
                shot = next(
                    (s for s in shots_data[video_id] if s.shot_id == shot_id),
                    None
                )
                if shot and shot.keyframe_paths:
                    labels = self.object_detector.get_objects_for_shot(shot)
                    detected = set(l.lower() for l in labels)
                    self._object_cache[key] = detected
                else:
                    detected = set()
            else:
                detected = set()

            if mode == "any" and (required & detected):
                filtered.append(result)
            elif mode == "all" and required.issubset(detected):
                filtered.append(result)

        return filtered
