"""
VRAG Shot Boundary Detection Module

Detects shot boundaries in videos using scene detection algorithms.
The paper uses master shot boundary detection to segment V3C1 dataset.
Supports PySceneDetect and TransNetV2 backends.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from vrag.utils.video_utils import Shot, get_video_info

logger = logging.getLogger(__name__)


class ShotBoundaryDetector:
    """
    Detect shot boundaries in videos.
    
    The VRAG paper uses master shot boundary detection for the V3C1 dataset.
    This implementation supports multiple detection methods.
    """

    def __init__(
        self,
        method: str = "pyscenedetect",
        threshold: float = 27.0,
        min_scene_len: int = 15,
    ):
        """
        Args:
            method: Detection method ("pyscenedetect" or "transnetv2").
            threshold: Scene change detection threshold.
            min_scene_len: Minimum scene length in frames.
        """
        self.method = method
        self.threshold = threshold
        self.min_scene_len = min_scene_len

    def detect(self, video_path: str, video_id: str) -> List[Shot]:
        """
        Detect shot boundaries in a video.

        Args:
            video_path: Path to the video file.
            video_id: Unique identifier for the video.

        Returns:
            List of Shot objects representing detected shots.
        """
        logger.info(f"Detecting shot boundaries for video: {video_id}")

        if self.method == "pyscenedetect":
            boundaries = self._detect_pyscenedetect(video_path)
        elif self.method == "transnetv2":
            boundaries = self._detect_transnetv2(video_path)
        elif self.method == "histogram":
            boundaries = self._detect_histogram(video_path)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")

        # Convert boundaries to Shot objects
        video_info = get_video_info(video_path)
        fps = video_info["fps"]
        shots = []

        for i, (start_frame, end_frame) in enumerate(boundaries):
            shot = Shot(
                video_id=video_id,
                shot_id=i,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_frame / fps,
                end_time=end_frame / fps,
            )
            shots.append(shot)

        logger.info(f"Detected {len(shots)} shots in video {video_id}")
        return shots

    def _detect_pyscenedetect(self, video_path: str) -> List[Tuple[int, int]]:
        """Detect scenes using PySceneDetect (ContentDetector)."""
        try:
            from scenedetect import detect, ContentDetector
        except ImportError:
            raise ImportError(
                "PySceneDetect is required. Install it with: "
                "pip install scenedetect[opencv]"
            )

        scene_list = detect(
            video_path,
            ContentDetector(
                threshold=self.threshold,
                min_scene_len=self.min_scene_len,
            ),
        )

        boundaries = []
        for scene in scene_list:
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            boundaries.append((start_frame, end_frame))

        return boundaries

    def _detect_transnetv2(self, video_path: str) -> List[Tuple[int, int]]:
        """Detect scenes using TransNetV2 neural network."""
        try:
            from transnetv2 import TransNetV2
        except ImportError:
            raise ImportError(
                "TransNetV2 is required. Install it with: "
                "pip install transnetv2"
            )

        model = TransNetV2()
        video_frames, single_frame_preds, all_frame_preds = model.predict_video(
            video_path
        )
        scene_list = model.predictions_to_scenes(single_frame_preds)

        boundaries = []
        for start, end in scene_list:
            boundaries.append((int(start), int(end)))

        return boundaries

    def _detect_histogram(self, video_path: str) -> List[Tuple[int, int]]:
        """
        Fallback: Detect scenes using histogram difference method.
        Simple but effective for basic shot boundary detection.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        boundaries = []
        prev_hist = None
        frame_idx = 0
        current_start = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Compute color histogram
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

            if prev_hist is not None:
                # Chi-Square distance between histograms
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR)

                if diff > self.threshold:
                    if frame_idx - current_start >= self.min_scene_len:
                        boundaries.append((current_start, frame_idx))
                        current_start = frame_idx

            prev_hist = hist
            frame_idx += 1

        # Add the last segment
        if frame_idx > current_start:
            boundaries.append((current_start, frame_idx))

        cap.release()
        return boundaries
