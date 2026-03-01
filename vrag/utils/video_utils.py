"""
VRAG Video Processing Utilities

Common utilities for video loading, frame extraction, shot manipulation,
and video segment operations used across all VRAG modules.
"""

import os
import cv2
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Shot:
    """Represents a single video shot (segment between scene boundaries)."""
    video_id: str
    shot_id: int
    start_frame: int
    end_frame: int
    start_time: float  # seconds
    end_time: float    # seconds
    keyframe_indices: List[int] = field(default_factory=list)
    keyframe_paths: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame

    def __repr__(self) -> str:
        return (f"Shot(video={self.video_id}, shot={self.shot_id}, "
                f"time={self.start_time:.1f}-{self.end_time:.1f}s)")


@dataclass
class VideoSegment:
    """Represents a merged video segment (multiple consecutive shots)."""
    video_id: str
    shots: List[Shot]
    start_time: float
    end_time: float
    relevance_score: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def get_all_keyframe_paths(self) -> List[str]:
        paths = []
        for shot in self.shots:
            paths.extend(shot.keyframe_paths)
        return paths

    def __repr__(self) -> str:
        return (f"VideoSegment(video={self.video_id}, "
                f"shots={len(self.shots)}, "
                f"time={self.start_time:.1f}-{self.end_time:.1f}s, "
                f"score={self.relevance_score:.3f})")


@dataclass
class RetrievalResult:
    """Represents a retrieval result with score information."""
    video_id: str
    shot: Shot
    score: float
    source: str  # Which retrieval module produced this result
    metadata: Dict = field(default_factory=dict)


@dataclass
class VQAQuery:
    """Decomposed VQA query with retrieval query and question."""
    original_query: str
    retrieval_query: str
    question: str
    video_id: Optional[str] = None


@dataclass
class VRAGResult:
    """Final VRAG pipeline result."""
    query: str
    answer: str
    video_id: Optional[str] = None
    retrieved_segments: List[VideoSegment] = field(default_factory=list)
    confidence: float = 0.0
    task_type: str = "vqa"  # "kis" or "vqa"
    # Per-chunk source references used to construct the answer.
    # Each entry contains 'chunk_id', 'start_time', 'end_time', 'confidence'.
    sources: List[Dict] = field(default_factory=list)


# =============================================================================
# Video I/O
# =============================================================================

def load_video(video_path: str) -> cv2.VideoCapture:
    """Load a video file and return the VideoCapture object."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    return cap


def get_video_info(video_path: str) -> Dict:
    """Get video metadata (fps, frame count, duration, resolution)."""
    cap = load_video(video_path)
    try:
        info = {
            "path": video_path,
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
        return info
    finally:
        cap.release()


def extract_frames(
    video_path: str,
    start_time: float = 0.0,
    end_time: Optional[float] = None,
    num_frames: int = 8,
    resize: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """
    Extract uniformly-sampled frames from a video segment.

    Args:
        video_path: Path to the video file.
        start_time: Start time in seconds.
        end_time: End time in seconds (None = end of video).
        num_frames: Number of frames to extract.
        resize: Optional (width, height) to resize frames.

    Returns:
        List of BGR numpy arrays (OpenCV format).
    """
    cap = load_video(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_time * fps)
        if end_time is not None:
            end_frame = min(int(end_time * fps), total_frames)
        else:
            end_frame = total_frames

        if end_frame <= start_frame:
            return []

        # Uniformly sample frame indices
        frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
        frame_indices = np.unique(frame_indices)

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                if resize:
                    frame = cv2.resize(frame, resize)
                frames.append(frame)

        return frames
    finally:
        cap.release()


def extract_frames_at_indices(
    video_path: str,
    frame_indices: List[int],
    resize: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """Extract specific frames by index."""
    cap = load_video(video_path)
    try:
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                if resize:
                    frame = cv2.resize(frame, resize)
                frames.append(frame)
        return frames
    finally:
        cap.release()


def save_keyframe(frame: np.ndarray, output_path: str) -> str:
    """Save a single keyframe to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame)
    return output_path


def extract_audio(video_path: str, output_path: str) -> str:
    """
    Extract audio track from a video file using ffmpeg.

    Args:
        video_path: Path to the video file.
        output_path: Path to save the extracted audio (WAV format).

    Returns:
        Path to the extracted audio file.
    """
    import subprocess
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        output_path, "-y"
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


# =============================================================================
# Shot Utilities
# =============================================================================

def merge_consecutive_shots(shots: List[Shot]) -> VideoSegment:
    """Merge a list of consecutive shots into a single VideoSegment."""
    if not shots:
        raise ValueError("Cannot merge empty list of shots")

    shots_sorted = sorted(shots, key=lambda s: s.start_time)
    return VideoSegment(
        video_id=shots_sorted[0].video_id,
        shots=shots_sorted,
        start_time=shots_sorted[0].start_time,
        end_time=shots_sorted[-1].end_time,
    )


def expand_shot_context(
    shot: Shot,
    all_shots: List[Shot],
    context_window: int = 3,
) -> List[Shot]:
    """
    Expand a shot by including N preceding and N succeeding shots.
    This is used in the re-ranking module (paper Section 3.2).

    Args:
        shot: The target shot.
        all_shots: All shots in the same video, sorted by time.
        context_window: Number of shots before/after to include.

    Returns:
        List of shots forming the expanded context.
    """
    # Find index of the target shot
    video_shots = [s for s in all_shots if s.video_id == shot.video_id]
    video_shots.sort(key=lambda s: s.start_time)

    target_idx = None
    for i, s in enumerate(video_shots):
        if s.shot_id == shot.shot_id:
            target_idx = i
            break

    if target_idx is None:
        return [shot]

    start_idx = max(0, target_idx - context_window)
    end_idx = min(len(video_shots), target_idx + context_window + 1)

    return video_shots[start_idx:end_idx]


def chunk_video(
    video_duration: float,
    chunk_size: float = 15.0,
    overlap: float = 5.0,
) -> List[Tuple[float, float]]:
    """
    Divide a video into overlapping chunks for the VQA Filtering Module.

    Args:
        video_duration: Total video duration in seconds.
        chunk_size: Size of each chunk in seconds.
        overlap: Overlap between consecutive chunks in seconds.

    Returns:
        List of (start_time, end_time) tuples.
    """
    chunks = []
    start = 0.0
    step = chunk_size - overlap

    while start < video_duration:
        end = min(start + chunk_size, video_duration)
        chunks.append((start, end))
        start += step
        if end >= video_duration:
            break

    return chunks


# =============================================================================
# JSON Serialization
# =============================================================================

def save_shots(shots: List[Shot], output_path: str):
    """Save shot data to JSON."""
    data = []
    for shot in shots:
        data.append({
            "video_id": shot.video_id,
            "shot_id": shot.shot_id,
            "start_frame": shot.start_frame,
            "end_frame": shot.end_frame,
            "start_time": shot.start_time,
            "end_time": shot.end_time,
            "keyframe_indices": shot.keyframe_indices,
            "keyframe_paths": shot.keyframe_paths,
            "metadata": shot.metadata,
        })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_shots(input_path: str) -> List[Shot]:
    """Load shot data from JSON."""
    with open(input_path, "r") as f:
        data = json.load(f)

    shots = []
    for item in data:
        shots.append(Shot(
            video_id=item["video_id"],
            shot_id=item["shot_id"],
            start_frame=item["start_frame"],
            end_frame=item["end_frame"],
            start_time=item["start_time"],
            end_time=item["end_time"],
            keyframe_indices=item.get("keyframe_indices", []),
            keyframe_paths=item.get("keyframe_paths", []),
            metadata=item.get("metadata", {}),
        ))
    return shots


def frames_to_pil_images(frames: List[np.ndarray]) -> list:
    """Convert OpenCV BGR frames to PIL RGB images."""
    from PIL import Image
    pil_images = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(rgb))
    return pil_images
