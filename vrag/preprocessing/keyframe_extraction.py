"""
VRAG Keyframe Extraction Module

Selects representative keyframes from each shot based on semantic feature analysis.
From the paper (Section 4.1.2): "We extracted semantic features using BEiT-3 and 
applied a threshold-based approach to determine the most representative keyframes."
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from vrag.utils.video_utils import Shot, extract_frames_at_indices, save_keyframe

logger = logging.getLogger(__name__)


class KeyframeExtractor:
    """
    Extracts representative keyframes from video shots using semantic features.
    
    The paper uses BEiT-3 features with a threshold-based approach to select
    keyframes that effectively capture essential content while reducing redundancy.
    """

    def __init__(
        self,
        method: str = "semantic",
        similarity_threshold: float = 0.85,
        max_keyframes_per_shot: int = 5,
        min_keyframes_per_shot: int = 1,
        feature_model: str = "beit3",
        device: str = "cuda",
    ):
        """
        Args:
            method: Extraction method ("semantic" or "uniform").
            similarity_threshold: Cosine similarity threshold for deduplication.
            max_keyframes_per_shot: Maximum keyframes per shot.
            min_keyframes_per_shot: Minimum keyframes per shot.
            feature_model: Model for semantic feature extraction.
            device: Computation device.
        """
        self.method = method
        self.similarity_threshold = similarity_threshold
        self.max_keyframes_per_shot = max_keyframes_per_shot
        self.min_keyframes_per_shot = min_keyframes_per_shot
        self.feature_model = feature_model
        self.device = device
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy-load the feature extraction model."""
        if self._model is not None:
            return

        if self.feature_model == "beit3":
            # Use BEiT-3 for semantic feature extraction (as in the paper)
            try:
                from transformers import AutoModel, AutoProcessor
                model_name = "microsoft/beit-base-patch16-224"
                self._processor = AutoProcessor.from_pretrained(model_name)
                self._model = AutoModel.from_pretrained(model_name).to(self.device)
                self._model.eval()
                logger.info(f"Loaded BEiT model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load BEiT-3, falling back to CLIP: {e}")
                self._load_clip_model()
        elif self.feature_model == "clip":
            self._load_clip_model()
        else:
            raise ValueError(f"Unknown feature model: {self.feature_model}")

    def _load_clip_model(self):
        """Load CLIP model as fallback."""
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai"
            )
            self._model = model.to(self.device)
            self._model.eval()
            self._processor = preprocess
            self.feature_model = "clip"
            logger.info("Loaded CLIP ViT-L-14 model")
        except ImportError:
            raise ImportError("Install open_clip: pip install open_clip_torch")

    @torch.no_grad()
    def _extract_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract semantic features from a batch of frames.

        Args:
            frames: List of BGR numpy array frames.

        Returns:
            Feature matrix of shape (num_frames, feature_dim).
        """
        self._load_model()
        from PIL import Image

        pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]

        if self.feature_model == "clip":
            # CLIP feature extraction
            tensors = torch.stack([self._processor(img) for img in pil_images])
            tensors = tensors.to(self.device)
            features = self._model.encode_image(tensors)
        else:
            # BEiT / Transformer feature extraction
            inputs = self._processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            # Use CLS token
            features = outputs.last_hidden_state[:, 0, :]

        features = features.cpu().numpy()
        # L2 normalize
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norms + 1e-8)
        return features

    def extract_keyframes(
        self,
        video_path: str,
        shot: Shot,
        output_dir: str,
    ) -> Shot:
        """
        Extract keyframes from a video shot.

        Strategy (from paper Section 4.1.2):
          1. Sample candidate frames uniformly from the shot.
          2. Extract BEiT-3 semantic features.
          3. Select keyframes using threshold-based deduplication:
             - Start with the first frame.
             - Add subsequent frames only if their similarity to all 
               selected frames is below the threshold.

        Args:
            video_path: Path to the video file.
            shot: Shot object defining the segment.
            output_dir: Directory to save keyframe images.

        Returns:
            Updated Shot object with keyframe indices and paths.
        """
        if self.method == "uniform":
            return self._extract_uniform(video_path, shot, output_dir)
        else:
            return self._extract_semantic(video_path, shot, output_dir)

    def _extract_semantic(
        self, video_path: str, shot: Shot, output_dir: str
    ) -> Shot:
        """Extract keyframes using semantic feature-based selection."""
        # Sample candidate frames from the shot
        num_candidates = min(
            self.max_keyframes_per_shot * 3,
            max(shot.num_frames, self.min_keyframes_per_shot)
        )
        candidate_indices = np.linspace(
            shot.start_frame, shot.end_frame - 1,
            num=num_candidates, dtype=int
        ).tolist()
        candidate_indices = list(set(candidate_indices))
        candidate_indices.sort()

        if not candidate_indices:
            candidate_indices = [shot.start_frame]

        # Extract frames
        frames = extract_frames_at_indices(video_path, candidate_indices)
        if not frames:
            logger.warning(f"No frames extracted for {shot}")
            return shot

        # Extract semantic features
        features = self._extract_features(frames)

        # Threshold-based keyframe selection
        selected = [0]  # Always include the first frame
        for i in range(1, len(features)):
            # Compute max similarity to all previously selected frames
            similarities = features[i] @ features[selected].T
            max_sim = similarities.max()
            if max_sim < self.similarity_threshold:
                selected.append(i)
            if len(selected) >= self.max_keyframes_per_shot:
                break

        # Ensure minimum keyframes
        if len(selected) < self.min_keyframes_per_shot:
            remaining = [i for i in range(len(features)) if i not in selected]
            if remaining:
                mid_idx = remaining[len(remaining) // 2]
                selected.append(mid_idx)

        # Save keyframes
        keyframe_indices = []
        keyframe_paths = []

        for sel_idx in selected:
            frame_idx = candidate_indices[sel_idx]
            frame = frames[sel_idx]

            kf_filename = f"{shot.video_id}_shot{shot.shot_id}_frame{frame_idx}.jpg"
            kf_path = os.path.join(output_dir, shot.video_id, kf_filename)
            save_keyframe(frame, kf_path)

            keyframe_indices.append(frame_idx)
            keyframe_paths.append(kf_path)

        shot.keyframe_indices = keyframe_indices
        shot.keyframe_paths = keyframe_paths
        logger.debug(
            f"Extracted {len(keyframe_indices)} keyframes for {shot}"
        )
        return shot

    def _extract_uniform(
        self, video_path: str, shot: Shot, output_dir: str
    ) -> Shot:
        """Extract keyframes at uniform intervals (fallback method)."""
        num_keyframes = min(self.max_keyframes_per_shot, max(1, shot.num_frames // 30))
        indices = np.linspace(
            shot.start_frame, shot.end_frame - 1,
            num=num_keyframes, dtype=int
        ).tolist()
        indices = list(set(indices))
        indices.sort()

        frames = extract_frames_at_indices(video_path, indices)

        keyframe_indices = []
        keyframe_paths = []

        for i, (frame_idx, frame) in enumerate(zip(indices, frames)):
            kf_filename = f"{shot.video_id}_shot{shot.shot_id}_frame{frame_idx}.jpg"
            kf_path = os.path.join(output_dir, shot.video_id, kf_filename)
            save_keyframe(frame, kf_path)

            keyframe_indices.append(frame_idx)
            keyframe_paths.append(kf_path)

        shot.keyframe_indices = keyframe_indices
        shot.keyframe_paths = keyframe_paths
        return shot
