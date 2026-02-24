"""
VRAG Re-ranking Module

Re-ranks retrieval candidates using a Multimodal Large Language Model (MLLM).
From the paper (Section 3.2): "The module begins by expanding each shot-level 
retrieval candidate X to include 3 preceding and 3 succeeding shots, which are 
merged into a short video segment V. A MLLM is then prompted to evaluate the 
relevance of V to the user query Q, returning a score between 0 and 1."
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from vrag.utils.video_utils import (
    Shot, VideoSegment, expand_shot_context, merge_consecutive_shots,
    extract_frames
)

logger = logging.getLogger(__name__)

# Default system prompt for re-ranking MLLM
RERANKING_PROMPT_TEMPLATE = """You are a video relevance assessor. Given a video segment and a text query, evaluate how relevant the video content is to the query.

Query: {query}

Instructions:
1. Watch/analyze the provided video frames carefully.
2. Assess how well the video content matches the query description.
3. Return ONLY a single floating-point number between 0.0 and 1.0:
   - 0.0 = completely irrelevant
   - 0.5 = partially relevant
   - 1.0 = perfectly matches the query

Your response must be ONLY a number (e.g., 0.75). No explanation."""


class Reranker:
    """
    Re-ranks retrieval candidates using a Multimodal LLM to assess
    visual relevance of video segments to user queries.
    
    Architecture (Paper Section 3.2):
    1. Expand each shot ±context_window neighboring shots
    2. Merge into a short video segment
    3. MLLM evaluates relevance (score 0-1)
    4. Rank by score descending, select top-K
    
    Best MLLM: InternVL2.5-78B achieved 40.5/45 in experiments.
    """

    def __init__(
        self,
        mllm_model: str = "InternVL2.5-8B",
        context_window: int = 3,
        top_k: int = 10,
        max_frames_per_segment: int = 16,
        batch_size: int = 4,
    ):
        """
        Args:
            mllm_model: Name/path of the MLLM for relevance scoring.
            context_window: Number of neighboring shots to expand (±N).
            top_k: Number of top results to keep after re-ranking.
            max_frames_per_segment: Max frames to sample per segment for MLLM.
            batch_size: Batch size for MLLM inference.
        """
        self.mllm_model = mllm_model
        self.context_window = context_window
        self.top_k = top_k
        self.max_frames_per_segment = max_frames_per_segment
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def _load_model(self):
        """Load the MLLM for re-ranking."""
        if self._model is not None:
            return

        self._load_internvl()

    def _load_internvl(self):
        """Load InternVL2.5 model for re-ranking."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            model_path = f"OpenGVLab/{self.mllm_model}"
            logger.info(f"Loading InternVL model: {model_path}")

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            )
            self._model.eval()
            self._model_type = "internvl"
            logger.info("InternVL loaded successfully")

        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

    def rerank(
        self,
        query: str,
        retrieval_results: List[Dict],
        shots_data: Dict[str, List[Shot]],
        video_dir: str = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Re-rank retrieval candidates using MLLM.

        Process (Paper Section 3.2):
        1. For each candidate shot, expand by ±context_window neighboring shots
        2. Merge expanded shots into a video segment
        3. Sample frames from segment
        4. MLLM assigns relevance score 0-1
        5. Sort by score, return top-K

        Args:
            query: User query.
            retrieval_results: List of retrieval result dicts (from multimodal retrieval).
            shots_data: Dict mapping video_id -> list of Shot objects.
            video_dir: Directory containing video files.
            top_k: Override for top-K.

        Returns:
            Re-ranked top-K results with relevance scores.
        """
        top_k = top_k or self.top_k

        if not retrieval_results:
            return []

        self._load_model()

        scored_results = []
        for i, result in enumerate(retrieval_results):
            video_id = result.get("video_id", "")
            shot_id = result.get("shot_id", -1)

            # Get the shot and its context
            all_shots = shots_data.get(video_id, [])
            target_shot = next(
                (s for s in all_shots if s.shot_id == shot_id), None
            )

            if not target_shot:
                logger.warning(f"Shot not found: {video_id}/{shot_id}")
                result["relevance_score"] = 0.0
                scored_results.append(result)
                continue

            # Step 1: Expand shot with context (±N neighboring shots)
            expanded_shots = expand_shot_context(
                target_shot, all_shots, context_window=self.context_window
            )

            # Step 2: Merge into video segment
            segment = self._create_segment(expanded_shots, video_id)

            # Step 3: Get frames for the segment
            frames = self._get_segment_frames(
                segment, target_shot, video_dir
            )

            # Step 4: Score with MLLM
            score = self._score_segment(query, frames)

            result["relevance_score"] = score
            result["segment_info"] = {
                "num_shots": len(expanded_shots),
                "start_time": segment.start_time if segment else 0,
                "end_time": segment.end_time if segment else 0,
            }
            scored_results.append(result)

            if (i + 1) % 10 == 0:
                logger.info(
                    f"Re-ranked {i + 1}/{len(retrieval_results)} candidates"
                )

        # Step 5: Sort by relevance score descending
        scored_results.sort(
            key=lambda x: x.get("relevance_score", 0),
            reverse=True,
        )

        top_msg = (
            f"{scored_results[0].get('relevance_score', 0):.3f}"
            if scored_results else "N/A"
        )
        logger.info(f"Re-ranking complete. Top score: {top_msg}")

        return scored_results[:top_k]

    def _create_segment(
        self,
        expanded_shots: List[Shot],
        video_id: str,
    ) -> Optional[VideoSegment]:
        """Create a merged video segment from expanded shots."""
        if not expanded_shots:
            return None

        return VideoSegment(
            video_id=video_id,
            shots=expanded_shots,
            start_time=expanded_shots[0].start_time,
            end_time=expanded_shots[-1].end_time,
        )

    def _get_segment_frames(
        self,
        segment: Optional[VideoSegment],
        target_shot: Shot,
        video_dir: str,
    ) -> List:
        """
        Get representative frames from a video segment.
        Prioritizes keyframes from the target shot and its context.
        """
        from PIL import Image
        import os

        frames = []

        if segment is None:
            return frames

        # Collect keyframe paths from all shots in segment
        all_keyframe_paths = []
        for shot in segment.shots:
            if shot.keyframe_paths:
                all_keyframe_paths.extend(shot.keyframe_paths)

        if all_keyframe_paths:
            # Sample up to max_frames_per_segment keyframes
            if len(all_keyframe_paths) > self.max_frames_per_segment:
                indices = np.linspace(
                    0, len(all_keyframe_paths) - 1,
                    self.max_frames_per_segment, dtype=int
                )
                sampled_paths = [all_keyframe_paths[i] for i in indices]
            else:
                sampled_paths = all_keyframe_paths

            for path in sampled_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    frames.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load keyframe: {path}: {e}")
        else:
            # Fall back to extracting from video file
            if video_dir and target_shot.video_id:
                video_path = os.path.join(
                    video_dir, f"{target_shot.video_id}.mp4"
                )
                if os.path.exists(video_path):
                    try:
                        raw_frames = extract_frames(
                            video_path,
                            start_time=segment.start_time,
                            end_time=segment.end_time,
                            num_frames=self.max_frames_per_segment,
                        )
                        from vrag.utils.video_utils import frames_to_pil_images
                        frames = frames_to_pil_images(raw_frames)
                    except Exception as e:
                        logger.warning(f"Failed to extract frames: {e}")

        return frames

    def _score_segment(
        self,
        query: str,
        frames: List,
    ) -> float:
        """
        Score a video segment's relevance to the query using MLLM.
        
        Returns a float between 0.0 and 1.0.
        """
        if not frames or self._model is None:
            return 0.0

        try:
            return self._score_internvl(
                RERANKING_PROMPT_TEMPLATE.format(query=query), frames
            )
        except Exception as e:
            logger.warning(f"Scoring failed: {e}")
            return 0.0

    def _score_internvl(self, prompt: str, frames: List) -> float:
        """Score using InternVL2.5."""
        import torch

        # Build pixel values from frames
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Use subset of frames
        sample_frames = frames[:min(len(frames), 8)]
        pixel_values = torch.stack([transform(f) for f in sample_frames])
        pixel_values = pixel_values.to(
            dtype=self._model.dtype, device=self._model.device
        )

        # Build image tags for the prompt
        num_frames = len(sample_frames)
        image_tags = "".join(
            [f"Frame {i+1}: <image>\n" for i in range(num_frames)]
        )
        full_prompt = image_tags + prompt

        # Generate
        generation_config = {"max_new_tokens": 10, "do_sample": False}
        response = self._model.chat(
            self._tokenizer,
            pixel_values,
            full_prompt,
            generation_config,
        )

        return self._parse_score(response)

    def _parse_score(self, response: str) -> float:
        """Parse MLLM response to extract float score."""
        import re

        response = response.strip()

        # Try to extract a float from the response
        match = re.search(r'(\d+\.?\d*)', response)
        if match:
            score = float(match.group(1))
            return float(np.clip(score, 0.0, 1.0))

        return 0.0
