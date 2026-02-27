"""
VRAG VQA Filtering Module

Filters video chunks for relevance to the user query using MLLM.
From the paper (Section 3.3): "The Filtering Module processes a long video by 
segmenting it into overlapping chunks and uses a MLLM to determine the binary 
relevance of each chunk."
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from vrag.utils.video_utils import (
    VideoSegment, Shot, chunk_video, extract_frames, frames_to_pil_images
)

logger = logging.getLogger(__name__)

# Prompt template for binary relevance decision
FILTERING_PROMPT_TEMPLATE = """You are a video relevance assessor. Given video frames and a query, determine if the video content is relevant to answering the query.

Query: {query}

Instructions:
1. Analyze the provided video frames carefully.
2. Determine if the video content contains information relevant to the query.
3. Respond with ONLY "YES" or "NO".
   - YES = the video content is relevant to the query
   - NO = the video content is NOT relevant to the query

Your response must be ONLY "YES" or "NO"."""


class FilteringModule:
    """
    VQA Filtering Module - Segments video and filters irrelevant chunks.
    
    Process (Paper Section 3.3):
    1. Segment the retrieved video into overlapping chunks (15s default, 5s overlap)
    2. MLLM makes binary relevance decision per chunk
    3. Relevant chunks are passed to the Answering Module
    
    Best configuration from paper: chunk_size=15s, VideoLLaMA3-7B
    Lightest: VideoLLaMA3-2B used here.
    """

    def __init__(
        self,
        mllm_model: str = "DAMO-NLP-SG/VideoLLaMA3-7B",
        chunk_size: float = 15.0,
        chunk_overlap: float = 5.0,
        max_frames_per_chunk: int = 16,
        confidence_threshold: float = 0.5,
    ):
        """
        Args:
            mllm_model: Full HuggingFace model path for relevance classification.
            chunk_size: Duration of each chunk in seconds.
            chunk_overlap: Overlap between consecutive chunks in seconds.
            max_frames_per_chunk: Frames sampled per chunk for MLLM.
            confidence_threshold: Threshold above which a chunk is considered relevant.
        """
        self.mllm_model = mllm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_frames_per_chunk = max_frames_per_chunk
        self.confidence_threshold = confidence_threshold
        self._model = None

    def _load_model(self):
        """Load the MLLM for relevance classification."""
        if self._model is not None:
            return

        self._load_videollama()

    def _load_videollama(self):
        """Load VideoLLaMA3 for filtering."""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            import torch

            model_path = self.mllm_model
            # Handle bare model names without org prefix
            if "/" not in model_path:
                model_path = f"DAMO-NLP-SG/{model_path}"
            logger.info(f"Loading VideoLLaMA3: {model_path}")

            self._processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            )
            self._model.eval()
            self._model_type = "videollama"
            logger.info("VideoLLaMA3 loaded successfully")

        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

    def filter_video(
        self,
        query: str,
        video_path: str,
        start_time: float = 0,
        end_time: float = None,
    ) -> List[Dict]:
        """
        Filter a video for relevant chunks.

        Args:
            query: User query.
            video_path: Path to the video file.
            start_time: Start time in seconds.
            end_time: End time in seconds.

        Returns:
            List of relevant chunk dicts with 'start_time', 'end_time', 'is_relevant'.
        """
        self._load_model()

        # Create chunks with overlap
        if end_time is not None:
            actual_end = end_time
        else:
            from vrag.utils.video_utils import get_video_info
            info = get_video_info(video_path)
            actual_end = info["duration"]
        duration = actual_end - start_time
        if duration <= 0:
            return []

        raw_chunks = chunk_video(
            video_duration=duration,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        )

        # Offset each chunk by the absolute start_time
        chunks = [
            (start_time + c_start, start_time + c_end)
            for c_start, c_end in raw_chunks
        ]

        relevant_chunks = []

        for i, (chunk_start, chunk_end) in enumerate(chunks):
            # Extract frames from this chunk
            try:
                raw_frames = extract_frames(
                    video_path,
                    start_time=chunk_start,
                    end_time=chunk_end,
                    num_frames=self.max_frames_per_chunk,
                )
                frames = frames_to_pil_images(raw_frames)
            except Exception as e:
                logger.warning(f"Failed to extract frames for chunk {i}: {e}")
                continue

            if not frames:
                continue

            # Binary relevance decision
            is_relevant, confidence = self._assess_relevance(query, frames)

            chunk_result = {
                "chunk_id": i,
                "start_time": chunk_start,
                "end_time": chunk_end,
                "is_relevant": is_relevant,
                "confidence": confidence,
            }

            if is_relevant:
                chunk_result["frames"] = frames
                relevant_chunks.append(chunk_result)

            if (i + 1) % 10 == 0:
                logger.info(
                    f"Filtered {i + 1}/{len(chunks)} chunks, "
                    f"{len(relevant_chunks)} relevant"
                )

        logger.info(
            f"Filtering complete: {len(relevant_chunks)}/{len(chunks)} "
            f"chunks are relevant"
        )
        return relevant_chunks

    def filter_segment(
        self,
        query: str,
        segment: VideoSegment,
        video_path: str,
    ) -> List[Dict]:
        """
        Filter a video segment (from re-ranker) for relevant chunks.
        """
        return self.filter_video(
            query=query,
            video_path=video_path,
            start_time=segment.start_time,
            end_time=segment.end_time,
        )

    def filter_shots(
        self,
        query: str,
        shots: List[Shot],
        video_path: str,
    ) -> List[Dict]:
        """
        Filter a list of shots, treating each shot as a chunk.
        """
        self._load_model()
        relevant = []

        for shot in shots:
            # Use keyframes if available
            frames = []
            if shot.keyframe_paths:
                from PIL import Image
                for path in shot.keyframe_paths[:self.max_frames_per_chunk]:
                    try:
                        frames.append(Image.open(path).convert("RGB"))
                    except Exception:
                        pass

            if not frames:
                try:
                    raw = extract_frames(
                        video_path,
                        start_time=shot.start_time,
                        end_time=shot.end_time,
                        num_frames=self.max_frames_per_chunk,
                    )
                    frames = frames_to_pil_images(raw)
                except Exception:
                    continue

            if not frames:
                continue

            is_relevant, confidence = self._assess_relevance(query, frames)

            if is_relevant:
                relevant.append({
                    "shot_id": shot.shot_id,
                    "start_time": shot.start_time,
                    "end_time": shot.end_time,
                    "is_relevant": True,
                    "confidence": confidence,
                    "frames": frames,
                })

        return relevant

    def _assess_relevance(
        self, query: str, frames: List
    ) -> Tuple[bool, float]:
        """
        MLLM binary relevance decision: YES/NO.

        Returns:
            (is_relevant, confidence)
        """
        try:
            return self._assess_mllm(query, frames)
        except Exception as e:
            logger.warning(f"Relevance assessment failed: {e}")
            return False, 0.0

    def _assess_mllm(self, query: str, frames: List) -> Tuple[bool, float]:
        """MLLM-based binary relevance decision."""
        prompt = FILTERING_PROMPT_TEMPLATE.format(query=query)
        response = self._generate_response(prompt, frames)

        # Parse YES/NO response
        response_clean = response.strip().upper()
        if "YES" in response_clean:
            return True, 0.9
        elif "NO" in response_clean:
            return False, 0.1
        else:
            logger.warning(f"Ambiguous MLLM response: {response}")
            return False, 0.5

    def _generate_response(self, prompt: str, frames: List) -> str:
        """Generate response using VideoLLaMA3 with proper chat template."""
        import torch

        # Build conversation messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            # Try using the processor's apply_chat_template if available
            if hasattr(self._processor, 'apply_chat_template'):
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._processor(
                    text=text, 
                    videos=frames,
                    return_tensors="pt",
                    padding=True,
                ).to(self._model.device)
            else:
                # Direct processing fallback
                inputs = self._processor(
                    messages, return_tensors="pt"
                ).to(self._model.device)

            with torch.no_grad():
                output = self._model.generate(
                    **inputs, max_new_tokens=10, do_sample=False
                )

            # Decode only the generated tokens (skip input)
            input_len = inputs.get("input_ids", inputs.get("input_token_ids")).shape[-1]
            response = self._processor.decode(
                output[0][input_len:], skip_special_tokens=True
            )
            return response
        except Exception as e:
            logger.warning(f"VideoLLaMA3 inference failed: {e}")
            return ""
