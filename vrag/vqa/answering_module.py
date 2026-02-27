"""
VRAG VQA Answering Module

Generates answers to user questions using relevant video segments.
From the paper (Section 3.3): "The Answering Module aggregates the relevant 
segments identified by the Filtering Module and uses a MLLM to generate the 
final answer to the user's question."
"""

import logging
from typing import Dict, List, Optional

from vrag.utils.video_utils import VRAGResult

logger = logging.getLogger(__name__)

# Answering prompt template
ANSWERING_PROMPT_TEMPLATE = """You are a video question answering assistant. Based on the provided video frames from relevant segments, answer the user's question accurately and concisely.

Question: {question}

Instructions:
1. Carefully analyze all provided video frames.
2. Base your answer ONLY on what you observe in the video content.
3. If the answer is not clearly visible in the frames, say so.
4. Provide a clear, direct answer.

Your answer:"""


class AnsweringModule:
    """
    VQA Answering Module - Generates answers from relevant video chunks.
    
    Process (Paper Section 3.3):
    1. Aggregate relevant chunks from the Filtering Module
    2. Sample representative frames from relevant chunks
    3. MLLM generates answer based on frames + question
    
    Best model from paper: VideoLLaMA3-7B achieved 4/5 VQA score
    Lightest: VideoLLaMA3-2B used here.
    """

    def __init__(
        self,
        mllm_model: str = "DAMO-NLP-SG/VideoLLaMA3-7B",
        max_frames: int = 32,
        max_chunks: int = 10,
    ):
        """
        Args:
            mllm_model: Full HuggingFace model path for answer generation.
            max_frames: Maximum total frames to provide to MLLM.
            max_chunks: Maximum number of chunks to use.
        """
        self.mllm_model = mllm_model
        self.max_frames = max_frames
        self.max_chunks = max_chunks
        self._model = None

    def _load_model(self):
        """Load the answering MLLM."""
        if self._model is not None:
            return

        self._load_videollama()

    def _load_videollama(self):
        """Load VideoLLaMA3 for answering."""
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

        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

    def answer(
        self,
        question: str,
        relevant_chunks: List[Dict],
        metadata: Optional[Dict] = None,
    ) -> VRAGResult:
        """
        Generate an answer based on relevant video chunks.

        Args:
            question: The specific question to answer.
            relevant_chunks: List of relevant chunk dicts from FilteringModule.
                Each chunk has 'frames', 'start_time', 'end_time', 'confidence'.
            metadata: Additional metadata (video_id, etc.).

        Returns:
            VRAGResult with answer and confidence.
        """
        self._load_model()

        if not relevant_chunks:
            return VRAGResult(
                query=question,
                answer="I could not find relevant video content to answer this question.",
                confidence=0.0,
                task_type="vqa",
            )

        # Sort by confidence and limit
        chunks = sorted(
            relevant_chunks,
            key=lambda x: x.get("confidence", 0),
            reverse=True,
        )[:self.max_chunks]

        # Aggregate frames from relevant chunks
        all_frames = self._aggregate_frames(chunks)

        if not all_frames:
            return VRAGResult(
                query=question,
                answer="No frames available from the relevant video segments.",
                confidence=0.0,
                task_type="vqa",
            )

        # Generate answer
        try:
            answer = self._generate_answer(question, all_frames)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            answer = f"Error generating answer: {str(e)}"

        # Build source references
        sources = [
            {
                "chunk_id": chunk.get("chunk_id", i),
                "start_time": chunk.get("start_time", 0),
                "end_time": chunk.get("end_time", 0),
                "confidence": chunk.get("confidence", 0),
            }
            for i, chunk in enumerate(chunks)
        ]

        avg_confidence = sum(c.get("confidence", 0) for c in chunks) / len(chunks)

        return VRAGResult(
            query=question,
            answer=answer,
            confidence=avg_confidence,
            task_type="vqa",
            sources=sources,
        )

    def _aggregate_frames(self, chunks: List[Dict]) -> List:
        """
        Aggregate and sample frames from multiple chunks.

        Distributes frames proportionally to chunk confidence scores.
        """
        total_frames = []

        # Calculate frames per chunk based on confidence
        total_confidence = sum(c.get("confidence", 0.5) for c in chunks)
        if total_confidence == 0:
            total_confidence = len(chunks)

        for chunk in chunks:
            chunk_frames = chunk.get("frames", [])
            if not chunk_frames:
                continue

            # Proportional allocation
            weight = chunk.get("confidence", 0.5) / total_confidence
            n_frames = max(1, int(self.max_frames * weight))

            if len(chunk_frames) > n_frames:
                import numpy as np
                indices = np.linspace(
                    0, len(chunk_frames) - 1, n_frames, dtype=int
                )
                selected = [chunk_frames[i] for i in indices]
            else:
                selected = chunk_frames

            total_frames.extend(selected)

        # Final cap
        if len(total_frames) > self.max_frames:
            import numpy as np
            indices = np.linspace(
                0, len(total_frames) - 1, self.max_frames, dtype=int
            )
            total_frames = [total_frames[i] for i in indices]

        return total_frames

    def _generate_answer(self, question: str, frames: List) -> str:
        """Generate answer using VideoLLaMA3 with proper chat template."""
        import torch

        prompt = ANSWERING_PROMPT_TEMPLATE.format(question=question)

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
                    **inputs, max_new_tokens=512, do_sample=False
                )

            # Decode only the generated tokens (skip input)
            input_len = inputs.get("input_ids", inputs.get("input_token_ids")).shape[-1]
            response = self._processor.decode(
                output[0][input_len:], skip_special_tokens=True
            )
            return response.strip()
        except Exception as e:
            logger.error(f"VideoLLaMA3 generation failed: {e}")
            return f"Error: {str(e)}"
