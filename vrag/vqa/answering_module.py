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
    """

    def __init__(
        self,
        mllm_model: str = "VideoLLaMA3-7B",
        max_frames: int = 32,
        max_chunks: int = 10,
    ):
        """
        Args:
            mllm_model: MLLM for answer generation.
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

        model_name = self.mllm_model.lower()

        if "videollama" in model_name:
            self._load_videollama()
        elif "internvl" in model_name:
            self._load_internvl()
        elif "gpt" in model_name:
            self._load_gpt()
        else:
            logger.warning(
                f"Model '{self.mllm_model}' not recognized. Using GPT fallback."
            )
            self._load_gpt()

    def _load_videollama(self):
        """Load VideoLLaMA3 for answering."""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            import torch

            model_path = f"DAMO-NLP-SG/{self.mllm_model}"
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

        except Exception as e:
            logger.warning(f"Failed to load VideoLLaMA3: {e}. Using GPT fallback.")
            self._load_gpt()

    def _load_internvl(self):
        """Load InternVL for answering."""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch

            model_path = f"OpenGVLab/{self.mllm_model}"
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

        except Exception as e:
            logger.warning(f"Failed to load InternVL: {e}. Using GPT fallback.")
            self._load_gpt()

    def _load_gpt(self):
        """Load GPT-4o for answering (API-based)."""
        try:
            import openai
            self._client = openai.OpenAI()
            self._model_type = "gpt"
            logger.info("GPT-4o initialized for VQA answering")
        except Exception as e:
            logger.error(f"Failed to init GPT client: {e}")
            self._model_type = "none"

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
                answer="I could not find relevant video content to answer this question.",
                confidence=0.0,
                task_type="vqa",
                sources=[],
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
                answer="No frames available from the relevant video segments.",
                confidence=0.0,
                task_type="vqa",
                sources=[],
            )

        # Generate answer
        model_type = getattr(self, "_model_type", "none")
        try:
            if model_type == "videollama":
                answer = self._answer_videollama(question, all_frames)
            elif model_type == "internvl":
                answer = self._answer_internvl(question, all_frames)
            elif model_type == "gpt":
                answer = self._answer_gpt(question, all_frames)
            else:
                answer = "Model not available for answering."
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

    def _answer_videollama(self, question: str, frames: List) -> str:
        """Generate answer using VideoLLaMA3."""
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

        inputs = self._processor(messages, return_tensors="pt").to(
            self._model.device
        )

        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=512, do_sample=False
            )

        response = self._processor.decode(
            output[0], skip_special_tokens=True
        )
        return response.strip()

    def _answer_internvl(self, question: str, frames: List) -> str:
        """Generate answer using InternVL."""
        import torch
        from torchvision import transforms

        prompt = ANSWERING_PROMPT_TEMPLATE.format(question=question)

        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        sample_frames = frames[:min(len(frames), 16)]
        pixel_values = torch.stack([transform(f) for f in sample_frames])
        pixel_values = pixel_values.to(
            dtype=self._model.dtype, device=self._model.device
        )

        num_frames = len(sample_frames)
        image_tags = "".join(
            [f"Frame {i+1}: <image>\n" for i in range(num_frames)]
        )
        full_prompt = image_tags + prompt

        generation_config = {"max_new_tokens": 512, "do_sample": False}
        response = self._model.chat(
            self._tokenizer, pixel_values, full_prompt, generation_config
        )
        return response.strip()

    def _answer_gpt(self, question: str, frames: List) -> str:
        """Generate answer using GPT-4o (via API with encoded images)."""
        import base64
        from io import BytesIO

        prompt = ANSWERING_PROMPT_TEMPLATE.format(question=question)

        # Encode frames as base64 for GPT-4o
        content = [{"type": "text", "text": prompt}]

        # Sample up to 8 frames for API efficiency
        if len(frames) > 8:
            import numpy as np
            indices = np.linspace(0, len(frames) - 1, 8, dtype=int)
            selected = [frames[i] for i in indices]
        else:
            selected = frames

        for frame in selected:
            try:
                buffered = BytesIO()
                frame.save(buffered, format="JPEG", quality=85)
                img_b64 = base64.b64encode(buffered.getvalue()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                        "detail": "low",
                    },
                })
            except Exception as e:
                logger.warning(f"Failed to encode frame: {e}")

        try:
            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                max_tokens=512,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"GPT-4o answering failed: {e}")
            return f"Error: {str(e)}"
