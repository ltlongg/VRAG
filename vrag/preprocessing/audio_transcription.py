"""
VRAG Audio Transcription Module

Transcribes spoken content in videos using OpenAI Whisper.
From the paper (Section 3.1): "Our system enables video retrieval based on spoken 
content by utilizing Whisper for automatic speech transcription."
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """
    Transcribe audio from videos using OpenAI Whisper.
    
    Enables retrieval of video segments based on spoken content.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        language: Optional[str] = None,
        device: str = "cuda",
        batch_size: int = 16,
    ):
        """
        Args:
            model_name: Whisper model identifier.
            language: Target language (None for auto-detect).
            device: Computation device.
            batch_size: Batch size for processing.
        """
        self.model_name = model_name
        self.language = language
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy-load the Whisper model."""
        if self._model is not None:
            return

        try:
            # Try using the faster-whisper implementation first
            from faster_whisper import WhisperModel
            compute_type = "float16" if self.device == "cuda" else "int8"
            
            # Map full model name to size
            size_map = {
                "openai/whisper-large-v3": "large-v3",
                "openai/whisper-large-v2": "large-v2",
                "openai/whisper-medium": "medium",
                "openai/whisper-small": "small",
                "openai/whisper-base": "base",
                "openai/whisper-tiny": "tiny",
            }
            model_size = size_map.get(self.model_name, "large-v3")
            
            self._model = WhisperModel(
                model_size, device=self.device, compute_type=compute_type
            )
            self._engine = "faster_whisper"
            logger.info(f"Loaded faster-whisper model: {model_size}")

        except ImportError:
            try:
                # Fallback to transformers Whisper
                import torch
                from transformers import (
                    WhisperProcessor,
                    WhisperForConditionalGeneration,
                )
                self._processor = WhisperProcessor.from_pretrained(self.model_name)
                self._model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                ).to(self.device)
                self._model.eval()
                self._engine = "transformers"
                logger.info(f"Loaded transformers Whisper: {self.model_name}")

            except ImportError:
                try:
                    # Fallback to openai-whisper
                    import whisper
                    size = self.model_name.split("-")[-1] if "-" in self.model_name else "base"
                    self._model = whisper.load_model(size, device=self.device)
                    self._engine = "openai_whisper"
                    logger.info(f"Loaded openai-whisper model: {size}")
                except ImportError:
                    raise ImportError(
                        "No Whisper implementation found. Install one of:\n"
                        "  pip install faster-whisper\n"
                        "  pip install transformers\n"
                        "  pip install openai-whisper"
                    )

    def transcribe(
        self,
        audio_path: str,
        word_timestamps: bool = True,
    ) -> Dict:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file (WAV recommended).
            word_timestamps: Whether to include word-level timestamps.

        Returns:
            Dictionary with keys:
              - 'text': Full transcription text.
              - 'segments': List of segments with timestamps.
              - 'language': Detected language.
        """
        self._load_model()

        if self._engine == "faster_whisper":
            return self._transcribe_faster_whisper(audio_path, word_timestamps)
        elif self._engine == "transformers":
            return self._transcribe_transformers(audio_path)
        else:
            return self._transcribe_openai_whisper(audio_path, word_timestamps)

    def _transcribe_faster_whisper(
        self, audio_path: str, word_timestamps: bool
    ) -> Dict:
        """Transcribe using faster-whisper."""
        segments, info = self._model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=word_timestamps,
            beam_size=5,
        )

        result_segments = []
        full_text = []

        for segment in segments:
            seg_data = {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
            }
            if word_timestamps and segment.words:
                seg_data["words"] = [
                    {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                    for w in segment.words
                ]
            result_segments.append(seg_data)
            full_text.append(segment.text.strip())

        return {
            "text": " ".join(full_text),
            "segments": result_segments,
            "language": info.language,
        }

    def _transcribe_transformers(self, audio_path: str) -> Dict:
        """Transcribe using HuggingFace transformers Whisper."""
        import torch
        import librosa

        audio, sr = librosa.load(audio_path, sr=16000)
        inputs = self._processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).to(self.device, torch.float16)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                return_timestamps=True,
                language=self.language,
            )

        transcription = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return {
            "text": transcription.strip(),
            "segments": [{"start": 0, "end": 0, "text": transcription.strip()}],
            "language": self.language or "auto",
        }

    def _transcribe_openai_whisper(
        self, audio_path: str, word_timestamps: bool
    ) -> Dict:
        """Transcribe using openai-whisper."""
        result = self._model.transcribe(
            audio_path,
            language=self.language,
            word_timestamps=word_timestamps,
        )

        segments = []
        for seg in result.get("segments", []):
            seg_data = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
            if word_timestamps and "words" in seg:
                seg_data["words"] = seg["words"]
            segments.append(seg_data)

        return {
            "text": result.get("text", "").strip(),
            "segments": segments,
            "language": result.get("language", "unknown"),
        }

    def transcribe_video(
        self,
        video_path: str,
        temp_dir: str = None,
    ) -> Dict:
        """
        Extract audio from a video and transcribe it.

        Args:
            video_path: Path to the video file.
            temp_dir: Temporary directory for extracted audio.

        Returns:
            Transcription result dictionary.
        """
        from vrag.utils.video_utils import extract_audio
        import tempfile

        if temp_dir is None:
            temp_dir = os.path.join(tempfile.gettempdir(), "vrag_audio")
        os.makedirs(temp_dir, exist_ok=True)
        video_name = Path(video_path).stem
        audio_path = os.path.join(temp_dir, f"{video_name}.wav")

        try:
            extract_audio(video_path, audio_path)
            result = self.transcribe(audio_path)
            return result
        except Exception as e:
            logger.error(f"Failed to transcribe video {video_path}: {e}")
            return {"text": "", "segments": [], "language": "unknown"}
        finally:
            # Cleanup temp audio
            if os.path.exists(audio_path):
                os.remove(audio_path)

    def get_transcript_for_timerange(
        self,
        transcription: Dict,
        start_time: float,
        end_time: float,
    ) -> str:
        """
        Get transcript text within a specific time range.

        Args:
            transcription: Full transcription result.
            start_time: Start time in seconds.
            end_time: End time in seconds.

        Returns:
            Transcript text within the time range.
        """
        texts = []
        for segment in transcription.get("segments", []):
            seg_start = segment.get("start", 0)
            seg_end = segment.get("end", 0)

            # Check overlap
            if seg_start < end_time and seg_end > start_time:
                texts.append(segment["text"])

        return " ".join(texts)
