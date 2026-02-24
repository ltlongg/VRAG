"""
VRAG OCR Extraction Module

Extracts on-screen text from video frames for text-based video retrieval.
From the paper (Section 3.1): "We employ optical character recognition (OCR) to 
extract textual content from video frames. Specifically, DeepSolo is utilized for 
text detection, while PARSeq is employed for text recognition."
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class OCRExtractor:
    """
    Extract on-screen text from video keyframes.
    
    Architecture (per paper):
      - Text Detection: DeepSolo
      - Text Recognition: PARSeq
    """

    def __init__(
        self,
        detector: str = "deepsolo",
        recognizer: str = "parseq",
        detection_confidence: float = 0.5,
        recognition_confidence: float = 0.7,
        device: str = "cuda",
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.detection_confidence = detection_confidence
        self.recognition_confidence = recognition_confidence
        self.device = device
        self._ocr_engine = None

    def _load_engine(self):
        """Load the OCR engine."""
        if self._ocr_engine is not None:
            return

        self._load_deepsolo_parseq()

    def _load_deepsolo_parseq(self):
        """
        Load DeepSolo (detection) + PARSeq (recognition).
        
        DeepSolo: 'Let Transformer Decoder with Explicit Points Solo for Text Spotting'
        PARSeq: 'Scene Text Recognition with Permuted Autoregressive Sequence Models'
        """
        try:
            # Try loading DeepSolo from detectron2/adet
            from adet.config import get_cfg
            from detectron2.engine import DefaultPredictor
            
            cfg = get_cfg()
            cfg.merge_from_file("configs/DeepSolo/R_50/TotalText/finetune_150k.yaml")
            cfg.MODEL.WEIGHTS = "models/deepsolo_totaltext.pth"
            cfg.MODEL.DEVICE = self.device
            self._detector_model = DefaultPredictor(cfg)
            
            # Load PARSeq
            from strhub.data.module import SceneTextDataModule
            import torch
            self._recognizer_model = torch.hub.load(
                'baudm/parseq', 'parseq', pretrained=True
            ).eval().to(self.device)
            self._ocr_engine = "deepsolo_parseq"
            logger.info("Loaded DeepSolo + PARSeq OCR pipeline")
            
        except ImportError:
            raise ImportError(
                "DeepSolo/PARSeq required. Install detectron2/adet and strhub."
            )

    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text from a single image/frame.

        Args:
            image: BGR numpy array (OpenCV format).

        Returns:
            List of dicts with keys: 'text', 'confidence', 'bbox'
        """
        self._load_engine()

        if self._ocr_engine is None:
            return []

        return self._extract_deepsolo_parseq(image)

    def _extract_deepsolo_parseq(self, image: np.ndarray) -> List[Dict]:
        """Extract text using DeepSolo detection + PARSeq recognition."""
        # Detection with DeepSolo
        outputs = self._detector_model(image)
        instances = outputs["instances"]
        
        extracted = []
        for i in range(len(instances)):
            score = instances.scores[i].item()
            if score < self.detection_confidence:
                continue
                
            # Get bounding box and crop
            bbox = instances.pred_boxes[i].tensor.cpu().numpy()[0]
            x1, y1, x2, y2 = map(int, bbox)
            text_crop = image[y1:y2, x1:x2]
            
            if text_crop.size == 0:
                continue
            
            # Recognition with PARSeq
            pil_crop = Image.fromarray(cv2.cvtColor(text_crop, cv2.COLOR_BGR2RGB))
            text, confidence = self._recognize_parseq(pil_crop)
            
            if confidence >= self.recognition_confidence:
                extracted.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                })
        
        return extracted

    def _recognize_parseq(self, image: Image.Image) -> Tuple[str, float]:
        """Recognize text in a cropped image using PARSeq."""
        import torch
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])
        
        tensor = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._recognizer_model(tensor)
            probs = logits.softmax(-1)
            preds, confidences = self._recognizer_model.tokenizer.decode(probs)
        
        return preds[0], confidences[0].mean().item()

    def extract_text_from_keyframes(
        self, 
        keyframe_paths: List[str],
    ) -> Dict[str, List[Dict]]:
        """
        Extract text from multiple keyframe images.

        Args:
            keyframe_paths: List of paths to keyframe images.

        Returns:
            Dict mapping keyframe path to list of text detections.
        """
        results = {}
        for path in keyframe_paths:
            image = cv2.imread(path)
            if image is None:
                logger.warning(f"Cannot read image: {path}")
                results[path] = []
                continue
            results[path] = self.extract_text(image)
        return results

    def extract_text_for_shot(
        self, shot, 
    ) -> str:
        """
        Extract and concatenate all text found in a shot's keyframes.

        Args:
            shot: Shot object with keyframe_paths populated.

        Returns:
            Concatenated string of all detected text.
        """
        all_text = []
        results = self.extract_text_from_keyframes(shot.keyframe_paths)
        for path, detections in results.items():
            for det in detections:
                all_text.append(det["text"])
        return " ".join(all_text)
