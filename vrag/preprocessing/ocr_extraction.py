"""
VRAG OCR Extraction Module

Extracts on-screen text from video frames for text-based video retrieval.
From the paper (Section 3.1): "We employ optical character recognition (OCR) to 
extract textual content from video frames. Specifically, DeepSolo is utilized for 
text detection, while PARSeq is employed for text recognition."

Falls back to EasyOCR when DeepSolo+PARSeq are not installed.
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
      - Primary: DeepSolo (detection) + PARSeq (recognition)
      - Fallback: EasyOCR (when DeepSolo/PARSeq not installed)
    """

    def __init__(
        self,
        detector: str = "deepsolo",
        recognizer: str = "parseq",
        detection_confidence: float = 0.5,
        recognition_confidence: float = 0.7,
        device: str = "cuda",
        deepsolo_config: Optional[str] = None,
        deepsolo_weights: Optional[str] = None,
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.detection_confidence = detection_confidence
        self.recognition_confidence = recognition_confidence
        self.device = device
        self.deepsolo_config = deepsolo_config
        self.deepsolo_weights = deepsolo_weights
        self._ocr_engine = None

    def _load_engine(self):
        """Load the OCR engine."""
        if self._ocr_engine is not None:
            return

        # Try DeepSolo + PARSeq first (paper's method)
        try:
            self._load_deepsolo_parseq()
            return
        except (ImportError, Exception) as e:
            logger.warning(
                f"DeepSolo+PARSeq not available ({e}). "
                "Falling back to EasyOCR."
            )

        # Fallback to EasyOCR
        try:
            self._load_easyocr()
            return
        except ImportError:
            raise ImportError(
                "No OCR engine available. Install one of:\n"
                "  1. DeepSolo + PARSeq (via detectron2/adet + strhub)\n"
                "  2. EasyOCR: pip install easyocr"
            )

    def _load_deepsolo_parseq(self):
        """
        Load DeepSolo (detection) + PARSeq (recognition).
        
        DeepSolo: 'Let Transformer Decoder with Explicit Points Solo for Text Spotting'
        PARSeq: 'Scene Text Recognition with Permuted Autoregressive Sequence Models'
        """
        from adet.config import get_cfg
        from detectron2.engine import DefaultPredictor
        
        cfg = get_cfg()
        
        # Use configurable paths instead of hardcoded
        config_file = self.deepsolo_config or \
            "configs/DeepSolo/R_50/TotalText/finetune_150k.yaml"
        weights_file = self.deepsolo_weights or \
            "models/deepsolo_totaltext.pth"
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"DeepSolo config not found: {config_file}"
            )
        
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = weights_file
        cfg.MODEL.DEVICE = self.device
        self._detector_model = DefaultPredictor(cfg)
        
        # Load PARSeq
        import torch
        self._recognizer_model = torch.hub.load(
            'baudm/parseq', 'parseq', pretrained=True
        ).eval().to(self.device)
        self._ocr_engine = "deepsolo_parseq"
        logger.info("Loaded DeepSolo + PARSeq OCR pipeline")

    def _load_easyocr(self):
        """Load EasyOCR as fallback."""
        import easyocr
        self._reader = easyocr.Reader(
            ['en'], gpu=(self.device == "cuda")
        )
        self._ocr_engine = "easyocr"
        logger.info("Loaded EasyOCR as OCR engine")

    def extract_text(self, image: np.ndarray) -> List[Dict]:
        """
        Extract text from a single image/frame.

        Args:
            image: BGR numpy array (OpenCV format).

        Returns:
            List of dicts with keys: 'text', 'confidence', 'bbox'
        """
        self._load_engine()

        if self._ocr_engine == "deepsolo_parseq":
            return self._extract_deepsolo_parseq(image)
        elif self._ocr_engine == "easyocr":
            return self._extract_easyocr(image)
        else:
            return []

    def _extract_deepsolo_parseq(self, image: np.ndarray) -> List[Dict]:
        """Extract text using DeepSolo detection + PARSeq recognition."""
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

    def _extract_easyocr(self, image: np.ndarray) -> List[Dict]:
        """Extract text using EasyOCR."""
        # EasyOCR expects BGR or RGB numpy array
        results = self._reader.readtext(image)
        
        extracted = []
        for (bbox, text, confidence) in results:
            if confidence < self.recognition_confidence:
                continue
            # bbox is a list of 4 corner points, convert to [x1,y1,x2,y2]
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            extracted.append({
                "text": text,
                "confidence": confidence,
                "bbox": [int(min(xs)), int(min(ys)), 
                         int(max(xs)), int(max(ys))],
            })
        
        return extracted

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
