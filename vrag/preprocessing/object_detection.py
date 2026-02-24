"""
VRAG Object Detection Module

Detects objects in video keyframes for object-based filtering.
From the paper (Section 3.1): "Object extraction is performed using Co-DETR 
to ensure the retrieved segments align with the query constraints."
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# COCO class names for reference
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


class ObjectDetector:
    """
    Detect objects in video keyframes using Co-DETR or fallback detectors.
    
    Used as a post-processing filter to ensure retrieved segments contain
    objects that match query constraints.
    """

    def __init__(
        self,
        model_name: str = "co-detr",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy-load the object detection model."""
        if self._model is not None:
            return

        if self.model_name == "co-detr":
            self._load_co_detr()
        else:
            self._load_fallback()

    def _load_co_detr(self):
        """
        Load Co-DETR (DETRs with Collaborative Hybrid Assignments Training).
        Falls back to DETA or DETR if Co-DETR is not available.
        """
        try:
            # Try Co-DETR from mmdetection or hub
            from transformers import AutoModelForObjectDetection, AutoImageProcessor
            
            # Use DETA as a close approximation of Co-DETR
            model_id = "jozhang97/deta-swin-large"
            self._processor = AutoImageProcessor.from_pretrained(model_id)
            self._model = AutoModelForObjectDetection.from_pretrained(model_id).to(
                self.device
            )
            self._model.eval()
            self._engine = "deta"
            logger.info(f"Loaded DETA model (Co-DETR approximation): {model_id}")

        except Exception as e:
            logger.warning(f"DETA not available ({e}), trying DETR fallback")
            try:
                from transformers import DetrForObjectDetection, DetrImageProcessor
                model_id = "facebook/detr-resnet-101"
                self._processor = DetrImageProcessor.from_pretrained(model_id)
                self._model = DetrForObjectDetection.from_pretrained(model_id).to(
                    self.device
                )
                self._model.eval()
                self._engine = "detr"
                logger.info(f"Loaded DETR model: {model_id}")
            except Exception as e2:
                logger.warning(f"DETR not available ({e2}), using YOLOv5 fallback")
                self._load_fallback()

    def _load_fallback(self):
        """Load YOLO or ultralytics as fallback."""
        try:
            from ultralytics import YOLO
            self._model = YOLO("yolov8x.pt")
            self._engine = "yolo"
            logger.info("Loaded YOLOv8x as fallback object detector")
        except ImportError:
            try:
                self._model = torch.hub.load("ultralytics/yolov5", "yolov5x")
                self._engine = "yolov5"
                logger.info("Loaded YOLOv5x as fallback object detector")
            except Exception:
                logger.error("No object detection model available!")
                self._model = None
                self._engine = None

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in a single image.

        Args:
            image: BGR numpy array (OpenCV format).

        Returns:
            List of dicts with keys: 'label', 'confidence', 'bbox' [x1, y1, x2, y2]
        """
        self._load_model()

        if self._model is None:
            return []

        if self._engine in ("deta", "detr"):
            return self._detect_detr(image)
        elif self._engine in ("yolo", "yolov5"):
            return self._detect_yolo(image)
        
        return []

    def _detect_detr(self, image: np.ndarray) -> List[Dict]:
        """Detect objects using DETR/DETA."""
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self._processor(images=pil_image, return_tensors="pt").to(self.device)
        outputs = self._model(**inputs)

        # Post-process
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
        results = self._processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.confidence_threshold
        )[0]

        detections = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            score = score.item()
            label_id = label.item()
            bbox = box.cpu().numpy().tolist()

            # Map label to class name
            label_name = self._model.config.id2label.get(label_id, f"class_{label_id}")

            detections.append({
                "label": label_name,
                "confidence": score,
                "bbox": bbox,
            })

        return detections

    def _detect_yolo(self, image: np.ndarray) -> List[Dict]:
        """Detect objects using YOLO."""
        if self._engine == "yolo":
            # ultralytics YOLOv8
            results = self._model(image, conf=self.confidence_threshold)
            detections = []
            for result in results:
                for box in result.boxes:
                    detections.append({
                        "label": result.names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].cpu().numpy().tolist(),
                    })
            return detections
        else:
            # YOLOv5
            results = self._model(image)
            detections = []
            for *xyxy, conf, cls in results.xyxy[0]:
                if conf >= self.confidence_threshold:
                    detections.append({
                        "label": results.names[int(cls)],
                        "confidence": float(conf),
                        "bbox": [float(x) for x in xyxy],
                    })
            return detections

    def detect_in_keyframes(
        self, keyframe_paths: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        Detect objects in multiple keyframe images.

        Args:
            keyframe_paths: List of paths to keyframe images.

        Returns:
            Dict mapping path to list of detections.
        """
        results = {}
        for path in keyframe_paths:
            image = cv2.imread(path)
            if image is None:
                logger.warning(f"Cannot read image: {path}")
                results[path] = []
                continue
            results[path] = self.detect(image)
        return results

    def get_objects_for_shot(self, shot) -> List[str]:
        """
        Get unique object labels detected in a shot's keyframes.

        Args:
            shot: Shot object with keyframe_paths.

        Returns:
            Sorted list of unique object labels.
        """
        detections = self.detect_in_keyframes(shot.keyframe_paths)
        labels = set()
        for path, dets in detections.items():
            for det in dets:
                labels.add(det["label"])
        return sorted(labels)

    def filter_by_objects(
        self,
        shots: list,
        required_objects: List[str],
        mode: str = "any",
    ) -> list:
        """
        Filter shots that contain (or don't contain) specific objects.

        Args:
            shots: List of Shot objects.
            required_objects: List of object labels to check.
            mode: "any" (at least one object present) or 
                  "all" (all objects present).

        Returns:
            Filtered list of shots.
        """
        required = set(obj.lower() for obj in required_objects)
        filtered = []

        for shot in shots:
            detected = set(obj.lower() for obj in self.get_objects_for_shot(shot))
            if mode == "any":
                if required & detected:  # Intersection is non-empty
                    filtered.append(shot)
            elif mode == "all":
                if required.issubset(detected):
                    filtered.append(shot)

        logger.info(
            f"Object filter: {len(shots)} -> {len(filtered)} shots "
            f"(required: {required_objects}, mode: {mode})"
        )
        return filtered
