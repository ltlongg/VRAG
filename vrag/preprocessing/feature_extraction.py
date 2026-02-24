"""
VRAG Feature Extraction Module

Extracts multi-modal features from keyframes using multiple vision-language models.
From the paper (Section 3.1): "Our multimodal retrieval system employs a late-fusion 
approach at the shot level, integrating results from InternVL-G, BLIP-2, BEiT-3, 
and CLIP."
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Multi-model feature extraction for keyframes.
    
    Supports: CLIP, BLIP-2, BEiT-3, InternVL
    Each model provides complementary semantic representations for late fusion.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._models = {}
        self._processors = {}
        self._tokenizers = {}

    # =========================================================================
    # CLIP Feature Extraction
    # =========================================================================

    def load_clip(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
    ):
        """Load OpenCLIP model."""
        if "clip" in self._models:
            return

        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            tokenizer = open_clip.get_tokenizer(model_name)
            self._models["clip"] = model.to(self.device).eval()
            self._processors["clip"] = preprocess
            self._tokenizers["clip"] = tokenizer
            logger.info(f"Loaded CLIP model: {model_name} ({pretrained})")
        except ImportError:
            raise ImportError("Install open_clip: pip install open_clip_torch")

    @torch.no_grad()
    def extract_clip_image_features(self, images: List[Image.Image]) -> np.ndarray:
        """Extract CLIP image features. Returns (N, D) normalized array."""
        self.load_clip()
        preprocess = self._processors["clip"]
        model = self._models["clip"]

        tensors = torch.stack([preprocess(img) for img in images]).to(self.device)
        features = model.encode_image(tensors)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    @torch.no_grad()
    def extract_clip_text_features(self, texts: List[str]) -> np.ndarray:
        """Extract CLIP text features. Returns (N, D) normalized array."""
        self.load_clip()
        tokenizer = self._tokenizers["clip"]
        model = self._models["clip"]

        tokens = tokenizer(texts).to(self.device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    # =========================================================================
    # BLIP-2 Feature Extraction
    # =========================================================================

    def load_blip2(self, model_name: str = "Salesforce/blip2-opt-2.7b"):
        """Load BLIP-2 model."""
        if "blip2" in self._models:
            return

        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            processor = Blip2Processor.from_pretrained(model_name)
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).to(self.device)
            model.eval()
            self._models["blip2"] = model
            self._processors["blip2"] = processor
            logger.info(f"Loaded BLIP-2 model: {model_name}")
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

    @torch.no_grad()
    def extract_blip2_features(self, images: List[Image.Image]) -> np.ndarray:
        """
        Extract BLIP-2 image-text matching features.
        Uses the image encoder to get visual features.
        """
        self.load_blip2()
        processor = self._processors["blip2"]
        model = self._models["blip2"]

        all_features = []
        batch_size = 8

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(
                self.device, torch.float16
            )
            # Extract image features from the vision model
            image_outputs = model.vision_model(pixel_values=inputs.pixel_values)
            # Use CLS token as the image feature
            features = image_outputs.last_hidden_state[:, 0, :]
            features = features.float()
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    @torch.no_grad()
    def extract_blip2_text_features(self, texts: List[str]) -> np.ndarray:
        """Extract BLIP-2 text features using the text encoder."""
        self.load_blip2()
        processor = self._processors["blip2"]
        model = self._models["blip2"]

        inputs = processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        # Use the language model to get text features
        with torch.no_grad():
            text_outputs = model.language_model.model.embed_tokens(inputs.input_ids)
            features = text_outputs.mean(dim=1)
            features = features.float()
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    # =========================================================================
    # BEiT-3 Feature Extraction
    # =========================================================================

    def load_beit3(self, model_name: str = "microsoft/beit-base-patch16-224"):
        """Load BEiT-3 model."""
        if "beit3" in self._models:
            return

        try:
            from transformers import BeitModel, BeitFeatureExtractor
            processor = BeitFeatureExtractor.from_pretrained(model_name)
            model = BeitModel.from_pretrained(model_name).to(self.device)
            model.eval()
            self._models["beit3"] = model
            self._processors["beit3"] = processor
            logger.info(f"Loaded BEiT model: {model_name}")
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

    @torch.no_grad()
    def extract_beit3_features(self, images: List[Image.Image]) -> np.ndarray:
        """Extract BEiT-3 image features. Returns (N, D) normalized array."""
        self.load_beit3()
        processor = self._processors["beit3"]
        model = self._models["beit3"]

        all_features = []
        batch_size = 16

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(self.device)
            outputs = model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    # =========================================================================
    # InternVL Feature Extraction
    # =========================================================================

    def load_internvl(self, model_name: str = "OpenGVLab/InternVL-14B-224px"):
        """
        Load InternVL model for vision-language feature extraction.
        Falls back to CLIP if InternVL is not available.
        """
        if "internvl" in self._models:
            return

        try:
            from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to(self.device)
            model.eval()
            processor = CLIPImageProcessor.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._models["internvl"] = model
            self._processors["internvl"] = processor
            self._tokenizers["internvl"] = tokenizer
            logger.info(f"Loaded InternVL model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load InternVL ({e}), will use CLIP as fallback")
            self.load_clip()
            self._models["internvl"] = self._models["clip"]
            self._processors["internvl"] = self._processors["clip"]

    @torch.no_grad()
    def extract_internvl_features(self, images: List[Image.Image]) -> np.ndarray:
        """Extract InternVL image features."""
        self.load_internvl()

        # If using CLIP as fallback
        if "clip" in self._models and self._models.get("internvl") is self._models.get("clip"):
            return self.extract_clip_image_features(images)

        processor = self._processors["internvl"]
        model = self._models["internvl"]

        all_features = []
        batch_size = 4

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(
                self.device, torch.float16
            )
            # Use the vision model component
            if hasattr(model, 'vision_model'):
                outputs = model.vision_model(pixel_values=inputs.pixel_values)
                features = outputs.last_hidden_state[:, 0, :]
            else:
                outputs = model.encode_image(inputs.pixel_values)
                features = outputs
            features = features.float()
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    # =========================================================================
    # Unified Feature Extraction Interface
    # =========================================================================

    def extract_features(
        self,
        images: List[Image.Image],
        model_names: List[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from multiple models.

        Args:
            images: List of PIL images.
            model_names: Which models to use (default: all available).

        Returns:
            Dictionary mapping model name to feature array (N, D).
        """
        if model_names is None:
            model_names = ["clip", "blip2", "beit3", "internvl"]

        results = {}
        for name in model_names:
            try:
                if name == "clip":
                    results[name] = self.extract_clip_image_features(images)
                elif name == "blip2":
                    results[name] = self.extract_blip2_features(images)
                elif name == "beit3":
                    results[name] = self.extract_beit3_features(images)
                elif name == "internvl":
                    results[name] = self.extract_internvl_features(images)
                else:
                    logger.warning(f"Unknown model: {name}")
            except Exception as e:
                logger.error(f"Failed to extract features with {name}: {e}")

        return results

    def extract_text_features(
        self,
        texts: List[str],
        model_names: List[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Extract text features from multiple models.

        Args:
            texts: List of text strings.
            model_names: Which models to use.

        Returns:
            Dictionary mapping model name to feature array (N, D).
        """
        if model_names is None:
            model_names = ["clip"]

        results = {}
        for name in model_names:
            try:
                if name == "clip":
                    results[name] = self.extract_clip_text_features(texts)
                elif name == "blip2":
                    results[name] = self.extract_blip2_text_features(texts)
                else:
                    logger.warning(f"Text feature extraction not available for: {name}")
            except Exception as e:
                logger.error(f"Failed to extract text features with {name}: {e}")

        return results
