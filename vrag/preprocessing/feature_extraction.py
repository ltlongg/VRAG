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
        self.device = device if torch.cuda.is_available() else "cpu"
        self._models = {}
        self._processors = {}
        self._tokenizers = {}

    # =========================================================================
    # CLIP Feature Extraction
    # =========================================================================

    def load_clip(
        self,
        model_name: str = "ViT-B-32",
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
        return features.cpu().float().numpy()

    @torch.no_grad()
    def extract_clip_text_features(self, texts: List[str]) -> np.ndarray:
        """Extract CLIP text features. Returns (N, D) normalized array."""
        self.load_clip()
        tokenizer = self._tokenizers["clip"]
        model = self._models["clip"]

        tokens = tokenizer(texts).to(self.device)
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().float().numpy()

    # =========================================================================
    # BLIP-2 Feature Extraction
    # =========================================================================

    def load_blip2(self, model_name: str = "Salesforce/blip2-opt-2.7b"):
        """Load BLIP-2 model."""
        if "blip2" in self._models:
            return

        try:
            from transformers import Blip2Processor, Blip2Model
            processor = Blip2Processor.from_pretrained(model_name)
            model = Blip2Model.from_pretrained(
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
        Extract BLIP-2 image features via ITC projection (256-dim).

        Uses Blip2Model.get_image_features() which returns ITC-projected
        vectors in the same embedding space as get_text_features().
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
            # get_image_features → Q-Former → ITC projection → (B, num_query, 256)
            image_features = model.get_image_features(
                pixel_values=inputs.pixel_values
            )
            # Mean-pool over query tokens to get a single vector
            features = image_features.mean(dim=1)  # (B, 256)
            features = features.float()
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    @torch.no_grad()
    def extract_blip2_text_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract BLIP-2 text features via ITC projection (256-dim).

        Uses Blip2Model.get_text_features() to produce vectors in the same
        embedding space as the image features from extract_blip2_features().
        """
        self.load_blip2()
        processor = self._processors["blip2"]
        model = self._models["blip2"]

        inputs = processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            # get_text_features → Q-Former text path → ITC → (batch, 256)
            features = model.get_text_features(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )
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
            all_features.append(features.cpu().float().numpy())

        return np.concatenate(all_features, axis=0)

    # =========================================================================
    # InternVL Feature Extraction
    # =========================================================================

    def load_internvl(self, model_name: str = "OpenGVLab/InternVL2-1B"):
        """
        Load InternVL2 model for vision-language feature extraction.
        Uses the vision encoder component to extract image features.
        """
        if "internvl" in self._models:
            return

        try:
            from transformers import AutoModel, AutoTokenizer, AutoProcessor
            logger.info(f"Loading InternVL model: {model_name}")
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(self.device)
            model.eval()

            # InternVL2 uses its own image processing
            try:
                processor = AutoProcessor.from_pretrained(
                    model_name, trust_remote_code=True
                )
            except Exception:
                from transformers import CLIPImageProcessor
                processor = CLIPImageProcessor.from_pretrained(model_name)

            tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self._models["internvl"] = model
            self._processors["internvl"] = processor
            self._tokenizers["internvl"] = tokenizer
            logger.info(f"Loaded InternVL model: {model_name}")
        except ImportError:
            raise ImportError("Install transformers: pip install transformers")

    @torch.no_grad()
    def extract_internvl_features(self, images: List[Image.Image]) -> np.ndarray:
        """Extract InternVL image features using the vision encoder."""
        self.load_internvl()

        processor = self._processors["internvl"]
        model = self._models["internvl"]

        all_features = []
        batch_size = 4

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            try:
                # Try processor with images
                if hasattr(processor, 'image_processor'):
                    inputs = processor.image_processor(
                        images=batch, return_tensors="pt"
                    ).to(self.device, torch.bfloat16)
                else:
                    inputs = processor(
                        images=batch, return_tensors="pt"
                    ).to(self.device, torch.bfloat16)
            except Exception:
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((448, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
                pixel_values = torch.stack([transform(img) for img in batch])
                inputs = {"pixel_values": pixel_values.to(
                    self.device, torch.bfloat16
                )}

            # Extract vision features
            pixel_values = inputs.get("pixel_values", inputs.pixel_values) \
                if not isinstance(inputs, dict) else inputs["pixel_values"]

            if hasattr(model, 'vision_model'):
                outputs = model.vision_model(pixel_values=pixel_values)
                features = outputs.last_hidden_state[:, 0, :]
            elif hasattr(model, 'encode_image'):
                features = model.encode_image(pixel_values)
            else:
                # Fallback: use forward with dummy text
                features = pixel_values.mean(dim=[2, 3])  # Global average pooling

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
