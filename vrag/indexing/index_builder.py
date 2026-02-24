"""
VRAG Index Builder Module

Builds and manages FAISS-based vector indices for efficient retrieval.
From the paper: Precomputed feature vectors are indexed for efficient 
nearest-neighbor search during retrieval.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Builds and manages FAISS vector indices for semantic search.
    
    Supports:
    - Multiple model indices (CLIP, BLIP-2, BEiT-3, InternVL)
    - IVF (Inverted File) and Flat index types
    - Metadata storage for shot mapping
    - Incremental updates
    """

    def __init__(
        self,
        index_dir: str = "data/indices",
        index_type: str = "IVFFlat",
        nlist: int = 100,
    ):
        """
        Args:
            index_dir: Directory to store/load indices.
            index_type: FAISS index type ("Flat", "IVFFlat", "IVFPQ").
            nlist: Number of clusters for IVF indices.
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_type = index_type
        self.nlist = nlist

        # Indices and metadata per model
        self._indices: Dict[str, object] = {}
        self._metadata: Dict[str, List[Dict]] = {}  # model -> list of {video_id, shot_id, ...}
        self._feature_extractors = {}

    def build_index(
        self,
        model_name: str,
        features: np.ndarray,
        metadata: List[Dict],
    ):
        """
        Build a FAISS index for a specific model.

        Args:
            model_name: Name of the model (e.g., "clip", "blip2").
            features: Feature matrix of shape (N, D).
            metadata: List of dicts, one per feature vector, with 
                      'video_id', 'shot_id', etc.
        """
        import faiss

        if len(features) == 0:
            logger.warning(f"No features to index for {model_name}")
            return

        features = features.astype("float32")
        dim = features.shape[1]

        # L2 normalize for cosine similarity via inner product
        faiss.normalize_L2(features)

        if self.index_type == "Flat":
            index = faiss.IndexFlatIP(dim)
        elif self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatIP(dim)
            nlist = min(self.nlist, len(features))
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            index.train(features)
            index.nprobe = min(10, nlist)
        elif self.index_type == "IVFPQ":
            quantizer = faiss.IndexFlatIP(dim)
            nlist = min(self.nlist, len(features))
            m = min(8, dim // 4)  # Number of sub-quantizers
            index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
            index.train(features)
            index.nprobe = min(10, nlist)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        index.add(features)

        self._indices[model_name] = index
        self._metadata[model_name] = metadata

        logger.info(
            f"Built {self.index_type} index for '{model_name}': "
            f"{features.shape[0]} vectors, dim={dim}"
        )

    def add_to_index(
        self,
        model_name: str,
        features: np.ndarray,
        metadata: List[Dict],
    ):
        """
        Add vectors to an existing index (incremental update).

        Args:
            model_name: Model name.
            features: New feature vectors.
            metadata: Metadata for new vectors.
        """
        import faiss

        if model_name not in self._indices:
            self.build_index(model_name, features, metadata)
            return

        features = features.astype("float32")
        faiss.normalize_L2(features)

        self._indices[model_name].add(features)
        self._metadata[model_name].extend(metadata)

        logger.info(
            f"Added {len(features)} vectors to '{model_name}' index"
        )

    def search(
        self,
        model_name: str,
        query_text: str,
        top_k: int = 100,
    ) -> List[Dict]:
        """
        Search an index using a text query.

        Args:
            model_name: Which model's index to search.
            query_text: Text query.
            top_k: Number of results.

        Returns:
            List of result dicts with score and metadata.
        """
        import faiss

        if model_name not in self._indices:
            logger.warning(f"No index for model: {model_name}")
            return []

        # Get text features using the appropriate model
        query_features = self._get_text_features(model_name, query_text)
        if query_features is None:
            return []

        query_features = query_features.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_features)

        index = self._indices[model_name]
        metadata = self._metadata[model_name]

        k = min(top_k, index.ntotal)
        scores, indices = index.search(query_features, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(metadata):
                result = metadata[idx].copy()
                result["score"] = float(score)
                results.append(result)

        return results

    def search_by_vector(
        self,
        model_name: str,
        query_vector: np.ndarray,
        top_k: int = 100,
    ) -> List[Dict]:
        """
        Search using a pre-computed feature vector.

        Args:
            model_name: Which model's index to search.
            query_vector: Query feature vector.
            top_k: Number of results.

        Returns:
            List of result dicts with score and metadata.
        """
        import faiss

        if model_name not in self._indices:
            return []

        query_vector = query_vector.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query_vector)

        index = self._indices[model_name]
        metadata = self._metadata[model_name]

        k = min(top_k, index.ntotal)
        scores, indices = index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(metadata):
                result = metadata[idx].copy()
                result["score"] = float(score)
                results.append(result)

        return results

    def search_by_image(
        self,
        model_name: str,
        image,
        top_k: int = 100,
    ) -> List[Dict]:
        """
        Search using an image query.

        Args:
            model_name: Which model's index.
            image: PIL Image.
            top_k: Number of results.

        Returns:
            List of result dicts.
        """
        image_features = self._get_image_features(model_name, image)
        if image_features is None:
            return []

        return self.search_by_vector(model_name, image_features, top_k)

    def has_index(self, model_name: str) -> bool:
        """Check if an index exists for a model."""
        return model_name in self._indices

    def save(self, prefix: str = ""):
        """
        Save all indices and metadata to disk.

        Args:
            prefix: Optional prefix for filenames.
        """
        import faiss

        for model_name, index in self._indices.items():
            index_path = self.index_dir / f"{prefix}{model_name}.index"
            meta_path = self.index_dir / f"{prefix}{model_name}_meta.json"

            faiss.write_index(index, str(index_path))

            with open(meta_path, "w") as f:
                json.dump(self._metadata[model_name], f)

            logger.info(f"Saved index: {index_path} ({index.ntotal} vectors)")

    def load(self, prefix: str = "", model_names: List[str] = None):
        """
        Load indices and metadata from disk.

        Args:
            prefix: Optional prefix for filenames.
            model_names: Specific models to load. If None, loads all found.
        """
        import faiss

        if model_names is None:
            # Find all index files
            model_names = []
            for path in self.index_dir.glob(f"{prefix}*.index"):
                name = path.stem.replace(prefix, "")
                model_names.append(name)

        for model_name in model_names:
            index_path = self.index_dir / f"{prefix}{model_name}.index"
            meta_path = self.index_dir / f"{prefix}{model_name}_meta.json"

            if not index_path.exists():
                logger.warning(f"Index not found: {index_path}")
                continue

            self._indices[model_name] = faiss.read_index(str(index_path))

            if meta_path.exists():
                with open(meta_path, "r") as f:
                    self._metadata[model_name] = json.load(f)
            else:
                self._metadata[model_name] = []

            logger.info(
                f"Loaded index: {model_name} "
                f"({self._indices[model_name].ntotal} vectors)"
            )

    def _get_text_features(
        self, model_name: str, text: str
    ) -> Optional[np.ndarray]:
        """Extract text features using the appropriate model."""
        if model_name not in self._feature_extractors:
            self._init_feature_extractor(model_name)

        extractor = self._feature_extractors.get(model_name)
        if extractor is None:
            return None

        try:
            features = extractor(text)
            return features
        except Exception as e:
            logger.warning(f"Text feature extraction failed for {model_name}: {e}")
            return None

    def _get_image_features(
        self, model_name: str, image
    ) -> Optional[np.ndarray]:
        """Extract image features using the appropriate model."""
        # For image search, use CLIP
        if "clip" not in self._feature_extractors:
            self._init_feature_extractor("clip")

        extractor = self._feature_extractors.get("clip")
        if extractor is None:
            return None

        try:
            features = extractor(image, is_image=True)
            return features
        except Exception as e:
            logger.warning(f"Image feature extraction failed: {e}")
            return None

    def _init_feature_extractor(self, model_name: str):
        """Initialize a feature extractor for a model."""
        try:
            if "clip" in model_name.lower():
                self._feature_extractors[model_name] = self._make_clip_extractor()
            elif "blip" in model_name.lower():
                self._feature_extractors[model_name] = self._make_blip_extractor()
            elif "beit" in model_name.lower():
                self._feature_extractors[model_name] = self._make_beit_extractor()
            elif "internvl" in model_name.lower():
                # InternVL uses CLIP-style features
                self._feature_extractors[model_name] = self._make_clip_extractor()
            else:
                logger.warning(f"No extractor for model: {model_name}")
                self._feature_extractors[model_name] = None
        except Exception as e:
            logger.warning(f"Failed to init extractor for {model_name}: {e}")
            self._feature_extractors[model_name] = None

    def _make_clip_extractor(self):
        """Create CLIP text/image feature extractor."""
        import torch
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        model.eval()

        def extract(input_data, is_image=False):
            with torch.no_grad():
                if is_image:
                    img_tensor = preprocess(input_data).unsqueeze(0)
                    features = model.encode_image(img_tensor)
                else:
                    tokens = tokenizer([input_data])
                    features = model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                return features.cpu().numpy().flatten()

        return extract

    def _make_blip_extractor(self):
        """Create BLIP-2 text feature extractor."""
        import torch
        from transformers import Blip2Model, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            "Salesforce/blip2-opt-2.7b"
        )
        model = Blip2Model.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

        def extract(text, is_image=False):
            with torch.no_grad():
                inputs = processor(text=text, return_tensors="pt").to(
                    model.device
                )
                text_features = model.get_text_features(**inputs)
                features = text_features.mean(dim=1)
                features = features / features.norm(dim=-1, keepdim=True)
                return features.cpu().numpy().flatten()

        return extract

    def _make_beit_extractor(self):
        """Create BEiT-3 text feature extractor (falls back to CLIP)."""
        return self._make_clip_extractor()

    def get_stats(self) -> Dict:
        """Get statistics about loaded indices."""
        stats = {}
        for model_name, index in self._indices.items():
            stats[model_name] = {
                "num_vectors": index.ntotal,
                "dimension": index.d,
                "num_metadata": len(self._metadata.get(model_name, [])),
            }
        return stats
