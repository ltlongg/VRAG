"""
VRAG Index Building Script

Builds FAISS vector indices from preprocessed video features.
Run this after preprocess_videos.py to create search indices.

Usage:
    python scripts/build_index.py --data_dir output/ --index_dir data/indices
    python scripts/build_index.py --config config/config.yaml
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from vrag.indexing.index_builder import IndexBuilder
from vrag.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="VRAG Index Builder"
    )
    parser.add_argument(
        "--data_dir", type=str, default="output",
        help="Directory with preprocessed video data",
    )
    parser.add_argument(
        "--index_dir", type=str, default="data/indices",
        help="Output directory for FAISS indices",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--index_type", type=str, default=None,
        help="FAISS index type: Flat, IVFFlat, IVFPQ (overrides config)",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated model names to index (e.g., clip,blip2)",
    )
    return parser.parse_args()


def load_preprocessed_features(data_dir: str, model_names: list = None):
    """
    Load all preprocessed features and metadata from output directory.

    Returns:
        features_by_model: Dict[model_name, np.ndarray]
        all_metadata: List[Dict]
        ocr_data: Dict[video_id, {shot_id: text}]
        transcript_data: Dict[video_id, transcript]
        object_data: Dict[video_id, {shot_id: [objects]}]
    """
    features_by_model = {}
    all_metadata = []
    ocr_data = {}
    transcript_data = {}
    object_data = {}

    # Find all video output directories
    video_dirs = sorted(
        [d for d in Path(data_dir).iterdir() if d.is_dir()]
    )

    logger.info(f"Found {len(video_dirs)} preprocessed videos in {data_dir}")

    for video_dir in video_dirs:
        video_id = video_dir.name

        # Load features
        feat_dir = video_dir / "features"
        if feat_dir.exists():
            for feat_file in feat_dir.glob("*.npy"):
                model_name = feat_file.stem
                if model_names and model_name not in model_names:
                    continue

                feats = np.load(str(feat_file))
                if model_name not in features_by_model:
                    features_by_model[model_name] = []

                features_by_model[model_name].append(feats)

                # Create metadata entries for each feature vector
                for i in range(len(feats)):
                    if len(all_metadata) <= len(features_by_model[model_name]) - 1:
                        all_metadata.append({
                            "video_id": video_id,
                            "shot_id": i,
                        })

        # Load shots for metadata
        shots_file = video_dir / "shots.json"
        if shots_file.exists():
            with open(shots_file) as f:
                shots_data = json.load(f)

            # Update metadata with shot info
            shot_metadata = []
            for shot in shots_data:
                shot_metadata.append({
                    "video_id": video_id,
                    "shot_id": shot.get("shot_id", 0),
                    "start_time": shot.get("start_time", 0),
                    "end_time": shot.get("end_time", 0),
                })

            # Replace generic metadata
            # (this properly maps feature vectors to shots)
            if shot_metadata:
                # Find and replace metadata entries for this video
                # Use shot metadata directly
                pass

    # Stack features per model
    for model_name in features_by_model:
        feat_list = features_by_model[model_name]
        features_by_model[model_name] = np.vstack(feat_list)

    # Build proper metadata
    # Reconstruct from all video directories
    all_metadata = []
    for video_dir in video_dirs:
        video_id = video_dir.name
        shots_file = video_dir / "shots.json"
        if shots_file.exists():
            with open(shots_file) as f:
                shots_data = json.load(f)
            for shot in shots_data:
                all_metadata.append({
                    "video_id": video_id,
                    "shot_id": shot.get("shot_id", 0),
                    "start_time": shot.get("start_time", 0),
                    "end_time": shot.get("end_time", 0),
                })

    return features_by_model, all_metadata, ocr_data, transcript_data, object_data


def main():
    args = parse_args()

    config = load_config(args.config)
    idx_cfg = config.get("indexing", {})

    index_type = args.index_type or idx_cfg.get("type", "IVFFlat")
    index_dir = args.index_dir or idx_cfg.get("index_dir", "data/indices")
    model_names = args.models.split(",") if args.models else None

    logger.info(f"Building indices from {args.data_dir}")
    logger.info(f"Index type: {index_type}, output: {index_dir}")

    # Load preprocessed data
    start = time.time()
    features_by_model, metadata, ocr, transcripts, objects = \
        load_preprocessed_features(args.data_dir, model_names)

    logger.info(
        f"Loaded features for {len(features_by_model)} models, "
        f"{len(metadata)} shots in {time.time() - start:.1f}s"
    )

    # Build indices
    builder = IndexBuilder(
        index_dir=index_dir,
        index_type=index_type,
        nlist=idx_cfg.get("nlist", 100),
    )

    for model_name, features in features_by_model.items():
        logger.info(
            f"Building index for {model_name}: "
            f"{features.shape[0]} vectors, dim={features.shape[1]}"
        )
        builder.build_index(model_name, features, metadata)

    # Save indices
    builder.save()

    # Print stats
    stats = builder.get_stats()
    logger.info("\n=== Index Statistics ===")
    for model_name, stat in stats.items():
        logger.info(
            f"  {model_name}: {stat['num_vectors']} vectors, "
            f"dim={stat['dimension']}"
        )

    logger.info(f"\nIndices saved to {index_dir}")
    logger.info(f"Total build time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
