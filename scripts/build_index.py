#!/usr/bin/env python
"""
VRAG Index Builder Script

Builds FAISS search indices from preprocessed video data (features + metadata).

Usage:
    python scripts/build_index.py --data_dir output/ --index_dir data/indices
    python scripts/build_index.py --data_dir output/ --index_type Flat
"""

import argparse
import glob
import json
import logging
import os
import sys
import time

import numpy as np

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="VRAG Index Builder")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing preprocessed video data (output/)"
    )
    parser.add_argument(
        "--index_dir", type=str, default="data/indices",
        help="Directory to save indices"
    )
    parser.add_argument(
        "--index_type", type=str, default="IVFFlat",
        choices=["Flat", "IVFFlat", "IVFPQ"],
        help="FAISS index type"
    )
    parser.add_argument(
        "--nlist", type=int, default=256,
        help="Number of clusters for IVF indices"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    if not os.path.isdir(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    from vrag.indexing.index_builder import IndexBuilder

    index_builder = IndexBuilder(
        index_dir=args.index_dir,
        index_type=args.index_type,
        nlist=args.nlist,
    )

    # Aggregate features from all preprocessed videos
    model_features = {}  # model_name -> list of feature arrays
    model_metadata = {}  # model_name -> list of metadata dicts

    video_dirs = sorted(glob.glob(os.path.join(args.data_dir, "*")))
    video_dirs = [d for d in video_dirs if os.path.isdir(d)]

    logger.info(f"Found {len(video_dirs)} processed video(s) in {args.data_dir}")

    for video_dir in video_dirs:
        video_id = os.path.basename(video_dir)
        feat_dir = os.path.join(video_dir, "features")

        if not os.path.isdir(feat_dir):
            logger.warning(f"No features directory for {video_id}, skipping")
            continue

        # Load per-model metadata
        meta_path = os.path.join(feat_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                per_model_meta = json.load(f)
        else:
            per_model_meta = {}

        # Load feature files
        for npy_file in glob.glob(os.path.join(feat_dir, "*.npy")):
            model_name = os.path.splitext(os.path.basename(npy_file))[0]
            features = np.load(npy_file)

            if model_name not in model_features:
                model_features[model_name] = []
                model_metadata[model_name] = []

            model_features[model_name].append(features)

            # Use per-model metadata if available, otherwise create basic metadata
            if model_name in per_model_meta:
                model_metadata[model_name].extend(per_model_meta[model_name])
            else:
                for i in range(len(features)):
                    model_metadata[model_name].append({
                        "video_id": video_id,
                        "shot_id": i,
                    })

        logger.info(f"Loaded features for {video_id}")

    # Build indices
    logger.info("Building FAISS indices...")
    start = time.time()

    for model_name in model_features:
        all_features = np.concatenate(model_features[model_name], axis=0)
        all_metadata = model_metadata[model_name]

        logger.info(
            f"Building index for '{model_name}': "
            f"{all_features.shape[0]} vectors, dim={all_features.shape[1]}"
        )

        index_builder.build_index(model_name, all_features, all_metadata)

    # Save indices
    index_builder.save()

    elapsed = time.time() - start
    stats = index_builder.get_stats()
    logger.info(f"\nIndex building complete ({elapsed:.1f}s):")
    for model_name, s in stats.items():
        logger.info(
            f"  {model_name}: {s['num_vectors']} vectors, "
            f"dim={s['dimension']}"
        )


if __name__ == "__main__":
    main()
