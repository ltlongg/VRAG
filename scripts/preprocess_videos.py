#!/usr/bin/env python
"""
VRAG Video Preprocessing Script

Preprocesses video files for the VRAG system:
1. Shot boundary detection
2. Keyframe extraction
3. Feature extraction (CLIP, BLIP-2, BEiT-3, InternVL)
4. OCR text extraction
5. Audio transcription (Whisper)
6. Object detection (Co-DETR / DETA)

Usage:
    # Single video
    python scripts/preprocess_videos.py --video path/to/video.mp4

    # Directory of videos
    python scripts/preprocess_videos.py --video_dir data/videos --output_dir output/

    # Skip already processed videos
    python scripts/preprocess_videos.py --video_dir data/videos --skip_existing
"""

import argparse
import glob
import logging
import os
import sys
import time

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging(level: str = "INFO"):
    """Set up logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def find_videos(video_dir: str) -> list:
    """Find all video files in a directory."""
    extensions = ["mp4", "avi", "mkv", "mov", "webm", "flv"]
    videos = []
    for ext in extensions:
        videos.extend(glob.glob(os.path.join(video_dir, f"*.{ext}")))
    return sorted(set(videos))


def main():
    parser = argparse.ArgumentParser(
        description="VRAG Video Preprocessing"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to a single video file"
    )
    parser.add_argument(
        "--video_dir", type=str, default=None,
        help="Directory containing video files"
    )
    parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Output directory for preprocessed data (default: output)"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip videos that have already been processed"
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    if not args.video and not args.video_dir:
        parser.error("Provide --video or --video_dir")

    # Collect video paths
    video_paths = []
    if args.video:
        if not os.path.isfile(args.video):
            logger.error(f"Video file not found: {args.video}")
            sys.exit(1)
        video_paths.append(args.video)
    if args.video_dir:
        if not os.path.isdir(args.video_dir):
            logger.error(f"Video directory not found: {args.video_dir}")
            sys.exit(1)
        video_paths.extend(find_videos(args.video_dir))

    if not video_paths:
        logger.error("No video files found")
        sys.exit(1)

    logger.info(f"Found {len(video_paths)} video(s) to process")

    # Initialize pipeline
    from vrag.pipeline import VRAGPipeline

    pipeline = VRAGPipeline(config_path=args.config)

    # Process each video
    total_start = time.time()
    processed = 0
    failed = 0

    for i, video_path in enumerate(video_paths, 1):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(args.output_dir, video_name)

        # Check if already processed
        if args.skip_existing and os.path.exists(
            os.path.join(video_output_dir, "shots.json")
        ):
            logger.info(
                f"[{i}/{len(video_paths)}] Skipping {video_name} "
                "(already processed)"
            )
            continue

        logger.info(
            f"[{i}/{len(video_paths)}] Processing: {video_name}"
        )
        start = time.time()

        try:
            result = pipeline.preprocess_video(
                video_path=video_path,
                output_dir=video_output_dir,
            )
            elapsed = time.time() - start
            n_shots = len(result.get("shots", []))
            logger.info(
                f"[{i}/{len(video_paths)}] Done: {video_name} "
                f"({n_shots} shots, {elapsed:.1f}s)"
            )
            processed += 1
        except Exception as e:
            logger.error(
                f"[{i}/{len(video_paths)}] Failed: {video_name}: {e}"
            )
            failed += 1

    total_elapsed = time.time() - total_start
    logger.info(
        f"\nPreprocessing complete: {processed} processed, "
        f"{failed} failed, {total_elapsed:.1f}s total"
    )


if __name__ == "__main__":
    main()
