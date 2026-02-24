"""
VRAG Video Preprocessing Script

Preprocesses videos for the VRAG system:
1. Shot boundary detection
2. Keyframe extraction  
3. Feature extraction (CLIP, BLIP-2, BEiT-3, InternVL)
4. OCR extraction
5. Audio transcription (Whisper)
6. Object detection (Co-DETR)

Usage:
    python scripts/preprocess_videos.py --video_dir data/videos --output_dir output/
    python scripts/preprocess_videos.py --video path/to/video.mp4
    python scripts/preprocess_videos.py --config config/config.yaml
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vrag.pipeline import VRAGPipeline
from vrag.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="VRAG Video Preprocessing Pipeline"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path to a single video file to preprocess",
    )
    parser.add_argument(
        "--video_dir", type=str, default=None,
        help="Directory containing video files to preprocess",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Output directory for preprocessed data",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--extensions", type=str, default="mp4,avi,mkv,mov,webm",
        help="Comma-separated video file extensions",
    )
    parser.add_argument(
        "--max_videos", type=int, default=None,
        help="Maximum number of videos to process",
    )
    parser.add_argument(
        "--skip_existing", action="store_true",
        help="Skip videos that already have preprocessed output",
    )
    return parser.parse_args()


def find_videos(video_dir: str, extensions: str) -> list:
    """Find all video files in a directory."""
    exts = extensions.split(",")
    videos = []
    for ext in exts:
        videos.extend(glob.glob(os.path.join(video_dir, f"*.{ext.strip()}")))
        videos.extend(glob.glob(os.path.join(video_dir, f"**/*.{ext.strip()}"), recursive=True))
    return sorted(set(videos))


def main():
    args = parse_args()

    # Collect video paths
    video_paths = []
    if args.video:
        if os.path.isfile(args.video):
            video_paths = [args.video]
        else:
            logger.error(f"Video file not found: {args.video}")
            sys.exit(1)
    elif args.video_dir:
        video_paths = find_videos(args.video_dir, args.extensions)
        logger.info(f"Found {len(video_paths)} videos in {args.video_dir}")
    else:
        # Try default from config
        config = load_config(args.config)
        default_dir = config.get("dataset", {}).get("video_dir", "data/videos")
        if os.path.isdir(default_dir):
            video_paths = find_videos(default_dir, args.extensions)
            logger.info(f"Found {len(video_paths)} videos in {default_dir}")
        else:
            logger.error("No video source specified. Use --video or --video_dir")
            sys.exit(1)

    if args.max_videos:
        video_paths = video_paths[:args.max_videos]

    if not video_paths:
        logger.error("No videos found!")
        sys.exit(1)

    # Initialize pipeline
    pipeline = VRAGPipeline(config_path=args.config)

    # Process videos
    all_results = []
    total_start = time.time()

    for i, video_path in enumerate(video_paths):
        video_id = Path(video_path).stem
        output_dir = os.path.join(args.output_dir, video_id)

        # Skip if already processed
        if args.skip_existing and os.path.exists(
            os.path.join(output_dir, "shots.json")
        ):
            logger.info(f"[{i+1}/{len(video_paths)}] Skipping {video_id} (exists)")
            continue

        logger.info(
            f"[{i+1}/{len(video_paths)}] Processing: {video_id}"
        )
        start = time.time()

        try:
            result = pipeline.preprocess_video(
                video_path=video_path,
                output_dir=output_dir,
            )
            all_results.append(result)
            elapsed = time.time() - start
            logger.info(
                f"[{i+1}/{len(video_paths)}] {video_id} completed in {elapsed:.1f}s"
            )
        except Exception as e:
            logger.error(f"[{i+1}/{len(video_paths)}] {video_id} failed: {e}")
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_start
    logger.info(
        f"\nPreprocessing complete!\n"
        f"  Videos processed: {len(all_results)}/{len(video_paths)}\n"
        f"  Total time: {total_elapsed:.1f}s\n"
        f"  Output directory: {args.output_dir}"
    )

    # Save summary
    summary = {
        "num_videos": len(all_results),
        "total_time": total_elapsed,
        "videos": [
            {
                "video_id": r["video_id"],
                "num_shots": len(r.get("shots", [])),
                "num_keyframes": sum(
                    len(s.keyframe_paths) for s in r.get("shots", [])
                ),
            }
            for r in all_results
        ],
    }
    summary_path = os.path.join(args.output_dir, "preprocessing_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
