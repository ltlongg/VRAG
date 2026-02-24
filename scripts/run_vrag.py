"""
VRAG Main Runner Script

Run the full VRAG pipeline for KIS (Known-Item Search) or VQA tasks.

Usage:
    # KIS task
    python scripts/run_vrag.py --task kis --query "a person riding a bicycle"
    
    # VQA task
    python scripts/run_vrag.py --task vqa --query "What color is the car?"
    
    # Interactive mode
    python scripts/run_vrag.py --interactive
    
    # Batch mode from file
    python scripts/run_vrag.py --queries_file queries.json
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from vrag.pipeline import VRAGPipeline
from vrag.utils.config import load_config
from vrag.utils.video_utils import load_shots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="VRAG Pipeline Runner")
    parser.add_argument(
        "--task", type=str, default="kis", choices=["kis", "vqa"],
        help="Task type: kis (Known-Item Search) or vqa (Video QA)",
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Query text",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Video file path (for VQA on a specific video)",
    )
    parser.add_argument(
        "--video_dir", type=str, default=None,
        help="Directory containing video files",
    )
    parser.add_argument(
        "--data_dir", type=str, default="output",
        help="Preprocessed data directory",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Configuration file",
    )
    parser.add_argument(
        "--top_n", type=int, default=100,
        help="Number of retrieval candidates",
    )
    parser.add_argument(
        "--top_k", type=int, default=10,
        help="Top-K after re-ranking",
    )
    parser.add_argument(
        "--queries_file", type=str, default=None,
        help="JSON file with batch queries",
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Output file for results",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--modules", type=str, default="retrieval,reranking,vqa,indexing",
        help="Comma-separated pipeline modules to initialize",
    )
    return parser.parse_args()


def load_all_shots(data_dir: str) -> dict:
    """Load all preprocessed shots data."""
    shots_data = {}
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        return shots_data

    for video_dir in data_path.iterdir():
        if video_dir.is_dir():
            shots_file = video_dir / "shots.json"
            if shots_file.exists():
                shots = load_shots(str(shots_file))
                shots_data[video_dir.name] = shots

    logger.info(f"Loaded shots for {len(shots_data)} videos")
    return shots_data


def run_kis(pipeline, query, shots_data, args):
    """Run KIS task."""
    results = pipeline.run_kis(
        query=query,
        top_n=args.top_n,
        top_k=args.top_k,
        shots_data=shots_data,
        video_dir=args.video_dir,
    )

    print(f"\n{'='*60}")
    print(f"KIS Results for: '{query}'")
    print(f"{'='*60}")

    for i, result in enumerate(results):
        score = result.get("relevance_score", result.get("score", 0))
        print(
            f"  {i+1}. Video: {result.get('video_id', 'N/A')}, "
            f"Shot: {result.get('shot_id', 'N/A')}, "
            f"Score: {score:.4f}"
        )
        if result.get("segment_info"):
            info = result["segment_info"]
            print(
                f"     Time: {info.get('start_time', 0):.1f}s - "
                f"{info.get('end_time', 0):.1f}s"
            )

    return results


def run_vqa(pipeline, query, shots_data, args):
    """Run VQA task."""
    result = pipeline.run_vqa(
        query=query,
        video_path=args.video,
        shots_data=shots_data,
        video_dir=args.video_dir,
        top_n=args.top_n,
        top_k=args.top_k,
    )

    print(f"\n{'='*60}")
    print(f"VQA Results for: '{query}'")
    print(f"{'='*60}")
    print(f"  Answer: {result.answer}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Sources: {len(result.sources)} chunks")
    for i, src in enumerate(result.sources[:5]):
        print(
            f"    {i+1}. Time: {src.get('start_time', 0):.1f}s - "
            f"{src.get('end_time', 0):.1f}s "
            f"(conf: {src.get('confidence', 0):.3f})"
        )

    return {
        "query": query,
        "answer": result.answer,
        "confidence": result.confidence,
        "sources": result.sources,
    }


def interactive_mode(pipeline, shots_data, args):
    """Run interactive query mode."""
    print(f"\n{'='*60}")
    print("VRAG Interactive Mode")
    print("Type 'quit' to exit, 'kis' or 'vqa' to switch task type")
    print(f"{'='*60}")

    task = args.task

    while True:
        try:
            user_input = input(f"\n[{task.upper()}] Enter query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() in ("kis", "vqa"):
            task = user_input.lower()
            print(f"Switched to {task.upper()} mode")
            continue

        if task == "kis":
            run_kis(pipeline, user_input, shots_data, args)
        else:
            run_vqa(pipeline, user_input, shots_data, args)


def main():
    args = parse_args()

    # Initialize pipeline
    logger.info("Initializing VRAG pipeline...")
    pipeline = VRAGPipeline(config_path=args.config)
    modules = [m.strip() for m in args.modules.split(",")]
    pipeline.initialize(modules=modules)

    # Load preprocessed shots data
    shots_data = load_all_shots(args.data_dir)

    # Interactive mode
    if args.interactive:
        interactive_mode(pipeline, shots_data, args)
        return

    # Batch mode
    if args.queries_file:
        with open(args.queries_file) as f:
            queries = json.load(f)

        all_results = []
        for q in queries:
            query_text = q if isinstance(q, str) else q.get("query", "")
            task = q.get("task", args.task) if isinstance(q, dict) else args.task

            if task == "kis":
                result = run_kis(pipeline, query_text, shots_data, args)
            else:
                result = run_vqa(pipeline, query_text, shots_data, args)
            all_results.append(result)

        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info(f"Results saved to {args.output_file}")

        return

    # Single query mode
    if not args.query:
        logger.error("No query specified. Use --query, --queries_file, or --interactive")
        sys.exit(1)

    if args.task == "kis":
        results = run_kis(pipeline, args.query, shots_data, args)
    else:
        results = run_vqa(pipeline, args.query, shots_data, args)

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
