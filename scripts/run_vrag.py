#!/usr/bin/env python
"""
VRAG Query Runner Script

Run KIS (Known-Item Search) or VQA (Video Question Answering) queries
against the VRAG system.

Usage:
    # Known-Item Search (KIS)
    python scripts/run_vrag.py --task kis --query "a person riding a bicycle near a lake"

    # Video Question Answering (VQA) 
    python scripts/run_vrag.py --task vqa --query "What color is the car?" --video data/videos/sample.mp4

    # Interactive mode
    python scripts/run_vrag.py --interactive

    # Batch queries from JSON file
    python scripts/run_vrag.py --queries_file queries.json --output_file results.json
"""

import argparse
import json
import logging
import os
import sys
import time

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_single_query(pipeline, task, query, video_path=None, top_n=100, top_k=10):
    """Run a single query and return the result."""
    start = time.time()

    if task == "kis":
        results = pipeline.run_kis(query=query, top_n=top_n, top_k=top_k)
        elapsed = time.time() - start
        return {
            "task": "kis",
            "query": query,
            "num_results": len(results),
            "elapsed_seconds": elapsed,
            "results": [
                {
                    "video_id": r.get("video_id", ""),
                    "shot_id": r.get("shot_id", -1),
                    "score": r.get("relevance_score", r.get("score", 0)),
                    "segment_info": r.get("segment_info", {}),
                }
                for r in results
            ],
        }

    elif task == "vqa":
        if not video_path:
            return {
                "error": "VQA task requires --video argument",
            }
        result = pipeline.run_vqa(query=query, video_path=video_path)
        elapsed = time.time() - start
        return {
            "task": "vqa",
            "query": query,
            "answer": result.answer,
            "confidence": result.confidence,
            "elapsed_seconds": elapsed,
            "sources": result.sources if hasattr(result, "sources") else [],
        }

    else:
        return {"error": f"Unknown task: {task}"}


def interactive_mode(pipeline):
    """Interactive query mode."""
    print("\n" + "=" * 60)
    print("  VRAG Interactive Mode")
    print("  Type 'quit' to exit, 'help' for commands")
    print("=" * 60 + "\n")

    current_task = "kis"
    current_video = None

    while True:
        try:
            prompt = f"[{current_task.upper()}] > "
            query = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() == "quit":
            print("Goodbye!")
            break
        elif query.lower() == "help":
            print("\nCommands:")
            print("  task kis     - Switch to Known-Item Search")
            print("  task vqa     - Switch to VQA mode")
            print("  video <path> - Set video for VQA")
            print("  quit         - Exit")
            print("  <anything>   - Run as query\n")
            continue
        elif query.lower().startswith("task "):
            new_task = query.split(maxsplit=1)[1].lower()
            if new_task in ("kis", "vqa"):
                current_task = new_task
                print(f"Switched to {current_task.upper()} mode")
            else:
                print(f"Unknown task: {new_task}. Use 'kis' or 'vqa'")
            continue
        elif query.lower().startswith("video "):
            current_video = query.split(maxsplit=1)[1]
            if os.path.isfile(current_video):
                print(f"Video set: {current_video}")
            else:
                print(f"Warning: file not found: {current_video}")
            continue

        # Run query
        print(f"\nProcessing {current_task.upper()} query...")
        result = run_single_query(
            pipeline, current_task, query, video_path=current_video
        )

        if "error" in result:
            print(f"Error: {result['error']}")
        elif current_task == "kis":
            print(f"\nFound {result['num_results']} results "
                  f"({result['elapsed_seconds']:.1f}s):")
            for i, r in enumerate(result["results"][:10], 1):
                print(f"  {i}. Video: {r['video_id']}, "
                      f"Shot: {r['shot_id']}, "
                      f"Score: {r['score']:.4f}")
        elif current_task == "vqa":
            print(f"\nAnswer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Time: {result['elapsed_seconds']:.1f}s")
        print()


def main():
    parser = argparse.ArgumentParser(description="VRAG Query Runner")
    parser.add_argument(
        "--task", type=str, default="kis",
        choices=["kis", "vqa"],
        help="Task type: kis (Known-Item Search) or vqa (Video QA)"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Text query"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Video path (required for VQA)"
    )
    parser.add_argument(
        "--queries_file", type=str, default=None,
        help="JSON file with batch queries"
    )
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Interactive query mode"
    )
    parser.add_argument(
        "--top_n", type=int, default=100,
        help="Number of retrieval candidates"
    )
    parser.add_argument(
        "--top_k", type=int, default=10,
        help="Number of re-ranked results"
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

    # Initialize pipeline
    from vrag.pipeline import VRAGPipeline

    logger.info("Initializing VRAG pipeline...")
    pipeline = VRAGPipeline(config_path=args.config)
    pipeline.initialize(
        modules=["retrieval", "reranking", "vqa", "indexing"]
    )

    # Interactive mode
    if args.interactive:
        interactive_mode(pipeline)
        return

    # Batch mode from file
    if args.queries_file:
        with open(args.queries_file, "r") as f:
            queries = json.load(f)

        results = []
        for i, q in enumerate(queries, 1):
            task = q.get("task", args.task)
            query = q.get("query", "")
            video = q.get("video", args.video)
            logger.info(f"[{i}/{len(queries)}] {task}: {query[:80]}...")

            result = run_single_query(
                pipeline, task, query, video_path=video,
                top_n=args.top_n, top_k=args.top_k,
            )
            results.append(result)

        if args.output_file:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {args.output_file}")
        else:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    # Single query
    if not args.query:
        parser.error("Provide --query, --queries_file, or --interactive")

    result = run_single_query(
        pipeline, args.task, args.query, video_path=args.video,
        top_n=args.top_n, top_k=args.top_k,
    )

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Result saved to {args.output_file}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
