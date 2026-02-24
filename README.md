# VRAG: Retrieval-Augmented Video Question Answering for Long-Form Videos

**Implementation of the VRAG system from CVPRW 2025**

VRAG is a modular retrieval-augmented system for video question answering over long-form video collections. It combines multi-modal retrieval, MLLM-based re-ranking, and video question answering into an end-to-end pipeline.

## Architecture

```
Query → Query Decomposition (Kimi Coding)
      → Multi-modal Retrieval System
          ├── Semantic Search (CLIP + BLIP-2 + BEiT-3 + InternVL late fusion)
          ├── On-screen Text Search (OCR → BM25)
          ├── Audio Search (Whisper → BM25)
          ├── Object Filtering (Co-DETR post-filter)
          └── Temporal Search (event relationships)
      → Re-ranking Module (MLLM relevance scoring 0-1)
      → VQA Module
          ├── Filtering (15s overlapping chunks → binary MLLM decisions)
          └── Answering (aggregate relevant chunks → MLLM generates answer)
```

## Project Structure

```
VRAG/
├── config/
│   └── config.yaml              # Central configuration
├── vrag/
│   ├── __init__.py
│   ├── pipeline.py              # Main VRAG pipeline orchestrator
│   ├── preprocessing/
│   │   ├── shot_detection.py    # Shot boundary detection (PySceneDetect/TransNetV2)
│   │   ├── keyframe_extraction.py  # Semantic keyframe selection (BEiT-3)
│   │   ├── feature_extraction.py   # Multi-model features (CLIP, BLIP-2, BEiT-3, InternVL)
│   │   ├── ocr_extraction.py      # OCR (DeepSolo+PARSeq / EasyOCR)
│   │   ├── audio_transcription.py  # Speech transcription (Whisper)
│   │   └── object_detection.py    # Object detection (Co-DETR / YOLOv8)
│   ├── retrieval/
│   │   ├── semantic_search.py     # Late-fusion semantic retrieval
│   │   ├── text_search.py         # BM25 on-screen text search
│   │   ├── audio_search.py        # BM25 audio transcript search
│   │   ├── object_filter.py       # Object-based post-filtering
│   │   ├── temporal_search.py     # Temporal constraint search
│   │   └── multimodal_retrieval.py # Unified multi-modal retrieval
│   ├── reranking/
│   │   └── reranker.py            # MLLM re-ranking (InternVL2.5-78B)
│   ├── vqa/
│   │   ├── query_decomposer.py    # Query decomposition (Kimi Coding)
│   │   ├── filtering_module.py    # Chunk-level relevance filtering
│   │   └── answering_module.py    # Answer generation (VideoLLaMA3-7B)
│   ├── indexing/
│   │   └── index_builder.py       # FAISS vector index manager
│   └── utils/
│       ├── video_utils.py         # Video I/O, data classes, utilities
│       └── config.py              # YAML configuration loader
├── scripts/
│   ├── preprocess_videos.py       # Batch video preprocessing
│   ├── build_index.py             # Build FAISS search indices
│   └── run_vrag.py                # Run KIS/VQA queries
├── requirements.txt
├── setup.py
└── README.md
```

## Quick Start

### 1. Installation

```bash
# Clone and install
cd VRAG
pip install -e ".[all]"

# Or install dependencies directly
pip install -r requirements.txt
```

### 2. Configure

Edit `config/config.yaml` to set paths and model preferences:

```yaml
general:
  device: "cuda"  # or "cpu"
  output_dir: "output"

dataset:
  video_dir: "data/videos"
```

Set your Kimi API key for query decomposition:
```bash
set KIMI_API_KEY=your-api-key-here
```

Install the Anthropic SDK (Kimi uses an Anthropic-compatible API):
```bash
pip install anthropic
```

### 3. Preprocess Videos

```bash
# Single video
python scripts/preprocess_videos.py --video path/to/video.mp4

# Directory of videos
python scripts/preprocess_videos.py --video_dir data/videos --output_dir output/

# Skip already processed videos
python scripts/preprocess_videos.py --video_dir data/videos --skip_existing
```

### 4. Build Search Indices

```bash
python scripts/build_index.py --data_dir output/ --index_dir data/indices
```

### 5. Run Queries

```bash
# Known-Item Search (KIS)
python scripts/run_vrag.py --task kis --query "a person riding a bicycle near a lake"

# Video Question Answering (VQA) 
python scripts/run_vrag.py --task vqa --query "What color is the car in the video?"

# Interactive mode
python scripts/run_vrag.py --interactive

# Batch queries from JSON file
python scripts/run_vrag.py --queries_file queries.json --output_file results.json
```

## Pipeline Usage (Python API)

```python
from vrag.pipeline import VRAGPipeline

# Initialize
pipeline = VRAGPipeline(config_path="config/config.yaml")
pipeline.initialize(modules=["retrieval", "reranking", "vqa", "indexing"])

# KIS: Find a specific video segment
results = pipeline.run_kis(
    query="a dog playing in the snow",
    top_n=100,
    top_k=10,
)

# VQA: Answer a question about a video
result = pipeline.run_vqa(
    query="How many people are in the meeting room?",
    video_path="data/videos/sample.mp4",
)
print(result.answer)
print(result.confidence)
```

## Key Models

| Component | Model | Paper Performance |
|-----------|-------|-------------------|
| Semantic Retrieval | CLIP ViT-L-14 + BLIP-2 + BEiT-3 + InternVL-G | Late fusion at shot level |
| OCR | DeepSolo + PARSeq | On-screen text extraction |
| Audio | Whisper | Speech transcription |
| Object Detection | Co-DETR | Object-based filtering |
| Re-ranking | InternVL2.5-78B | 40.5/45 KIS score |
| VQA Filtering | VideoLLaMA3-7B | Binary relevance decisions |
| VQA Answering | VideoLLaMA3-7B | 4/5 VQA score |
| Query Decomposition | Kimi Coding (Anthropic-compatible API) | retrieval_query + question |

## Design Decisions from Paper

- **Shot-level retrieval**: Features are averaged per shot (not per frame) for efficiency
- **Late fusion**: Each model retrieves independently, results combined via Reciprocal Rank Fusion
- **Re-ranking context**: Each candidate expanded by ±3 neighboring shots before MLLM scoring
- **VQA chunk size**: 15 seconds with 5s overlap (best configuration from ablation)
- **Keyframe selection**: Cosine similarity threshold (0.85) using BEiT-3 features to remove near-duplicates
- **Dataset**: V3C1 (7,475 videos, ~1,000 hours) with 2,143,361 extracted keyframes

## Evaluation

### KIS Task
Score formula: $S = 1 - \frac{r-1}{n}$ where $r$ = rank of correct result, $n$ = 10

### VQA Task
Human assessors rate answer correctness on a scale of 0-5.

## Citation

```bibtex
@inproceedings{vrag2025,
  title={VRAG: Retrieval-Augmented Video Question Answering for Long-Form Videos},
  booktitle={CVPR Workshops},
  year={2025}
}
```

## License

This is a research implementation. Please respect the licenses of all underlying models and datasets.
