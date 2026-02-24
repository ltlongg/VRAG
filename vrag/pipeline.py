"""
VRAG Main Pipeline

End-to-end pipeline orchestrating all VRAG modules: 
Preprocessing → Indexing → Retrieval → Re-ranking → VQA

From the paper: "We present VRAG (Retrieval-Augmented Video Question Answering 
for Long-Form Videos), a modular system with three main components: 
(1) a Multi-modal Retrieval System, (2) a Re-ranking Module, and 
(3) a Video Question Answering (VQA) Module."
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from vrag.utils.config import load_config
from vrag.utils.video_utils import (
    Shot, VideoSegment, VQAQuery, VRAGResult, save_shots, load_shots
)

logger = logging.getLogger(__name__)


class VRAGPipeline:
    """
    Main VRAG Pipeline integrating all modules.
    
    Supports two primary tasks:
    1. KIS (Known-Item Search): Find a specific video segment from a text query
    2. VQA (Video Question Answering): Answer questions about long-form videos
    
    Architecture:
        Query → Query Decomposition (VQA only)
              → Multi-modal Retrieval System
                  - Semantic Search (CLIP, BLIP-2, BEiT-3, InternVL)
                  - On-screen Text Search (OCR)
                  - Audio Search (Whisper)
                  - Object Filtering (Co-DETR)
                  - Temporal Search
              → Re-ranking Module (MLLM-based)
              → VQA Module (Filtering + Answering)
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Args:
            config_path: Path to YAML configuration file.
        """
        self.config = load_config(config_path)
        self._modules = {}
        self._initialized = False

    def initialize(self, modules: List[str] = None):
        """
        Initialize pipeline modules.

        Args:
            modules: Specific modules to init. If None, initializes all.
                Options: "retrieval", "reranking", "vqa", "indexing"
        """
        if modules is None:
            modules = ["retrieval", "reranking", "vqa", "indexing"]

        logger.info(f"Initializing VRAG pipeline with modules: {modules}")

        if "indexing" in modules:
            self._init_indexing()

        if "retrieval" in modules:
            self._init_retrieval()

        if "reranking" in modules:
            self._init_reranking()

        if "vqa" in modules:
            self._init_vqa()

        self._initialized = True
        logger.info("VRAG pipeline initialized successfully")

    def _init_indexing(self):
        """Initialize the indexing module."""
        from vrag.indexing.index_builder import IndexBuilder

        idx_cfg = self.config.get("indexing", {})
        faiss_cfg = idx_cfg.get("faiss", {})
        self._modules["index_builder"] = IndexBuilder(
            index_dir=self.config.get("dataset", {}).get("index_dir", "data/indices"),
            index_type=faiss_cfg.get("index_type", "IVFFlat"),
            nlist=faiss_cfg.get("nlist", 256),
        )

        # Try to load existing indices
        try:
            self._modules["index_builder"].load()
        except Exception as e:
            logger.info(f"No existing indices found: {e}")

    def _init_retrieval(self):
        """Initialize retrieval modules."""
        from vrag.retrieval.semantic_search import SemanticSearch
        from vrag.retrieval.text_search import TextSearch
        from vrag.retrieval.audio_search import AudioSearch
        from vrag.retrieval.object_filter import ObjectFilter
        from vrag.retrieval.temporal_search import TemporalSearch
        from vrag.retrieval.multimodal_retrieval import MultimodalRetrievalSystem

        ret_cfg = self.config.get("retrieval", {})
        sem_cfg = ret_cfg.get("semantic", {})

        index_builder = self._modules.get("index_builder")

        semantic_search = SemanticSearch(
            index_manager=index_builder,
            models=sem_cfg.get("models", ["clip", "blip2", "beit3", "internvl"]),
            fusion_method=sem_cfg.get("fusion_method", "late_fusion"),
            fusion_weights=sem_cfg.get("fusion_weights", {}),
            top_n=ret_cfg.get("top_n_candidates", 100),
        )

        text_search = TextSearch(
            method=ret_cfg.get("text_search", {}).get("method", "bm25"),
            top_n=ret_cfg.get("top_n_candidates", 100),
        )

        audio_search = AudioSearch(
            method=ret_cfg.get("audio_search", {}).get("method", "bm25"),
            top_n=ret_cfg.get("top_n_candidates", 100),
        )

        object_filter = ObjectFilter()
        temporal_search = TemporalSearch()

        self._modules["multimodal_retrieval"] = MultimodalRetrievalSystem(
            semantic_search=semantic_search,
            text_search=text_search,
            audio_search=audio_search,
            object_filter=object_filter,
            temporal_search=temporal_search,
            top_n=ret_cfg.get("top_n_candidates", 100),
        )

        self._modules["semantic_search"] = semantic_search
        self._modules["text_search"] = text_search
        self._modules["audio_search"] = audio_search
        self._modules["object_filter"] = object_filter
        self._modules["temporal_search"] = temporal_search

    def _init_reranking(self):
        """Initialize the re-ranking module."""
        from vrag.reranking.reranker import Reranker

        rr_cfg = self.config.get("reranking", {})

        mllm_cfg = rr_cfg.get("mllm", {})
        self._modules["reranker"] = Reranker(
            mllm_model=mllm_cfg.get("model_name", "InternVL2.5-8B"),
            context_window=rr_cfg.get("context_window", 3),
            top_k=rr_cfg.get("top_k", 10),
            max_frames_per_segment=mllm_cfg.get("max_frames_per_segment", 16),
        )

    def _init_vqa(self):
        """Initialize VQA modules."""
        from vrag.vqa.query_decomposer import QueryDecomposer
        from vrag.vqa.filtering_module import FilteringModule
        from vrag.vqa.answering_module import AnsweringModule

        vqa_cfg = self.config.get("vqa", {})
        decomp_cfg = vqa_cfg.get("query_decomposer", {})
        filt_cfg = vqa_cfg.get("filtering", {})
        ans_cfg = vqa_cfg.get("answering", {})

        self._modules["query_decomposer"] = QueryDecomposer(
            model=decomp_cfg.get("model", "kimi-for-coding"),
            api_key=decomp_cfg.get("api_key"),
            api_base=decomp_cfg.get("api_base", "https://api.kimi.com/coding/"),
        )

        filt_mllm_cfg = filt_cfg.get("mllm", {})
        self._modules["filtering"] = FilteringModule(
            mllm_model=filt_mllm_cfg.get("model_name", "VideoLLaMA3-7B"),
            chunk_size=filt_cfg.get("chunk_size_seconds", 15.0),
            chunk_overlap=filt_cfg.get("chunk_overlap_seconds", 5.0),
            max_frames_per_chunk=filt_mllm_cfg.get("max_frames_per_chunk", 8),
        )

        ans_mllm_cfg = ans_cfg.get("mllm", {})
        self._modules["answering"] = AnsweringModule(
            mllm_model=ans_mllm_cfg.get("model_name", "VideoLLaMA3-7B"),
            max_frames=ans_mllm_cfg.get("max_frames", 32),
        )

    # ========== Auto Preparation ==========

    def _auto_prepare(
        self,
        video_path: str = None,
        video_dir: str = None,
        shots_data: Dict[str, List[Shot]] = None,
    ) -> Dict[str, List[Shot]]:
        """
        Automatically preprocess video(s) and build indices if not already done.
        This enables true end-to-end usage: just call run_kis/run_vqa with a
        video path and everything is handled automatically.

        Args:
            video_path: Path to a single video file.
            video_dir: Directory of video files.
            shots_data: Pre-existing shots data (skips preprocessing if provided).

        Returns:
            shots_data dict mapping video_id -> List[Shot].
        """
        if shots_data is None:
            shots_data = {}

        # Collect video paths to process
        video_paths = []
        if video_path and os.path.isfile(video_path):
            video_paths.append(video_path)
        if video_dir and os.path.isdir(video_dir):
            import glob
            for ext in ["mp4", "avi", "mkv", "mov", "webm"]:
                video_paths.extend(glob.glob(os.path.join(video_dir, f"*.{ext}")))
            video_paths = sorted(set(video_paths))

        if not video_paths:
            return shots_data

        # Check which videos need preprocessing
        index_builder = self._modules.get("index_builder")
        has_any_index = index_builder and any(
            index_builder.has_index(m) for m in ["clip", "blip2", "beit3", "internvl"]
        )

        videos_to_process = []
        for vp in video_paths:
            vid = Path(vp).stem
            if vid not in shots_data:
                videos_to_process.append(vp)

        if not videos_to_process and has_any_index:
            # Everything already prepared
            return shots_data

        # Preprocess videos that need it
        all_preprocessed = []
        for vp in videos_to_process:
            vid = Path(vp).stem
            output_dir = str(
                Path(self.config.get("general", {}).get("output_dir", "output")) / vid
            )

            # Check if preprocessing output already exists on disk
            shots_file = os.path.join(output_dir, "shots.json")
            if os.path.exists(shots_file):
                logger.info(f"[{vid}] Loading existing preprocessed data from {output_dir}")
                loaded_shots = load_shots(shots_file)
                shots_data[vid] = loaded_shots

                # Try to load features for index building
                feat_dir = os.path.join(output_dir, "features")
                if os.path.isdir(feat_dir):
                    import numpy as np
                    import json as _json
                    features = {}
                    num_vectors = 0
                    for feat_file in Path(feat_dir).glob("*.npy"):
                        feat_array = np.load(str(feat_file))
                        features[feat_file.stem] = list(feat_array)
                        num_vectors = max(num_vectors, len(feat_array))

                    # Prefer saved metadata file for exact alignment
                    meta_file = os.path.join(feat_dir, "metadata.json")
                    if os.path.exists(meta_file):
                        with open(meta_file, "r") as mf:
                            metadata = _json.load(mf)
                    else:
                        # Fallback: derive from shot list, capped to feature count
                        metadata = [
                            {"video_id": vid, "shot_id": s.shot_id}
                            for s in loaded_shots[:num_vectors]
                        ]
                    all_preprocessed.append({
                        "video_id": vid,
                        "shots": loaded_shots,
                        "features": features,
                        "metadata": metadata,
                        "ocr": {},
                        "transcript": {"text": "", "segments": []},
                        "objects": {},
                    })
            else:
                logger.info(f"[{vid}] Auto-preprocessing video: {vp}")
                result = self.preprocess_video(video_path=vp, output_dir=output_dir)
                shots_data[vid] = result["shots"]
                all_preprocessed.append(result)

        # Build/rebuild indices if we have new data
        if all_preprocessed:
            logger.info("Auto-building search indices...")
            self.build_indices(preprocessed_data=all_preprocessed, save=True)

        return shots_data

    # ========== KIS Task ==========

    def run_kis(
        self,
        query: str,
        top_n: int = 100,
        top_k: int = 10,
        shots_data: Dict[str, List[Shot]] = None,
        video_path: str = None,
        video_dir: str = None,
    ) -> List[Dict]:
        """
        Run Known-Item Search (KIS) task.

        Fully end-to-end: if no index exists, automatically preprocesses
        video(s) and builds indices before running retrieval.

        Args:
            query: KIS query text description.
            top_n: Number of candidates from retrieval.
            top_k: Final top-K after re-ranking.
            shots_data: Video shots data (auto-generated if not provided).
            video_path: Path to video file (for auto-preprocessing).
            video_dir: Video files directory.

        Returns:
            Top-K ranked retrieval results.
        """
        logger.info(f"Running KIS task: '{query}'")
        start_time = time.time()

        # Auto-prepare: preprocess + index if needed
        shots_data = self._auto_prepare(
            video_path=video_path,
            video_dir=video_dir,
            shots_data=shots_data,
        )
        if not video_dir and video_path:
            video_dir = str(Path(video_path).parent)

        # Step 1: Multi-modal Retrieval
        retrieval = self._modules.get("multimodal_retrieval")
        if not retrieval:
            raise RuntimeError("Retrieval module not initialized")

        candidates = retrieval.search_for_kis(query, top_n=top_n)
        logger.info(f"Retrieval: {len(candidates)} candidates")

        # Step 2: Re-ranking
        reranker = self._modules.get("reranker")
        if reranker and shots_data:
            results = reranker.rerank(
                query=query,
                retrieval_results=candidates,
                shots_data=shots_data,
                video_dir=video_dir,
                top_k=top_k,
            )
            logger.info(f"Re-ranking: {len(results)} final results")
        else:
            results = candidates[:top_k]
            logger.info("Skipping re-ranking (no reranker or shots_data)")

        elapsed = time.time() - start_time
        logger.info(f"KIS completed in {elapsed:.2f}s")

        return results

    # ========== VQA Task ==========

    def run_vqa(
        self,
        query: str,
        video_path: str = None,
        shots_data: Dict[str, List[Shot]] = None,
        video_dir: str = None,
        top_n: int = 100,
        top_k: int = 10,
    ) -> VRAGResult:
        """
        Run Video Question Answering (VQA) task.

        Fully end-to-end: if no index exists, automatically preprocesses
        video(s) and builds indices before running the full pipeline.

        Pipeline (Paper Section 3.3):
        0. Auto-prepare (preprocess + build index if needed)
        1. Query Decomposition (Kimi Coding)
        2. Multi-modal Retrieval
        3. Re-ranking
        4. Filtering (chunk video, binary relevance)
        5. Answering (aggregate relevant chunks, generate answer)

        Args:
            query: User question about a video.
            video_path: Path to the target video.
            shots_data: Pre-computed shots data (auto-generated if not provided).
            video_dir: Directory containing video files.
            top_n: Retrieval candidates.
            top_k: Re-ranked candidates.

        Returns:
            VRAGResult with answer and confidence.
        """
        logger.info(f"Running VQA task: '{query}'")
        start_time = time.time()

        # Step 0: Auto-prepare - preprocess + build index if needed
        shots_data = self._auto_prepare(
            video_path=video_path,
            video_dir=video_dir,
            shots_data=shots_data,
        )
        if not video_dir and video_path:
            video_dir = str(Path(video_path).parent)

        # Step 1: Query Decomposition
        decomposer = self._modules.get("query_decomposer")
        if decomposer:
            vqa_query = decomposer.decompose(query)
            logger.info(
                f"Decomposed: retrieval='{vqa_query.retrieval_query}', "
                f"question='{vqa_query.question}'"
            )
        else:
            vqa_query = VQAQuery(
                original_query=query,
                retrieval_query=query,
                question=query,
            )

        # Step 2: Multi-modal Retrieval
        retrieval = self._modules.get("multimodal_retrieval")
        if retrieval:
            candidates = retrieval.search(
                query=vqa_query.retrieval_query,
                modalities=["semantic", "text", "audio"],
                top_n=top_n,
            )
        else:
            candidates = []

        logger.info(f"Retrieval: {len(candidates)} candidates")

        # Step 3: Re-ranking
        reranker = self._modules.get("reranker")
        if reranker and shots_data and candidates:
            reranked = reranker.rerank(
                query=vqa_query.retrieval_query,
                retrieval_results=candidates,
                shots_data=shots_data,
                video_dir=video_dir,
                top_k=top_k,
            )
        else:
            reranked = candidates[:top_k]

        logger.info(f"Re-ranking: {len(reranked)} candidates")

        # Step 4: Filtering
        filtering = self._modules.get("filtering")
        relevant_chunks = []

        if filtering and reranked:
            # Determine video path for top result
            top_video_id = reranked[0].get("video_id", "")
            if not video_path and video_dir:
                video_path = os.path.join(video_dir, f"{top_video_id}.mp4")

            if video_path and os.path.exists(video_path):
                # Get time range from re-ranked results
                start = min(
                    r.get("segment_info", {}).get("start_time", 0)
                    for r in reranked
                )
                end = max(
                    r.get("segment_info", {}).get("end_time", 300)
                    for r in reranked
                )

                relevant_chunks = filtering.filter_video(
                    query=vqa_query.question,
                    video_path=video_path,
                    start_time=start,
                    end_time=end,
                )
            else:
                logger.warning(
                    f"Video not found for filtering: {video_path}"
                )
                # Use keyframes from shots as fallback
                if shots_data:
                    for result in reranked:
                        vid = result.get("video_id", "")
                        sid = result.get("shot_id", -1)
                        all_shots = shots_data.get(vid, [])
                        shot = next(
                            (s for s in all_shots if s.shot_id == sid), None
                        )
                        if shot and shot.keyframe_paths:
                            from PIL import Image
                            frames = []
                            for p in shot.keyframe_paths[:8]:
                                try:
                                    frames.append(Image.open(p).convert("RGB"))
                                except Exception:
                                    pass
                            if frames:
                                relevant_chunks.append({
                                    "chunk_id": sid,
                                    "start_time": shot.start_time,
                                    "end_time": shot.end_time,
                                    "is_relevant": True,
                                    "confidence": result.get(
                                        "relevance_score", 0.5
                                    ),
                                    "frames": frames,
                                })
        elif video_path and os.path.exists(video_path) and filtering:
            # No retrieval results but we have a video - filter the whole video
            logger.info("No retrieval results. Filtering entire video...")
            relevant_chunks = filtering.filter_video(
                query=vqa_query.question if hasattr(vqa_query, 'question') else query,
                video_path=video_path,
            )

        logger.info(f"Filtering: {len(relevant_chunks)} relevant chunks")

        # Step 5: Answering
        answering = self._modules.get("answering")
        if answering and relevant_chunks:
            result = answering.answer(
                question=vqa_query.question,
                relevant_chunks=relevant_chunks,
            )
        else:
            result = VRAGResult(
                query=query,
                answer="Could not generate answer - no relevant content found.",
                confidence=0.0,
                task_type="vqa",
            )

        elapsed = time.time() - start_time
        logger.info(f"VQA completed in {elapsed:.2f}s. Answer: {result.answer[:100]}")

        return result

    # ========== Preprocessing ==========

    def preprocess_video(
        self,
        video_path: str,
        output_dir: str = None,
    ) -> Dict:
        """
        Run full preprocessing pipeline on a single video.

        Steps:
        1. Shot boundary detection
        2. Keyframe extraction
        3. Feature extraction (for all models)
        4. OCR extraction
        5. Audio transcription
        6. Object detection

        Args:
            video_path: Path to video file.
            output_dir: Directory for output artifacts.

        Returns:
            Dict with preprocessing results.
        """
        from vrag.preprocessing.shot_detection import ShotBoundaryDetector
        from vrag.preprocessing.keyframe_extraction import KeyframeExtractor
        from vrag.preprocessing.feature_extraction import FeatureExtractor
        from vrag.preprocessing.ocr_extraction import OCRExtractor
        from vrag.preprocessing.audio_transcription import AudioTranscriber
        from vrag.preprocessing.object_detection import ObjectDetector

        video_id = Path(video_path).stem
        if output_dir is None:
            output_dir = str(
                Path(self.config.get("general", {}).get("output_dir", "output"))
                / video_id
            )
        os.makedirs(output_dir, exist_ok=True)

        pp_cfg = self.config.get("preprocessing", {})
        results = {"video_id": video_id, "video_path": video_path}

        # 1. Shot Detection
        logger.info(f"[{video_id}] Step 1: Shot boundary detection")
        detector = ShotBoundaryDetector(
            method=pp_cfg.get("shot_detection", {}).get("method", "pyscenedetect"),
            threshold=pp_cfg.get("shot_detection", {}).get("threshold", 27.0),
        )
        shots = detector.detect(video_path, video_id)
        results["shots"] = shots
        save_shots(shots, os.path.join(output_dir, "shots.json"))
        logger.info(f"[{video_id}] Detected {len(shots)} shots")

        # 2. Keyframe Extraction
        logger.info(f"[{video_id}] Step 2: Keyframe extraction")
        kf_cfg = pp_cfg.get("keyframe_extraction", {})
        kf_extractor = KeyframeExtractor(
            method=kf_cfg.get("method", "semantic"),
            similarity_threshold=kf_cfg.get("similarity_threshold", 0.85),
            max_keyframes_per_shot=kf_cfg.get("max_keyframes_per_shot", 5),
            min_keyframes_per_shot=kf_cfg.get("min_keyframes_per_shot", 1),
        )
        keyframe_dir = os.path.join(output_dir, "keyframes")
        os.makedirs(keyframe_dir, exist_ok=True)
        for idx, shot in enumerate(shots):
            shots[idx] = kf_extractor.extract_keyframes(video_path, shot, keyframe_dir)
        results["shots"] = shots
        total_kf = sum(len(s.keyframe_paths) for s in shots)
        logger.info(f"[{video_id}] Extracted {total_kf} keyframes")

        # 3. Feature Extraction
        logger.info(f"[{video_id}] Step 3: Feature extraction")
        fe_cfg = pp_cfg.get("feature_extraction", {})
        # Derive model names from config sub-keys (clip, blip2, beit3, internvl)
        model_names = [k for k in fe_cfg if k in ("clip", "blip2", "beit3", "internvl")]
        if not model_names:
            model_names = ["clip"]
        device = self.config.get("general", {}).get("device", "cuda")
        feat_extractor = FeatureExtractor(device=device)

        features_per_model = {}
        metadata_list = []

        for shot in shots:
            if shot.keyframe_paths:
                from PIL import Image
                images = []
                for p in shot.keyframe_paths:
                    try:
                        images.append(Image.open(p).convert("RGB"))
                    except Exception:
                        pass

                if images:
                    shot_features = feat_extractor.extract_features(
                        images, model_names
                    )
                    if shot_features:
                        for model_name, feats in shot_features.items():
                            if model_name not in features_per_model:
                                features_per_model[model_name] = []
                            # Average features across keyframes for shot-level representation
                            import numpy as np
                            avg_feat = np.mean(feats, axis=0)
                            features_per_model[model_name].append(avg_feat)

                        # Only append metadata when features were actually extracted
                        metadata_list.append({
                            "video_id": video_id,
                            "shot_id": shot.shot_id,
                        })

        results["features"] = features_per_model
        results["metadata"] = metadata_list

        # Save features and metadata
        import numpy as np
        import json as _json
        feat_dir = os.path.join(output_dir, "features")
        os.makedirs(feat_dir, exist_ok=True)
        for model_name, feats in features_per_model.items():
            feat_array = np.array(feats)
            np.save(os.path.join(feat_dir, f"{model_name}.npy"), feat_array)
        # Save metadata so it can be reloaded aligned with features
        with open(os.path.join(feat_dir, "metadata.json"), "w") as f:
            _json.dump(metadata_list, f)

        logger.info(
            f"[{video_id}] Extracted features for {len(metadata_list)} shots "
            f"across {len(features_per_model)} models"
        )

        # 4. OCR Extraction
        logger.info(f"[{video_id}] Step 4: OCR extraction")
        try:
            ocr_cfg = pp_cfg.get("ocr", {})
            ocr = OCRExtractor(
                detector=ocr_cfg.get("text_detector", "deepsolo"),
                recognizer=ocr_cfg.get("text_recognizer", "parseq"),
                detection_confidence=ocr_cfg.get("detection_confidence", 0.5),
                recognition_confidence=ocr_cfg.get("recognition_confidence", 0.7),
            )
            ocr_data = {}
            for shot in shots:
                text = ocr.extract_text_for_shot(shot)  # returns str
                if text:
                    ocr_data[shot.shot_id] = text
            results["ocr"] = ocr_data
            logger.info(
                f"[{video_id}] OCR extracted for {len(ocr_data)} shots"
            )
        except Exception as e:
            logger.warning(f"[{video_id}] OCR failed: {e}")
            results["ocr"] = {}

        # 5. Audio Transcription
        logger.info(f"[{video_id}] Step 5: Audio transcription")
        try:
            audio_cfg = pp_cfg.get("audio", {})
            transcriber = AudioTranscriber(
                model_name=audio_cfg.get("model_name", "openai/whisper-large-v3"),
                language=audio_cfg.get("language"),
                batch_size=audio_cfg.get("batch_size", 16),
            )
            transcript = transcriber.transcribe_video(video_path)
            results["transcript"] = transcript
            logger.info(
                f"[{video_id}] Transcribed "
                f"{len(transcript.get('segments', []))} segments"
            )
        except Exception as e:
            logger.warning(f"[{video_id}] Transcription failed: {e}")
            results["transcript"] = {"text": "", "segments": []}

        # 6. Object Detection
        logger.info(f"[{video_id}] Step 6: Object detection")
        try:
            obj_cfg = pp_cfg.get("object_detection", {})
            obj_detector = ObjectDetector(
                model_name=obj_cfg.get("model_name", "co-detr"),
                confidence_threshold=obj_cfg.get("confidence_threshold", 0.5),
                nms_threshold=obj_cfg.get("nms_threshold", 0.5),
            )
            object_data = {}
            for shot in shots:
                objects = obj_detector.get_objects_for_shot(shot)
                if objects:
                    object_data[shot.shot_id] = objects
            results["objects"] = object_data
            logger.info(
                f"[{video_id}] Objects detected in {len(object_data)} shots"
            )
        except Exception as e:
            logger.warning(f"[{video_id}] Object detection failed: {e}")
            results["objects"] = {}

        logger.info(f"[{video_id}] Preprocessing complete!")
        return results

    def build_indices(
        self,
        preprocessed_data: List[Dict],
        save: bool = True,
    ):
        """
        Build search indices from preprocessed video data.

        Args:
            preprocessed_data: List of results from preprocess_video().
            save: Whether to save indices to disk.
        """
        import numpy as np

        index_builder = self._modules.get("index_builder")
        if not index_builder:
            self._init_indexing()
            index_builder = self._modules["index_builder"]

        # Aggregate features and metadata across all videos
        all_features = {}  # model -> list of feature vectors
        all_metadata = []
        all_ocr = {}  # video_id -> {shot_id: text}
        all_transcripts = {}  # video_id -> transcript
        all_objects = {}  # video_id -> {shot_id: [objects]}

        for data in preprocessed_data:
            video_id = data["video_id"]

            # Features
            for model_name, feats in data.get("features", {}).items():
                if model_name not in all_features:
                    all_features[model_name] = []
                all_features[model_name].extend(feats)

            all_metadata.extend(data.get("metadata", []))

            # OCR
            if data.get("ocr"):
                all_ocr[video_id] = data["ocr"]

            # Transcripts
            if data.get("transcript"):
                all_transcripts[video_id] = data["transcript"]

            # Objects
            if data.get("objects"):
                all_objects[video_id] = data["objects"]

        # Build FAISS indices
        for model_name, feats in all_features.items():
            feat_array = np.array(feats)
            index_builder.build_index(model_name, feat_array, all_metadata)

        # Build text/audio search indices
        text_search = self._modules.get("text_search")
        if text_search and all_ocr:
            text_search.build_index(all_ocr)

        audio_search = self._modules.get("audio_search")
        if audio_search and all_transcripts:
            audio_search.build_index(all_transcripts)

        object_filter = self._modules.get("object_filter")
        if object_filter and all_objects:
            object_filter.load_object_data(all_objects)

        if save:
            index_builder.save()

        logger.info(
            f"Indices built: {len(all_features)} models, "
            f"{len(all_metadata)} total vectors"
        )

    def get_module(self, name: str):
        """Get a specific module by name."""
        return self._modules.get(name)
