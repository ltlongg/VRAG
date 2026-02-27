#!/usr/bin/env python
"""
VRAG Smoke Test

Verifies that all modules can be imported and basic functionality works
without requiring GPU models or large dependencies.

Run with: python -m pytest tests/test_smoke.py -v
Or: python tests/test_smoke.py
"""

import os
import sys
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImports(unittest.TestCase):
    """Test that all modules can be imported."""

    def test_import_utils_config(self):
        from vrag.utils.config import load_config
        self.assertTrue(callable(load_config))

    def test_import_utils_video(self):
        from vrag.utils.video_utils import (
            Shot, VideoSegment, VQAQuery, VRAGResult,
            save_shots, load_shots, chunk_video,
            expand_shot_context, merge_consecutive_shots,
        )
        self.assertTrue(callable(chunk_video))

    def test_import_preprocessing(self):
        from vrag.preprocessing.shot_detection import ShotBoundaryDetector
        from vrag.preprocessing.keyframe_extraction import KeyframeExtractor
        from vrag.preprocessing.feature_extraction import FeatureExtractor
        from vrag.preprocessing.ocr_extraction import OCRExtractor
        from vrag.preprocessing.audio_transcription import AudioTranscriber
        from vrag.preprocessing.object_detection import ObjectDetector
        self.assertTrue(True)

    def test_import_retrieval(self):
        from vrag.retrieval.semantic_search import SemanticSearch
        from vrag.retrieval.text_search import TextSearch
        from vrag.retrieval.audio_search import AudioSearch
        from vrag.retrieval.object_filter import ObjectFilter
        from vrag.retrieval.temporal_search import TemporalSearch
        from vrag.retrieval.multimodal_retrieval import MultimodalRetrievalSystem
        self.assertTrue(True)

    def test_import_reranking(self):
        from vrag.reranking.reranker import Reranker
        self.assertTrue(True)

    def test_import_vqa(self):
        from vrag.vqa.query_decomposer import QueryDecomposer
        from vrag.vqa.filtering_module import FilteringModule
        from vrag.vqa.answering_module import AnsweringModule
        self.assertTrue(True)

    def test_import_indexing(self):
        from vrag.indexing.index_builder import IndexBuilder
        self.assertTrue(True)

    def test_import_pipeline(self):
        from vrag.pipeline import VRAGPipeline
        self.assertTrue(True)


class TestDataClasses(unittest.TestCase):
    """Test data class instantiation."""

    def test_shot_creation(self):
        from vrag.utils.video_utils import Shot
        shot = Shot(
            video_id="test_vid",
            shot_id=0,
            start_frame=0,
            end_frame=100,
            start_time=0.0,
            end_time=3.33,
        )
        self.assertEqual(shot.video_id, "test_vid")
        self.assertEqual(shot.shot_id, 0)
        self.assertAlmostEqual(shot.end_time, 3.33)

    def test_vqa_query_creation(self):
        from vrag.utils.video_utils import VQAQuery
        query = VQAQuery(
            original_query="What color is the car?",
            retrieval_query="car",
            question="What color is the car?",
        )
        self.assertEqual(query.original_query, "What color is the car?")

    def test_vrag_result_creation(self):
        from vrag.utils.video_utils import VRAGResult
        result = VRAGResult(
            query="test",
            answer="blue",
            confidence=0.95,
            task_type="vqa",
        )
        self.assertEqual(result.answer, "blue")


class TestConfigLoading(unittest.TestCase):
    """Test configuration loading."""

    def test_load_config(self):
        from vrag.utils.config import load_config
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "config.yaml"
        )
        if os.path.exists(config_path):
            config = load_config(config_path)
            self.assertIn("general", config)
            self.assertIn("preprocessing", config)
            self.assertIn("retrieval", config)
            self.assertIn("reranking", config)
            self.assertIn("vqa", config)
            self.assertIn("indexing", config)
        else:
            self.skipTest("config.yaml not found")


class TestChunkVideo(unittest.TestCase):
    """Test the chunk_video utility."""

    def test_chunk_video_basic(self):
        from vrag.utils.video_utils import chunk_video
        chunks = chunk_video(
            video_duration=60.0,
            chunk_size=15.0,
            overlap=5.0,
        )
        self.assertGreater(len(chunks), 0)
        # All chunks should have valid ranges
        for start, end in chunks:
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, 60.0)
            self.assertLess(start, end)

    def test_chunk_video_short(self):
        from vrag.utils.video_utils import chunk_video
        chunks = chunk_video(
            video_duration=5.0,
            chunk_size=15.0,
            overlap=5.0,
        )
        self.assertGreater(len(chunks), 0)


class TestPipelineConstruction(unittest.TestCase):
    """Test that pipeline can be constructed (no model loading)."""

    def test_pipeline_constructor(self):
        from vrag.pipeline import VRAGPipeline
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "config.yaml"
        )
        if os.path.exists(config_path):
            pipeline = VRAGPipeline(config_path=config_path)
            self.assertIsNotNone(pipeline.config)
        else:
            self.skipTest("config.yaml not found")


class TestModuleInstantiation(unittest.TestCase):
    """Test that module classes can be instantiated without loading models."""

    def test_semantic_search(self):
        from vrag.retrieval.semantic_search import SemanticSearch
        ss = SemanticSearch(top_n=50)
        self.assertEqual(ss.top_n, 50)

    def test_text_search(self):
        from vrag.retrieval.text_search import TextSearch
        ts = TextSearch(method="bm25", top_n=50)
        self.assertEqual(ts.method, "bm25")

    def test_audio_search(self):
        from vrag.retrieval.audio_search import AudioSearch
        aus = AudioSearch(method="bm25", top_n=50)
        self.assertEqual(aus.method, "bm25")

    def test_object_filter(self):
        from vrag.retrieval.object_filter import ObjectFilter
        of = ObjectFilter()
        self.assertEqual(len(of._object_cache), 0)

    def test_temporal_search(self):
        from vrag.retrieval.temporal_search import TemporalSearch
        ts = TemporalSearch()
        self.assertEqual(len(ts.shots_data), 0)

    def test_reranker(self):
        from vrag.reranking.reranker import Reranker
        rr = Reranker(
            mllm_model="OpenGVLab/InternVL2_5-1B",
            context_window=3,
            top_k=10,
        )
        self.assertEqual(rr.context_window, 3)

    def test_query_decomposer(self):
        from vrag.vqa.query_decomposer import QueryDecomposer
        qd = QueryDecomposer(model="kimi-for-coding")
        self.assertEqual(qd.model, "kimi-for-coding")

    def test_filtering_module(self):
        from vrag.vqa.filtering_module import FilteringModule
        fm = FilteringModule(
            mllm_model="DAMO-NLP-SG/VideoLLaMA3-2B",
            chunk_size=15.0,
        )
        self.assertAlmostEqual(fm.chunk_size, 15.0)

    def test_answering_module(self):
        from vrag.vqa.answering_module import AnsweringModule
        am = AnsweringModule(
            mllm_model="DAMO-NLP-SG/VideoLLaMA3-2B",
            max_frames=32,
        )
        self.assertEqual(am.max_frames, 32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
