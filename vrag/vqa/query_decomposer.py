"""
VRAG Query Decomposition Module

Decomposes user queries into retrieval-optimized and question components using LLM.
From the paper (Section 3.3): "A large language model decomposes the user query 
into retrieval_query and question, where retrieval_query is optimized for video 
retrieval and question isolates the specific question to be answered."

Uses Kimi coding (Anthropic-compatible API) as the LLM backend.
"""

import json
import logging
import os
from typing import Dict, Optional, Tuple

from vrag.utils.video_utils import VQAQuery

logger = logging.getLogger(__name__)

# Default system prompt for query decomposition
DECOMPOSITION_SYSTEM_PROMPT = """You are a query decomposition assistant for a video question answering system.

Given a user query about a video, decompose it into two parts:
1. retrieval_query: A descriptive text optimized for retrieving relevant video segments. This should describe the visual/audio content to search for.
2. question: The specific question to be answered after finding the relevant video segments.

Respond ONLY in valid JSON format:
{
    "retrieval_query": "description of visual/audio content to search for",
    "question": "the specific question to answer"
}"""

DECOMPOSITION_USER_PROMPT = """Decompose this query:
"{query}"

Remember to respond ONLY in JSON format with "retrieval_query" and "question" fields."""


class QueryDecomposer:
    """
    Decomposes user queries into retrieval and answering components.
    
    Uses Kimi coding (Anthropic-compatible API) for query decomposition.
    The decomposed query enables:
      - retrieval_query: optimized for the multi-modal retrieval system
      - question: focused question for the answering module
    """

    def __init__(
        self,
        model: str = "kimi-for-coding",
        api_key: Optional[str] = None,
        api_base: Optional[str] = "https://api.kimi.com/coding/",
        temperature: float = 0.0,
    ):
        """
        Args:
            model: LLM model for decomposition (default: kimi-for-coding).
            api_key: API key for Kimi. Falls back to KIMI_API_KEY env var,
                     then to built-in default key.
            api_base: Base URL for the Kimi API.
            temperature: Sampling temperature.
        """
        self.model = model
        self.api_key = api_key or os.environ.get(
            "KIMI_API_KEY",
            "sk-kimi-B2k7ZUdzhJQaXUnVl1KEmc9czAGzw09jwrHLqKxTbdDSO4h4ZrVhlYn6nL6xvnGS"
        )
        self.api_base = api_base
        self.temperature = temperature
        self._client = None

    def _init_client(self):
        """Initialize the Kimi (Anthropic-compatible) client."""
        if self._client is not None:
            return

        try:
            from anthropic import Anthropic

            self._client = Anthropic(
                api_key=self.api_key,
                base_url=self.api_base,
            )
            self._client_type = "kimi"
            logger.info(f"Initialized Kimi client with model: {self.model}")

        except ImportError:
            logger.warning("anthropic package not available. Using rule-based decomposition.")
            self._client_type = "rule_based"

    def decompose(self, query: str) -> VQAQuery:
        """
        Decompose a user query into retrieval_query and question.

        Args:
            query: Original user query.

        Returns:
            VQAQuery with original_query, retrieval_query, and question.
        """
        self._init_client()

        if self._client_type == "kimi":
            return self._decompose_llm(query)
        else:
            return self._decompose_rule_based(query)

    def _decompose_llm(self, query: str) -> VQAQuery:
        """Decompose using Kimi coding (Anthropic-compatible API)."""
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=256,
                system=DECOMPOSITION_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": DECOMPOSITION_USER_PROMPT.format(query=query),
                    },
                ],
            )

            content = message.content[0].text.strip()

            # Parse JSON response
            parsed = self._parse_json_response(content)

            return VQAQuery(
                original_query=query,
                retrieval_query=parsed.get("retrieval_query", query),
                question=parsed.get("question", query),
            )

        except Exception as e:
            logger.warning(f"LLM decomposition failed: {e}. Using rule-based.")
            return self._decompose_rule_based(query)

    def _decompose_rule_based(self, query: str) -> VQAQuery:
        """
        Rule-based query decomposition fallback.
        
        Heuristic rules:
        - If query contains '?', split at the question
        - Extract descriptive parts for retrieval
        - Keep question part for answering
        """
        query = query.strip()

        # Check for question marks
        if "?" in query:
            # Find the question part
            parts = query.split("?")
            question_part = parts[-1].strip() if parts[-1].strip() else parts[-2].strip() + "?"

            # For retrieval, extract the descriptive/visual part
            retrieval_parts = []
            for part in parts[:-1]:
                part = part.strip()
                if part:
                    retrieval_parts.append(part)

            retrieval_query = " ".join(retrieval_parts) if retrieval_parts else query
            question = question_part if question_part.endswith("?") else question_part + "?"

        else:
            # No question mark - treat as both retrieval and question
            retrieval_query = query
            question = query

        # Clean up common prefixes
        for prefix in ["what is", "what are", "where is", "who is", "how"]:
            if retrieval_query.lower().startswith(prefix):
                # Remove question prefix for better retrieval
                remaining = retrieval_query[len(prefix):].strip()
                if remaining:
                    retrieval_query = remaining
                break

        return VQAQuery(
            original_query=query,
            retrieval_query=retrieval_query,
            question=question if "?" in question else query,
        )

    def _parse_json_response(self, content: str) -> Dict:
        """Parse JSON from LLM response, handling various formats."""
        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        import re
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try extracting JSON object
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse JSON from response: {content[:200]}")
        return {}

    def decompose_batch(self, queries: list) -> list:
        """Decompose multiple queries."""
        return [self.decompose(q) for q in queries]
