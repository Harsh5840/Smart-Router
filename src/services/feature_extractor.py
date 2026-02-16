"""
Feature extraction service for query analysis
"""

import re
import asyncio
from typing import Dict, Any, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from src.models.schemas import QueryFeatures
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# PHASE 2: Feature Extraction Service
# ============================================================================


class FeatureExtractor:
    """
    Extracts features from user queries for routing decisions
    Features include: token count, domain classification, embeddings
    """

    def __init__(self):
        self.embedding_model: Optional[SentenceTransformer] = None
        self._load_embedding_model()

        # Domain keywords
        self.coding_keywords = {
            "code", "function", "class", "debug", "error", "api", "implementation",
            "algorithm", "python", "javascript", "java", "sql", "bug", "syntax",
            "variable", "loop", "array", "object", "import", "package"
        }

        self.analytical_keywords = {
            "analyze", "compare", "evaluate", "calculate", "explain", "why",
            "data", "statistics", "trend", "pattern", "insight", "metric",
            "performance", "optimization", "breakdown", "impact"
        }

        self.creative_keywords = {
            "write", "story", "create", "imagine", "design", "brainstorm",
            "idea", "creative", "poem", "blog", "article", "narrative",
            "describe", "visualize", "concept"
        }

    def _load_embedding_model(self) -> None:
        """Load sentence transformer model for embeddings"""
        try:
            # Use a lightweight model for production
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("embedding_model_loaded", model="all-MiniLM-L6-v2")
        except Exception as e:
            logger.error("embedding_model_load_failed", error=str(e))
            self.embedding_model = None

    async def extract_features(self, query: str) -> QueryFeatures:
        """
        Extract all features from a query

        Args:
            query: User query string

        Returns:
            QueryFeatures object with all extracted features
        """
        # Basic metrics
        token_count = self._count_tokens(query)
        query_length = len(query)
        word_count = len(query.split())
        sentence_count = self._count_sentences(query)

        # Domain classification
        is_coding = self._is_coding_query(query)
        is_analytical = self._is_analytical_query(query)
        is_creative = self._is_creative_query(query)
        has_code_block = self._has_code_block(query)

        features = QueryFeatures(
            token_count=token_count,
            query_length=query_length,
            word_count=word_count,
            sentence_count=sentence_count,
            is_coding=is_coding,
            is_analytical=is_analytical,
            is_creative=is_creative,
            has_code_block=has_code_block,
        )

        logger.info(
            "features_extracted",
            tokens=token_count,
            words=word_count,
            coding=is_coding,
            analytical=is_analytical,
            creative=is_creative,
        )

        return features

    def _count_tokens(self, text: str) -> int:
        """Rough token count approximation"""
        # Simple approximation: ~1.3 tokens per word on average
        return int(len(text.split()) * 1.3)

    def _count_sentences(self, text: str) -> int:
        """Count sentences in text"""
        sentences = re.split(r'[.!?]+', text)
        return len([s for s in sentences if s.strip()])

    def _is_coding_query(self, query: str) -> bool:
        """Detect if query is coding-related"""
        query_lower = query.lower()
        
        # Check for code blocks
        if '```' in query or 'def ' in query or 'function ' in query:
            return True
        
        # Check for coding keywords
        matches = sum(1 for keyword in self.coding_keywords if keyword in query_lower)
        return matches >= 2

    def _is_analytical_query(self, query: str) -> bool:
        """Detect if query requires analysis"""
        query_lower = query.lower()
        matches = sum(1 for keyword in self.analytical_keywords if keyword in query_lower)
        return matches >= 2

    def _is_creative_query(self, query: str) -> bool:
        """Detect if query is creative"""
        query_lower = query.lower()
        matches = sum(1 for keyword in self.creative_keywords if keyword in query_lower)
        return matches >= 2

    def _has_code_block(self, query: str) -> bool:
        """Check for code blocks in query"""
        return '```' in query or bool(re.search(r'(def |class |function |import )', query))

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding vector for text
        Used for semantic similarity search in Phase 6
        """
        if self.embedding_model is None:
            logger.warning("embedding_model_not_available")
            return None

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self.embedding_model.encode,
                text
            )
            return embedding.tolist()
        except Exception as e:
            logger.error("embedding_generation_failed", error=str(e))
            return None

    def calculate_complexity_score(self, features: QueryFeatures) -> float:
        """
        Calculate a complexity score from features
        Used in Phase 3 for rule-based routing

        Returns:
            Float between 0 (simple) and 1 (complex)
        """
        score = 0.0

        # Token count contribution (normalized to 0-0.3)
        token_score = min(features.token_count / 500, 1.0) * 0.3
        score += token_score

        # Sentence count contribution (normalized to 0-0.2)
        sentence_score = min(features.sentence_count / 5, 1.0) * 0.2
        score += sentence_score

        # Domain complexity (0-0.5)
        if features.is_coding or features.has_code_block:
            score += 0.3
        if features.is_analytical:
            score += 0.2

        return min(score, 1.0)


# Global feature extractor instance
feature_extractor = FeatureExtractor()
